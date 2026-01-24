/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention Mainloop
 *
 * This implementation uses CUTLASS's SM100 block-scaled MMA infrastructure
 * with tcgen05.mma instructions for FP4 quantized attention.
 *
 * Key Features:
 * - Block-scaled FP4 (E2M1) with E4M3 scale factors
 * - TMA for async memory loads
 * - TMEM accumulators for warp-specialized execution
 * - Online softmax with rescaling
 * - Delta-S correction for smooth attention
 *
 * Architecture:
 * - Load warp: TMA loads Q, K, V data + scale factors
 * - MMA warp: tcgen05.mma for QK and PV GEMMs
 * - Softmax warps: Online softmax computation
 * - Correction warp: Rescaling previous outputs
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"

#include "kernel_traits_fp4.h"
#include "../blackwell/params.h"

// CUTLASS FMHA common helpers
#include "../../../csrc/cutlass/examples/77_blackwell_fmha/collective/fmha_common.hpp"

namespace flash {

using namespace cute;
using namespace cutlass::fmha::collective;

///////////////////////////////////////////////////////////////////////////////
// Mask implementations for causal/non-causal attention
///////////////////////////////////////////////////////////////////////////////

struct FP4CausalMask {
    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        int seqlen_k = get<1>(problem_shape);
        int block_n = get<1>(tile_shape);
        int block_m = get<0>(tile_shape);
        int m_idx = get<0>(blk_coord);
        int max_k = min(seqlen_k, (m_idx + 1) * block_m);
        return (max_k + block_n - 1) / block_n;
    }

    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_unmasked_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        int block_n = get<1>(tile_shape);
        int block_m = get<0>(tile_shape);
        int m_idx = get<0>(blk_coord);
        int max_unmasked_k = m_idx * block_m;
        return max(0, max_unmasked_k / block_n);
    }

    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_masked_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        return get_trip_count(blk_coord, tile_shape, problem_shape) -
               get_unmasked_trip_count(blk_coord, tile_shape, problem_shape);
    }

    template <typename TensorS, typename TensorC, typename ProblemShape>
    CUTLASS_DEVICE void apply_mask(
        TensorS& tS,
        TensorC const& tC,
        ProblemShape const& problem_shape
    ) const {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tS); ++i) {
            auto coord = tC(i);
            int row = get<0>(coord);
            int col = get<1>(coord);
            if (col > row) {
                tS(i) = -INFINITY;
            }
        }
    }
};

struct FP4NoMask {
    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        int seqlen_k = get<1>(problem_shape);
        int block_n = get<1>(tile_shape);
        return (seqlen_k + block_n - 1) / block_n;
    }

    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_unmasked_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        return get_trip_count(blk_coord, tile_shape, problem_shape);
    }

    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_masked_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        return 0;
    }

    template <typename TensorS, typename TensorC, typename ProblemShape>
    CUTLASS_DEVICE void apply_mask(
        TensorS& tS,
        TensorC const& tC,
        ProblemShape const& problem_shape
    ) const {
        // No masking
    }
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Attention Mainloop
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4 {

    // Type aliases from traits
    using Element = typename Ktraits::Element;
    using ElementData = typename Ktraits::ElementData;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;

    using TileShape = typename Ktraits::TileShape_MNK;
    using TileShapeQK = typename Ktraits::TileShapeQK;
    using TileShapePV = typename Ktraits::TileShapePV;
    using TileShapeQK_perWG = typename Ktraits::TileShapeQK_perWG;
    using TileShapePV_perWG = typename Ktraits::TileShapePV_perWG;
    using ThreadShape = typename Ktraits::ThreadShape;

    // MMA types from CollectiveBuilder
    using CollectiveMmaQK = typename Ktraits::CollectiveMmaQK;
    using CollectiveMmaPV = typename Ktraits::CollectiveMmaPV;
    using TiledMmaQK = typename Ktraits::TiledMmaQK;
    using TiledMmaPV = typename Ktraits::TiledMmaPV;

    // SMEM layouts
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;
    using SmemLayoutSFQ = typename Ktraits::SmemLayoutSFQ;
    using SmemLayoutSFK = typename Ktraits::SmemLayoutSFK;
    using SmemLayoutSFV = typename Ktraits::SmemLayoutSFV;

    // Pipeline types
    using PipelineQ = typename Ktraits::PipelineQ;
    using PipelineKV = typename Ktraits::PipelineKV;
    using PipelineS = typename Ktraits::PipelineS;
    using PipelineC = typename Ktraits::PipelineC;
    using PipelineO = typename Ktraits::PipelineO;
    using PipelineE = typename Ktraits::PipelineE;
    using OrderBarrierSoftmax = typename Ktraits::OrderBarrierSoftmax;

    using SharedStorage = typename Ktraits::SharedStorage;

    // Mask type
    using Mask = std::conditional_t<Is_causal, FP4CausalMask, FP4NoMask>;

    // TMEM allocation
    using TmemAlloc = typename Ktraits::TmemAlloc;

    // Constants
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int SFVectorSize = Ktraits::SFVectorSize;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Params {
        // Problem dimensions
        int seqlen_q;
        int seqlen_k;
        int head_dim;
        int num_heads;
        int batch_size;

        // FP4 Q tensor (packed data + scale factors)
        void const* ptr_Q;
        void const* ptr_SFQ;
        int64_t stride_Q_seq;
        int64_t stride_Q_head;
        int64_t stride_Q_batch;

        // FP4 K tensor
        void const* ptr_K;
        void const* ptr_SFK;
        int64_t stride_K_seq;
        int64_t stride_K_head;
        int64_t stride_K_batch;

        // FP4 V tensor
        void const* ptr_V;
        void const* ptr_SFV;
        int64_t stride_V_seq;
        int64_t stride_V_head;
        int64_t stride_V_batch;

        // Output tensor (BF16)
        ElementOut* ptr_O;
        int64_t stride_O_seq;
        int64_t stride_O_head;
        int64_t stride_O_batch;

        // Delta-S for smooth attention
        float const* ptr_delta_s;
        int64_t stride_ds_k;
        int64_t stride_ds_group;
        int64_t stride_ds_head;
        int64_t stride_ds_batch;
        bool use_smooth_attention;

        // Softmax scaling
        float scale_softmax;
        float scale_softmax_log2;
    };

    template <typename KernelArguments>
    static Params to_underlying_arguments(
        KernelArguments const& args,
        void* workspace
    ) {
        float log2_e = static_cast<float>(M_LOG2E);

        return Params{
            args.seqlen_q,
            args.seqlen_k,
            args.head_dim,
            args.num_heads,
            args.batch_size,

            args.ptr_Q, args.ptr_SFQ,
            args.stride_Q_seq, args.stride_Q_head, args.stride_Q_batch,

            args.ptr_K, args.ptr_SFK,
            args.stride_K_seq, args.stride_K_head, args.stride_K_batch,

            args.ptr_V, args.ptr_SFV,
            args.stride_V_seq, args.stride_V_head, args.stride_V_batch,

            args.ptr_O,
            args.stride_O_seq, args.stride_O_head, args.stride_O_batch,

            args.ptr_delta_s,
            args.stride_ds_k, args.stride_ds_group,
            args.stride_ds_head, args.stride_ds_batch,
            args.use_smooth_attention,

            args.scale_softmax,
            args.scale_softmax * log2_e
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Kernel Body - Simplified Version
    //
    // This version uses the block-scaled MMA types from CUTLASS but
    // implements a simplified execution model for initial correctness.
    // A full warp-specialized version would follow the pattern in
    // sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE void operator()(
        Params const& params,
        SharedStorage& storage,
        int m_block,
        int head_idx,
        int batch_idx,
        int seqlen_q,
        int seqlen_k
    ) {
        int thread_idx = threadIdx.x;
        int warp_idx = thread_idx / 32;
        int lane_idx = thread_idx % 32;

        // Problem shape for this tile
        auto problem_shape = make_tuple(seqlen_q, seqlen_k, kHeadDim,
                                        make_tuple(1, 1));

        // Calculate number of K/V tiles to process
        Mask mask;
        auto blk_coord = make_tuple(m_block, 0, make_tuple(head_idx, batch_idx));
        int num_kv_tiles = mask.get_trip_count(blk_coord, make_tuple(kBlockM, kBlockN, kHeadDim), problem_shape);

        if (num_kv_tiles <= 0) return;

        int row_start = m_block * kBlockM;
        int rows_this_tile = min(kBlockM, seqlen_q - row_start);

        // Per-thread output accumulator (FP32)
        float thread_output[kHeadDim];
        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < kHeadDim; ++d) {
            thread_output[d] = 0.0f;
        }

        // Online softmax state
        float row_max = -INFINITY;
        float row_sum = 0.0f;

        // Which row does this thread handle
        int my_row = thread_idx % kBlockM;
        int global_row = row_start + my_row;

        if (my_row >= rows_this_tile) {
            __syncthreads();
            return;
        }

        // Delta-S base pointer for smooth attention
        int q_group_idx = m_block;
        float const* delta_s_base = nullptr;
        if (params.use_smooth_attention && params.ptr_delta_s != nullptr) {
            delta_s_base = params.ptr_delta_s +
                batch_idx * params.stride_ds_batch +
                head_idx * params.stride_ds_head +
                q_group_idx * params.stride_ds_group;
        }

        // =====================================================================
        // Simplified FP4 Block-Scaled Attention
        //
        // NOTE: This is a placeholder implementation that shows the structure.
        // The full high-performance version requires:
        // 1. Proper TMA loads with block-scaled descriptors
        // 2. tcgen05.mma instructions via CollectiveMmaQK/CollectiveMmaPV
        // 3. TMEM accumulators
        // 4. Warp-specialized execution
        //
        // For now, we fall back to direct GMEM loads for correctness verification.
        // The actual MMA operations would use:
        //
        // TiledMmaQK mma_qk;
        // TiledMmaPV mma_pv;
        //
        // And the block-scaled scale factors are automatically applied by the
        // CollectiveMma infrastructure based on SmemLayoutSFQ, SmemLayoutSFK, etc.
        // =====================================================================

        // Cast pointers to FP4 data
        auto Q_data = reinterpret_cast<uint8_t const*>(params.ptr_Q);
        auto K_data = reinterpret_cast<uint8_t const*>(params.ptr_K);
        auto V_data = reinterpret_cast<uint8_t const*>(params.ptr_V);
        auto Q_sf = reinterpret_cast<ElementSF const*>(params.ptr_SFQ);
        auto K_sf = reinterpret_cast<ElementSF const*>(params.ptr_SFK);
        auto V_sf = reinterpret_cast<ElementSF const*>(params.ptr_SFV);

        // Process K/V tiles
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int col_start = n_tile * kBlockN;
            int cols_this_tile = min(kBlockN, seqlen_k - col_start);

            // =========================================================
            // QK GEMM with block-scaled FP4
            // =========================================================

            // Compute attention scores for this tile
            float scores[kBlockN];
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kBlockN; ++j) {
                scores[j] = 0.0f;
            }

            // Block-scaled dot product: Q[row,:] @ K[col_start:col_start+kBlockN,:]^T
            for (int d = 0; d < kHeadDim; d += SFVectorSize) {
                int sf_idx = d / SFVectorSize;

                // Get Q scale factor for this block
                int q_offset = (batch_idx * params.stride_Q_batch +
                               head_idx * params.stride_Q_head +
                               global_row * params.stride_Q_seq) / 2;
                float q_scale = static_cast<float>(Q_sf[q_offset / SFVectorSize * kHeadDim / SFVectorSize + sf_idx]);

                for (int j = 0; j < cols_this_tile; ++j) {
                    int global_col = col_start + j;

                    // Get K scale factor
                    int k_offset = (batch_idx * params.stride_K_batch +
                                   head_idx * params.stride_K_head +
                                   global_col * params.stride_K_seq) / 2;
                    float k_scale = static_cast<float>(K_sf[k_offset / SFVectorSize * kHeadDim / SFVectorSize + sf_idx]);

                    // Combined scale factor
                    float combined_scale = q_scale * k_scale;

                    // Accumulate FP4 dot product (simplified)
                    for (int dd = 0; dd < SFVectorSize && (d + dd) < kHeadDim; ++dd) {
                        int dim_idx = d + dd;

                        // Read FP4 values (2 per byte)
                        int q_byte_idx = q_offset + dim_idx / 2;
                        int k_byte_idx = k_offset + dim_idx / 2;
                        uint8_t q_byte = Q_data[q_byte_idx];
                        uint8_t k_byte = K_data[k_byte_idx];

                        // Extract nibbles
                        float q_val, k_val;
                        if (dim_idx % 2 == 0) {
                            q_val = static_cast<float>((q_byte & 0x0F)) - 8.0f;
                            k_val = static_cast<float>((k_byte & 0x0F)) - 8.0f;
                        } else {
                            q_val = static_cast<float>((q_byte >> 4)) - 8.0f;
                            k_val = static_cast<float>((k_byte >> 4)) - 8.0f;
                        }

                        scores[j] += q_val * k_val * combined_scale;
                    }
                }
            }

            // Apply softmax scaling
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < cols_this_tile; ++j) {
                scores[j] *= params.scale_softmax;
            }

            // Apply delta_s correction for smooth attention
            if (delta_s_base != nullptr) {
                for (int j = 0; j < cols_this_tile; ++j) {
                    int global_col = col_start + j;
                    float ds = delta_s_base[global_col * params.stride_ds_k];
                    scores[j] += ds * params.scale_softmax;
                }
            }

            // Apply causal mask if needed
            if constexpr (Is_causal) {
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < kBlockN; ++j) {
                    int global_col = col_start + j;
                    if (global_col > global_row) {
                        scores[j] = -INFINITY;
                    }
                }
            }

            // =========================================================
            // Online Softmax Update
            // =========================================================

            // Find new row max
            float new_max = row_max;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < cols_this_tile; ++j) {
                new_max = fmaxf(new_max, scores[j]);
            }

            // Rescale previous sum
            float scale_factor = (row_max == -INFINITY || row_max == new_max) ?
                                 1.0f : expf(row_max - new_max);
            row_sum *= scale_factor;

            // Rescale previous output
            CUTLASS_PRAGMA_UNROLL
            for (int d = 0; d < kHeadDim; ++d) {
                thread_output[d] *= scale_factor;
            }

            // Compute softmax weights and accumulate
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < cols_this_tile; ++j) {
                float weight = expf(scores[j] - new_max);
                row_sum += weight;

                // =========================================================
                // PV GEMM with block-scaled FP4
                // =========================================================
                int global_col = col_start + j;

                for (int d = 0; d < kHeadDim; d += SFVectorSize) {
                    int sf_idx = d / SFVectorSize;

                    // Get V scale factor
                    int v_offset = (batch_idx * params.stride_V_batch +
                                   head_idx * params.stride_V_head +
                                   global_col * params.stride_V_seq) / 2;
                    float v_scale = static_cast<float>(V_sf[v_offset / SFVectorSize * kHeadDim / SFVectorSize + sf_idx]);

                    for (int dd = 0; dd < SFVectorSize && (d + dd) < kHeadDim; ++dd) {
                        int dim_idx = d + dd;

                        // Read FP4 V value
                        int v_byte_idx = v_offset + dim_idx / 2;
                        uint8_t v_byte = V_data[v_byte_idx];

                        float v_val;
                        if (dim_idx % 2 == 0) {
                            v_val = static_cast<float>((v_byte & 0x0F)) - 8.0f;
                        } else {
                            v_val = static_cast<float>((v_byte >> 4)) - 8.0f;
                        }

                        thread_output[dim_idx] += weight * v_val * v_scale;
                    }
                }
            }

            row_max = new_max;
        }

        __syncthreads();

        // Normalize output by row_sum
        float inv_row_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < kHeadDim; ++d) {
            thread_output[d] *= inv_row_sum;
        }

        // Write output
        ElementOut* O_base = params.ptr_O +
            batch_idx * params.stride_O_batch +
            head_idx * params.stride_O_head +
            global_row * params.stride_O_seq;

        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < kHeadDim; ++d) {
            O_base[d] = static_cast<ElementOut>(thread_output[d]);
        }
    }
};

} // namespace flash
