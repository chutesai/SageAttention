/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention Mainloop
 *
 * Simplified implementation for FP4 block-scaled attention.
 * This is a scalar fallback implementation that demonstrates the structure.
 * A full high-performance implementation would use:
 * - TMA for async loads
 * - tcgen05.mma via TiledMmaQK/TiledMmaPV
 * - TMEM accumulators
 * - Warp-specialized execution
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"

#include "kernel_traits_fp4.h"

namespace flash {

using namespace cute;

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
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Attention Mainloop
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4 {

    // Type aliases from traits
    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using SharedStorage = typename Ktraits::SharedStorage;

    // Mask type
    using Mask = std::conditional_t<Is_causal, FP4CausalMask, FP4NoMask>;

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
        Element const* ptr_Q;
        ElementSF const* ptr_SFQ;
        int64_t stride_Q_seq;
        int64_t stride_Q_head;
        int64_t stride_Q_batch;

        // FP4 K tensor
        Element const* ptr_K;
        ElementSF const* ptr_SFK;
        int64_t stride_K_seq;
        int64_t stride_K_head;
        int64_t stride_K_batch;

        // FP4 V tensor
        Element const* ptr_V;
        ElementSF const* ptr_SFV;
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
    // FP4 Decode Helpers
    //
    // FP4 E2M1 format:
    // - 1 sign bit, 2 exponent bits, 1 mantissa bit
    // - Values: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float decode_fp4(uint8_t packed, int which) {
        // Lookup table for FP4 E2M1 -> FP32 conversion
        // Index 0-15 maps to the 16 possible FP4 values
        // Defined inside function to avoid CUDA device code issues with static constexpr
        constexpr float fp4_lut[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        // Extract nibble (which = 0 for low nibble, 1 for high nibble)
        uint8_t nibble = which ? (packed >> 4) : (packed & 0x0F);
        return fp4_lut[nibble];
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Kernel Body - Simplified Scalar Implementation
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
        // Local constexpr copies to avoid CUDA device code ODR issues
        constexpr int HeadDim = Ktraits::kHeadDim;
        constexpr int BlockM = Ktraits::kBlockM;
        constexpr int BlockN = Ktraits::kBlockN;
        constexpr int SFVecSize = Ktraits::SFVectorSize;

        int thread_idx = threadIdx.x;

        // Problem shape for this tile
        auto problem_shape = make_tuple(seqlen_q, seqlen_k, HeadDim,
                                        make_tuple(1, 1));

        // Calculate number of K/V tiles to process
        Mask mask;
        auto blk_coord = make_tuple(m_block, 0, make_tuple(head_idx, batch_idx));
        int num_kv_tiles = mask.get_trip_count(blk_coord, make_tuple(BlockM, BlockN, HeadDim), problem_shape);

        if (num_kv_tiles <= 0) return;

        int row_start = m_block * BlockM;
        int rows_this_tile = min(BlockM, seqlen_q - row_start);

        // Which row does this thread handle
        int my_row = thread_idx % BlockM;
        int global_row = row_start + my_row;

        if (my_row >= rows_this_tile) {
            __syncthreads();
            return;
        }

        // Per-thread output accumulator (FP32)
        float thread_output[HeadDim];
        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < HeadDim; ++d) {
            thread_output[d] = 0.0f;
        }

        // Online softmax state
        float row_max = -INFINITY;
        float row_sum = 0.0f;

        // Delta-S base pointer for smooth attention
        int q_group_idx = m_block;
        float const* delta_s_base = nullptr;
        if (params.use_smooth_attention && params.ptr_delta_s != nullptr) {
            delta_s_base = params.ptr_delta_s +
                batch_idx * params.stride_ds_batch +
                head_idx * params.stride_ds_head +
                q_group_idx * params.stride_ds_group;
        }

        // Cast to byte pointers for packed FP4 access
        auto Q_data = reinterpret_cast<uint8_t const*>(params.ptr_Q);
        auto K_data = reinterpret_cast<uint8_t const*>(params.ptr_K);
        auto V_data = reinterpret_cast<uint8_t const*>(params.ptr_V);

        // Process K/V tiles
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int col_start = n_tile * BlockN;
            int cols_this_tile = min(BlockN, seqlen_k - col_start);

            // =========================================================
            // QK GEMM with block-scaled FP4
            // =========================================================

            float scores[BlockN];
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < BlockN; ++j) {
                scores[j] = 0.0f;
            }

            // Block-scaled dot product: Q[row,:] @ K[col_start:col_start+BlockN,:]^T
            for (int sf_block = 0; sf_block < HeadDim / SFVecSize; ++sf_block) {
                int d_start = sf_block * SFVecSize;

                // Get Q scale factor for this block
                int q_sf_offset = (batch_idx * params.stride_Q_batch +
                                   head_idx * params.stride_Q_head +
                                   global_row * params.stride_Q_seq) / SFVecSize + sf_block;
                float q_scale = static_cast<float>(params.ptr_SFQ[q_sf_offset]);

                for (int j = 0; j < cols_this_tile; ++j) {
                    int global_col = col_start + j;

                    // Get K scale factor for this block
                    int k_sf_offset = (batch_idx * params.stride_K_batch +
                                       head_idx * params.stride_K_head +
                                       global_col * params.stride_K_seq) / SFVecSize + sf_block;
                    float k_scale = static_cast<float>(params.ptr_SFK[k_sf_offset]);

                    // Combined scale factor
                    float combined_scale = q_scale * k_scale;

                    // Accumulate FP4 dot product within this SF block
                    float block_sum = 0.0f;
                    for (int dd = 0; dd < SFVecSize; dd += 2) {
                        int dim_idx = d_start + dd;

                        // Read packed FP4 values (2 per byte)
                        int q_byte_offset = (batch_idx * params.stride_Q_batch +
                                            head_idx * params.stride_Q_head +
                                            global_row * params.stride_Q_seq) / 2 + dim_idx / 2;
                        int k_byte_offset = (batch_idx * params.stride_K_batch +
                                            head_idx * params.stride_K_head +
                                            global_col * params.stride_K_seq) / 2 + dim_idx / 2;

                        uint8_t q_byte = Q_data[q_byte_offset];
                        uint8_t k_byte = K_data[k_byte_offset];

                        // Decode and accumulate both values in the byte
                        block_sum += decode_fp4(q_byte, 0) * decode_fp4(k_byte, 0);
                        block_sum += decode_fp4(q_byte, 1) * decode_fp4(k_byte, 1);
                    }
                    scores[j] += block_sum * combined_scale;
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
                for (int j = 0; j < BlockN; ++j) {
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

            // Rescale previous sum and output
            float scale_factor = (row_max == -INFINITY || row_max == new_max) ?
                                 1.0f : expf(row_max - new_max);
            row_sum *= scale_factor;

            CUTLASS_PRAGMA_UNROLL
            for (int d = 0; d < HeadDim; ++d) {
                thread_output[d] *= scale_factor;
            }

            // Compute softmax weights and accumulate PV
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < cols_this_tile; ++j) {
                float weight = expf(scores[j] - new_max);
                row_sum += weight;

                // =========================================================
                // PV GEMM with block-scaled FP4
                // =========================================================
                int global_col = col_start + j;

                for (int sf_block = 0; sf_block < HeadDim / SFVecSize; ++sf_block) {
                    int d_start = sf_block * SFVecSize;

                    // Get V scale factor for this block
                    int v_sf_offset = (batch_idx * params.stride_V_batch +
                                       head_idx * params.stride_V_head +
                                       global_col * params.stride_V_seq) / SFVecSize + sf_block;
                    float v_scale = static_cast<float>(params.ptr_SFV[v_sf_offset]);

                    for (int dd = 0; dd < SFVecSize; dd += 2) {
                        int dim_idx = d_start + dd;

                        // Read packed FP4 V values
                        int v_byte_offset = (batch_idx * params.stride_V_batch +
                                            head_idx * params.stride_V_head +
                                            global_col * params.stride_V_seq) / 2 + dim_idx / 2;
                        uint8_t v_byte = V_data[v_byte_offset];

                        thread_output[dim_idx] += weight * decode_fp4(v_byte, 0) * v_scale;
                        thread_output[dim_idx + 1] += weight * decode_fp4(v_byte, 1) * v_scale;
                    }
                }
            }

            row_max = new_max;
        }

        __syncthreads();

        // Normalize output by row_sum
        float inv_row_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < HeadDim; ++d) {
            thread_output[d] *= inv_row_sum;
        }

        // Write output
        ElementOut* O_base = params.ptr_O +
            batch_idx * params.stride_O_batch +
            head_idx * params.stride_O_head +
            global_row * params.stride_O_seq;

        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < HeadDim; ++d) {
            O_base[d] = static_cast<ElementOut>(thread_output[d]);
        }
    }
};

} // namespace flash
