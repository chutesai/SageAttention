/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Mainloop for FlashAttention.
 *
 * This implements FP4 attention using SM100's block-scaled tcgen05.mma
 * instructions. Uses CUTLASS CollectiveMma infrastructure.
 *
 * Key constraints:
 * - HeadDim = 256 (FP4 MMA K=256 requirement)
 * - BlockN = 256 (FP4 PV matmul K=256 requirement)
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "kernel_traits_fp4.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Causal mask for FP4 attention
///////////////////////////////////////////////////////////////////////////////

struct CausalMaskFP4 {
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

struct NoMaskFP4 {
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

    using Element = typename Ktraits::Element;
    using ElementData = typename Ktraits::ElementData;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;

    using TileShapeQK = typename Ktraits::TileShapeQK;
    using TileShapePV = typename Ktraits::TileShapePV;

    // CollectiveMma types from kernel traits
    using CollectiveMmaQK = typename Ktraits::CollectiveMmaQK;
    using CollectiveMmaPV = typename Ktraits::CollectiveMmaPV;

    using TiledMmaQK = typename Ktraits::TiledMmaQK;
    using TiledMmaPV = typename Ktraits::TiledMmaPV;

    // SMEM layouts
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;
    using SmemLayoutSFA = typename Ktraits::SmemLayoutSFA;
    using SmemLayoutSFB_QK = typename Ktraits::SmemLayoutSFB_QK;
    using SmemLayoutSFB_PV = typename Ktraits::SmemLayoutSFB_PV;

    // Strides
    using StrideQ = typename Ktraits::StrideQ;
    using StrideK = typename Ktraits::StrideK;
    using StrideV = typename Ktraits::StrideV;
    using LayoutSFA = typename Ktraits::LayoutSFA;
    using LayoutSFB = typename Ktraits::LayoutSFB;

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kNThreads = Ktraits::kNThreads;

    // Scale factor configuration
    static constexpr int kSFVectorSize = Ktraits::SFVectorSize;

    // Mask type
    using Mask = std::conditional_t<Is_causal, CausalMaskFP4, NoMaskFP4>;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters - simplified for FP4 attention
    // TMA construction is complex for block-scaled ops; we use a simpler
    // direct GMEM approach initially to get functional correctness
    ///////////////////////////////////////////////////////////////////////////

    struct Params {
        // Q data and scale factors
        ElementData const* ptr_Q;
        ElementSF const* ptr_SFQ;
        int64_t stride_Q_seq;
        int64_t stride_Q_head;
        int64_t stride_Q_batch;

        // K data and scale factors
        ElementData const* ptr_K;
        ElementSF const* ptr_SFK;
        int64_t stride_K_seq;
        int64_t stride_K_head;
        int64_t stride_K_batch;

        // V data and scale factors
        ElementData const* ptr_V;
        ElementSF const* ptr_SFV;
        int64_t stride_V_seq;
        int64_t stride_V_head;
        int64_t stride_V_batch;

        // Output tensor
        ElementOut* ptr_O;
        int64_t stride_O_seq;
        int64_t stride_O_head;
        int64_t stride_O_batch;

        float scale_softmax;
        float scale_softmax_log2;
    };

    // Forward declaration for kernel Arguments type
    template <typename KernelArguments>
    static Params to_underlying_arguments(
        KernelArguments const& args,
        void* workspace
    ) {
        float log2_e = static_cast<float>(M_LOG2E);

        return Params{
            args.ptr_Q,
            args.ptr_SFQ,
            args.stride_Q_seq,
            args.stride_Q_head,
            args.stride_Q_batch,

            args.ptr_K,
            args.ptr_SFK,
            args.stride_K_seq,
            args.stride_K_head,
            args.stride_K_batch,

            args.ptr_V,
            args.ptr_SFV,
            args.stride_V_seq,
            args.stride_V_head,
            args.stride_V_batch,

            args.ptr_O,
            args.stride_O_seq,
            args.stride_O_head,
            args.stride_O_batch,

            args.scale_softmax,
            args.scale_softmax * log2_e
        };
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        // No TMA descriptors in simplified version
        // Will be added when we implement TMA-based loading
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Kernel Body
    ///////////////////////////////////////////////////////////////////////////

    template <typename SharedStorage>
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
        int num_kv_tiles = mask.get_trip_count(blk_coord, TileShapeQK{}, problem_shape);

        // Early exit if no tiles to process
        if (num_kv_tiles <= 0) return;

        // Row start in Q for this block
        int row_start = m_block * kBlockM;
        int rows_this_tile = min(kBlockM, seqlen_q - row_start);

        // =====================================================================
        // SIMPLIFIED FP4 ATTENTION IMPLEMENTATION
        // This uses direct GMEM access instead of TMA for initial correctness.
        // Full warp-specialized TMA implementation will follow.
        // =====================================================================

        // Per-thread accumulator for output (FP32)
        // Each thread handles a subset of the output elements
        constexpr int kRowsPerThread = kBlockM / kNThreads * kHeadDim;
        float thread_output[kHeadDim];  // One row per thread for simplicity

        // Initialize output to zero
        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < kHeadDim; ++d) {
            thread_output[d] = 0.0f;
        }

        // Online softmax state per row handled by this thread
        float row_max = -INFINITY;
        float row_sum = 0.0f;

        // Which row does this thread handle
        int my_row = thread_idx % kBlockM;
        int global_row = row_start + my_row;

        if (my_row >= rows_this_tile) {
            // This thread doesn't have valid work
            __syncthreads();
            return;
        }

        // Compute base pointers for this batch/head
        ElementData const* Q_base = params.ptr_Q +
            batch_idx * params.stride_Q_batch +
            head_idx * params.stride_Q_head +
            global_row * params.stride_Q_seq;

        ElementSF const* SFQ_base = params.ptr_SFQ +
            batch_idx * params.stride_Q_batch / kSFVectorSize +
            head_idx * params.stride_Q_head / kSFVectorSize +
            global_row * params.stride_Q_seq / kSFVectorSize;

        // Loop over K/V tiles
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int k_start = n_tile * kBlockN;
            int cols_this_tile = min(kBlockN, seqlen_k - k_start);

            // For causal: check if this K tile has any valid positions
            if constexpr (Is_causal) {
                if (k_start > global_row) {
                    continue;  // Skip tiles entirely after causal boundary
                }
            }

            // Compute S = Q @ K^T for this tile
            // For each valid K position in this tile
            for (int k_col = 0; k_col < cols_this_tile; ++k_col) {
                int global_k = k_start + k_col;

                // Causal mask check
                if constexpr (Is_causal) {
                    if (global_k > global_row) {
                        continue;
                    }
                }

                // Compute dot product Q[my_row] @ K[k_col]
                ElementData const* K_base = params.ptr_K +
                    batch_idx * params.stride_K_batch +
                    head_idx * params.stride_K_head +
                    global_k * params.stride_K_seq;

                ElementSF const* SFK_base = params.ptr_SFK +
                    batch_idx * params.stride_K_batch / kSFVectorSize +
                    head_idx * params.stride_K_head / kSFVectorSize +
                    global_k * params.stride_K_seq / kSFVectorSize;

                // Block-scaled dot product: sum over blocks
                float dot = 0.0f;
                for (int blk = 0; blk < kHeadDim / kSFVectorSize; ++blk) {
                    // Get scale factors for this block
                    float sf_q = static_cast<float>(SFQ_base[blk]);
                    float sf_k = static_cast<float>(SFK_base[blk]);
                    float scale = sf_q * sf_k;

                    // Dot product within block (FP4 values)
                    for (int i = 0; i < kSFVectorSize; ++i) {
                        int idx = blk * kSFVectorSize + i;
                        float q_val = static_cast<float>(Q_base[idx]);
                        float k_val = static_cast<float>(K_base[idx]);
                        dot += q_val * k_val * scale;
                    }
                }

                // Apply softmax scale
                float s = dot * params.scale_softmax;

                // Online softmax update
                float old_max = row_max;
                row_max = fmaxf(row_max, s);
                float correction = expf(old_max - row_max);
                row_sum = row_sum * correction + expf(s - row_max);

                // Correct previous output accumulator
                CUTLASS_PRAGMA_UNROLL
                for (int d = 0; d < kHeadDim; ++d) {
                    thread_output[d] *= correction;
                }

                // Add contribution from this K position
                // P[my_row, k_col] = exp(s - row_max)
                float p = expf(s - row_max);

                // V contribution: O += P * V
                ElementData const* V_base = params.ptr_V +
                    batch_idx * params.stride_V_batch +
                    head_idx * params.stride_V_head +
                    global_k * params.stride_V_seq;

                ElementSF const* SFV_base = params.ptr_SFV +
                    batch_idx * params.stride_V_batch / kSFVectorSize +
                    head_idx * params.stride_V_head / kSFVectorSize +
                    global_k * params.stride_V_seq / kSFVectorSize;

                for (int blk = 0; blk < kHeadDim / kSFVectorSize; ++blk) {
                    float sf_v = static_cast<float>(SFV_base[blk]);
                    for (int i = 0; i < kSFVectorSize; ++i) {
                        int d = blk * kSFVectorSize + i;
                        float v_val = static_cast<float>(V_base[d]) * sf_v;
                        thread_output[d] += p * v_val;
                    }
                }
            }
        }

        // Normalize output by row_sum
        if (row_sum > 0.0f) {
            float inv_sum = 1.0f / row_sum;
            CUTLASS_PRAGMA_UNROLL
            for (int d = 0; d < kHeadDim; ++d) {
                thread_output[d] *= inv_sum;
            }
        }

        __syncthreads();

        // Write output to GMEM
        // Each thread writes one row (thread_idx % kBlockM)
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
