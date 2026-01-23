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

    // Helper to decode linear 4-bit value to float
    // Input: 4-bit value (0-15), mapped from [-6, +6] via (x/6 * 7.5 + 7.5)
    // Output: float in roughly [-1, +1] range (before scale factor)
    CUTLASS_DEVICE static float decode_fp4_linear(uint8_t nibble) {
        // Reverse the encoding: (nibble - 7.5) / 7.5 * 6.0
        // Simplified: (nibble - 7.5) * 0.8
        return (static_cast<float>(nibble) - 7.5f) * 0.8f;
    }

    struct Params {
        // Q data (packed uint8, 2 FP4 values per byte) and scale factors
        uint8_t const* ptr_Q;
        ElementSF const* ptr_SFQ;
        int64_t stride_Q_seq;  // In packed bytes (HeadDim/2)
        int64_t stride_Q_head;
        int64_t stride_Q_batch;

        // K data and scale factors
        uint8_t const* ptr_K;
        ElementSF const* ptr_SFK;
        int64_t stride_K_seq;
        int64_t stride_K_head;
        int64_t stride_K_batch;

        // V data and scale factors
        uint8_t const* ptr_V;
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
            reinterpret_cast<uint8_t const*>(args.ptr_Q),
            args.ptr_SFQ,
            args.stride_Q_seq,
            args.stride_Q_head,
            args.stride_Q_batch,

            reinterpret_cast<uint8_t const*>(args.ptr_K),
            args.ptr_SFK,
            args.stride_K_seq,
            args.stride_K_head,
            args.stride_K_batch,

            reinterpret_cast<uint8_t const*>(args.ptr_V),
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
        (void)thread_idx;  // Used below

        // Use local constants to avoid device code issues with class static members
        constexpr int BlockM = Ktraits::kBlockM;
        constexpr int BlockN = Ktraits::kBlockN;
        constexpr int HeadDim = Ktraits::kHeadDim;
        constexpr int NThreads = Ktraits::kNThreads;
        constexpr int SFVecSize = Ktraits::SFVectorSize;

        // Problem shape for this tile
        auto problem_shape = make_tuple(seqlen_q, seqlen_k, HeadDim,
                                        make_tuple(1, 1));

        // Calculate number of K/V tiles to process
        Mask mask;
        auto blk_coord = make_tuple(m_block, 0, make_tuple(head_idx, batch_idx));
        int num_kv_tiles = mask.get_trip_count(blk_coord, TileShapeQK{}, problem_shape);

        // Early exit if no tiles to process
        if (num_kv_tiles <= 0) return;

        // Row start in Q for this block
        int row_start = m_block * BlockM;
        int rows_this_tile = min(BlockM, seqlen_q - row_start);

        // =====================================================================
        // SIMPLIFIED FP4 ATTENTION IMPLEMENTATION
        // This uses direct GMEM access instead of TMA for initial correctness.
        // Full warp-specialized TMA implementation will follow.
        // =====================================================================

        // Per-thread accumulator for output (FP32)
        // Each thread handles one row
        float thread_output[HeadDim];

        // Initialize output to zero
        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < HeadDim; ++d) {
            thread_output[d] = 0.0f;
        }

        // Online softmax state per row handled by this thread
        float row_max = -INFINITY;
        float row_sum = 0.0f;

        // Which row does this thread handle
        int my_row = thread_idx % BlockM;
        int global_row = row_start + my_row;

        if (my_row >= rows_this_tile) {
            // This thread doesn't have valid work
            __syncthreads();
            return;
        }

        // Compute base pointers for this batch/head
        // Data is packed as uint8 (2 FP4 values per byte), strides are in packed bytes
        uint8_t const* Q_base = params.ptr_Q +
            batch_idx * params.stride_Q_batch +
            head_idx * params.stride_Q_head +
            global_row * params.stride_Q_seq;

        // Scale factor layout is [batch, heads, seqlen, num_sf] where num_sf = HeadDim/16 = 16
        // SF strides: seq_stride=16, head_stride=seqlen*16, batch_stride=heads*seqlen*16
        constexpr int NumSF = HeadDim / SFVecSize;  // 16
        int64_t sf_seq_stride = NumSF;
        int64_t sf_head_stride = seqlen_q * sf_seq_stride;
        int64_t sf_batch_stride = (params.stride_Q_batch / params.stride_Q_seq) * sf_head_stride;

        ElementSF const* SFQ_base = params.ptr_SFQ +
            batch_idx * sf_batch_stride +
            head_idx * sf_head_stride +
            global_row * sf_seq_stride;

        // Loop over K/V tiles
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int k_start = n_tile * BlockN;
            int cols_this_tile = min(BlockN, seqlen_k - k_start);

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
                uint8_t const* K_base = params.ptr_K +
                    batch_idx * params.stride_K_batch +
                    head_idx * params.stride_K_head +
                    global_k * params.stride_K_seq;

                // K scale factors - same layout as Q
                int64_t sfk_seq_stride = NumSF;
                int64_t sfk_head_stride = seqlen_k * sfk_seq_stride;
                int64_t sfk_batch_stride = (params.stride_K_batch / params.stride_K_seq) * sfk_head_stride;

                ElementSF const* SFK_base = params.ptr_SFK +
                    batch_idx * sfk_batch_stride +
                    head_idx * sfk_head_stride +
                    global_k * sfk_seq_stride;

                // Block-scaled dot product: sum over blocks
                // Data is packed: 2 FP4 values per byte, HeadDim/2 bytes per row
                float dot = 0.0f;
                for (int blk = 0; blk < HeadDim / SFVecSize; ++blk) {
                    // Get scale factors for this block
                    float sf_q = static_cast<float>(SFQ_base[blk]);
                    float sf_k = static_cast<float>(SFK_base[blk]);
                    float scale = sf_q * sf_k;

                    // Dot product within block (16 FP4 values = 8 packed bytes)
                    for (int i = 0; i < SFVecSize; i += 2) {
                        int byte_idx = (blk * SFVecSize + i) / 2;
                        uint8_t q_packed = Q_base[byte_idx];
                        uint8_t k_packed = K_base[byte_idx];

                        // Unpack low nibble (even index)
                        float q_lo = decode_fp4_linear(q_packed & 0x0F);
                        float k_lo = decode_fp4_linear(k_packed & 0x0F);
                        dot += q_lo * k_lo * scale;

                        // Unpack high nibble (odd index)
                        float q_hi = decode_fp4_linear((q_packed >> 4) & 0x0F);
                        float k_hi = decode_fp4_linear((k_packed >> 4) & 0x0F);
                        dot += q_hi * k_hi * scale;
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
                for (int d = 0; d < HeadDim; ++d) {
                    thread_output[d] *= correction;
                }

                // Add contribution from this K position
                // P[my_row, k_col] = exp(s - row_max)
                float p = expf(s - row_max);

                // V contribution: O += P * V
                uint8_t const* V_base = params.ptr_V +
                    batch_idx * params.stride_V_batch +
                    head_idx * params.stride_V_head +
                    global_k * params.stride_V_seq;

                // V scale factors - same layout as K
                ElementSF const* SFV_base = params.ptr_SFV +
                    batch_idx * sfk_batch_stride +
                    head_idx * sfk_head_stride +
                    global_k * sfk_seq_stride;

                for (int blk = 0; blk < HeadDim / SFVecSize; ++blk) {
                    float sf_v = static_cast<float>(SFV_base[blk]);
                    for (int i = 0; i < SFVecSize; i += 2) {
                        int byte_idx = (blk * SFVecSize + i) / 2;
                        uint8_t v_packed = V_base[byte_idx];

                        // Unpack and accumulate
                        int d_lo = blk * SFVecSize + i;
                        int d_hi = d_lo + 1;
                        float v_lo = decode_fp4_linear(v_packed & 0x0F) * sf_v;
                        float v_hi = decode_fp4_linear((v_packed >> 4) & 0x0F) * sf_v;
                        thread_output[d_lo] += p * v_lo;
                        thread_output[d_hi] += p * v_hi;
                    }
                }
            }
        }

        // Normalize output by row_sum
        if (row_sum > 0.0f) {
            float inv_sum = 1.0f / row_sum;
            CUTLASS_PRAGMA_UNROLL
            for (int d = 0; d < HeadDim; ++d) {
                thread_output[d] *= inv_sum;
            }
        }

        __syncthreads();

        // Write output to GMEM
        // Each thread writes one row (thread_idx % BlockM)
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
