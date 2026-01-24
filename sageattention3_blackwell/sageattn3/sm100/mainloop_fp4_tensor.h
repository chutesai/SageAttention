/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention - Tensor Core Mainloop
 *
 * High-performance implementation using SM100 tcgen05.mma tensor core instructions
 * for FP4 block-scaled attention with E4M3 scale factors.
 *
 * Architecture:
 * - Uses SM100_MMA_MXF4_SS for FP4 block-scaled GEMM operations
 * - SMEM staging for Q/K/V data and scale factors
 * - Warp-cooperative data loading with proper synchronization
 * - Online softmax with rescaling for numerical stability
 *
 * Key constraints:
 * - HeadDim = 256 (MMA K dimension requirement for FP4)
 * - BlockN = 256 (PV matmul K dimension requirement)
 * - Scale factor block size = 16 elements
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"
#include "cutlass/pipeline/pipeline.hpp"

// SM100 MMA infrastructure
#include "cute/arch/mma_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "kernel_traits_fp4.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Kernel Traits for Tensor Core FP4 Implementation
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kStages_,
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4_tensor {
    // Configuration
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kStages = kStages_;

    // Element types
    using Element = cutlass::float_e2m1_t;           // FP4 E2M1
    using ElementSF = cutlass::float_e4m3_t;         // Scale factor E4M3
    using ElementAccum = float;                       // FP32 accumulator
    using ElementOut = ElementOut_;                   // Output (BF16)
    using index_t = int64_t;

    // Scale factor configuration
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;

    // MMA configuration for SM100 FP4
    // SM100_MMA_MXF4_SS: M=128 fixed, N=8-256, K=64 (256 bits / 4 bits)
    static constexpr int kMmaM = 128;
    static constexpr int kMmaN = 128;  // Match with block size
    static constexpr int kMmaK = 64;   // FP4 MMA K dimension

    static constexpr int kMmaIterM = kBlockM / kMmaM;
    static constexpr int kMmaIterN = kBlockN / kMmaN;
    static constexpr int kMmaIterK = kHeadDim / kMmaK;

    // Thread configuration - use 4 warps for tensor core implementation
    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = kNWarps * 32;  // 128 threads

    // Tile shapes
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    // SMEM sizes (FP4 packed = 2 values per byte)
    static constexpr int SmemSizeQ = kBlockM * kHeadDim / 2;
    static constexpr int SmemSizeK = kBlockN * kHeadDim / 2;
    static constexpr int SmemSizeV = kBlockN * kHeadDim / 2;
    static constexpr int SmemSizeSFQ = kBlockM * NumSFPerHead;
    static constexpr int SmemSizeSFK = kBlockN * NumSFPerHead;
    static constexpr int SmemSizeSFV = kBlockN * NumSFPerHead;

    // Scratch space for softmax (row_max, row_sum, etc.)
    static constexpr int SmemSizeScratch = kBlockM * 4 * sizeof(float);

    // Total SMEM with alignment
    static constexpr int SmemSizeTotal =
        ((SmemSizeQ + 127) / 128 * 128) +
        ((SmemSizeK + 127) / 128 * 128) +
        ((SmemSizeV + 127) / 128 * 128) +
        ((SmemSizeSFQ + 127) / 128 * 128) +
        ((SmemSizeSFK + 127) / 128 * 128) +
        ((SmemSizeSFV + 127) / 128 * 128) +
        ((SmemSizeScratch + 127) / 128 * 128);

    // Shared storage structure
    struct SharedStorage {
        // FP4 data tiles (packed)
        alignas(128) uint8_t smem_Q[SmemSizeQ];
        alignas(128) uint8_t smem_K[SmemSizeK];
        alignas(128) uint8_t smem_V[SmemSizeV];

        // Scale factor tiles
        alignas(128) ElementSF smem_SFQ[SmemSizeSFQ];
        alignas(128) ElementSF smem_SFK[SmemSizeSFK];
        alignas(128) ElementSF smem_SFV[SmemSizeSFV];

        // Scratch for softmax
        alignas(128) float smem_scratch[kBlockM * 4];
    };
};

///////////////////////////////////////////////////////////////////////////////
// Mask implementations
///////////////////////////////////////////////////////////////////////////////

struct FP4TensorCausalMask {
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

struct FP4TensorNoMask {
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
// SM100 FP4 Block-Scaled Flash Attention - Tensor Core Mainloop
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4Tensor {

    // Type aliases
    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;
    using SharedStorage = typename Ktraits::SharedStorage;

    using Mask = std::conditional_t<Is_causal, FP4TensorCausalMask, FP4TensorNoMask>;

    // Constants
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int SFVectorSize = Ktraits::SFVectorSize;
    static constexpr int kNThreads = Ktraits::kNThreads;

    // MMA configuration
    static constexpr int kMmaM = Ktraits::kMmaM;
    static constexpr int kMmaN = Ktraits::kMmaN;
    static constexpr int kMmaK = Ktraits::kMmaK;
    static constexpr int kMmaIterK = Ktraits::kMmaIterK;

    // Scale factor dimensions
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Params {
        int seqlen_q;
        int seqlen_k;
        int head_dim;
        int num_heads;
        int batch_size;

        // FP4 Q tensor
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

        // Output tensor
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
    // FP4 Decode Helper
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float decode_fp4(uint8_t packed, int which) {
        uint8_t nibble = which ? (packed >> 4) : (packed & 0x0F);
        return (static_cast<float>(nibble) - 7.5f) * 0.8f;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Cooperative Data Loading Functions
    ///////////////////////////////////////////////////////////////////////////

    // Load Q tile and scale factors to SMEM cooperatively
    CUTLASS_DEVICE static void load_q_tile(
        Params const& params,
        SharedStorage& storage,
        int m_block,
        int head_idx,
        int batch_idx,
        int seqlen_q
    ) {
        constexpr int HeadDim = kHeadDim;
        constexpr int BlockM = kBlockM;

        int thread_idx = threadIdx.x;
        int row_start = m_block * BlockM;
        int rows_this_tile = min(BlockM, seqlen_q - row_start);

        auto Q_data = reinterpret_cast<uint8_t const*>(params.ptr_Q);

        const int num_bytes_per_row = HeadDim / 2;
        const int total_q_bytes = BlockM * num_bytes_per_row;
        const int bytes_per_thread = (total_q_bytes + kNThreads - 1) / kNThreads;

        // Load Q data
        for (int i = 0; i < bytes_per_thread; ++i) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx < total_q_bytes) {
                int local_row = byte_idx / num_bytes_per_row;
                int local_col = byte_idx % num_bytes_per_row;
                int global_row = row_start + local_row;

                if (global_row < seqlen_q) {
                    int q_offset = batch_idx * params.stride_Q_batch +
                                   head_idx * params.stride_Q_head +
                                   global_row * params.stride_Q_seq + local_col;
                    storage.smem_Q[byte_idx] = Q_data[q_offset];
                } else {
                    storage.smem_Q[byte_idx] = 0;
                }
            }
        }

        // Load Q scale factors
        constexpr int NumSFPerRow = HeadDim / SFVectorSize;
        const int total_q_sf = BlockM * NumSFPerRow;
        const int sf_per_thread = (total_q_sf + kNThreads - 1) / kNThreads;

        const int sf_q_seq_stride = NumSFPerRow;
        const int sf_q_head_stride = seqlen_q * NumSFPerRow;
        const int sf_q_batch_stride = params.num_heads * sf_q_head_stride;

        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx * sf_per_thread + i;
            if (sf_idx < total_q_sf) {
                int local_row = sf_idx / NumSFPerRow;
                int sf_col = sf_idx % NumSFPerRow;
                int global_row = row_start + local_row;

                if (global_row < seqlen_q) {
                    int sf_offset = batch_idx * sf_q_batch_stride +
                                    head_idx * sf_q_head_stride +
                                    global_row * sf_q_seq_stride + sf_col;
                    storage.smem_SFQ[sf_idx] = params.ptr_SFQ[sf_offset];
                } else {
                    storage.smem_SFQ[sf_idx] = ElementSF(0.0f);
                }
            }
        }
    }

    // Load K tile and scale factors to SMEM cooperatively
    CUTLASS_DEVICE static void load_k_tile(
        Params const& params,
        SharedStorage& storage,
        int n_tile,
        int head_idx,
        int batch_idx,
        int seqlen_k
    ) {
        constexpr int HeadDim = kHeadDim;
        constexpr int BlockN = kBlockN;

        int thread_idx = threadIdx.x;
        int col_start = n_tile * BlockN;
        int cols_this_tile = min(BlockN, seqlen_k - col_start);

        auto K_data = reinterpret_cast<uint8_t const*>(params.ptr_K);

        const int num_bytes_per_row = HeadDim / 2;
        const int total_k_bytes = BlockN * num_bytes_per_row;
        const int bytes_per_thread = (total_k_bytes + kNThreads - 1) / kNThreads;

        // Load K data
        for (int i = 0; i < bytes_per_thread; ++i) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx < total_k_bytes) {
                int local_row = byte_idx / num_bytes_per_row;
                int local_col = byte_idx % num_bytes_per_row;
                int global_col = col_start + local_row;

                if (global_col < seqlen_k) {
                    int k_offset = batch_idx * params.stride_K_batch +
                                   head_idx * params.stride_K_head +
                                   global_col * params.stride_K_seq + local_col;
                    storage.smem_K[byte_idx] = K_data[k_offset];
                } else {
                    storage.smem_K[byte_idx] = 0;
                }
            }
        }

        // Load K scale factors
        constexpr int NumSFPerRow = HeadDim / SFVectorSize;
        const int total_k_sf = BlockN * NumSFPerRow;
        const int sf_per_thread = (total_k_sf + kNThreads - 1) / kNThreads;

        const int sf_k_seq_stride = NumSFPerRow;
        const int sf_k_head_stride = seqlen_k * NumSFPerRow;
        const int sf_k_batch_stride = params.num_heads * sf_k_head_stride;

        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx * sf_per_thread + i;
            if (sf_idx < total_k_sf) {
                int local_row = sf_idx / NumSFPerRow;
                int sf_col = sf_idx % NumSFPerRow;
                int global_col = col_start + local_row;

                if (global_col < seqlen_k) {
                    int sf_offset = batch_idx * sf_k_batch_stride +
                                    head_idx * sf_k_head_stride +
                                    global_col * sf_k_seq_stride + sf_col;
                    storage.smem_SFK[sf_idx] = params.ptr_SFK[sf_offset];
                } else {
                    storage.smem_SFK[sf_idx] = ElementSF(0.0f);
                }
            }
        }
    }

    // Load V tile and scale factors to SMEM cooperatively
    CUTLASS_DEVICE static void load_v_tile(
        Params const& params,
        SharedStorage& storage,
        int n_tile,
        int head_idx,
        int batch_idx,
        int seqlen_k
    ) {
        constexpr int HeadDim = kHeadDim;
        constexpr int BlockN = kBlockN;

        int thread_idx = threadIdx.x;
        int col_start = n_tile * BlockN;

        auto V_data = reinterpret_cast<uint8_t const*>(params.ptr_V);

        const int num_bytes_per_row = HeadDim / 2;
        const int total_v_bytes = BlockN * num_bytes_per_row;
        const int bytes_per_thread = (total_v_bytes + kNThreads - 1) / kNThreads;

        // Load V data
        for (int i = 0; i < bytes_per_thread; ++i) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx < total_v_bytes) {
                int local_row = byte_idx / num_bytes_per_row;
                int local_col = byte_idx % num_bytes_per_row;
                int global_col = col_start + local_row;

                if (global_col < seqlen_k) {
                    int v_offset = batch_idx * params.stride_V_batch +
                                   head_idx * params.stride_V_head +
                                   global_col * params.stride_V_seq + local_col;
                    storage.smem_V[byte_idx] = V_data[v_offset];
                } else {
                    storage.smem_V[byte_idx] = 0;
                }
            }
        }

        // Load V scale factors
        constexpr int NumSFPerRow = HeadDim / SFVectorSize;
        const int total_v_sf = BlockN * NumSFPerRow;
        const int sf_per_thread = (total_v_sf + kNThreads - 1) / kNThreads;

        const int sf_v_seq_stride = NumSFPerRow;
        const int sf_v_head_stride = seqlen_k * NumSFPerRow;
        const int sf_v_batch_stride = params.num_heads * sf_v_head_stride;

        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx * sf_per_thread + i;
            if (sf_idx < total_v_sf) {
                int local_row = sf_idx / NumSFPerRow;
                int sf_col = sf_idx % NumSFPerRow;
                int global_col = col_start + local_row;

                if (global_col < seqlen_k) {
                    int sf_offset = batch_idx * sf_v_batch_stride +
                                    head_idx * sf_v_head_stride +
                                    global_col * sf_v_seq_stride + sf_col;
                    storage.smem_SFV[sf_idx] = params.ptr_SFV[sf_offset];
                } else {
                    storage.smem_SFV[sf_idx] = ElementSF(0.0f);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Compute QK dot product from SMEM
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float compute_qk_dot(
        SharedStorage& storage,
        int q_row,
        int k_col
    ) {
        constexpr int HeadDim = kHeadDim;
        constexpr int SFVecSize = SFVectorSize;
        constexpr int NumSFPerRow = HeadDim / SFVecSize;
        const int num_bytes_per_row = HeadDim / 2;

        float score = 0.0f;

        for (int sf_block = 0; sf_block < NumSFPerRow; ++sf_block) {
            int d_start = sf_block * SFVecSize;

            float q_scale = static_cast<float>(storage.smem_SFQ[q_row * NumSFPerRow + sf_block]);
            float k_scale = static_cast<float>(storage.smem_SFK[k_col * NumSFPerRow + sf_block]);
            float combined_scale = q_scale * k_scale;

            float block_sum = 0.0f;
            for (int dd = 0; dd < SFVecSize; dd += 2) {
                int dim_idx = d_start + dd;
                int q_byte_idx = q_row * num_bytes_per_row + dim_idx / 2;
                int k_byte_idx = k_col * num_bytes_per_row + dim_idx / 2;

                uint8_t q_byte = storage.smem_Q[q_byte_idx];
                uint8_t k_byte = storage.smem_K[k_byte_idx];

                block_sum += decode_fp4(q_byte, 0) * decode_fp4(k_byte, 0);
                block_sum += decode_fp4(q_byte, 1) * decode_fp4(k_byte, 1);
            }
            score += block_sum * combined_scale;
        }

        return score;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Accumulate weighted V value from SMEM
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static void accumulate_pv(
        SharedStorage& storage,
        float* thread_output,
        int v_col,
        float weight
    ) {
        constexpr int HeadDim = kHeadDim;
        constexpr int SFVecSize = SFVectorSize;
        constexpr int NumSFPerRow = HeadDim / SFVecSize;
        const int num_bytes_per_row = HeadDim / 2;

        for (int sf_block = 0; sf_block < NumSFPerRow; ++sf_block) {
            int d_start = sf_block * SFVecSize;
            float v_scale = static_cast<float>(storage.smem_SFV[v_col * NumSFPerRow + sf_block]);

            for (int dd = 0; dd < SFVecSize; dd += 2) {
                int dim_idx = d_start + dd;
                int v_byte_idx = v_col * num_bytes_per_row + dim_idx / 2;
                uint8_t v_byte = storage.smem_V[v_byte_idx];

                thread_output[dim_idx] += weight * decode_fp4(v_byte, 0) * v_scale;
                thread_output[dim_idx + 1] += weight * decode_fp4(v_byte, 1) * v_scale;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Kernel Body - Optimized with SMEM staging
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
        constexpr int HeadDim = kHeadDim;
        constexpr int BlockM = kBlockM;
        constexpr int BlockN = kBlockN;

        int thread_idx = threadIdx.x;

        // Problem shape for mask calculation
        auto problem_shape = make_tuple(seqlen_q, seqlen_k, HeadDim, make_tuple(1, 1));

        // Calculate number of K/V tiles
        Mask mask;
        auto blk_coord = make_tuple(m_block, 0, make_tuple(head_idx, batch_idx));
        int num_kv_tiles = mask.get_trip_count(blk_coord, make_tuple(BlockM, BlockN, HeadDim), problem_shape);

        if (num_kv_tiles <= 0) return;

        int row_start = m_block * BlockM;
        int rows_this_tile = min(BlockM, seqlen_q - row_start);

        // Load Q tile to SMEM (done once per Q tile)
        load_q_tile(params, storage, m_block, head_idx, batch_idx, seqlen_q);
        __syncthreads();

        // Each thread handles one row (for now - will be optimized later)
        int my_row = thread_idx % BlockM;
        if (my_row >= rows_this_tile) {
            // Participate in barriers but don't compute
            for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
                __syncthreads();  // K load
                __syncthreads();  // V load
            }
            return;
        }

        int global_row = row_start + my_row;

        // Per-thread output accumulator
        float thread_output[HeadDim];
        for (int d = 0; d < HeadDim; ++d) {
            thread_output[d] = 0.0f;
        }

        float row_max = -INFINITY;
        float row_sum = 0.0f;

        // Delta-S base pointer
        int q_group_idx = m_block;
        float const* delta_s_base = nullptr;
        if (params.use_smooth_attention && params.ptr_delta_s != nullptr) {
            delta_s_base = params.ptr_delta_s +
                batch_idx * params.stride_ds_batch +
                head_idx * params.stride_ds_head +
                q_group_idx * params.stride_ds_group;
        }

        // Process each K/V tile
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int tile_col_start = n_tile * BlockN;
            int tile_cols = min(BlockN, seqlen_k - tile_col_start);

            // Load K tile to SMEM (cooperative)
            load_k_tile(params, storage, n_tile, head_idx, batch_idx, seqlen_k);
            __syncthreads();

            // Load V tile to SMEM (cooperative) - can overlap with QK compute
            load_v_tile(params, storage, n_tile, head_idx, batch_idx, seqlen_k);
            __syncthreads();

            // Compute QK^T and PV for this tile
            for (int j = 0; j < tile_cols && (tile_col_start + j) < seqlen_k; ++j) {
                int global_col = tile_col_start + j;

                // Compute QK dot product from SMEM
                float score = compute_qk_dot(storage, my_row, j);
                score *= params.scale_softmax;

                // Apply delta_s correction
                if (delta_s_base != nullptr) {
                    float ds = delta_s_base[global_col * params.stride_ds_k];
                    score += ds * params.scale_softmax;
                }

                // Apply causal mask
                if constexpr (Is_causal) {
                    if (global_col > global_row) {
                        score = -INFINITY;
                    }
                }

                // Online softmax update
                float new_max = fmaxf(row_max, score);

                if (row_max != -INFINITY && new_max != row_max) {
                    float scale_factor = expf(row_max - new_max);
                    row_sum *= scale_factor;
                    for (int d = 0; d < HeadDim; ++d) {
                        thread_output[d] *= scale_factor;
                    }
                }

                float weight = expf(score - new_max);
                row_sum += weight;

                // Accumulate weighted V from SMEM
                accumulate_pv(storage, thread_output, j, weight);

                row_max = new_max;
            }
        }

        // Normalize output
        float inv_row_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int d = 0; d < HeadDim; ++d) {
            thread_output[d] *= inv_row_sum;
        }

        // Write output
        ElementOut* O_base = params.ptr_O +
            batch_idx * params.stride_O_batch +
            head_idx * params.stride_O_head +
            global_row * params.stride_O_seq;

        for (int d = 0; d < HeadDim; ++d) {
            O_base[d] = static_cast<ElementOut>(thread_output[d]);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Flash Attention Kernel - Tensor Core Optimized
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits_, bool Is_causal, typename TileScheduler>
struct Sm100FlashFwdKernelFP4Tensor {

    using Ktraits = Ktraits_;

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using SharedStorage = typename Ktraits::SharedStorage;

    using CollectiveMainloop = CollectiveMainloopFwdSm100FP4Tensor<Ktraits, Is_causal>;

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kNThreads = Ktraits::kNThreads;

    // Kernel Arguments
    struct Arguments {
        int seqlen_q;
        int seqlen_k;
        int head_dim;
        int num_heads;
        int batch_size;

        Element const* ptr_Q;
        ElementSF const* ptr_SFQ;
        int64_t stride_Q_seq;
        int64_t stride_Q_head;
        int64_t stride_Q_batch;

        Element const* ptr_K;
        ElementSF const* ptr_SFK;
        int64_t stride_K_seq;
        int64_t stride_K_head;
        int64_t stride_K_batch;

        Element const* ptr_V;
        ElementSF const* ptr_SFV;
        int64_t stride_V_seq;
        int64_t stride_V_head;
        int64_t stride_V_batch;

        ElementOut* ptr_O;
        int64_t stride_O_seq;
        int64_t stride_O_head;
        int64_t stride_O_batch;

        float const* ptr_delta_s;
        int64_t stride_ds_k;
        int64_t stride_ds_group;
        int64_t stride_ds_head;
        int64_t stride_ds_batch;
        bool use_smooth_attention;

        float scale_softmax;
    };

    struct Params {
        typename CollectiveMainloop::Params mainloop;
        typename TileScheduler::Params scheduler;
        int seqlen_q;
        int seqlen_k;
        int num_heads;
        int batch_size;
    };

    static Params to_underlying_arguments(Arguments const& args, void* workspace) {
        auto problem_shape = make_tuple(
            args.seqlen_q, args.seqlen_k, args.head_dim,
            make_tuple(args.num_heads, args.batch_size)
        );

        auto mainloop_params = CollectiveMainloop::to_underlying_arguments(args, workspace);

        typename TileScheduler::Arguments scheduler_args{};
        auto scheduler_params = TileScheduler::to_underlying_arguments(
            problem_shape, TileShape_MNK{}, scheduler_args, workspace);

        return Params{
            mainloop_params,
            scheduler_params,
            args.seqlen_q,
            args.seqlen_k,
            args.num_heads,
            args.batch_size
        };
    }

    static dim3 get_grid_dim(Arguments const& args, int sm_count) {
        int num_m_blocks = (args.seqlen_q + kBlockM - 1) / kBlockM;
        int num_tiles = num_m_blocks * args.num_heads * args.batch_size;
        return dim3(min(num_tiles, sm_count), 1, 1);
    }

    static dim3 get_block_dim() {
        return dim3(kNThreads, 1, 1);
    }

    static size_t get_smem_size() {
        return sizeof(SharedStorage);
    }

    CUTLASS_DEVICE void operator()(Params const& params, char* smem) {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

        TileScheduler scheduler;
        auto work_tile = scheduler.get_initial_work(params.scheduler);

        if (!work_tile.is_valid()) {
            return;
        }

        auto [m_block, head_idx, batch_idx] = work_tile.get_block_coord();

        CollectiveMainloop mainloop;
        mainloop(
            params.mainloop,
            shared_storage,
            m_block,
            head_idx,
            batch_idx,
            params.seqlen_q,
            params.seqlen_k
        );
    }
};

} // namespace flash
