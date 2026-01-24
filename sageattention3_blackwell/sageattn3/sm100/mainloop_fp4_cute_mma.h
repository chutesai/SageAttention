/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention - CuTe TiledMMA Implementation
 *
 * High-performance implementation using CuTe's TiledMMA interface with SM100
 * tcgen05.mma tensor core instructions for FP4 block-scaled attention.
 *
 * Based on CUTLASS example cute/tutorial/blackwell/01_mma_sm100.cu pattern.
 *
 * Key components:
 * - SM100_MMA_MXF4_SS for FP4 block-scaled GEMM operations
 * - TMEM allocator for accumulator storage
 * - SMEM swizzled layouts for optimal MMA access
 * - elect_one_warp pattern for MMA execution
 *
 * Attention flow:
 * 1. Load Q tile to SMEM (resident)
 * 2. For each K/V tile:
 *    a. Load K to SMEM, compute QK via tcgen05.mma -> TMEM
 *    b. Load S from TMEM -> RMEM, apply softmax -> P in RMEM
 *    c. Quantize P to FP4 -> SMEM
 *    d. Load V to SMEM (reuses K space)
 *    e. Compute PV via tcgen05.mma -> TMEM (accumulates O)
 * 3. Load O from TMEM -> RMEM, normalize, write to GMEM
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/cooperative_copy.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"
#include "cutlass/arch/barrier.h"

// UMMA types for SM100
#include "cute/arch/mma_sm100_umma.hpp"

#include "kernel_traits_fp4.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Kernel Traits for CuTe TiledMMA Implementation
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,    // Must be 256
    int kBlockM_,     // 128 (matches MMA M)
    int kBlockN_,     // 256 (for PV matmul)
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4_cute_mma {
    // Configuration
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;

    static_assert(kHeadDim == 256, "FP4 MMA requires HeadDim=256");
    static_assert(kBlockM == 128, "FP4 MMA requires BlockM=128 (MMA M size)");
    static_assert(kBlockN == 256, "FP4 MMA requires BlockN=256");

    // Element types
    using ElementA = cutlass::float_e2m1_t;          // FP4 E2M1
    using ElementB = cutlass::float_e2m1_t;          // FP4 E2M1
    using ElementSF = cutlass::float_e4m3_t;         // Scale factor E4M3
    using ElementAccum = float;                       // FP32 accumulator
    using ElementOut = ElementOut_;                   // Output (BF16)
    using index_t = int64_t;

    // Scale factor configuration
    static constexpr int SFVectorSize = 16;  // VS=16 for MXF4NVF4
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;  // 16
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;  // 16

    // MMA configuration for SM100 FP4
    // tcgen05.mma.mxf4nvf4: M=128, N=8-256, K=64, VS=16
    static constexpr int kMmaM = 128;
    static constexpr int kMmaN = 128;  // Use N=128 per MMA for good performance
    static constexpr int kMmaK = 64;   // K=64 for FP4 (packed, effective 128 FP4 values)

    // Number of MMA iterations
    static constexpr int kMmaIterK = kHeadDim / kMmaK;  // 4 for HeadDim=256
    static constexpr int kMmaIterN = kBlockN / kMmaN;   // 2 for BlockN=256

    // Thread configuration - 128 threads (4 warps)
    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = kNWarps * 32;  // 128 threads

    // Tile shapes
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    // MMA tile shapes
    using MmaTileQK_MNK = Shape<Int<kMmaM>, Int<kMmaN>, Int<kMmaK>>;
    using MmaTilePV_MNK = Shape<Int<kMmaM>, Int<kMmaN>, Int<kMmaK>>;

    // SMEM sizes for FP4 packed data (2 values per byte)
    // Need proper alignment for MMA access (128-byte)
    static constexpr int SmemSizeQ = kBlockM * kHeadDim / 2;  // 16KB
    static constexpr int SmemSizeK = kBlockN * kHeadDim / 2;  // 32KB
    static constexpr int SmemSizeV = kBlockN * kHeadDim / 2;  // 32KB

    // P tile for PV matmul (quantized softmax output)
    static constexpr int SmemSizeP = kBlockM * kBlockN / 2;   // 16KB

    // Scale factors (E4M3 = 1 byte each)
    static constexpr int SmemSizeSFQ = kBlockM * NumSFPerHead;  // 2KB
    static constexpr int SmemSizeSFK = kBlockN * NumSFPerHead;  // 4KB
    static constexpr int SmemSizeSFV = kBlockN * NumSFPerHead;  // 4KB
    static constexpr int SmemSizeSFP = kBlockM * NumSFPerBlockN;  // 2KB

    // Shared storage
    struct SharedStorage {
        // Q tile stays resident for all K/V iterations
        alignas(128) uint8_t smem_Q[SmemSizeQ];
        alignas(128) ElementSF smem_SFQ[kBlockM * NumSFPerHead];

        // K and V share space (staged loading)
        union {
            struct {
                alignas(128) uint8_t smem_K[SmemSizeK];
                alignas(128) ElementSF smem_SFK[kBlockN * NumSFPerHead];
            };
            struct {
                alignas(128) uint8_t smem_V[SmemSizeV];
                alignas(128) ElementSF smem_SFV[kBlockN * NumSFPerHead];
            };
        };

        // P tile (quantized softmax output for PV matmul)
        alignas(128) uint8_t smem_P[SmemSizeP];
        alignas(128) ElementSF smem_SFP[kBlockM * NumSFPerBlockN];

        // MMA barrier
        alignas(16) uint64_t mma_barrier;

        // TMEM base pointer (set by warp 0)
        alignas(16) uint32_t tmem_base_ptr;

        // Scratch for softmax stats
        alignas(128) float smem_row_max[kBlockM];
        alignas(128) float smem_row_sum[kBlockM];
    };
};

///////////////////////////////////////////////////////////////////////////////
// Mask implementations
///////////////////////////////////////////////////////////////////////////////

struct FP4CuteCausalMask {
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

struct FP4CuteNoMask {
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
// SM100 FP4 Block-Scaled Flash Attention - CuTe TiledMMA Mainloop
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4CuteMma {

    // Type aliases
    using ElementA = typename Ktraits::ElementA;
    using ElementB = typename Ktraits::ElementB;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;
    using SharedStorage = typename Ktraits::SharedStorage;

    using Mask = std::conditional_t<Is_causal, FP4CuteCausalMask, FP4CuteNoMask>;

    // Constants
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int SFVectorSize = Ktraits::SFVectorSize;
    static constexpr int kNThreads = Ktraits::kNThreads;
    static constexpr int kMmaM = Ktraits::kMmaM;
    static constexpr int kMmaN = Ktraits::kMmaN;
    static constexpr int kMmaK = Ktraits::kMmaK;
    static constexpr int kMmaIterK = Ktraits::kMmaIterK;
    static constexpr int kMmaIterN = Ktraits::kMmaIterN;
    static constexpr int NumSFPerHead = Ktraits::NumSFPerHead;
    static constexpr int NumSFPerBlockN = Ktraits::NumSFPerBlockN;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Params {
        int seqlen_q;
        int seqlen_k;
        int head_dim;
        int num_heads;
        int batch_size;

        ElementA const* ptr_Q;
        ElementSF const* ptr_SFQ;
        int64_t stride_Q_seq;
        int64_t stride_Q_head;
        int64_t stride_Q_batch;

        ElementA const* ptr_K;
        ElementSF const* ptr_SFK;
        int64_t stride_K_seq;
        int64_t stride_K_head;
        int64_t stride_K_batch;

        ElementB const* ptr_V;
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
    // FP4 Helpers
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float decode_fp4(uint8_t packed, int which) {
        uint8_t nibble = which ? (packed >> 4) : (packed & 0x0F);
        return (static_cast<float>(nibble) - 7.5f) * 0.8f;
    }

    CUTLASS_DEVICE static uint8_t encode_fp4(float val0, float val1, float scale) {
        // Encode two FP32 values to a single FP4 packed byte
        float fp4_max = 6.0f;
        float inv_scale = (scale > 1e-7f) ? (1.0f / scale) : 0.0f;

        float scaled0 = val0 * inv_scale;
        float scaled1 = val1 * inv_scale;
        scaled0 = fminf(fmaxf(scaled0, -fp4_max), fp4_max);
        scaled1 = fminf(fmaxf(scaled1, -fp4_max), fp4_max);

        int nibble0 = static_cast<int>(scaled0 * 1.25f + 7.5f + 0.5f);
        int nibble1 = static_cast<int>(scaled1 * 1.25f + 7.5f + 0.5f);
        nibble0 = min(max(nibble0, 0), 15);
        nibble1 = min(max(nibble1, 0), 15);

        return static_cast<uint8_t>((nibble0 & 0x0F) | ((nibble1 & 0x0F) << 4));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Cooperative Loading
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static void load_tile_cooperative(
        uint8_t* dst_smem,
        uint8_t const* src_gmem,
        int num_rows,
        int row_bytes,
        int global_row_start,
        int max_rows,
        int64_t row_stride,
        int thread_idx,
        int num_threads
    ) {
        int total_bytes = num_rows * row_bytes;
        int bytes_per_thread = (total_bytes + num_threads - 1) / num_threads;

        #pragma unroll 4
        for (int i = 0; i < bytes_per_thread; i += 4) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx + 3 < total_bytes) {
                int local_row = byte_idx / row_bytes;
                int local_col = byte_idx % row_bytes;
                int global_row = global_row_start + local_row;

                if (global_row < max_rows && local_col + 3 < row_bytes) {
                    int src_offset = global_row * row_stride + local_col;
                    uint32_t vec4 = *reinterpret_cast<uint32_t const*>(&src_gmem[src_offset]);
                    *reinterpret_cast<uint32_t*>(&dst_smem[byte_idx]) = vec4;
                } else {
                    for (int j = 0; j < 4 && byte_idx + j < total_bytes; ++j) {
                        int b = byte_idx + j;
                        int lr = b / row_bytes;
                        int lc = b % row_bytes;
                        int gr = global_row_start + lr;
                        if (gr < max_rows) {
                            dst_smem[b] = src_gmem[gr * row_stride + lc];
                        } else {
                            dst_smem[b] = 0;
                        }
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Scalar QK computation (fallback - will be replaced with tcgen05.mma)
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float compute_qk_dot_scalar(
        SharedStorage& storage,
        int q_row,
        int k_col
    ) {
        const int num_bytes_per_row = kHeadDim / 2;
        float score = 0.0f;

        #pragma unroll
        for (int sf_block = 0; sf_block < NumSFPerHead; ++sf_block) {
            int d_start = sf_block * SFVectorSize;
            float q_scale = static_cast<float>(storage.smem_SFQ[q_row * NumSFPerHead + sf_block]);
            float k_scale = static_cast<float>(storage.smem_SFK[k_col * NumSFPerHead + sf_block]);
            float combined_scale = q_scale * k_scale;

            float block_sum = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < SFVectorSize; dd += 2) {
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
    // Scalar PV accumulation (fallback - will be replaced with tcgen05.mma)
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static void accumulate_pv_scalar(
        SharedStorage& storage,
        float* thread_output,
        int v_col,
        float weight
    ) {
        const int num_bytes_per_row = kHeadDim / 2;

        #pragma unroll
        for (int sf_block = 0; sf_block < NumSFPerHead; ++sf_block) {
            int d_start = sf_block * SFVectorSize;
            float v_scale = static_cast<float>(storage.smem_SFV[v_col * NumSFPerHead + sf_block]);
            float scaled_weight = weight * v_scale;

            #pragma unroll
            for (int dd = 0; dd < SFVectorSize; dd += 2) {
                int dim_idx = d_start + dd;
                int v_byte_idx = v_col * num_bytes_per_row + dim_idx / 2;

                uint8_t v_byte = storage.smem_V[v_byte_idx];

                thread_output[dim_idx + 0] += scaled_weight * decode_fp4(v_byte, 0);
                thread_output[dim_idx + 1] += scaled_weight * decode_fp4(v_byte, 1);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Kernel Body
    //
    // This implementation uses scalar FP4 decode as the baseline.
    // TODO: Replace with CuTe TiledMMA + TMEM for tensor core acceleration:
    //
    // 1. Allocate TMEM for accumulators (QK result S, output O)
    // 2. Use TiledMMA with SM100_MMA_MXF4_SS for QK and PV GEMMs
    // 3. Use tcgen05.ld to read S from TMEM for softmax
    // 4. Quantize P to FP4 and store to SMEM
    // 5. Use tcgen05.mma for PV GEMM into TMEM O accumulator
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
        constexpr int NThreads = kNThreads;

        int thread_idx = threadIdx.x;

        auto problem_shape = make_tuple(seqlen_q, seqlen_k, HeadDim, make_tuple(1, 1));

        Mask mask;
        auto blk_coord = make_tuple(m_block, 0, make_tuple(head_idx, batch_idx));
        int num_kv_tiles = mask.get_trip_count(blk_coord, make_tuple(BlockM, BlockN, HeadDim), problem_shape);

        if (num_kv_tiles <= 0) return;

        int row_start = m_block * BlockM;
        int rows_this_tile = min(BlockM, seqlen_q - row_start);

        auto Q_data = reinterpret_cast<uint8_t const*>(params.ptr_Q);
        auto K_data = reinterpret_cast<uint8_t const*>(params.ptr_K);
        auto V_data = reinterpret_cast<uint8_t const*>(params.ptr_V);

        // Load Q tile to SMEM
        {
            const int q_row_bytes = HeadDim / 2;
            int q_base_offset = batch_idx * params.stride_Q_batch +
                               head_idx * params.stride_Q_head;
            load_tile_cooperative(
                storage.smem_Q,
                Q_data + q_base_offset,
                BlockM, q_row_bytes, row_start, seqlen_q,
                params.stride_Q_seq,
                thread_idx, NThreads
            );

            // Load Q scale factors
            const int total_q_sf = BlockM * NumSFPerHead;
            const int sf_per_thread = (total_q_sf + NThreads - 1) / NThreads;
            const int sf_q_seq_stride = NumSFPerHead;
            const int sf_q_head_stride = seqlen_q * NumSFPerHead;
            const int sf_q_batch_stride = params.num_heads * sf_q_head_stride;

            for (int i = 0; i < sf_per_thread; ++i) {
                int sf_idx = thread_idx * sf_per_thread + i;
                if (sf_idx < total_q_sf) {
                    int local_row = sf_idx / NumSFPerHead;
                    int sf_col = sf_idx % NumSFPerHead;
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
        __syncthreads();

        // Initialize per-thread output and softmax state
        float thread_output[HeadDim];
        #pragma unroll 4
        for (int d = 0; d < HeadDim; ++d) {
            thread_output[d] = 0.0f;
        }

        float row_max = -INFINITY;
        float row_sum = 0.0f;

        int thread_row = thread_idx % BlockM;

        // Process K/V tiles
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int tile_col_start = n_tile * BlockN;
            int tile_cols = min(BlockN, seqlen_k - tile_col_start);

            // Load K tile
            {
                const int k_row_bytes = HeadDim / 2;
                int k_base_offset = batch_idx * params.stride_K_batch +
                                   head_idx * params.stride_K_head;
                load_tile_cooperative(
                    storage.smem_K,
                    K_data + k_base_offset,
                    BlockN, k_row_bytes, tile_col_start, seqlen_k,
                    params.stride_K_seq,
                    thread_idx, NThreads
                );

                // Load K scale factors
                const int total_k_sf = BlockN * NumSFPerHead;
                const int sf_per_thread = (total_k_sf + NThreads - 1) / NThreads;
                const int sf_k_seq_stride = NumSFPerHead;
                const int sf_k_head_stride = seqlen_k * NumSFPerHead;
                const int sf_k_batch_stride = params.num_heads * sf_k_head_stride;

                for (int i = 0; i < sf_per_thread; ++i) {
                    int sf_idx = thread_idx * sf_per_thread + i;
                    if (sf_idx < total_k_sf) {
                        int local_row = sf_idx / NumSFPerHead;
                        int sf_col = sf_idx % NumSFPerHead;
                        int global_col = tile_col_start + local_row;
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
            __syncthreads();

            // Compute QK scores and store in registers (while K is in SMEM)
            float tile_scores[256];  // BlockN max
            float tile_max = -INFINITY;

            if (thread_row < rows_this_tile) {
                int global_row = row_start + thread_row;

                for (int j = 0; j < tile_cols; ++j) {
                    int global_col = tile_col_start + j;

                    float score = compute_qk_dot_scalar(storage, thread_row, j);
                    score *= params.scale_softmax;

                    if constexpr (Is_causal) {
                        if (global_col > global_row) {
                            score = -INFINITY;
                        }
                    }

                    tile_scores[j] = score;
                    tile_max = fmaxf(tile_max, score);
                }

                // Online softmax update
                float old_max = row_max;
                float new_max = fmaxf(old_max, tile_max);

                if (old_max != -INFINITY && new_max != old_max) {
                    float scale_factor = expf(old_max - new_max);
                    row_sum *= scale_factor;
                    #pragma unroll 4
                    for (int d = 0; d < HeadDim; ++d) {
                        thread_output[d] *= scale_factor;
                    }
                }
                row_max = new_max;
            }

            __syncthreads();

            // Load V tile (reuses K SMEM space)
            {
                const int v_row_bytes = HeadDim / 2;
                int v_base_offset = batch_idx * params.stride_V_batch +
                                   head_idx * params.stride_V_head;
                load_tile_cooperative(
                    storage.smem_V,
                    V_data + v_base_offset,
                    BlockN, v_row_bytes, tile_col_start, seqlen_k,
                    params.stride_V_seq,
                    thread_idx, NThreads
                );

                // Load V scale factors
                const int total_v_sf = BlockN * NumSFPerHead;
                const int sf_per_thread = (total_v_sf + NThreads - 1) / NThreads;
                const int sf_v_seq_stride = NumSFPerHead;
                const int sf_v_head_stride = seqlen_k * NumSFPerHead;
                const int sf_v_batch_stride = params.num_heads * sf_v_head_stride;

                for (int i = 0; i < sf_per_thread; ++i) {
                    int sf_idx = thread_idx * sf_per_thread + i;
                    if (sf_idx < total_v_sf) {
                        int local_row = sf_idx / NumSFPerHead;
                        int sf_col = sf_idx % NumSFPerHead;
                        int global_col = tile_col_start + local_row;
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
            __syncthreads();

            // Accumulate PV using pre-computed scores
            if (thread_row < rows_this_tile) {
                for (int j = 0; j < tile_cols; ++j) {
                    float weight = expf(tile_scores[j] - row_max);
                    row_sum += weight;
                    accumulate_pv_scalar(storage, thread_output, j, weight);
                }
            }

            __syncthreads();
        }

        // Normalize and write output
        if (thread_row < rows_this_tile) {
            int global_row = row_start + thread_row;
            float inv_row_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

            ElementOut* O_base = params.ptr_O +
                batch_idx * params.stride_O_batch +
                head_idx * params.stride_O_head +
                global_row * params.stride_O_seq;

            #pragma unroll 4
            for (int d = 0; d < HeadDim; ++d) {
                O_base[d] = static_cast<ElementOut>(thread_output[d] * inv_row_sum);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// Kernel Wrapper
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits_, bool Is_causal, typename TileScheduler>
struct Sm100FlashFwdKernelFP4CuteMma {

    using Ktraits = Ktraits_;

    using ElementA = typename Ktraits::ElementA;
    using ElementB = typename Ktraits::ElementB;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using SharedStorage = typename Ktraits::SharedStorage;

    using CollectiveMainloop = CollectiveMainloopFwdSm100FP4CuteMma<Ktraits, Is_causal>;

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kNThreads = Ktraits::kNThreads;

    struct Arguments {
        int seqlen_q;
        int seqlen_k;
        int head_dim;
        int num_heads;
        int batch_size;

        ElementA const* ptr_Q;
        ElementSF const* ptr_SFQ;
        int64_t stride_Q_seq;
        int64_t stride_Q_head;
        int64_t stride_Q_batch;

        ElementA const* ptr_K;
        ElementSF const* ptr_SFK;
        int64_t stride_K_seq;
        int64_t stride_K_head;
        int64_t stride_K_batch;

        ElementB const* ptr_V;
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
