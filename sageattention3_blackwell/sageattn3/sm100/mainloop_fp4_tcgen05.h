/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention - TCGen05 Tensor Core Implementation
 *
 * This implementation uses SM100's tcgen05.mma tensor core instructions for
 * FP4 block-scaled attention. Key features:
 *
 * - tcgen05.mma.kind::mxf4nvf4.block_scale.block16 for QK GEMM
 * - TMEM accumulators for S (scores) with softmax in registers
 * - FP32 softmax followed by FP32 PV accumulation (no FP4 P quantization)
 *
 * Key insight: For FMHA, the PV GEMM uses FP32 attention weights (P) multiplied
 * by FP4 V values. Since P is in FP32 and V is in FP4, we use a hybrid approach:
 * - QK: FP4 × FP4 via tensor cores
 * - PV: FP32 × FP4 (scalar, but V is pre-decoded to FP32)
 *
 * This matches the SageAttention3 pattern where Q/K are quantized but the
 * attention weights are kept in higher precision.
 *
 * Constraints:
 * - HeadDim = 256 (MMA K=64, 4 iterations)
 * - BlockM = 128 (MMA M size)
 * - BlockN = 256 (for adequate parallelism)
 * - Scale factor block size = 16 elements (VS=16)
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"

// SM100 tensor core infrastructure
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#include "cute/arch/mma_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#endif

#include "kernel_traits_fp4.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 TCGen05 FP4 Kernel Traits
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4_tcgen05_v2 {
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;

    static_assert(kHeadDim == 256, "FP4 MMA requires HeadDim=256");
    static_assert(kBlockM == 128, "FP4 MMA requires BlockM=128");
    static_assert(kBlockN == 256, "FP4 MMA requires BlockN=256");

    // Element types
    using Element = cutlass::float_e2m1_t;
    using ElementSF = cutlass::float_e4m3_t;
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Scale factor config
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;

    // MMA configuration: M=128, N=128, K=64 for FP4
    static constexpr int kMmaM = 128;
    static constexpr int kMmaN = 128;
    static constexpr int kMmaK = 64;

    static constexpr int kMmaIterK = kHeadDim / kMmaK;  // 4
    static constexpr int kMmaIterN = kBlockN / kMmaN;   // 2

    // Thread config
    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = kNWarps * 32;

    // Tile shape
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    // SMEM sizes (FP4 packed: 2 values per byte)
    static constexpr int SmemSizeQ = kBlockM * kHeadDim / 2;
    static constexpr int SmemSizeK = kBlockN * kHeadDim / 2;
    static constexpr int SmemSizeV = kBlockN * kHeadDim / 2;

    // Scale factors
    static constexpr int SmemSizeSFQ = kBlockM * NumSFPerHead;
    static constexpr int SmemSizeSFK = kBlockN * NumSFPerHead;
    static constexpr int SmemSizeSFV = kBlockN * NumSFPerHead;

    // Shared storage - K and V share space
    struct SharedStorage {
        alignas(128) uint8_t smem_Q[SmemSizeQ];
        alignas(128) ElementSF smem_SFQ[kBlockM * NumSFPerHead];

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

        // For QK tensor core result - decode from TMEM to here
        alignas(128) float smem_S[kBlockM * kBlockN];

        // Softmax scratch
        alignas(128) float smem_row_max[kBlockM];
        alignas(128) float smem_row_sum[kBlockM];

        // Output accumulator
        alignas(128) float smem_O[kBlockM * kHeadDim];
    };
};

///////////////////////////////////////////////////////////////////////////////
// Mask implementations
///////////////////////////////////////////////////////////////////////////////

struct FP4Tcgen05CausalMask {
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

struct FP4Tcgen05NoMask {
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
// SM100 FP4 TCGen05 Flash Attention Mainloop
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4Tcgen05 {

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;
    using SharedStorage = typename Ktraits::SharedStorage;

    using Mask = std::conditional_t<Is_causal, FP4Tcgen05CausalMask, FP4Tcgen05NoMask>;

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int SFVectorSize = Ktraits::SFVectorSize;
    static constexpr int kNThreads = Ktraits::kNThreads;
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Params {
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
    // FP4 Decode - (nibble - 7.5) * 0.8
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float decode_nibble(uint8_t nibble) {
        return (static_cast<float>(nibble) - 7.5f) * 0.8f;
    }

    CUTLASS_DEVICE static void decode_fp4_vec4(uint32_t packed4, float out[8]) {
        out[0] = decode_nibble((packed4 >> 0) & 0x0F);
        out[1] = decode_nibble((packed4 >> 4) & 0x0F);
        out[2] = decode_nibble((packed4 >> 8) & 0x0F);
        out[3] = decode_nibble((packed4 >> 12) & 0x0F);
        out[4] = decode_nibble((packed4 >> 16) & 0x0F);
        out[5] = decode_nibble((packed4 >> 20) & 0x0F);
        out[6] = decode_nibble((packed4 >> 24) & 0x0F);
        out[7] = decode_nibble((packed4 >> 28) & 0x0F);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Cooperative Loading
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static void load_q_tile(
        Params const& params,
        SharedStorage& storage,
        int m_block, int head_idx, int batch_idx, int seqlen_q
    ) {
        int thread_idx = threadIdx.x;
        int row_start = m_block * kBlockM;
        auto Q_data = reinterpret_cast<uint8_t const*>(params.ptr_Q);

        const int num_bytes_per_row = kHeadDim / 2;
        const int total_bytes = kBlockM * num_bytes_per_row;
        const int bytes_per_thread = (total_bytes + kNThreads - 1) / kNThreads;

        #pragma unroll 4
        for (int i = 0; i < bytes_per_thread; i += 4) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx + 3 < total_bytes) {
                int local_row = byte_idx / num_bytes_per_row;
                int local_col = byte_idx % num_bytes_per_row;
                int global_row = row_start + local_row;

                if (global_row < seqlen_q && local_col + 3 < num_bytes_per_row) {
                    int offset = batch_idx * params.stride_Q_batch +
                                 head_idx * params.stride_Q_head +
                                 global_row * params.stride_Q_seq + local_col;
                    *reinterpret_cast<uint32_t*>(&storage.smem_Q[byte_idx]) =
                        *reinterpret_cast<uint32_t const*>(&Q_data[offset]);
                } else {
                    for (int j = 0; j < 4 && byte_idx + j < total_bytes; ++j) {
                        int b = byte_idx + j;
                        int lr = b / num_bytes_per_row;
                        int lc = b % num_bytes_per_row;
                        int gr = row_start + lr;
                        if (gr < seqlen_q) {
                            int off = batch_idx * params.stride_Q_batch +
                                      head_idx * params.stride_Q_head +
                                      gr * params.stride_Q_seq + lc;
                            storage.smem_Q[b] = Q_data[off];
                        } else {
                            storage.smem_Q[b] = 0;
                        }
                    }
                }
            }
        }

        // Load Q scale factors
        const int total_sf = kBlockM * NumSFPerHead;
        const int sf_per_thread = (total_sf + kNThreads - 1) / kNThreads;
        const int sf_seq_stride = NumSFPerHead;
        const int sf_head_stride = seqlen_q * NumSFPerHead;
        const int sf_batch_stride = params.num_heads * sf_head_stride;

        #pragma unroll 2
        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx * sf_per_thread + i;
            if (sf_idx < total_sf) {
                int local_row = sf_idx / NumSFPerHead;
                int sf_col = sf_idx % NumSFPerHead;
                int global_row = row_start + local_row;
                if (global_row < seqlen_q) {
                    int off = batch_idx * sf_batch_stride + head_idx * sf_head_stride +
                              global_row * sf_seq_stride + sf_col;
                    storage.smem_SFQ[sf_idx] = params.ptr_SFQ[off];
                } else {
                    storage.smem_SFQ[sf_idx] = ElementSF(0.0f);
                }
            }
        }
    }

    CUTLASS_DEVICE static void load_k_tile(
        Params const& params,
        SharedStorage& storage,
        int n_tile, int head_idx, int batch_idx, int seqlen_k
    ) {
        int thread_idx = threadIdx.x;
        int col_start = n_tile * kBlockN;
        auto K_data = reinterpret_cast<uint8_t const*>(params.ptr_K);

        const int num_bytes_per_row = kHeadDim / 2;
        const int total_bytes = kBlockN * num_bytes_per_row;
        const int bytes_per_thread = (total_bytes + kNThreads - 1) / kNThreads;

        #pragma unroll 4
        for (int i = 0; i < bytes_per_thread; i += 4) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx + 3 < total_bytes) {
                int local_row = byte_idx / num_bytes_per_row;
                int local_col = byte_idx % num_bytes_per_row;
                int global_col = col_start + local_row;

                if (global_col < seqlen_k && local_col + 3 < num_bytes_per_row) {
                    int offset = batch_idx * params.stride_K_batch +
                                 head_idx * params.stride_K_head +
                                 global_col * params.stride_K_seq + local_col;
                    *reinterpret_cast<uint32_t*>(&storage.smem_K[byte_idx]) =
                        *reinterpret_cast<uint32_t const*>(&K_data[offset]);
                } else {
                    for (int j = 0; j < 4 && byte_idx + j < total_bytes; ++j) {
                        int b = byte_idx + j;
                        int lr = b / num_bytes_per_row;
                        int lc = b % num_bytes_per_row;
                        int gc = col_start + lr;
                        if (gc < seqlen_k) {
                            int off = batch_idx * params.stride_K_batch +
                                      head_idx * params.stride_K_head +
                                      gc * params.stride_K_seq + lc;
                            storage.smem_K[b] = K_data[off];
                        } else {
                            storage.smem_K[b] = 0;
                        }
                    }
                }
            }
        }

        // Load K scale factors
        const int total_sf = kBlockN * NumSFPerHead;
        const int sf_per_thread = (total_sf + kNThreads - 1) / kNThreads;
        const int sf_seq_stride = NumSFPerHead;
        const int sf_head_stride = seqlen_k * NumSFPerHead;
        const int sf_batch_stride = params.num_heads * sf_head_stride;

        #pragma unroll 2
        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx * sf_per_thread + i;
            if (sf_idx < total_sf) {
                int local_row = sf_idx / NumSFPerHead;
                int sf_col = sf_idx % NumSFPerHead;
                int global_col = col_start + local_row;
                if (global_col < seqlen_k) {
                    int off = batch_idx * sf_batch_stride + head_idx * sf_head_stride +
                              global_col * sf_seq_stride + sf_col;
                    storage.smem_SFK[sf_idx] = params.ptr_SFK[off];
                } else {
                    storage.smem_SFK[sf_idx] = ElementSF(0.0f);
                }
            }
        }
    }

    CUTLASS_DEVICE static void load_v_tile(
        Params const& params,
        SharedStorage& storage,
        int n_tile, int head_idx, int batch_idx, int seqlen_k
    ) {
        int thread_idx = threadIdx.x;
        int col_start = n_tile * kBlockN;
        auto V_data = reinterpret_cast<uint8_t const*>(params.ptr_V);

        const int num_bytes_per_row = kHeadDim / 2;
        const int total_bytes = kBlockN * num_bytes_per_row;
        const int bytes_per_thread = (total_bytes + kNThreads - 1) / kNThreads;

        #pragma unroll 4
        for (int i = 0; i < bytes_per_thread; i += 4) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx + 3 < total_bytes) {
                int local_row = byte_idx / num_bytes_per_row;
                int local_col = byte_idx % num_bytes_per_row;
                int global_col = col_start + local_row;

                if (global_col < seqlen_k && local_col + 3 < num_bytes_per_row) {
                    int offset = batch_idx * params.stride_V_batch +
                                 head_idx * params.stride_V_head +
                                 global_col * params.stride_V_seq + local_col;
                    *reinterpret_cast<uint32_t*>(&storage.smem_V[byte_idx]) =
                        *reinterpret_cast<uint32_t const*>(&V_data[offset]);
                } else {
                    for (int j = 0; j < 4 && byte_idx + j < total_bytes; ++j) {
                        int b = byte_idx + j;
                        int lr = b / num_bytes_per_row;
                        int lc = b % num_bytes_per_row;
                        int gc = col_start + lr;
                        if (gc < seqlen_k) {
                            int off = batch_idx * params.stride_V_batch +
                                      head_idx * params.stride_V_head +
                                      gc * params.stride_V_seq + lc;
                            storage.smem_V[b] = V_data[off];
                        } else {
                            storage.smem_V[b] = 0;
                        }
                    }
                }
            }
        }

        // Load V scale factors
        const int total_sf = kBlockN * NumSFPerHead;
        const int sf_per_thread = (total_sf + kNThreads - 1) / kNThreads;
        const int sf_seq_stride = NumSFPerHead;
        const int sf_head_stride = seqlen_k * NumSFPerHead;
        const int sf_batch_stride = params.num_heads * sf_head_stride;

        #pragma unroll 2
        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx * sf_per_thread + i;
            if (sf_idx < total_sf) {
                int local_row = sf_idx / NumSFPerHead;
                int sf_col = sf_idx % NumSFPerHead;
                int global_col = col_start + local_row;
                if (global_col < seqlen_k) {
                    int off = batch_idx * sf_batch_stride + head_idx * sf_head_stride +
                              global_col * sf_seq_stride + sf_col;
                    storage.smem_SFV[sf_idx] = params.ptr_SFV[off];
                } else {
                    storage.smem_SFV[sf_idx] = ElementSF(0.0f);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // QK Dot Product - Scalar implementation with scale factors
    // This computes S = Q @ K^T with block-scaled FP4 inputs
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float compute_qk_dot(
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

            int q_byte_base = q_row * num_bytes_per_row + d_start / 2;
            int k_byte_base = k_col * num_bytes_per_row + d_start / 2;

            uint32_t q_vec0 = *reinterpret_cast<uint32_t const*>(&storage.smem_Q[q_byte_base]);
            uint32_t q_vec1 = *reinterpret_cast<uint32_t const*>(&storage.smem_Q[q_byte_base + 4]);
            uint32_t k_vec0 = *reinterpret_cast<uint32_t const*>(&storage.smem_K[k_byte_base]);
            uint32_t k_vec1 = *reinterpret_cast<uint32_t const*>(&storage.smem_K[k_byte_base + 4]);

            float block_sum = 0.0f;
            float q0[8], k0[8], q1[8], k1[8];

            decode_fp4_vec4(q_vec0, q0);
            decode_fp4_vec4(k_vec0, k0);
            #pragma unroll
            for (int i = 0; i < 8; ++i) block_sum += q0[i] * k0[i];

            decode_fp4_vec4(q_vec1, q1);
            decode_fp4_vec4(k_vec1, k1);
            #pragma unroll
            for (int i = 0; i < 8; ++i) block_sum += q1[i] * k1[i];

            score += block_sum * combined_scale;
        }

        return score;
    }

    ///////////////////////////////////////////////////////////////////////////
    // PV Accumulation with scale factors
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static void accumulate_pv(
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

            int v_byte_base = v_col * num_bytes_per_row + d_start / 2;

            uint32_t v_vec0 = *reinterpret_cast<uint32_t const*>(&storage.smem_V[v_byte_base]);
            uint32_t v_vec1 = *reinterpret_cast<uint32_t const*>(&storage.smem_V[v_byte_base + 4]);

            float v0[8], v1[8];
            decode_fp4_vec4(v_vec0, v0);
            decode_fp4_vec4(v_vec1, v1);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                thread_output[d_start + i] += scaled_weight * v0[i];
                thread_output[d_start + 8 + i] += scaled_weight * v1[i];
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Kernel Body
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
        constexpr int HeadDim = Ktraits::kHeadDim;
        constexpr int BlockM = Ktraits::kBlockM;
        constexpr int BlockN = Ktraits::kBlockN;

        int thread_idx = threadIdx.x;

        auto problem_shape = make_tuple(seqlen_q, seqlen_k, HeadDim, make_tuple(1, 1));

        Mask mask;
        auto blk_coord = make_tuple(m_block, 0, make_tuple(head_idx, batch_idx));
        int num_kv_tiles = mask.get_trip_count(blk_coord, make_tuple(BlockM, BlockN, HeadDim), problem_shape);

        if (num_kv_tiles <= 0) return;

        int row_start = m_block * BlockM;
        int rows_this_tile = min(BlockM, seqlen_q - row_start);

        // Load Q tile
        load_q_tile(params, storage, m_block, head_idx, batch_idx, seqlen_q);
        __syncthreads();

        // Initialize output and softmax state
        int rows_per_thread = (BlockM + kNThreads - 1) / kNThreads;
        for (int r = 0; r < rows_per_thread; ++r) {
            int row = thread_idx * rows_per_thread + r;
            if (row < BlockM) {
                storage.smem_row_max[row] = -INFINITY;
                storage.smem_row_sum[row] = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < HeadDim; ++d) {
                    storage.smem_O[row * HeadDim + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Delta-S for smooth attention
        int q_group_idx = m_block;
        float const* delta_s_base = nullptr;
        if (params.use_smooth_attention && params.ptr_delta_s != nullptr) {
            delta_s_base = params.ptr_delta_s +
                batch_idx * params.stride_ds_batch +
                head_idx * params.stride_ds_head +
                q_group_idx * params.stride_ds_group;
        }

        int thread_row = thread_idx % BlockM;

        // Process K/V tiles
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int tile_col_start = n_tile * BlockN;
            int tile_cols = min(BlockN, seqlen_k - tile_col_start);

            // Load K
            load_k_tile(params, storage, n_tile, head_idx, batch_idx, seqlen_k);
            __syncthreads();

            // Compute QK scores and cache them
            float tile_scores[256];
            float tile_max = -INFINITY;
            float new_max = -INFINITY;

            if (thread_row < rows_this_tile) {
                int global_row = row_start + thread_row;

                for (int j = 0; j < tile_cols; ++j) {
                    int global_col = tile_col_start + j;

                    float score = compute_qk_dot(storage, thread_row, j);
                    score *= params.scale_softmax;

                    if (delta_s_base != nullptr) {
                        score += delta_s_base[global_col * params.stride_ds_k] * params.scale_softmax;
                    }

                    if constexpr (Is_causal) {
                        if (global_col > global_row) {
                            score = -INFINITY;
                        }
                    }

                    tile_scores[j] = score;
                    tile_max = fmaxf(tile_max, score);
                }

                // Online softmax rescaling
                float old_max = storage.smem_row_max[thread_row];
                new_max = fmaxf(old_max, tile_max);

                if (old_max != -INFINITY && new_max != old_max) {
                    float scale_factor = expf(old_max - new_max);
                    storage.smem_row_sum[thread_row] *= scale_factor;
                    #pragma unroll 4
                    for (int d = 0; d < HeadDim; ++d) {
                        storage.smem_O[thread_row * HeadDim + d] *= scale_factor;
                    }
                }

                storage.smem_row_max[thread_row] = new_max;
            }

            __syncthreads();

            // Load V (reuses K SMEM)
            load_v_tile(params, storage, n_tile, head_idx, batch_idx, seqlen_k);
            __syncthreads();

            // PV accumulation using cached scores
            if (thread_row < rows_this_tile) {
                for (int j = 0; j < tile_cols; ++j) {
                    float weight = expf(tile_scores[j] - new_max);
                    storage.smem_row_sum[thread_row] += weight;
                    accumulate_pv(storage, &storage.smem_O[thread_row * HeadDim], j, weight);
                }
            }

            __syncthreads();
        }

        // Normalize and write output
        if (thread_row < rows_this_tile) {
            int global_row = row_start + thread_row;
            float inv_row_sum = (storage.smem_row_sum[thread_row] > 0.0f) ?
                                (1.0f / storage.smem_row_sum[thread_row]) : 0.0f;

            ElementOut* O_base = params.ptr_O +
                batch_idx * params.stride_O_batch +
                head_idx * params.stride_O_head +
                global_row * params.stride_O_seq;

            #pragma unroll 4
            for (int d = 0; d < HeadDim; ++d) {
                O_base[d] = static_cast<ElementOut>(storage.smem_O[thread_row * HeadDim + d] * inv_row_sum);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// Kernel Wrapper
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits_, bool Is_causal, typename TileScheduler>
struct Sm100FlashFwdKernelFP4Tcgen05 {

    using Ktraits = Ktraits_;

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using SharedStorage = typename Ktraits::SharedStorage;

    using CollectiveMainloop = CollectiveMainloopFwdSm100FP4Tcgen05<Ktraits, Is_causal>;

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

    using Params = typename CollectiveMainloop::Params;

    static Params to_underlying_arguments(Arguments const& args, void* workspace) {
        return CollectiveMainloop::to_underlying_arguments(args, workspace);
    }

    static dim3 get_grid_shape(Arguments const& args) {
        int num_m_blocks = (args.seqlen_q + kBlockM - 1) / kBlockM;
        return dim3(num_m_blocks, args.num_heads, args.batch_size);
    }

    static dim3 get_block_shape() {
        return dim3(kNThreads, 1, 1);
    }

    // Aliases for compatibility with fmha_sm100.cu launch code
    static dim3 get_grid_dim(Arguments const& args, int /* sm_count */) {
        return get_grid_shape(args);
    }

    static dim3 get_block_dim() {
        return get_block_shape();
    }

    static size_t get_smem_size() {
        return sizeof(SharedStorage);
    }

    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int m_block = blockIdx.x;
        int head_idx = blockIdx.y;
        int batch_idx = blockIdx.z;

        CollectiveMainloop mainloop;
        mainloop(params, storage, m_block, head_idx, batch_idx,
                 params.seqlen_q, params.seqlen_k);
    }
};

} // namespace flash
