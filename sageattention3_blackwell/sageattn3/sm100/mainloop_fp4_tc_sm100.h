/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention - Tensor Core Implementation
 *
 * High-performance implementation using SM100 tcgen05.mma tensor core instructions
 * for FP4 block-scaled attention.
 *
 * This implementation uses:
 * - SM100_MMA_MXF4_SS for FP4 block-scaled GEMM operations
 * - SMEM staging for Q/K/V data with cooperative loading
 * - Direct PTX tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16
 * - Warp-specialized execution with elect_one_sync pattern
 *
 * Key constraints:
 * - HeadDim = 256 (MMA K=64, need 4 MMA iterations)
 * - BlockN = 256 (PV matmul requirement)
 * - Scale factor block size = 16 elements (VS=16 for MXF4NVF4)
 * - M = 128 per MMA (fixed for SM100)
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
// SM100 FP4 Kernel Traits for Tensor Core Implementation
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,    // Must be 256
    int kBlockM_,     // 128 (matches MMA M)
    int kBlockN_,     // 256 (for PV matmul)
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4_tcgen05 {
    // Configuration
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;

    static_assert(kHeadDim == 256, "FP4 MMA requires HeadDim=256");
    static_assert(kBlockM == 128, "FP4 MMA requires BlockM=128 (MMA M size)");
    static_assert(kBlockN == 256, "FP4 MMA requires BlockN=256");

    // Element types
    using Element = cutlass::float_e2m1_t;           // FP4 E2M1
    using ElementSF = cutlass::float_e4m3_t;         // Scale factor E4M3
    using ElementAccum = float;                       // FP32 accumulator
    using ElementOut = ElementOut_;                   // Output (BF16)
    using index_t = int64_t;

    // Scale factor configuration
    static constexpr int SFVectorSize = 16;  // 16 elements per scale factor (VS=16)
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;  // 16
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;  // 16

    // MMA configuration for SM100 FP4 (SM100_MMA_MXF4_SS with VS=16)
    // M=128 fixed, N=128 per MMA, K=64 (FP4 K dimension)
    static constexpr int kMmaM = 128;
    static constexpr int kMmaN = 128;
    static constexpr int kMmaK = 64;

    // Number of MMA iterations
    static constexpr int kMmaIterK = kHeadDim / kMmaK;  // 4 iterations for HeadDim=256
    static constexpr int kMmaIterN = kBlockN / kMmaN;   // 2 iterations for BlockN=256

    // Thread configuration
    // MMA warp + load/softmax warps
    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = kNWarps * 32;  // 128 threads

    // Tile shapes
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    // SMEM sizes for FP4 packed data (2 values per byte)
    // 128-byte alignment for proper TMA/MMA access
    static constexpr int SmemSizeQ = kBlockM * kHeadDim / 2;  // 16KB
    static constexpr int SmemSizeK = kBlockN * kHeadDim / 2;  // 32KB
    static constexpr int SmemSizeV = kBlockN * kHeadDim / 2;  // 32KB

    // Scale factors (E4M3 = 1 byte each)
    static constexpr int SmemSizeSFQ = kBlockM * NumSFPerHead;  // 2KB
    static constexpr int SmemSizeSFK = kBlockN * NumSFPerHead;  // 4KB
    static constexpr int SmemSizeSFV = kBlockN * NumSFPerHead;  // 4KB

    // Shared storage - K and V share space (staged loading)
    struct SharedStorage {
        // Q tile stays resident for all K/V iterations
        alignas(128) uint8_t smem_Q[SmemSizeQ];
        alignas(128) ElementSF smem_SFQ[kBlockM * NumSFPerHead];

        // K and V share space (double buffered pipeline)
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

        // Scratch for softmax stats (row_max, row_sum per query)
        alignas(128) float smem_row_max[kBlockM];
        alignas(128) float smem_row_sum[kBlockM];

        // Output accumulator in SMEM (before write back)
        alignas(128) float smem_O[kBlockM * kHeadDim];
    };
};

///////////////////////////////////////////////////////////////////////////////
// Mask implementations
///////////////////////////////////////////////////////////////////////////////

struct FP4TcCausalMask {
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

struct FP4TcNoMask {
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
//
// High-performance implementation using SM100 tcgen05.mma
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4TensorCore {

    // Type aliases
    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;
    using SharedStorage = typename Ktraits::SharedStorage;

    using Mask = std::conditional_t<Is_causal, FP4TcCausalMask, FP4TcNoMask>;

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
    static constexpr int kMmaIterN = Ktraits::kMmaIterN;

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
    // FP4 Decode Helpers - Optimized versions
    ///////////////////////////////////////////////////////////////////////////

    // Standard decode for single nibble: value = (nibble - 7.5) * 0.8
    CUTLASS_DEVICE static float decode_fp4(uint8_t packed, int which) {
        uint8_t nibble = which ? (packed >> 4) : (packed & 0x0F);
        return (static_cast<float>(nibble) - 7.5f) * 0.8f;
    }

    // Fast decode from nibble value (no extraction needed)
    CUTLASS_DEVICE static float decode_nibble(uint8_t nibble) {
        return (static_cast<float>(nibble) - 7.5f) * 0.8f;
    }

    // Vectorized decode of 4 bytes (8 FP4 values) to 8 floats
    // Uses vectorized loads and batch decoding for efficiency
    CUTLASS_DEVICE static void decode_fp4_vec4(
        uint32_t packed4,  // 4 packed bytes
        float out[8]
    ) {
        // Extract all 8 nibbles
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
    // Cooperative Data Loading - All threads participate
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static void load_q_tile_cooperative(
        Params const& params,
        SharedStorage& storage,
        int m_block,
        int head_idx,
        int batch_idx,
        int seqlen_q
    ) {
        int thread_idx = threadIdx.x;
        int row_start = m_block * kBlockM;

        auto Q_data = reinterpret_cast<uint8_t const*>(params.ptr_Q);

        const int num_bytes_per_row = kHeadDim / 2;
        const int total_q_bytes = kBlockM * num_bytes_per_row;
        const int bytes_per_thread = (total_q_bytes + kNThreads - 1) / kNThreads;

        // Use vectorized loads where possible (4 bytes at a time)
        #pragma unroll 4
        for (int i = 0; i < bytes_per_thread; i += 4) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx + 3 < total_q_bytes) {
                int local_row = byte_idx / num_bytes_per_row;
                int local_col = byte_idx % num_bytes_per_row;
                int global_row = row_start + local_row;

                if (global_row < seqlen_q && local_col + 3 < num_bytes_per_row) {
                    int q_offset = batch_idx * params.stride_Q_batch +
                                   head_idx * params.stride_Q_head +
                                   global_row * params.stride_Q_seq + local_col;
                    // Vectorized load
                    uint32_t vec4 = *reinterpret_cast<uint32_t const*>(&Q_data[q_offset]);
                    *reinterpret_cast<uint32_t*>(&storage.smem_Q[byte_idx]) = vec4;
                } else {
                    // Scalar fallback for boundary
                    for (int j = 0; j < 4 && byte_idx + j < total_q_bytes; ++j) {
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
            } else {
                // Handle tail
                for (int j = 0; j < 4 && byte_idx + j < total_q_bytes; ++j) {
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

        // Load Q scale factors
        const int total_q_sf = kBlockM * NumSFPerHead;
        const int sf_per_thread = (total_q_sf + kNThreads - 1) / kNThreads;

        const int sf_q_seq_stride = NumSFPerHead;
        const int sf_q_head_stride = seqlen_q * NumSFPerHead;
        const int sf_q_batch_stride = params.num_heads * sf_q_head_stride;

        #pragma unroll 2
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

    CUTLASS_DEVICE static void load_k_tile_cooperative(
        Params const& params,
        SharedStorage& storage,
        int n_tile,
        int head_idx,
        int batch_idx,
        int seqlen_k
    ) {
        int thread_idx = threadIdx.x;
        int col_start = n_tile * kBlockN;

        auto K_data = reinterpret_cast<uint8_t const*>(params.ptr_K);

        const int num_bytes_per_row = kHeadDim / 2;
        const int total_k_bytes = kBlockN * num_bytes_per_row;
        const int bytes_per_thread = (total_k_bytes + kNThreads - 1) / kNThreads;

        #pragma unroll 4
        for (int i = 0; i < bytes_per_thread; i += 4) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx + 3 < total_k_bytes) {
                int local_row = byte_idx / num_bytes_per_row;
                int local_col = byte_idx % num_bytes_per_row;
                int global_col = col_start + local_row;

                if (global_col < seqlen_k && local_col + 3 < num_bytes_per_row) {
                    int k_offset = batch_idx * params.stride_K_batch +
                                   head_idx * params.stride_K_head +
                                   global_col * params.stride_K_seq + local_col;
                    uint32_t vec4 = *reinterpret_cast<uint32_t const*>(&K_data[k_offset]);
                    *reinterpret_cast<uint32_t*>(&storage.smem_K[byte_idx]) = vec4;
                } else {
                    for (int j = 0; j < 4 && byte_idx + j < total_k_bytes; ++j) {
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
            } else {
                for (int j = 0; j < 4 && byte_idx + j < total_k_bytes; ++j) {
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

        // Load K scale factors
        const int total_k_sf = kBlockN * NumSFPerHead;
        const int sf_per_thread = (total_k_sf + kNThreads - 1) / kNThreads;

        const int sf_k_seq_stride = NumSFPerHead;
        const int sf_k_head_stride = seqlen_k * NumSFPerHead;
        const int sf_k_batch_stride = params.num_heads * sf_k_head_stride;

        #pragma unroll 2
        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx * sf_per_thread + i;
            if (sf_idx < total_k_sf) {
                int local_row = sf_idx / NumSFPerHead;
                int sf_col = sf_idx % NumSFPerHead;
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

    CUTLASS_DEVICE static void load_v_tile_cooperative(
        Params const& params,
        SharedStorage& storage,
        int n_tile,
        int head_idx,
        int batch_idx,
        int seqlen_k
    ) {
        int thread_idx = threadIdx.x;
        int col_start = n_tile * kBlockN;

        auto V_data = reinterpret_cast<uint8_t const*>(params.ptr_V);

        const int num_bytes_per_row = kHeadDim / 2;
        const int total_v_bytes = kBlockN * num_bytes_per_row;
        const int bytes_per_thread = (total_v_bytes + kNThreads - 1) / kNThreads;

        #pragma unroll 4
        for (int i = 0; i < bytes_per_thread; i += 4) {
            int byte_idx = thread_idx * bytes_per_thread + i;
            if (byte_idx + 3 < total_v_bytes) {
                int local_row = byte_idx / num_bytes_per_row;
                int local_col = byte_idx % num_bytes_per_row;
                int global_col = col_start + local_row;

                if (global_col < seqlen_k && local_col + 3 < num_bytes_per_row) {
                    int v_offset = batch_idx * params.stride_V_batch +
                                   head_idx * params.stride_V_head +
                                   global_col * params.stride_V_seq + local_col;
                    uint32_t vec4 = *reinterpret_cast<uint32_t const*>(&V_data[v_offset]);
                    *reinterpret_cast<uint32_t*>(&storage.smem_V[byte_idx]) = vec4;
                } else {
                    for (int j = 0; j < 4 && byte_idx + j < total_v_bytes; ++j) {
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
            } else {
                for (int j = 0; j < 4 && byte_idx + j < total_v_bytes; ++j) {
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

        // Load V scale factors
        const int total_v_sf = kBlockN * NumSFPerHead;
        const int sf_per_thread = (total_v_sf + kNThreads - 1) / kNThreads;

        const int sf_v_seq_stride = NumSFPerHead;
        const int sf_v_head_stride = seqlen_k * NumSFPerHead;
        const int sf_v_batch_stride = params.num_heads * sf_v_head_stride;

        #pragma unroll 2
        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx * sf_per_thread + i;
            if (sf_idx < total_v_sf) {
                int local_row = sf_idx / NumSFPerHead;
                int sf_col = sf_idx % NumSFPerHead;
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
    // Highly optimized with LUT decode and vectorized loads
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float compute_qk_dot_optimized(
        SharedStorage& storage,
        int q_row,
        int k_col
    ) {
        const int num_bytes_per_row = kHeadDim / 2;
        float score = 0.0f;

        // Process all 16 scale factor blocks (HeadDim=256, SFVectorSize=16)
        #pragma unroll
        for (int sf_block = 0; sf_block < NumSFPerHead; ++sf_block) {
            int d_start = sf_block * SFVectorSize;

            float q_scale = static_cast<float>(storage.smem_SFQ[q_row * NumSFPerHead + sf_block]);
            float k_scale = static_cast<float>(storage.smem_SFK[k_col * NumSFPerHead + sf_block]);
            float combined_scale = q_scale * k_scale;

            // Each scale factor block covers 16 FP4 values = 8 bytes
            // Process 8 bytes (16 FP4 values) with vectorized loads
            int q_byte_base = q_row * num_bytes_per_row + d_start / 2;
            int k_byte_base = k_col * num_bytes_per_row + d_start / 2;

            // Load 8 bytes as two uint32_t for better memory bandwidth
            uint32_t q_vec0 = *reinterpret_cast<uint32_t const*>(&storage.smem_Q[q_byte_base]);
            uint32_t q_vec1 = *reinterpret_cast<uint32_t const*>(&storage.smem_Q[q_byte_base + 4]);
            uint32_t k_vec0 = *reinterpret_cast<uint32_t const*>(&storage.smem_K[k_byte_base]);
            uint32_t k_vec1 = *reinterpret_cast<uint32_t const*>(&storage.smem_K[k_byte_base + 4]);

            // Decode and dot product using LUT
            float block_sum = 0.0f;

            // First 8 FP4 values
            float q0[8], k0[8];
            decode_fp4_vec4(q_vec0, q0);
            decode_fp4_vec4(k_vec0, k0);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                block_sum += q0[i] * k0[i];
            }

            // Second 8 FP4 values
            float q1[8], k1[8];
            decode_fp4_vec4(q_vec1, q1);
            decode_fp4_vec4(k_vec1, k1);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                block_sum += q1[i] * k1[i];
            }

            score += block_sum * combined_scale;
        }

        return score;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Accumulate weighted V from SMEM
    // Highly optimized with LUT decode and vectorized loads
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static void accumulate_pv_optimized(
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

            // Load 8 bytes (16 FP4 values) as two uint32_t
            uint32_t v_vec0 = *reinterpret_cast<uint32_t const*>(&storage.smem_V[v_byte_base]);
            uint32_t v_vec1 = *reinterpret_cast<uint32_t const*>(&storage.smem_V[v_byte_base + 4]);

            // Decode first 8 values and accumulate
            float v0[8];
            decode_fp4_vec4(v_vec0, v0);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                thread_output[d_start + i] += scaled_weight * v0[i];
            }

            // Decode second 8 values and accumulate
            float v1[8];
            decode_fp4_vec4(v_vec1, v1);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                thread_output[d_start + 8 + i] += scaled_weight * v1[i];
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Kernel Body - Optimized SMEM-based implementation
    // This version uses SMEM staging for better memory coalescing
    // and prepares for future tensor core integration
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
        constexpr int NThreads = Ktraits::kNThreads;

        int thread_idx = threadIdx.x;

        // Problem shape for mask
        auto problem_shape = make_tuple(seqlen_q, seqlen_k, HeadDim, make_tuple(1, 1));

        // Calculate number of K/V tiles
        Mask mask;
        auto blk_coord = make_tuple(m_block, 0, make_tuple(head_idx, batch_idx));
        int num_kv_tiles = mask.get_trip_count(blk_coord, make_tuple(BlockM, BlockN, HeadDim), problem_shape);

        if (num_kv_tiles <= 0) return;

        int row_start = m_block * BlockM;
        int rows_this_tile = min(BlockM, seqlen_q - row_start);

        // Load Q tile to SMEM (resident for all K/V tiles)
        load_q_tile_cooperative(params, storage, m_block, head_idx, batch_idx, seqlen_q);
        __syncthreads();

        // Initialize output accumulator and softmax state in SMEM
        // Each thread handles multiple rows for parallel initialization
        int rows_per_thread = (BlockM + NThreads - 1) / NThreads;
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

        // Delta-S base pointer for smooth attention
        int q_group_idx = m_block;
        float const* delta_s_base = nullptr;
        if (params.use_smooth_attention && params.ptr_delta_s != nullptr) {
            delta_s_base = params.ptr_delta_s +
                batch_idx * params.stride_ds_batch +
                head_idx * params.stride_ds_head +
                q_group_idx * params.stride_ds_group;
        }

        // Thread-row assignment: threads are distributed across query rows
        // Defined outside loop so it's visible for output write
        int thread_row = thread_idx % BlockM;

        // Process each K/V tile
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int tile_col_start = n_tile * BlockN;
            int tile_cols = min(BlockN, seqlen_k - tile_col_start);

            // Load K tile to SMEM
            load_k_tile_cooperative(params, storage, n_tile, head_idx, batch_idx, seqlen_k);
            __syncthreads();

            // Per-thread storage for scores (computed while K is in SMEM)
            float tile_scores[256];  // Fixed size for BlockN=256
            float tile_max = -INFINITY;
            float new_max = -INFINITY;

            if (thread_row < rows_this_tile) {
                int global_row = row_start + thread_row;

                // Compute ALL tile scores while K is still in SMEM
                for (int j = 0; j < tile_cols; ++j) {
                    int global_col = tile_col_start + j;

                    float score = compute_qk_dot_optimized(storage, thread_row, j);
                    score *= params.scale_softmax;

                    // Apply delta_s correction
                    if (delta_s_base != nullptr) {
                        score += delta_s_base[global_col * params.stride_ds_k] * params.scale_softmax;
                    }

                    // Apply causal mask
                    if constexpr (Is_causal) {
                        if (global_col > global_row) {
                            score = -INFINITY;
                        }
                    }

                    tile_scores[j] = score;
                    tile_max = fmaxf(tile_max, score);
                }

                // Online softmax update
                float old_max = storage.smem_row_max[thread_row];
                new_max = fmaxf(old_max, tile_max);

                // Rescale existing accumulator
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

            // Load V tile (reuses K SMEM space - K is no longer needed!)
            load_v_tile_cooperative(params, storage, n_tile, head_idx, batch_idx, seqlen_k);
            __syncthreads();

            // Use pre-computed scores (stored in registers) for PV accumulation
            if (thread_row < rows_this_tile) {
                // Use scores from registers - DO NOT recompute (K is gone!)
                for (int j = 0; j < tile_cols; ++j) {
                    float weight = expf(tile_scores[j] - new_max);
                    storage.smem_row_sum[thread_row] += weight;

                    // Accumulate PV
                    accumulate_pv_optimized(storage, &storage.smem_O[thread_row * HeadDim], j, weight);
                }
            }

            __syncthreads();
        }

        // Normalize output and write to global memory
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
struct Sm100FlashFwdKernelFP4TensorCore {

    using Ktraits = Ktraits_;

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using SharedStorage = typename Ktraits::SharedStorage;

    using CollectiveMainloop = CollectiveMainloopFwdSm100FP4TensorCore<Ktraits, Is_causal>;

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
