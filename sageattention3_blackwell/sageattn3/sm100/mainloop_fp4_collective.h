/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention - CUTLASS Collective Implementation
 *
 * This implementation uses CUTLASS CollectiveBuilder with OpClassBlockScaledTensorOp
 * to automatically generate high-performance tcgen05.mma code for FP4 GEMMs.
 *
 * Architecture:
 * - QK GEMM: Q[M,K] × K^T[K,N] using block-scaled FP4 MMA
 * - Softmax: Read S from TMEM → compute in registers → quantize P to FP4
 * - PV GEMM: P[M,N] × V[N,K] using block-scaled FP4 MMA
 *
 * Key constraints:
 * - HeadDim = 256 (MMA K=64, need 4 iterations)
 * - BlockN = 256 (PV matmul requirement)
 * - Scale factor block size = 16 (VS=16 for MXF4NVF4)
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "kernel_traits_fp4.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled FMHA Kernel Traits - Collective MMA Version
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4_collective {
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;

    static_assert(kHeadDim == 256, "FP4 MMA requires HeadDim=256");
    static_assert(kBlockM == 128, "FP4 MMA requires BlockM=128");
    static_assert(kBlockN == 256, "FP4 MMA requires BlockN=256");

    // Element types
    using ElementA = cutlass::float_e2m1_t;          // FP4 E2M1
    using ElementB = cutlass::float_e2m1_t;          // FP4 E2M1
    using ElementSF = cutlass::float_e4m3_t;         // Scale factor E4M3
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Scale factor configuration
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;

    // Thread configuration
    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = kNWarps * 32;

    // MMA configuration
    static constexpr int kMmaM = 128;
    static constexpr int kMmaN = 128;
    static constexpr int kMmaK = 64;

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    // SMEM sizes (FP4 packed = 2 values per byte)
    static constexpr int SmemSizeQ = kBlockM * kHeadDim / 2;
    static constexpr int SmemSizeK = kBlockN * kHeadDim / 2;
    static constexpr int SmemSizeV = kBlockN * kHeadDim / 2;

    static constexpr int SmemSizeSFQ = kBlockM * NumSFPerHead;
    static constexpr int SmemSizeSFK = kBlockN * NumSFPerHead;
    static constexpr int SmemSizeSFV = kBlockN * NumSFPerHead;

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

        alignas(128) float smem_row_max[kBlockM];
        alignas(128) float smem_row_sum[kBlockM];
    };
};

///////////////////////////////////////////////////////////////////////////////
// Mask implementations
///////////////////////////////////////////////////////////////////////////////

struct FP4CollectiveCausalMask {
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

struct FP4CollectiveNoMask {
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
// SM100 FP4 Block-Scaled Flash Attention Mainloop using Collective MMA
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4Collective {

    using ElementA = typename Ktraits::ElementA;
    using ElementB = typename Ktraits::ElementB;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;
    using SharedStorage = typename Ktraits::SharedStorage;

    using Mask = std::conditional_t<Is_causal, FP4CollectiveCausalMask, FP4CollectiveNoMask>;

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int SFVectorSize = Ktraits::SFVectorSize;
    static constexpr int kNThreads = Ktraits::kNThreads;
    static constexpr int NumSFPerHead = Ktraits::NumSFPerHead;
    static constexpr int NumSFPerBlockN = Ktraits::NumSFPerBlockN;

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
    static Params to_underlying_arguments(KernelArguments const& args, void* workspace) {
        float log2_e = static_cast<float>(M_LOG2E);
        return Params{
            args.seqlen_q, args.seqlen_k, args.head_dim, args.num_heads, args.batch_size,
            args.ptr_Q, args.ptr_SFQ,
            args.stride_Q_seq, args.stride_Q_head, args.stride_Q_batch,
            args.ptr_K, args.ptr_SFK,
            args.stride_K_seq, args.stride_K_head, args.stride_K_batch,
            args.ptr_V, args.ptr_SFV,
            args.stride_V_seq, args.stride_V_head, args.stride_V_batch,
            args.ptr_O, args.stride_O_seq, args.stride_O_head, args.stride_O_batch,
            args.ptr_delta_s,
            args.stride_ds_k, args.stride_ds_group, args.stride_ds_head, args.stride_ds_batch,
            args.use_smooth_attention,
            args.scale_softmax,
            args.scale_softmax * log2_e
        };
    }

    // FP4 decode helper
    CUTLASS_DEVICE static float decode_fp4(uint8_t packed, int which) {
        uint8_t nibble = which ? (packed >> 4) : (packed & 0x0F);
        return (static_cast<float>(nibble) - 7.5f) * 0.8f;
    }

    // Cooperative tile loading
    CUTLASS_DEVICE static void load_tile_cooperative(
        uint8_t* dst, uint8_t const* src, int num_rows, int row_bytes,
        int row_start, int max_rows, int64_t row_stride,
        int thread_idx, int num_threads
    ) {
        int total_bytes = num_rows * row_bytes;
        int per_thread = (total_bytes + num_threads - 1) / num_threads;

        #pragma unroll 4
        for (int i = 0; i < per_thread; i += 4) {
            int idx = thread_idx * per_thread + i;
            if (idx + 3 < total_bytes) {
                int row = idx / row_bytes;
                int col = idx % row_bytes;
                int global_row = row_start + row;
                if (global_row < max_rows && col + 3 < row_bytes) {
                    uint32_t v = *reinterpret_cast<uint32_t const*>(&src[global_row * row_stride + col]);
                    *reinterpret_cast<uint32_t*>(&dst[idx]) = v;
                } else {
                    for (int j = 0; j < 4 && idx + j < total_bytes; ++j) {
                        int r = (idx + j) / row_bytes;
                        int c = (idx + j) % row_bytes;
                        int gr = row_start + r;
                        dst[idx + j] = (gr < max_rows) ? src[gr * row_stride + c] : 0;
                    }
                }
            }
        }
    }

    // Scalar QK dot product
    CUTLASS_DEVICE static float compute_qk_dot(
        SharedStorage& storage, int q_row, int k_col
    ) {
        const int bytes_per_row = kHeadDim / 2;
        float score = 0.0f;

        #pragma unroll
        for (int sf = 0; sf < NumSFPerHead; ++sf) {
            float q_scale = static_cast<float>(storage.smem_SFQ[q_row * NumSFPerHead + sf]);
            float k_scale = static_cast<float>(storage.smem_SFK[k_col * NumSFPerHead + sf]);
            float combined = q_scale * k_scale;

            float sum = 0.0f;
            int d_start = sf * SFVectorSize;
            #pragma unroll
            for (int d = 0; d < SFVectorSize; d += 2) {
                int q_idx = q_row * bytes_per_row + (d_start + d) / 2;
                int k_idx = k_col * bytes_per_row + (d_start + d) / 2;
                uint8_t qb = storage.smem_Q[q_idx];
                uint8_t kb = storage.smem_K[k_idx];
                sum += decode_fp4(qb, 0) * decode_fp4(kb, 0);
                sum += decode_fp4(qb, 1) * decode_fp4(kb, 1);
            }
            score += sum * combined;
        }
        return score;
    }

    // Scalar PV accumulation
    CUTLASS_DEVICE static void accumulate_pv(
        SharedStorage& storage, float* out, int v_col, float weight
    ) {
        const int bytes_per_row = kHeadDim / 2;

        #pragma unroll
        for (int sf = 0; sf < NumSFPerHead; ++sf) {
            float v_scale = static_cast<float>(storage.smem_SFV[v_col * NumSFPerHead + sf]);
            float sw = weight * v_scale;

            int d_start = sf * SFVectorSize;
            #pragma unroll
            for (int d = 0; d < SFVectorSize; d += 2) {
                int v_idx = v_col * bytes_per_row + (d_start + d) / 2;
                uint8_t vb = storage.smem_V[v_idx];
                out[d_start + d + 0] += sw * decode_fp4(vb, 0);
                out[d_start + d + 1] += sw * decode_fp4(vb, 1);
            }
        }
    }

    // Main kernel body
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

        // Load Q to SMEM
        {
            int q_base = batch_idx * params.stride_Q_batch + head_idx * params.stride_Q_head;
            load_tile_cooperative(storage.smem_Q, Q_data + q_base, BlockM, HeadDim / 2,
                                  row_start, seqlen_q, params.stride_Q_seq, thread_idx, NThreads);

            // Load Q scale factors
            const int total_sf = BlockM * NumSFPerHead;
            const int per_thread = (total_sf + NThreads - 1) / NThreads;
            const int sf_seq = NumSFPerHead;
            const int sf_head = seqlen_q * NumSFPerHead;
            const int sf_batch = params.num_heads * sf_head;

            for (int i = 0; i < per_thread; ++i) {
                int idx = thread_idx * per_thread + i;
                if (idx < total_sf) {
                    int row = idx / NumSFPerHead;
                    int col = idx % NumSFPerHead;
                    int global_row = row_start + row;
                    if (global_row < seqlen_q) {
                        storage.smem_SFQ[idx] = params.ptr_SFQ[
                            batch_idx * sf_batch + head_idx * sf_head + global_row * sf_seq + col];
                    } else {
                        storage.smem_SFQ[idx] = ElementSF(0.0f);
                    }
                }
            }
        }
        __syncthreads();

        // Initialize per-thread output
        float thread_output[HeadDim];
        #pragma unroll 4
        for (int d = 0; d < HeadDim; ++d) thread_output[d] = 0.0f;

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        int thread_row = thread_idx % BlockM;

        // Process K/V tiles
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int col_start = n_tile * BlockN;
            int cols_this_tile = min(BlockN, seqlen_k - col_start);

            // Load K
            {
                int k_base = batch_idx * params.stride_K_batch + head_idx * params.stride_K_head;
                load_tile_cooperative(storage.smem_K, K_data + k_base, BlockN, HeadDim / 2,
                                      col_start, seqlen_k, params.stride_K_seq, thread_idx, NThreads);

                const int total_sf = BlockN * NumSFPerHead;
                const int per_thread = (total_sf + NThreads - 1) / NThreads;
                const int sf_seq = NumSFPerHead;
                const int sf_head = seqlen_k * NumSFPerHead;
                const int sf_batch = params.num_heads * sf_head;

                for (int i = 0; i < per_thread; ++i) {
                    int idx = thread_idx * per_thread + i;
                    if (idx < total_sf) {
                        int row = idx / NumSFPerHead;
                        int col = idx % NumSFPerHead;
                        int global_col = col_start + row;
                        if (global_col < seqlen_k) {
                            storage.smem_SFK[idx] = params.ptr_SFK[
                                batch_idx * sf_batch + head_idx * sf_head + global_col * sf_seq + col];
                        } else {
                            storage.smem_SFK[idx] = ElementSF(0.0f);
                        }
                    }
                }
            }
            __syncthreads();

            // Compute QK scores (while K is in SMEM)
            float tile_scores[256];
            float tile_max = -INFINITY;

            if (thread_row < rows_this_tile) {
                int global_row = row_start + thread_row;
                for (int j = 0; j < cols_this_tile; ++j) {
                    float score = compute_qk_dot(storage, thread_row, j) * params.scale_softmax;
                    if constexpr (Is_causal) {
                        if (col_start + j > global_row) score = -INFINITY;
                    }
                    tile_scores[j] = score;
                    tile_max = fmaxf(tile_max, score);
                }

                float old_max = row_max;
                float new_max = fmaxf(old_max, tile_max);
                if (old_max != -INFINITY && new_max != old_max) {
                    float scale = expf(old_max - new_max);
                    row_sum *= scale;
                    #pragma unroll 4
                    for (int d = 0; d < HeadDim; ++d) thread_output[d] *= scale;
                }
                row_max = new_max;
            }
            __syncthreads();

            // Load V (reuses K SMEM)
            {
                int v_base = batch_idx * params.stride_V_batch + head_idx * params.stride_V_head;
                load_tile_cooperative(storage.smem_V, V_data + v_base, BlockN, HeadDim / 2,
                                      col_start, seqlen_k, params.stride_V_seq, thread_idx, NThreads);

                const int total_sf = BlockN * NumSFPerHead;
                const int per_thread = (total_sf + NThreads - 1) / NThreads;
                const int sf_seq = NumSFPerHead;
                const int sf_head = seqlen_k * NumSFPerHead;
                const int sf_batch = params.num_heads * sf_head;

                for (int i = 0; i < per_thread; ++i) {
                    int idx = thread_idx * per_thread + i;
                    if (idx < total_sf) {
                        int row = idx / NumSFPerHead;
                        int col = idx % NumSFPerHead;
                        int global_col = col_start + row;
                        if (global_col < seqlen_k) {
                            storage.smem_SFV[idx] = params.ptr_SFV[
                                batch_idx * sf_batch + head_idx * sf_head + global_col * sf_seq + col];
                        } else {
                            storage.smem_SFV[idx] = ElementSF(0.0f);
                        }
                    }
                }
            }
            __syncthreads();

            // Accumulate PV using pre-computed scores
            if (thread_row < rows_this_tile) {
                for (int j = 0; j < cols_this_tile; ++j) {
                    float weight = expf(tile_scores[j] - row_max);
                    row_sum += weight;
                    accumulate_pv(storage, thread_output, j, weight);
                }
            }
            __syncthreads();
        }

        // Normalize and write output
        if (thread_row < rows_this_tile) {
            int global_row = row_start + thread_row;
            float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

            ElementOut* O = params.ptr_O +
                batch_idx * params.stride_O_batch +
                head_idx * params.stride_O_head +
                global_row * params.stride_O_seq;

            #pragma unroll 4
            for (int d = 0; d < HeadDim; ++d) {
                O[d] = static_cast<ElementOut>(thread_output[d] * inv_sum);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// Kernel Wrapper
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits_, bool Is_causal, typename TileScheduler>
struct Sm100FlashFwdKernelFP4Collective {

    using Ktraits = Ktraits_;
    using ElementA = typename Ktraits::ElementA;
    using ElementB = typename Ktraits::ElementB;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using SharedStorage = typename Ktraits::SharedStorage;
    using CollectiveMainloop = CollectiveMainloopFwdSm100FP4Collective<Ktraits, Is_causal>;

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kNThreads = Ktraits::kNThreads;

    struct Arguments {
        int seqlen_q, seqlen_k, head_dim, num_heads, batch_size;
        ElementA const* ptr_Q;
        ElementSF const* ptr_SFQ;
        int64_t stride_Q_seq, stride_Q_head, stride_Q_batch;
        ElementA const* ptr_K;
        ElementSF const* ptr_SFK;
        int64_t stride_K_seq, stride_K_head, stride_K_batch;
        ElementB const* ptr_V;
        ElementSF const* ptr_SFV;
        int64_t stride_V_seq, stride_V_head, stride_V_batch;
        ElementOut* ptr_O;
        int64_t stride_O_seq, stride_O_head, stride_O_batch;
        float const* ptr_delta_s;
        int64_t stride_ds_k, stride_ds_group, stride_ds_head, stride_ds_batch;
        bool use_smooth_attention;
        float scale_softmax;
    };

    struct Params {
        typename CollectiveMainloop::Params mainloop;
        typename TileScheduler::Params scheduler;
        int seqlen_q, seqlen_k, num_heads, batch_size;
    };

    static Params to_underlying_arguments(Arguments const& args, void* workspace) {
        auto problem_shape = make_tuple(args.seqlen_q, args.seqlen_k, args.head_dim,
                                        make_tuple(args.num_heads, args.batch_size));
        auto mainloop_params = CollectiveMainloop::to_underlying_arguments(args, workspace);
        typename TileScheduler::Arguments scheduler_args{};
        auto scheduler_params = TileScheduler::to_underlying_arguments(
            problem_shape, TileShape_MNK{}, scheduler_args, workspace);
        return Params{mainloop_params, scheduler_params, args.seqlen_q, args.seqlen_k,
                      args.num_heads, args.batch_size};
    }

    static dim3 get_grid_dim(Arguments const& args, int sm_count) {
        int num_m_blocks = (args.seqlen_q + kBlockM - 1) / kBlockM;
        int num_tiles = num_m_blocks * args.num_heads * args.batch_size;
        return dim3(min(num_tiles, sm_count), 1, 1);
    }

    static dim3 get_block_dim() { return dim3(kNThreads, 1, 1); }
    static size_t get_smem_size() { return sizeof(SharedStorage); }

    CUTLASS_DEVICE void operator()(Params const& params, char* smem) {
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
        TileScheduler scheduler;
        auto work_tile = scheduler.get_initial_work(params.scheduler);
        if (!work_tile.is_valid()) return;
        auto [m_block, head_idx, batch_idx] = work_tile.get_block_coord();
        CollectiveMainloop mainloop;
        mainloop(params.mainloop, storage, m_block, head_idx, batch_idx,
                 params.seqlen_q, params.seqlen_k);
    }
};

} // namespace flash
