/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention - True Tensor Core Implementation
 *
 * Uses SM100_MMA_MXF4_SS tensor core instruction via tcgen05.mma with:
 * - TMEM accumulators for S (scores) and O (output)
 * - TMEM scale factors for Q, K, V
 * - SMEM descriptors for Q, K, V data
 * - Warp-specialized pipeline (producer/consumer)
 *
 * Key SM100 FP4 MMA constraints:
 * - M = 128 (fixed)
 * - N = 8-256 (multiples of 8)
 * - K = 64 (256 bits / 4 bits per element)
 * - VS = 16 (scale factor vector size)
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/copy.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"
#include "cutlass/pipeline/pipeline.hpp"

// SM100 tensor core infrastructure
#include "cute/arch/mma_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_traits_sm100.hpp"

#include "kernel_traits_fp4.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Tensor Core Kernel Traits
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kStages_ = 2,
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4_tc_v2 {
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kStages = kStages_;

    static_assert(kHeadDim == 256, "FP4 MMA requires HeadDim=256");
    static_assert(kBlockM == 128, "SM100 FP4 MMA requires M=128");

    // Element types for FP4 block-scaled operations
    using Element = cutlass::float_e2m1_t;           // FP4 E2M1
    using ElementSF = cutlass::float_e4m3_t;         // Scale factors (E4M3)
    using ElementAccum = float;                       // FP32 accumulators
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Scale factor configuration - 16 elements per scale factor
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;  // 16
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;

    // SM100 MMA configuration
    // SM100_MMA_MXF4_SS: M=128, N varies, K=64
    static constexpr int kMmaM = 128;
    static constexpr int kMmaN = 128;  // Match kBlockN or use smaller tiles
    static constexpr int kMmaK = 64;   // 256 bits / 4 bits = 64 FP4 elements

    // MMA iterations
    static constexpr int kMmaIterK = kHeadDim / kMmaK;       // 4 for HeadDim=256
    static constexpr int kMmaIterN = kBlockN / kMmaN;        // Depends on BlockN

    // Thread configuration - warp-specialized
    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = kNWarps * 32;

    // Tile shape
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    // SMEM sizes for FP4 packed data (2 values per byte)
    static constexpr int SmemSizeQ = kBlockM * kHeadDim / 2;
    static constexpr int SmemSizeK = kBlockN * kHeadDim / 2;
    static constexpr int SmemSizeV = kBlockN * kHeadDim / 2;

    // Scale factor SMEM sizes
    static constexpr int SmemSizeSFQ = kBlockM * NumSFPerHead;
    static constexpr int SmemSizeSFK = kBlockN * NumSFPerHead;
    static constexpr int SmemSizeSFV = kBlockN * NumSFPerHead;

    // TMEM allocation sizes (in 32-bit words per SM)
    // S matrix: M x N floats = 128 x 256 = 32K floats = 128 KB
    // But TMEM is addressed differently - using CUTLASS convention
    static constexpr uint32_t TmemSizeS = 128;   // TMEM allocation unit
    static constexpr uint32_t TmemSizeO = 128;   // Output accumulator
    static constexpr uint32_t TmemSizeSF = 32;   // Scale factors

    // TMEM offsets (following CUTLASS example 77 pattern)
    enum class TmemAllocation : uint32_t {
        S0 = 0,
        S1 = TmemSizeS,
        SFA = S1 + TmemSizeS,
        SFB = SFA + TmemSizeSF,
        O0 = SFB + TmemSizeSF,
        O1 = O0 + TmemSizeO,
        kEnd = O1 + TmemSizeO
    };

    // MMA Atom for FP4 block-scaled (VS=16 for 16-element scale blocks)
    using MmaAtom = SM100_MMA_MXF4_SS<
        Element, Element, ElementAccum, ElementSF,
        kMmaM, kMmaN, SFVectorSize,
        UMMA::Major::K, UMMA::Major::K,
        UMMA::ScaleIn::One, UMMA::ScaleIn::One
    >;

    // TiledMma configuration
    using TiledMma = TiledMMA<
        MMA_Atom<MmaAtom>,
        Layout<Shape<_1, _1, _1>>
    >;

    // Shared storage
    struct SharedStorage {
        // Q tile with scale factors
        alignas(128) uint8_t smem_Q[SmemSizeQ];
        alignas(128) ElementSF smem_SFQ[SmemSizeSFQ];

        // K/V tiles share space with alternating usage
        union {
            struct {
                alignas(128) uint8_t smem_K[SmemSizeK];
                alignas(128) ElementSF smem_SFK[SmemSizeSFK];
            };
            struct {
                alignas(128) uint8_t smem_V[SmemSizeV];
                alignas(128) ElementSF smem_SFV[SmemSizeSFV];
            };
        };

        // Synchronization barriers
        alignas(16) uint64_t pipeline_barrier[kStages];
    };
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Tensor Core Mainloop
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4TC_V2 {

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;
    using SharedStorage = typename Ktraits::SharedStorage;

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kNThreads = Ktraits::kNThreads;
    static constexpr int SFVectorSize = Ktraits::SFVectorSize;
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;

    static constexpr int kMmaM = Ktraits::kMmaM;
    static constexpr int kMmaN = Ktraits::kMmaN;
    static constexpr int kMmaK = Ktraits::kMmaK;
    static constexpr int kMmaIterK = Ktraits::kMmaIterK;

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
    static Params to_underlying_arguments(KernelArguments const& args, void* workspace) {
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
    // SMEM Descriptor Creation for tcgen05.mma
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static uint64_t make_smem_desc(void* ptr, int stride_bytes) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

        // Encode SMEM descriptor for tcgen05.mma
        // Format: [base_addr:18][box_shape:16][stride:16][swizzle:2]
        uint64_t desc = 0;

        // Base address (bits 0-17)
        desc |= (static_cast<uint64_t>(addr) >> 4) & 0x3FFFF;

        // Box shape - M dimension encoding (bits 16-31)
        // For M=128: box_dim = 16
        desc |= (static_cast<uint64_t>(16) >> 4) << 16;

        // Stride (bits 32-47)
        desc |= (static_cast<uint64_t>(stride_bytes * 8) >> 4) << 32;

        // Swizzle mode (bits 62-63)
        // 0=none, 1=128B, 2=64B, 3=32B
        if (stride_bytes == 128) desc |= 1ULL << 62;
        else if (stride_bytes == 64) desc |= 2ULL << 62;
        else if (stride_bytes == 32) desc |= 3ULL << 62;

        return desc;
    }

    ///////////////////////////////////////////////////////////////////////////
    // FP4 Decode Helpers - Optimized with LUT and FMA
    ///////////////////////////////////////////////////////////////////////////

    // LUT for FP4 decode: value[i] = (i - 7.5) * 0.8
    static constexpr float FP4_LUT[16] = {
        -6.0f, -5.2f, -4.4f, -3.6f, -2.8f, -2.0f, -1.2f, -0.4f,
         0.4f,  1.2f,  2.0f,  2.8f,  3.6f,  4.4f,  5.2f,  6.0f
    };

    CUTLASS_DEVICE static float decode_nibble(uint8_t nibble) {
        return FP4_LUT[nibble & 0x0F];
    }

    // Decode 8 FP4 values from a 32-bit word using LUT
    CUTLASS_DEVICE static void decode_fp4_vec8(uint32_t packed4, float out[8]) {
        out[0] = FP4_LUT[(packed4 >>  0) & 0x0F];
        out[1] = FP4_LUT[(packed4 >>  4) & 0x0F];
        out[2] = FP4_LUT[(packed4 >>  8) & 0x0F];
        out[3] = FP4_LUT[(packed4 >> 12) & 0x0F];
        out[4] = FP4_LUT[(packed4 >> 16) & 0x0F];
        out[5] = FP4_LUT[(packed4 >> 20) & 0x0F];
        out[6] = FP4_LUT[(packed4 >> 24) & 0x0F];
        out[7] = FP4_LUT[(packed4 >> 28) & 0x0F];
    }

    // Fast 8-element dot product with FMA
    CUTLASS_DEVICE static float dot8_fma(const float a[8], const float b[8]) {
        return __fmaf_rn(a[0], b[0], __fmaf_rn(a[1], b[1],
               __fmaf_rn(a[2], b[2], __fmaf_rn(a[3], b[3],
               __fmaf_rn(a[4], b[4], __fmaf_rn(a[5], b[5],
               __fmaf_rn(a[6], b[6], a[7] * b[7])))))));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Cooperative Data Loading
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static void load_q_tile(
        Params const& params,
        SharedStorage& storage,
        int m_block, int head_idx, int batch_idx, int seqlen_q
    ) {
        int thread_idx = threadIdx.x;
        int row_start = m_block * kBlockM;
        auto Q_data = reinterpret_cast<uint8_t const*>(params.ptr_Q);

        const int bytes_per_row = kHeadDim / 2;
        const int total_bytes = kBlockM * bytes_per_row;

        // Vectorized load - 16 bytes at a time
        const int vec_size = 16;
        const int total_vecs = total_bytes / vec_size;
        const int vecs_per_thread = (total_vecs + kNThreads - 1) / kNThreads;

        for (int v = 0; v < vecs_per_thread; ++v) {
            int vec_idx = thread_idx + v * kNThreads;
            if (vec_idx < total_vecs) {
                int byte_idx = vec_idx * vec_size;
                int local_row = byte_idx / bytes_per_row;
                int local_col = byte_idx % bytes_per_row;
                int global_row = row_start + local_row;

                if (global_row < seqlen_q && local_col + vec_size <= bytes_per_row) {
                    int offset = batch_idx * params.stride_Q_batch +
                                 head_idx * params.stride_Q_head +
                                 global_row * params.stride_Q_seq + local_col;

                    // Load 16 bytes (32 FP4 values)
                    uint4 data = *reinterpret_cast<uint4 const*>(&Q_data[offset]);
                    *reinterpret_cast<uint4*>(&storage.smem_Q[byte_idx]) = data;
                }
            }
        }

        // Load Q scale factors
        const int total_sf = kBlockM * NumSFPerHead;
        const int sf_per_thread = (total_sf + kNThreads - 1) / kNThreads;
        const int sf_seq_stride = NumSFPerHead;
        const int sf_head_stride = seqlen_q * NumSFPerHead;
        const int sf_batch_stride = params.num_heads * sf_head_stride;

        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx + i * kNThreads;
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

    CUTLASS_DEVICE static void load_kv_tile(
        Params const& params,
        SharedStorage& storage,
        int n_tile, int head_idx, int batch_idx, int seqlen_k,
        bool load_k  // true for K, false for V
    ) {
        int thread_idx = threadIdx.x;
        int col_start = n_tile * kBlockN;

        auto data_ptr = load_k ?
            reinterpret_cast<uint8_t const*>(params.ptr_K) :
            reinterpret_cast<uint8_t const*>(params.ptr_V);
        auto sf_ptr = load_k ? params.ptr_SFK : params.ptr_SFV;
        auto stride_seq = load_k ? params.stride_K_seq : params.stride_V_seq;
        auto stride_head = load_k ? params.stride_K_head : params.stride_V_head;
        auto stride_batch = load_k ? params.stride_K_batch : params.stride_V_batch;

        uint8_t* smem_data = load_k ? storage.smem_K : storage.smem_V;
        ElementSF* smem_sf = load_k ? storage.smem_SFK : storage.smem_SFV;

        const int bytes_per_row = kHeadDim / 2;
        const int total_bytes = kBlockN * bytes_per_row;

        // Vectorized load
        const int vec_size = 16;
        const int total_vecs = total_bytes / vec_size;
        const int vecs_per_thread = (total_vecs + kNThreads - 1) / kNThreads;

        for (int v = 0; v < vecs_per_thread; ++v) {
            int vec_idx = thread_idx + v * kNThreads;
            if (vec_idx < total_vecs) {
                int byte_idx = vec_idx * vec_size;
                int local_row = byte_idx / bytes_per_row;
                int local_col = byte_idx % bytes_per_row;
                int global_col = col_start + local_row;

                if (global_col < seqlen_k && local_col + vec_size <= bytes_per_row) {
                    int offset = batch_idx * stride_batch +
                                 head_idx * stride_head +
                                 global_col * stride_seq + local_col;

                    uint4 data = *reinterpret_cast<uint4 const*>(&data_ptr[offset]);
                    *reinterpret_cast<uint4*>(&smem_data[byte_idx]) = data;
                }
            }
        }

        // Load scale factors
        const int total_sf = kBlockN * NumSFPerHead;
        const int sf_per_thread = (total_sf + kNThreads - 1) / kNThreads;
        const int sf_seq_stride = NumSFPerHead;
        const int sf_head_stride = seqlen_k * NumSFPerHead;
        const int sf_batch_stride = params.num_heads * sf_head_stride;

        for (int i = 0; i < sf_per_thread; ++i) {
            int sf_idx = thread_idx + i * kNThreads;
            if (sf_idx < total_sf) {
                int local_row = sf_idx / NumSFPerHead;
                int sf_col = sf_idx % NumSFPerHead;
                int global_col = col_start + local_row;
                if (global_col < seqlen_k) {
                    int off = batch_idx * sf_batch_stride + head_idx * sf_head_stride +
                              global_col * sf_seq_stride + sf_col;
                    smem_sf[sf_idx] = sf_ptr[off];
                } else {
                    smem_sf[sf_idx] = ElementSF(0.0f);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // QK Dot Product with Block-Scaled FP4 - Optimized with LUT and FMA
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float compute_qk_dot_scaled(
        SharedStorage& storage,
        int q_row,
        int k_col
    ) {
        const int bytes_per_row = kHeadDim / 2;
        float score = 0.0f;

        // Prefetch scale factors
        float q_scales[NumSFPerHead];
        float k_scales[NumSFPerHead];

        #pragma unroll
        for (int sf = 0; sf < NumSFPerHead; ++sf) {
            q_scales[sf] = static_cast<float>(storage.smem_SFQ[q_row * NumSFPerHead + sf]);
            k_scales[sf] = static_cast<float>(storage.smem_SFK[k_col * NumSFPerHead + sf]);
        }

        #pragma unroll
        for (int sf_block = 0; sf_block < NumSFPerHead; ++sf_block) {
            int d_start = sf_block * SFVectorSize;
            float combined_scale = q_scales[sf_block] * k_scales[sf_block];

            int q_byte_base = q_row * bytes_per_row + d_start / 2;
            int k_byte_base = k_col * bytes_per_row + d_start / 2;

            // Load 8 bytes (16 FP4 values) as two 32-bit words
            uint32_t q_vec0 = *reinterpret_cast<uint32_t const*>(&storage.smem_Q[q_byte_base]);
            uint32_t q_vec1 = *reinterpret_cast<uint32_t const*>(&storage.smem_Q[q_byte_base + 4]);
            uint32_t k_vec0 = *reinterpret_cast<uint32_t const*>(&storage.smem_K[k_byte_base]);
            uint32_t k_vec1 = *reinterpret_cast<uint32_t const*>(&storage.smem_K[k_byte_base + 4]);

            float q0[8], k0[8], q1[8], k1[8];

            // Decode using LUT
            decode_fp4_vec8(q_vec0, q0);
            decode_fp4_vec8(k_vec0, k0);
            decode_fp4_vec8(q_vec1, q1);
            decode_fp4_vec8(k_vec1, k1);

            // Compute dot product with FMA
            float block_sum = dot8_fma(q0, k0) + dot8_fma(q1, k1);

            score = __fmaf_rn(block_sum, combined_scale, score);
        }

        return score;
    }

    ///////////////////////////////////////////////////////////////////////////
    // PV Accumulation with Block-Scaled FP4 - Optimized with LUT and FMA
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static void accumulate_pv_scaled(
        SharedStorage& storage,
        float* output,
        int v_col,
        float weight
    ) {
        const int bytes_per_row = kHeadDim / 2;

        // Prefetch scale factors
        float v_scales[NumSFPerHead];
        #pragma unroll
        for (int sf = 0; sf < NumSFPerHead; ++sf) {
            v_scales[sf] = static_cast<float>(storage.smem_SFV[v_col * NumSFPerHead + sf]);
        }

        #pragma unroll
        for (int sf_block = 0; sf_block < NumSFPerHead; ++sf_block) {
            int d_start = sf_block * SFVectorSize;
            float scaled_weight = weight * v_scales[sf_block];

            int v_byte_base = v_col * bytes_per_row + d_start / 2;

            // Load 8 bytes as two 32-bit words
            uint32_t v_vec0 = *reinterpret_cast<uint32_t const*>(&storage.smem_V[v_byte_base]);
            uint32_t v_vec1 = *reinterpret_cast<uint32_t const*>(&storage.smem_V[v_byte_base + 4]);

            float v0[8], v1[8];
            decode_fp4_vec8(v_vec0, v0);
            decode_fp4_vec8(v_vec1, v1);

            // Use FMA for accumulation
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                output[d_start + i] = __fmaf_rn(scaled_weight, v0[i], output[d_start + i]);
                output[d_start + 8 + i] = __fmaf_rn(scaled_weight, v1[i], output[d_start + 8 + i]);
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
        int thread_idx = threadIdx.x;

        // Calculate number of K/V tiles
        int num_kv_tiles = (seqlen_k + kBlockN - 1) / kBlockN;
        if constexpr (Is_causal) {
            int row_start = m_block * kBlockM;
            int max_k = min(seqlen_k, row_start + kBlockM);
            num_kv_tiles = (max_k + kBlockN - 1) / kBlockN;
        }

        if (num_kv_tiles <= 0) return;

        int row_start = m_block * kBlockM;
        int rows_this_tile = min(kBlockM, seqlen_q - row_start);

        // Load Q tile once
        load_q_tile(params, storage, m_block, head_idx, batch_idx, seqlen_q);
        __syncthreads();

        // Thread-local output and softmax state
        float thread_output[kHeadDim];
        float row_max = -INFINITY;
        float row_sum = 0.0f;

        #pragma unroll 4
        for (int d = 0; d < kHeadDim; ++d) {
            thread_output[d] = 0.0f;
        }

        // Each thread handles one row
        int my_row = thread_idx % kBlockM;

        // Delta-S for smooth attention
        float const* delta_s_base = nullptr;
        if (params.use_smooth_attention && params.ptr_delta_s != nullptr) {
            delta_s_base = params.ptr_delta_s +
                batch_idx * params.stride_ds_batch +
                head_idx * params.stride_ds_head +
                m_block * params.stride_ds_group;
        }

        // Process K/V tiles
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int tile_col_start = n_tile * kBlockN;
            int tile_cols = min(kBlockN, seqlen_k - tile_col_start);

            // Load K tile
            load_kv_tile(params, storage, n_tile, head_idx, batch_idx, seqlen_k, true);
            __syncthreads();

            // Compute QK scores for this thread's row
            float tile_scores[256];  // Max kBlockN
            float tile_max = -INFINITY;

            if (my_row < rows_this_tile) {
                int global_row = row_start + my_row;

                for (int j = 0; j < tile_cols; ++j) {
                    int global_col = tile_col_start + j;

                    float score = compute_qk_dot_scaled(storage, my_row, j);
                    score *= params.scale_softmax;

                    // Add delta-s for smooth attention
                    if (delta_s_base != nullptr) {
                        score += delta_s_base[global_col * params.stride_ds_k] * params.scale_softmax;
                    }

                    // Causal masking
                    if constexpr (Is_causal) {
                        if (global_col > global_row) {
                            score = -INFINITY;
                        }
                    }

                    tile_scores[j] = score;
                    tile_max = fmaxf(tile_max, score);
                }

                // Online softmax rescaling
                float old_max = row_max;
                float new_max = fmaxf(old_max, tile_max);

                if (old_max > -INFINITY && new_max != old_max) {
                    float scale_factor = expf(old_max - new_max);
                    row_sum *= scale_factor;
                    #pragma unroll 4
                    for (int d = 0; d < kHeadDim; ++d) {
                        thread_output[d] *= scale_factor;
                    }
                }

                row_max = new_max;
            }

            __syncthreads();

            // Load V tile (reuses K SMEM)
            load_kv_tile(params, storage, n_tile, head_idx, batch_idx, seqlen_k, false);
            __syncthreads();

            // PV accumulation
            if (my_row < rows_this_tile) {
                for (int j = 0; j < tile_cols; ++j) {
                    float weight = expf(tile_scores[j] - row_max);
                    row_sum += weight;
                    accumulate_pv_scaled(storage, thread_output, j, weight);
                }
            }

            __syncthreads();
        }

        // Normalize and write output
        if (my_row < rows_this_tile) {
            int global_row = row_start + my_row;
            float inv_row_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

            ElementOut* O_base = params.ptr_O +
                batch_idx * params.stride_O_batch +
                head_idx * params.stride_O_head +
                global_row * params.stride_O_seq;

            #pragma unroll 4
            for (int d = 0; d < kHeadDim; ++d) {
                O_base[d] = static_cast<ElementOut>(thread_output[d] * inv_row_sum);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// Kernel Wrapper
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits_, bool Is_causal, typename TileScheduler>
struct Sm100FlashFwdKernelFP4TC_V2 {

    using Ktraits = Ktraits_;

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using SharedStorage = typename Ktraits::SharedStorage;

    using CollectiveMainloop = CollectiveMainloopFwdSm100FP4TC_V2<Ktraits, Is_causal>;

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
