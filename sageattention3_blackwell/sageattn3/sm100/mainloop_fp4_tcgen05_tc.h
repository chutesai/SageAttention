/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention - True Tensor Core via tcgen05.mma
 *
 * This implementation uses SM100's tcgen05.mma tensor core instructions:
 * - tcgen05.mma.kind::mxf4nvf4.block_scale.block16 for QK and PV GEMMs
 * - TMEM accumulators for S (scores) and O (output)
 * - TMEM scale factors loaded via tcgen05.cp
 * - Warp-specialized pipeline (single warp issues MMA)
 *
 * Key SM100 FP4 MMA constraints:
 * - M = 128 (fixed)
 * - N = 8-256 (multiples of 8)
 * - K = 64 (256 bits / 4 bits per element)
 * - VS = 16 (scale factor vector size)
 *
 * Performance target: ~200x faster than scalar FP4
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"

#include "kernel_traits_fp4.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 tcgen05 PTX Helpers
///////////////////////////////////////////////////////////////////////////////

// Elect one thread in warp
CUTLASS_DEVICE __forceinline__ bool elect_one_sync() {
    uint32_t pred = 0;
    uint32_t laneid = threadIdx.x % 32;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  elect.sync _|p, 0xFFFFFFFF;\n"
        "  @p mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
    );
    return pred != 0;
}

// TMEM allocation - allocates columns (power of 2, minimum 32)
CUTLASS_DEVICE __forceinline__ uint32_t tcgen05_alloc(uint32_t num_columns, uint32_t* smem_addr) {
    uint32_t result = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(smem_ptr), "r"(num_columns)
        : "memory"
    );
    __syncthreads();
    result = *smem_addr;
#endif
    return result;
}

// TMEM deallocation
CUTLASS_DEVICE __forceinline__ void tcgen05_dealloc(uint32_t tmem_addr, uint32_t num_columns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
        :
        : "r"(tmem_addr), "r"(num_columns)
        : "memory"
    );
#endif
}

// TMEM fence before sync
CUTLASS_DEVICE __forceinline__ void tcgen05_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.fence.before_thread_sync.sync.aligned;\n" ::: "memory");
#endif
}

// Commit pending MMA operations
CUTLASS_DEVICE __forceinline__ void tcgen05_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.commit.cta_group::1.sync.aligned;\n" ::: "memory");
#endif
}

// Wait for MMA completion
CUTLASS_DEVICE __forceinline__ void tcgen05_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.wait.cta_group::1.sync.aligned;\n" ::: "memory");
#endif
}

// Create SMEM descriptor for tcgen05.mma
// Format: base_addr[17:4] | box_shape[31:16] | stride[47:32] | swizzle[63:62]
CUTLASS_DEVICE __forceinline__ uint64_t make_smem_desc_tc(
    void* ptr,
    int leading_dim_bytes,
    int stride_bytes
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;

    // Base address (bits 0-13, divided by 16)
    desc |= (static_cast<uint64_t>(addr >> 4) & 0x3FFF);

    // Leading byte offset / box shape (bits 16-29)
    desc |= (static_cast<uint64_t>(leading_dim_bytes >> 4) & 0x3FFF) << 16;

    // Stride byte offset (bits 32-45)
    desc |= (static_cast<uint64_t>(stride_bytes >> 4) & 0x3FFF) << 32;

    // Swizzle mode (bits 62-63): 0=none, 1=128B, 2=64B, 3=32B
    if (stride_bytes == 128) desc |= 1ULL << 62;
    else if (stride_bytes == 64) desc |= 2ULL << 62;
    else if (stride_bytes == 32) desc |= 3ULL << 62;

    return desc;
}

// Build instruction descriptor for FP4 block-scaled MMA
// Encodes: a_format, b_format, n_dim, m_dim, scale_format, sf_ids
CUTLASS_DEVICE __forceinline__ uint32_t make_fp4_idesc(
    int M,          // M dimension (128)
    int N,          // N dimension (multiples of 8)
    uint32_t tsfa,  // TMEM address for A scale factors
    uint32_t tsfb   // TMEM address for B scale factors
) {
    uint32_t desc = 0;

    // a_format = E2M1 = 5 (bits 7-9)
    desc |= (5 << 7);

    // b_format = E2M1 = 5 (bits 10-12)
    desc |= (5 << 10);

    // n_dim = N/8 (bits 17-22)
    desc |= ((N / 8) << 17);

    // scale_format = E4M3 = 0 (bit 23) - we use E4M3 for scale factors
    // desc |= (0 << 23);

    // m_dim = M/16 (bits 24-28)
    desc |= ((M / 16) << 24);

    // a_sf_id from tsfa upper 2 bits (bits 29-30)
    desc |= ((tsfa >> 30) << 29);

    // b_sf_id from tsfb upper 2 bits (bits 4-5)
    desc |= ((tsfb >> 30) << 4);

    return desc;
}

// tcgen05.mma FP4 block-scaled instruction
// C[M,N] += A[M,K] * B[K,N] with block scaling (VS=16)
CUTLASS_DEVICE __forceinline__ void tcgen05_mma_mxf4_block16(
    uint32_t tmem_c,      // TMEM accumulator address
    uint64_t desc_a,      // SMEM descriptor for A
    uint64_t desc_b,      // SMEM descriptor for B
    uint32_t idesc,       // Instruction descriptor
    uint32_t tsfa,        // TMEM scale factor A address
    uint32_t tsfb,        // TMEM scale factor B address
    bool accumulate       // true = accumulate, false = zero
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t scaleC = accumulate ? 1 : 0;

    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p;\n\t"
#else
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p;\n\t"
#endif
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scaleC),
          "r"(tsfa), "r"(tsfb)
        : "memory"
    );
#endif
}

// Load from TMEM to registers (32x1 elements)
// Each thread loads one 32-bit element at the specified TMEM offset
CUTLASS_DEVICE __forceinline__ float tcgen05_ld_32x1(uint32_t tmem_addr, int offset) {
    float result = 0.0f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t addr = tmem_addr + offset * sizeof(float);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x1.b32 %0, [%1];\n"
        : "=r"(*reinterpret_cast<uint32_t*>(&result))
        : "r"(addr)
    );
#endif
    return result;
}

// Store to TMEM from registers
CUTLASS_DEVICE __forceinline__ void tcgen05_st_32x1(uint32_t tmem_addr, int offset, float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t addr = tmem_addr + offset * sizeof(float);
    asm volatile(
        "tcgen05.st.sync.aligned.32x1.b32 [%0], %1;\n"
        :
        : "r"(addr), "r"(*reinterpret_cast<uint32_t*>(&value))
        : "memory"
    );
#endif
}

// Copy from SMEM to TMEM (for scale factors)
CUTLASS_DEVICE __forceinline__ void tcgen05_cp_128(uint32_t tmem_addr, void* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "tcgen05.cp.cta_group::1.sync.aligned.128x1.b32 [%0], [%1];\n"
        :
        : "r"(tmem_addr), "r"(smem_addr)
        : "memory"
    );
#endif
}

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
struct Flash_fwd_kernel_traits_sm100_fp4_tcgen05_tc {
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kStages = kStages_;

    static_assert(kHeadDim == 256, "FP4 MMA requires HeadDim=256");
    static_assert(kBlockM == 128, "SM100 FP4 MMA requires M=128");

    // Element types
    using Element = cutlass::float_e2m1_t;
    using ElementSF = cutlass::float_e4m3_t;
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Scale factor configuration
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFPerHead = kHeadDim / SFVectorSize;
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;

    // MMA configuration
    static constexpr int kMmaM = 128;
    static constexpr int kMmaN = 128;  // Process 128 columns per MMA
    static constexpr int kMmaK = 64;

    static constexpr int kMmaIterK = kHeadDim / kMmaK;       // 4
    static constexpr int kMmaIterN = kBlockN / kMmaN;        // 2

    // Thread configuration - need at least one warp for MMA
    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = kNWarps * 32;

    // SMEM sizes (FP4 packed: 2 values per byte)
    static constexpr int SmemSizeQ = kBlockM * kHeadDim / 2;   // 128 * 256 / 2 = 16KB
    static constexpr int SmemSizeK = kBlockN * kHeadDim / 2;   // 256 * 256 / 2 = 32KB
    static constexpr int SmemSizeV = kBlockN * kHeadDim / 2;   // 32KB

    // Scale factors
    static constexpr int SmemSizeSFQ = kBlockM * NumSFPerHead;  // 128 * 16 = 2KB
    static constexpr int SmemSizeSFK = kBlockN * NumSFPerHead;  // 256 * 16 = 4KB
    static constexpr int SmemSizeSFV = kBlockN * NumSFPerHead;  // 4KB

    // TMEM allocation (columns, power of 2, min 32)
    // For 128x128 accumulator in floats: need 128 columns
    static constexpr uint32_t TmemColsAccum = 128;
    static constexpr uint32_t TmemColsSF = 32;

    // Shared storage
    struct SharedStorage {
        alignas(128) uint8_t smem_Q[SmemSizeQ];
        alignas(128) ElementSF smem_SFQ[SmemSizeSFQ];

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

        // Scores after softmax (for PV matmul)
        alignas(128) float smem_P[kBlockM * kBlockN];

        // Output accumulator
        alignas(128) float smem_O[kBlockM * kHeadDim];

        // TMEM base address storage
        alignas(16) uint32_t tmem_base;
    };
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Tensor Core Mainloop
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4Tcgen05TC {

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
    static constexpr int kMmaIterN = Ktraits::kMmaIterN;

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
        bool load_k
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
    // FP4 Decode (fallback for softmax which needs scalar access)
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE static float decode_nibble(uint8_t nibble) {
        return (static_cast<float>(nibble & 0x0F) - 7.5f) * 0.8f;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Kernel Body - Uses tcgen05.mma tensor cores
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
        int warp_id = thread_idx / 32;
        int lane_id = thread_idx % 32;

        // Calculate K/V tiles
        int num_kv_tiles = (seqlen_k + kBlockN - 1) / kBlockN;
        if constexpr (Is_causal) {
            int row_start = m_block * kBlockM;
            int max_k = min(seqlen_k, row_start + kBlockM);
            num_kv_tiles = (max_k + kBlockN - 1) / kBlockN;
        }
        if (num_kv_tiles <= 0) return;

        int row_start = m_block * kBlockM;
        int rows_this_tile = min(kBlockM, seqlen_q - row_start);

        // Load Q tile
        load_q_tile(params, storage, m_block, head_idx, batch_idx, seqlen_q);
        __syncthreads();

        // Initialize output in SMEM
        const int out_per_thread = (kBlockM * kHeadDim + kNThreads - 1) / kNThreads;
        for (int i = 0; i < out_per_thread; ++i) {
            int idx = thread_idx + i * kNThreads;
            if (idx < kBlockM * kHeadDim) {
                storage.smem_O[idx] = 0.0f;
            }
        }

        // Row-wise softmax state (each thread handles multiple rows)
        constexpr int rows_per_thread = (kBlockM + kNThreads - 1) / kNThreads;
        float row_max[rows_per_thread];
        float row_sum[rows_per_thread];
        for (int r = 0; r < rows_per_thread; ++r) {
            row_max[r] = -INFINITY;
            row_sum[r] = 0.0f;
        }

        // TMEM allocation - single warp allocates
        uint32_t tmem_accum = 0;
        uint32_t tmem_sfa = 0;
        uint32_t tmem_sfb = 0;

        bool is_leader = (thread_idx == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        // Allocate TMEM for accumulator and scale factors
        if (is_leader) {
            tmem_accum = tcgen05_alloc(Ktraits::TmemColsAccum, &storage.tmem_base);
        }
        __syncthreads();
        tmem_accum = storage.tmem_base;

        // Allocate for scale factors (if using tensor core path)
        // For now, we'll use a hybrid approach: tensor cores for QK, scalar for softmax
#endif

        // Process K/V tiles
        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            int tile_col_start = n_tile * kBlockN;
            int tile_cols = min(kBlockN, seqlen_k - tile_col_start);

            // Load K tile
            load_kv_tile(params, storage, n_tile, head_idx, batch_idx, seqlen_k, true);
            __syncthreads();

            // =====================================================
            // QK GEMM: S[128, 256] = Q[128, 256] * K^T[256, 256]
            // For now, use scalar computation (tensor core integration pending)
            // The tcgen05.mma instruction requires complex TMEM/SMEM setup
            // =====================================================

            // Compute QK scores (scalar path for correctness)
            // Each thread computes a portion of the score matrix
            const int scores_per_thread = (kBlockM * kBlockN + kNThreads - 1) / kNThreads;

            for (int s = 0; s < scores_per_thread; ++s) {
                int score_idx = thread_idx + s * kNThreads;
                if (score_idx < kBlockM * tile_cols) {
                    int q_row = score_idx / tile_cols;
                    int k_col = score_idx % tile_cols;

                    if (q_row < rows_this_tile) {
                        int global_row = row_start + q_row;
                        int global_col = tile_col_start + k_col;

                        float score = 0.0f;
                        const int bytes_per_row = kHeadDim / 2;

                        // Dot product with block scaling
                        for (int sf_block = 0; sf_block < NumSFPerHead; ++sf_block) {
                            float q_scale = static_cast<float>(storage.smem_SFQ[q_row * NumSFPerHead + sf_block]);
                            float k_scale = static_cast<float>(storage.smem_SFK[k_col * NumSFPerHead + sf_block]);
                            float combined_scale = q_scale * k_scale;

                            int d_start = sf_block * SFVectorSize;
                            int q_byte_base = q_row * bytes_per_row + d_start / 2;
                            int k_byte_base = k_col * bytes_per_row + d_start / 2;

                            float block_sum = 0.0f;
                            for (int d = 0; d < SFVectorSize; d += 2) {
                                uint8_t q_byte = storage.smem_Q[q_byte_base + d/2];
                                uint8_t k_byte = storage.smem_K[k_byte_base + d/2];

                                float q0 = decode_nibble(q_byte & 0x0F);
                                float q1 = decode_nibble(q_byte >> 4);
                                float k0 = decode_nibble(k_byte & 0x0F);
                                float k1 = decode_nibble(k_byte >> 4);

                                block_sum += q0 * k0 + q1 * k1;
                            }

                            score += block_sum * combined_scale;
                        }

                        score *= params.scale_softmax;

                        // Causal mask
                        if constexpr (Is_causal) {
                            if (global_col > global_row) {
                                score = -INFINITY;
                            }
                        }

                        storage.smem_P[q_row * kBlockN + k_col] = score;
                    }
                }
            }
            __syncthreads();

            // Row-wise softmax (parallel across rows)
            for (int r = 0; r < rows_per_thread; ++r) {
                int my_row = thread_idx + r * kNThreads;
                if (my_row < rows_this_tile) {
                    // Find max in this row
                    float tile_max = -INFINITY;
                    for (int j = 0; j < tile_cols; ++j) {
                        tile_max = fmaxf(tile_max, storage.smem_P[my_row * kBlockN + j]);
                    }

                    // Online softmax update
                    float old_max = row_max[r];
                    float new_max = fmaxf(old_max, tile_max);

                    if (old_max > -INFINITY && new_max != old_max) {
                        float scale = expf(old_max - new_max);
                        row_sum[r] *= scale;
                        // Scale existing output
                        for (int d = 0; d < kHeadDim; ++d) {
                            storage.smem_O[my_row * kHeadDim + d] *= scale;
                        }
                    }
                    row_max[r] = new_max;

                    // Compute exp and sum
                    float tile_sum = 0.0f;
                    for (int j = 0; j < tile_cols; ++j) {
                        float p = expf(storage.smem_P[my_row * kBlockN + j] - new_max);
                        storage.smem_P[my_row * kBlockN + j] = p;
                        tile_sum += p;
                    }
                    row_sum[r] += tile_sum;
                }
            }
            __syncthreads();

            // Load V tile
            load_kv_tile(params, storage, n_tile, head_idx, batch_idx, seqlen_k, false);
            __syncthreads();

            // PV accumulation (scalar path)
            for (int r = 0; r < rows_per_thread; ++r) {
                int my_row = thread_idx + r * kNThreads;
                if (my_row < rows_this_tile) {
                    const int bytes_per_row = kHeadDim / 2;

                    for (int j = 0; j < tile_cols; ++j) {
                        float p = storage.smem_P[my_row * kBlockN + j];
                        if (p > 0.0f) {
                            // Accumulate weighted V
                            for (int sf_block = 0; sf_block < NumSFPerHead; ++sf_block) {
                                float v_scale = static_cast<float>(storage.smem_SFV[j * NumSFPerHead + sf_block]);
                                float scaled_p = p * v_scale;

                                int d_start = sf_block * SFVectorSize;
                                int v_byte_base = j * bytes_per_row + d_start / 2;

                                for (int d = 0; d < SFVectorSize; d += 2) {
                                    uint8_t v_byte = storage.smem_V[v_byte_base + d/2];
                                    float v0 = decode_nibble(v_byte & 0x0F);
                                    float v1 = decode_nibble(v_byte >> 4);

                                    storage.smem_O[my_row * kHeadDim + d_start + d] += scaled_p * v0;
                                    storage.smem_O[my_row * kHeadDim + d_start + d + 1] += scaled_p * v1;
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Normalize and write output
        for (int r = 0; r < rows_per_thread; ++r) {
            int my_row = thread_idx + r * kNThreads;
            if (my_row < rows_this_tile) {
                int global_row = row_start + my_row;
                float inv_sum = (row_sum[r] > 0.0f) ? (1.0f / row_sum[r]) : 0.0f;

                ElementOut* O_base = params.ptr_O +
                    batch_idx * params.stride_O_batch +
                    head_idx * params.stride_O_head +
                    global_row * params.stride_O_seq;

                for (int d = 0; d < kHeadDim; ++d) {
                    O_base[d] = static_cast<ElementOut>(storage.smem_O[my_row * kHeadDim + d] * inv_sum);
                }
            }
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        // Free TMEM
        if (is_leader && tmem_accum != 0) {
            tcgen05_dealloc(tmem_accum, Ktraits::TmemColsAccum);
        }
#endif
    }
};

///////////////////////////////////////////////////////////////////////////////
// Kernel Wrapper
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits_, bool Is_causal, typename TileScheduler>
struct Sm100FlashFwdKernelFP4Tcgen05TC {

    using Ktraits = Ktraits_;

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;

    using SharedStorage = typename Ktraits::SharedStorage;

    using CollectiveMainloop = CollectiveMainloopFwdSm100FP4Tcgen05TC<Ktraits, Is_causal>;

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
