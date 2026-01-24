/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 tcgen05.mma block-scaled FP4 tensor core wrappers
 *
 * Similar to wgmma.cuh for SM90, this provides low-level PTX wrappers
 * for SM100's tcgen05.mma block-scaled FP4 instructions.
 */

#pragma once
#include <cuda.h>
#include <cuda_fp8.h>

namespace tcgen05 {

///////////////////////////////////////////////////////////////////////////////
// SMEM Descriptor Creation (similar to wgmma descriptor but for tcgen05)
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

// Create 64-bit SMEM descriptor for tcgen05.mma
// stride: leading dimension stride in bytes (32, 64, or 128 for swizzle modes)
template <int stride, typename T>
__device__ uint64_t make_smem_desc(T* ptr) {
    static_assert(stride == 32 || stride == 64 || stride == 128);
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;  // Box shape
    desc |= matrix_descriptor_encode((uint64_t)(8 * stride)) << 32;  // Stride
    desc |= ((stride == 128) ? 1llu : (stride == 64) ? 2llu : 3llu) << 62;  // Swizzle
    return desc;
}

// Simpler non-swizzled descriptor for FP4 packed data
__device__ __forceinline__ uint64_t make_smem_desc_fp4(void* ptr, int stride_bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)stride_bytes) << 32;
    // No swizzle for now (can add later for performance)
    return desc;
}

///////////////////////////////////////////////////////////////////////////////
// TMEM Address Helpers
///////////////////////////////////////////////////////////////////////////////

// Get TMEM address (SM100 has 64KB TMEM per SM, organized as 32-bit addresses)
__device__ __forceinline__ uint32_t tmem_address(uint32_t offset) {
    return offset;
}

///////////////////////////////////////////////////////////////////////////////
// tcgen05.mma Fence and Sync
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void tcgen05_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.fence.before_thread_sync.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.commit.cta_group::1.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.wait.cta_group::1.sync.aligned;\n" ::: "memory");
#endif
}

///////////////////////////////////////////////////////////////////////////////
// TMEM Allocation/Deallocation
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ uint32_t tcgen05_alloc(uint32_t size_bytes) {
    uint32_t addr;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned %0, %1;\n"
        : "=r"(addr)
        : "r"(size_bytes)
    );
#else
    addr = 0;
#endif
    return addr;
}

__device__ __forceinline__ void tcgen05_dealloc(uint32_t addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned %0;\n"
        :
        : "r"(addr)
    );
#endif
}

///////////////////////////////////////////////////////////////////////////////
// Block-Scaled FP4 MMA (mxf4nvf4 with VS=16)
//
// tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16
//
// C[M,N] = A[M,K] * B[K,N] with block scaling
// - M = 128 (fixed)
// - N = configurable (8-256, multiples of 8)
// - K = 64 per MMA iteration
// - VS = 16 (scale factor applies to 16 elements)
///////////////////////////////////////////////////////////////////////////////

// Block-scaled FP4 MMA: accumulate mode
// tmem_c: TMEM address for accumulator (M x N floats)
// desc_a: 64-bit SMEM descriptor for A (M x K FP4)
// desc_b: 64-bit SMEM descriptor for B (K x N FP4)
// tsfa: TMEM address for A scale factors
// tsfb: TMEM address for B scale factors
template<int N = 128>
__device__ __forceinline__ void mma_mxf4_block16_acc(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t tsfa,
    uint32_t tsfb
) {
    static_assert(N >= 8 && N <= 256 && N % 8 == 0, "N must be 8-256, multiple of 8");

#if defined(CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
    // Only thread 0 in warp issues the MMA
    if (threadIdx.x % 32 == 0) {
        uint32_t scaleC = 1;  // Accumulate (don't zero)
        uint32_t idescE_hi = 0;  // Instruction descriptor (set N size, etc.)

        // Encode N dimension in idescE
        // Format depends on CUDA version
        idescE_hi = (N / 8 - 1) << 8;  // N encoding

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
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idescE_hi), "r"(scaleC),
              "r"(tsfa), "r"(tsfb)
        );
    }
#endif
}

// Block-scaled FP4 MMA: zero accumulator mode
template<int N = 128>
__device__ __forceinline__ void mma_mxf4_block16_zero(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t tsfa,
    uint32_t tsfb
) {
    static_assert(N >= 8 && N <= 256 && N % 8 == 0, "N must be 8-256, multiple of 8");

#if defined(CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
    if (threadIdx.x % 32 == 0) {
        uint32_t scaleC = 0;  // Zero accumulator
        uint32_t idescE_hi = (N / 8 - 1) << 8;

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
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idescE_hi), "r"(scaleC),
              "r"(tsfa), "r"(tsfb)
        );
    }
#endif
}

///////////////////////////////////////////////////////////////////////////////
// TMEM Load/Store
///////////////////////////////////////////////////////////////////////////////

// Load from TMEM to registers
template<typename T, int N>
__device__ __forceinline__ void tmem_load(T* dst, uint32_t tmem_addr, int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Each thread reads its portion based on lane ID
    uint32_t offset = tmem_addr + lane * sizeof(T);
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x1.b32 %0, [%1];\n"
            : "=r"(reinterpret_cast<uint32_t*>(dst)[i])
            : "r"(offset + i * 32 * sizeof(T))
        );
    }
#endif
}

// Store from registers to TMEM
template<typename T, int N>
__device__ __forceinline__ void tmem_store(uint32_t tmem_addr, T* src, int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t offset = tmem_addr + lane * sizeof(T);
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x1.b32 [%0], %1;\n"
            :
            : "r"(offset + i * 32 * sizeof(T)), "r"(reinterpret_cast<uint32_t*>(src)[i])
        );
    }
#endif
}

///////////////////////////////////////////////////////////////////////////////
// TMA Load (Tensor Memory Accelerator)
///////////////////////////////////////////////////////////////////////////////

// Async TMA load from global to shared memory
template<typename T>
__device__ __forceinline__ void tma_load_4d(
    T* smem_dst,
    void const* tma_desc,
    uint64_t* mbar,
    int c0, int c1, int c2, int c3
) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(tma_desc);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));

    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
        : "memory"
    );
}

///////////////////////////////////////////////////////////////////////////////
// Mbarrier Operations
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void mbar_init(uint64_t* bar, int expected_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(bar_ptr), "r"(expected_count)
    );
}

template<uint32_t bytes>
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(bar_ptr), "n"(bytes)
    );
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@!P1 bra.uni LAB_WAIT;\n\t"
        "}\n"
        :
        : "r"(bar_ptr), "r"(phase)
        : "memory"
    );
}

} // namespace tcgen05
