/*
 * Test harness for SM100 tcgen05.mma FP4 block-scaled MMA
 *
 * This file tests the raw PTX instruction to verify it works correctly
 * before integrating into the full FMHA kernel.
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace tcgen05_test {

// TMEM allocation
__device__ __forceinline__ uint32_t tmem_alloc(uint32_t num_columns) {
    uint32_t tmem_addr = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Allocate to a SMEM location first, then read back
    __shared__ uint32_t smem_tmem_addr;
    if (threadIdx.x == 0) {
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :
            : "r"(smem_ptr), "r"(num_columns)
        );
    }
    __syncthreads();
    tmem_addr = smem_tmem_addr;
#endif
    return tmem_addr;
}

// TMEM deallocation
__device__ __forceinline__ void tmem_free(uint32_t tmem_addr, uint32_t num_columns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_addr), "r"(num_columns)
        );
    }
    __syncthreads();
#endif
}

// Create SMEM descriptor for tcgen05.mma
// Format: [start_addr:14][lbo:14][sbo:14][version:2][base_off:3][lbo_mode:1][layout:3]
__device__ __forceinline__ uint64_t make_smem_desc(
    void* smem_ptr,
    int leading_byte_offset,
    int stride_byte_offset
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    uint64_t desc = 0;

    // Start address (bits 0-13, 4 LSB not included)
    desc |= static_cast<uint64_t>((addr >> 4) & 0x3FFF);

    // Leading byte offset (bits 16-29, 4 LSB not included)
    desc |= static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 16;

    // Stride byte offset (bits 32-45, 4 LSB not included)
    desc |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3FFF) << 32;

    // Layout type (bits 61-63) - 0 = SWIZZLE_NONE
    // desc |= 0ULL << 61;

    return desc;
}

// Build instruction descriptor for FP4 block-scaled MMA
// M=128, N=128, K=64, E2M1 format, VS=16
__device__ __forceinline__ uint32_t make_fp4_instr_desc(
    uint32_t tsfa_addr,
    uint32_t tsfb_addr
) {
    uint32_t desc = 0;

    // a_format = E2M1 = 5 (bits 7-9)
    desc |= (5 << 7);

    // b_format = E2M1 = 5 (bits 10-12)
    desc |= (5 << 10);

    // n_dim = N/8 = 128/8 = 16 (bits 17-22)
    desc |= (16 << 17);

    // scale_format = E4M3 = 0 (bit 23) - using E4M3 for scale factors
    // desc |= (0 << 23);

    // m_dim = M/16 = 128/16 = 8 (bits 24-28)
    desc |= (8 << 24);

    // a_sf_id from tsfa_addr upper 2 bits (bits 29-30)
    desc |= ((tsfa_addr >> 30) << 29);

    // b_sf_id from tsfb_addr upper 2 bits (bits 4-5)
    desc |= ((tsfb_addr >> 30) << 4);

    return desc;
}

// Execute tcgen05.mma FP4 block-scaled MMA
// C[128, 128] += A[128, 64] * B[64, 128] with block scaling (VS=16)
__device__ __forceinline__ void tcgen05_mma_fp4(
    uint32_t tmem_c,      // TMEM address for accumulator
    uint64_t desc_a,      // SMEM descriptor for A
    uint64_t desc_b,      // SMEM descriptor for B
    uint32_t tsfa_addr,   // TMEM address for A scale factors
    uint32_t tsfb_addr,   // TMEM address for B scale factors
    bool accumulate       // false = zero, true = accumulate
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t idesc = make_fp4_instr_desc(tsfa_addr, tsfb_addr);
    uint32_t scaleC = accumulate ? 1 : 0;

    // Only one thread per warp issues the MMA (elect_one)
    if ((threadIdx.x & 31) == 0) {
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
              "r"(tsfa_addr), "r"(tsfb_addr)
        );
    }
#endif
}

// Fence before thread sync
__device__ __forceinline__ void tcgen05_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.fence.before_thread_sync.sync.aligned;\n" ::: "memory");
#endif
}

// Commit MMA operations
__device__ __forceinline__ void tcgen05_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.commit.cta_group::1.sync.aligned;\n" ::: "memory");
#endif
}

// Wait for MMA completion
__device__ __forceinline__ void tcgen05_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.wait.cta_group::1.sync.aligned;\n" ::: "memory");
#endif
}

// Load from TMEM to registers
// Each thread gets a portion based on lane ID
__device__ __forceinline__ void tmem_load_row(
    float* dst,           // Output buffer (per-thread)
    uint32_t tmem_addr,   // Base TMEM address
    int row,              // Which row to load
    int cols              // Number of columns (128)
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x / 32;

    // TMEM is organized as 128 DP (depth planes) x 512 columns
    // Each element is 32 bits (float)
    // Row stride depends on M dimension

    // For M=128, each row is stored across the columns
    // Thread mapping: each thread loads cols/32 elements
    int elems_per_thread = cols / 32;

    for (int i = 0; i < elems_per_thread; ++i) {
        int col = lane + i * 32;
        uint32_t offset = tmem_addr + row * cols * 4 + col * 4;  // byte offset

        asm volatile(
            "tcgen05.ld.sync.aligned.32x1.b32 %0, [%1];\n"
            : "=r"(reinterpret_cast<uint32_t&>(dst[i]))
            : "r"(offset)
        );
    }
#endif
}

// Store to TMEM from registers
__device__ __forceinline__ void tmem_store_row(
    uint32_t tmem_addr,
    float* src,
    int row,
    int cols
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    int lane = threadIdx.x & 31;
    int elems_per_thread = cols / 32;

    for (int i = 0; i < elems_per_thread; ++i) {
        int col = lane + i * 32;
        uint32_t offset = tmem_addr + row * cols * 4 + col * 4;

        asm volatile(
            "tcgen05.st.sync.aligned.32x1.b32 [%0], %1;\n"
            :
            : "r"(offset), "r"(reinterpret_cast<uint32_t&>(src[i]))
        );
    }
#endif
}

// Test kernel for tcgen05.mma FP4
__global__ void test_tcgen05_mma_fp4_kernel(
    uint8_t* A,           // FP4 packed [128, 64] -> [128, 32] bytes
    uint8_t* B,           // FP4 packed [64, 128] -> [64, 64] bytes
    uint8_t* SF_A,        // Scale factors for A [128, 4] (4 scale blocks per row for K=64, VS=16)
    uint8_t* SF_B,        // Scale factors for B [64, 8] (8 scale blocks per row for N=128, VS=16)
    float* C,             // Output [128, 128]
    bool verbose
) {
    // Shared memory for A, B, and scale factors
    extern __shared__ uint8_t smem[];

    uint8_t* smem_A = smem;                          // [128, 32] bytes
    uint8_t* smem_B = smem_A + 128 * 32;             // [64, 64] bytes
    uint8_t* smem_SFA = smem_B + 64 * 64;            // [128, 4] bytes
    uint8_t* smem_SFB = smem_SFA + 128 * 4;          // [64, 8] bytes

    // Load A, B, scale factors to SMEM cooperatively
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Load A (4096 bytes)
    for (int i = tid; i < 128 * 32; i += num_threads) {
        smem_A[i] = A[i];
    }

    // Load B (4096 bytes)
    for (int i = tid; i < 64 * 64; i += num_threads) {
        smem_B[i] = B[i];
    }

    // Load scale factors
    for (int i = tid; i < 128 * 4; i += num_threads) {
        smem_SFA[i] = SF_A[i];
    }
    for (int i = tid; i < 64 * 8; i += num_threads) {
        smem_SFB[i] = SF_B[i];
    }

    __syncthreads();

    if (verbose && tid == 0) {
        printf("Data loaded to SMEM\n");
    }

    // Allocate TMEM for accumulator and scale factors
    // Accumulator: 128 x 128 floats = 64KB -> need 128 columns (each column is 128 DP x 32 bits)
    // Scale factors: small, need 32 columns each
    uint32_t tmem_accum = tmem_alloc(128);  // For 128x128 accumulator
    uint32_t tmem_sfa = tmem_alloc(32);     // For A scale factors
    uint32_t tmem_sfb = tmem_alloc(32);     // For B scale factors

    if (verbose && tid == 0) {
        printf("TMEM allocated: accum=%u, sfa=%u, sfb=%u\n", tmem_accum, tmem_sfa, tmem_sfb);
    }

    // TODO: Copy scale factors from SMEM to TMEM
    // This requires UTCCP (User Tensor Core Copy) instructions
    // For now, this is a placeholder

    // Create SMEM descriptors
    // A: [128, 64] FP4 packed as [128, 32] bytes, K-major
    // Leading dimension = 32 bytes (K/2), stride = 32 bytes
    uint64_t desc_a = make_smem_desc(smem_A, 32, 32);

    // B: [64, 128] FP4 packed as [64, 64] bytes, K-major
    // Leading dimension = 64 bytes (N/2), stride = 64 bytes
    uint64_t desc_b = make_smem_desc(smem_B, 64, 64);

    if (verbose && tid == 0) {
        printf("SMEM descriptors: A=0x%llx, B=0x%llx\n",
               (unsigned long long)desc_a, (unsigned long long)desc_b);
    }

    // Execute MMA
    tcgen05_mma_fp4(tmem_accum, desc_a, desc_b, tmem_sfa, tmem_sfb, false);

    // Commit and wait
    tcgen05_commit();
    tcgen05_fence();
    tcgen05_wait();

    __syncthreads();

    if (verbose && tid == 0) {
        printf("MMA executed\n");
    }

    // Load results from TMEM to global memory
    // Each thread loads a portion of the result
    for (int row = tid; row < 128; row += num_threads) {
        float row_data[4];  // 128 / 32 = 4 elements per thread
        tmem_load_row(row_data, tmem_accum, row, 128);

        int lane = tid & 31;
        for (int i = 0; i < 4; ++i) {
            C[row * 128 + lane + i * 32] = row_data[i];
        }
    }

    __syncthreads();

    // Free TMEM
    tmem_free(tmem_accum, 128);
    tmem_free(tmem_sfa, 32);
    tmem_free(tmem_sfb, 32);

    if (verbose && tid == 0) {
        printf("Test complete\n");
    }
}

} // namespace tcgen05_test
