/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SM100 (B200/B300) Blockscaled FP4 MMA atoms using tcgen05 instructions.
 * Unlike SM120 which uses mma.sync.aligned, SM100 uses tcgen05.mma with
 * Tensor Memory (TMEM) for accumulator storage.
 */

#pragma once

#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/util/type_traits.hpp"

#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/detail/helper_macros.hpp"

namespace cute {

///////////////////////////////////////////////////////////////////////////////
// SM100 SMEM Descriptor for tcgen05.mma
//
// 64-bit descriptor encoding:
// - Bits 0-13:   Start address (upper bits, addr >> 4)
// - Bits 14-15:  Reserved
// - Bits 16-29:  Leading Byte Offset (LBO)
// - Bits 30-31:  Reserved
// - Bits 32-45:  Stride Byte Offset (SBO)
// - Bit 46:      Valid flag (must be 1)
// - Bits 47-60:  Reserved
// - Bits 61-63:  Swizzle mode (0=none, 1=64B, 2=128B, 3=256B)
///////////////////////////////////////////////////////////////////////////////

struct SmemDescriptorSm100 {
    uint64_t desc_;

    CUTE_HOST_DEVICE constexpr
    SmemDescriptorSm100() : desc_(0) {}

    CUTE_HOST_DEVICE constexpr
    SmemDescriptorSm100(uint64_t desc) : desc_(desc) {}

    CUTE_HOST_DEVICE
    uint64_t get() const { return desc_; }

    // Create descriptor from SMEM pointer and layout info
    // swizzle_mode: 0=none, 1=64B, 2=128B (recommended), 3=256B
    CUTE_DEVICE static
    SmemDescriptorSm100 make(void const* smem_ptr,
                              uint32_t leading_byte_offset,
                              uint32_t stride_byte_offset,
                              uint32_t swizzle_mode = 2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

        uint64_t desc = 0;
        // Start address: bits 0-13 (addr >> 4, keeping upper 14 bits)
        desc |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFF);
        // Leading Byte Offset: bits 16-29
        desc |= static_cast<uint64_t>(leading_byte_offset & 0x3FFF) << 16;
        // Stride Byte Offset: bits 32-45
        desc |= static_cast<uint64_t>(stride_byte_offset & 0x3FFF) << 32;
        // Valid flag: bit 46
        desc |= (1ULL << 46);
        // Swizzle mode: bits 61-63
        desc |= static_cast<uint64_t>(swizzle_mode & 0x7) << 61;

        return SmemDescriptorSm100(desc);
#else
        return SmemDescriptorSm100(0);
#endif
    }

    // Simplified constructor for common case with 128B swizzle
    CUTE_DEVICE static
    SmemDescriptorSm100 make_128B_swizzle(void const* smem_ptr,
                                          int rows, int cols, int elem_bytes) {
        // For 128B swizzle with row-major layout:
        // LBO = cols * elem_bytes (stride between rows)
        // SBO = 8 * cols * elem_bytes (stride for 8-row groups, used in swizzle)
        uint32_t lbo = cols * elem_bytes;
        uint32_t sbo = 8 * cols * elem_bytes;
        return make(smem_ptr, lbo, sbo, 2);
    }
};

///////////////////////////////////////////////////////////////////////////////
// SM100 Instruction Descriptor for tcgen05.mma
//
// 32-bit descriptor encoding MMA operation parameters:
// - Bits 0-3:   Reserved
// - Bits 4-6:   Output dtype (0=FP16, 1=FP32, 2=BF16)
// - Bits 7-9:   Input A dtype
// - Bits 10-12: Input B dtype
// - Bits 13-16: Reserved
// - Bits 17-23: MMA_N >> 3
// - Bits 24-30: MMA_M >> 4
// - Bit 31:     Reserved
///////////////////////////////////////////////////////////////////////////////

struct InstrDescriptorSm100 {
    uint32_t desc_;

    CUTE_HOST_DEVICE constexpr
    InstrDescriptorSm100() : desc_(0) {}

    CUTE_HOST_DEVICE constexpr
    InstrDescriptorSm100(uint32_t desc) : desc_(desc) {}

    CUTE_HOST_DEVICE
    uint32_t get() const { return desc_; }

    // Create instruction descriptor for blockscaled mxf4nvf4
    // Output is always FP32, inputs are e2m1 (FP4)
    CUTE_HOST_DEVICE static constexpr
    InstrDescriptorSm100 make_mxf4nvf4(uint32_t M, uint32_t N) {
        uint32_t desc = 0;
        // Output dtype FP32 = 1
        desc |= (1U << 4);
        // Input A dtype (mxf4 encoding)
        desc |= (4U << 7);   // FP4 e2m1 type
        // Input B dtype (nvf4 encoding)
        desc |= (4U << 10);  // FP4 e2m1 type
        // MMA_N dimension
        desc |= ((N >> 3) & 0x7F) << 17;
        // MMA_M dimension
        desc |= ((M >> 4) & 0x7F) << 24;
        return InstrDescriptorSm100(desc);
    }
};

///////////////////////////////////////////////////////////////////////////////
// TMEM Allocator for SM100
// TMEM has 128 rows × 512 columns of 32-bit cells per SM
///////////////////////////////////////////////////////////////////////////////

struct TmemAllocatorSm100 {
    static constexpr uint32_t kTmemRows = 128;
    static constexpr uint32_t kTmemCols = 512;
    static constexpr uint32_t kTmemCellBytes = 4;  // 32-bit cells

    uint32_t base_addr_;
    uint32_t num_cols_;

    CUTE_HOST_DEVICE constexpr
    TmemAllocatorSm100() : base_addr_(0), num_cols_(0) {}

    // Allocate TMEM columns
    // Returns base address, or 0 on failure
    CUTE_DEVICE
    uint32_t allocate(uint32_t num_cols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        uint32_t addr = 0;
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 %0, %1;\n"
            : "=r"(addr)
            : "r"(num_cols)
            : "memory"
        );
        if (addr != 0) {
            base_addr_ = addr;
            num_cols_ = num_cols;
        }
        return addr;
#else
        return 0;
#endif
    }

    // Calculate required columns for a given tile size
    CUTE_HOST_DEVICE static constexpr
    uint32_t cols_for_tile(uint32_t rows, uint32_t cols, uint32_t elem_bytes = 4) {
        // Each TMEM column holds 128 rows of 32-bit values
        // Total elements = rows * cols
        // Columns needed = ceil(rows * cols * elem_bytes / (128 * 4))
        return (rows * cols * elem_bytes + (kTmemRows * kTmemCellBytes - 1)) /
               (kTmemRows * kTmemCellBytes);
    }

    // Deallocate TMEM (MUST be called before kernel exit)
    CUTE_DEVICE
    void deallocate() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        if (base_addr_ != 0) {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                :
                : "r"(base_addr_), "r"(num_cols_)
                : "memory"
            );
            base_addr_ = 0;
            num_cols_ = 0;
        }
#endif
    }

    // Compute TMEM address for a specific (row, col) position
    CUTE_DEVICE static
    uint32_t addr_at(uint32_t base, uint32_t row, uint32_t col) {
        // TMEM address encoding: bits [31:16] = row, bits [15:0] = column
        return base + (row << 16) + col;
    }
};

///////////////////////////////////////////////////////////////////////////////
// TMEM Load/Store Operations
///////////////////////////////////////////////////////////////////////////////

// Load 8 float values from TMEM to registers
CUTE_DEVICE
void tmem_load_8xf32(float* dst, uint32_t tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr)
        : "memory"
    );
#endif
}

// Load 4 float values from TMEM
CUTE_DEVICE
void tmem_load_4xf32(float* dst, uint32_t tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr)
        : "memory"
    );
#endif
}

// Load single float from TMEM
CUTE_DEVICE
float tmem_load_1xf32(uint32_t tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    float result;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];\n"
        : "=f"(result)
        : "r"(tmem_addr)
        : "memory"
    );
    return result;
#else
    return 0.0f;
#endif
}

// Wait for TMEM loads to complete
CUTE_DEVICE
void tmem_load_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
#endif
}

// Store to TMEM (8 floats)
CUTE_DEVICE
void tmem_store_8xf32(uint32_t tmem_addr, float const* src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "
        "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
        :
        : "r"(tmem_addr),
          "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
          "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7])
        : "memory"
    );
#endif
}

///////////////////////////////////////////////////////////////////////////////
// mbarrier Operations for tcgen05 Synchronization
///////////////////////////////////////////////////////////////////////////////

// Initialize mbarrier
CUTE_DEVICE
void mbarrier_init(uint32_t mbar_smem_addr, uint32_t expected_arrivals) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(mbar_smem_addr), "r"(expected_arrivals)
        : "memory"
    );
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
}

// Signal MMA completion via mbarrier (called after tcgen05.mma)
CUTE_DEVICE
void umma_commit(uint32_t mbar_smem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :
        : "r"(mbar_smem_addr)
        : "memory"
    );
#endif
}

// Wait on mbarrier with phase
CUTE_DEVICE
void mbarrier_wait(uint32_t mbar_smem_addr, uint32_t phase) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t ticks = 0x989680;  // Timeout ticks
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :
        : "r"(mbar_smem_addr), "r"(phase), "r"(ticks)
        : "memory"
    );
#endif
}

// Arrive at mbarrier (non-tcgen05)
CUTE_DEVICE
void mbarrier_arrive(uint32_t mbar_smem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
        :
        : "r"(mbar_smem_addr)
        : "memory"
    );
#endif
}

///////////////////////////////////////////////////////////////////////////////
// Copy Scale Factors from SMEM to TMEM
///////////////////////////////////////////////////////////////////////////////

// Copy scale factors from SMEM to TMEM using tcgen05.cp
// This handles the format conversion automatically
CUTE_DEVICE
void copy_sf_smem_to_tmem(uint32_t tmem_addr, void const* smem_ptr,
                          uint32_t num_rows, uint32_t num_cols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    // tcgen05.cp copies scale factors with automatic layout transformation
    asm volatile(
        "tcgen05.cp.cta_group::1.warpx4::02_13.b32 [%0], [%1];\n"
        :
        : "r"(tmem_addr), "r"(smem_addr)
        : "memory"
    );
#endif
}

///////////////////////////////////////////////////////////////////////////////
// SM100 Blockscaled FP4 MMA Operation
//
// tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
// - A, B: e2m1 (FP4) packed format
// - SFA, SFB: FP8 E4M3 scale factors
// - C, D: FP32 accumulators in TMEM
///////////////////////////////////////////////////////////////////////////////

struct Sm100BlockscaledMma {
    // Issue blockscaled FP4 MMA instruction
    // d_tmem: Output TMEM address
    // a_desc, b_desc: 64-bit SMEM descriptors
    // idesc: 32-bit instruction descriptor
    // sfa_tmem, sfb_tmem: Scale factor TMEM addresses
    // accumulate: If true, D = C + A*B; if false, D = A*B
    CUTE_DEVICE static
    void mma(uint32_t d_tmem,
             uint64_t a_desc,
             uint64_t b_desc,
             uint32_t idesc,
             uint32_t sfa_tmem,
             uint32_t sfb_tmem,
             bool accumulate) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        if (accumulate) {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
                "[%0], %1, %2, %3, [%4], [%5], 1;\n"
                :
                : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
                  "r"(sfa_tmem), "r"(sfb_tmem)
                : "memory"
            );
        } else {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
                "[%0], %1, %2, %3, [%4], [%5], 0;\n"
                :
                : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
                  "r"(sfa_tmem), "r"(sfb_tmem)
                : "memory"
            );
        }
#endif
    }

    // MMA for 2-SM cluster
    CUTE_DEVICE static
    void mma_2sm(uint32_t d_tmem,
                 uint64_t a_desc,
                 uint64_t b_desc,
                 uint32_t idesc,
                 uint32_t sfa_tmem,
                 uint32_t sfb_tmem,
                 bool accumulate) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        if (accumulate) {
            asm volatile(
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
                "[%0], %1, %2, %3, [%4], [%5], 1;\n"
                :
                : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
                  "r"(sfa_tmem), "r"(sfb_tmem)
                : "memory"
            );
        } else {
            asm volatile(
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
                "[%0], %1, %2, %3, [%4], [%5], 0;\n"
                :
                : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
                  "r"(sfa_tmem), "r"(sfb_tmem)
                : "memory"
            );
        }
#endif
    }
};

///////////////////////////////////////////////////////////////////////////////
// Helper to create SMEM descriptor for Q, K, V matrices
///////////////////////////////////////////////////////////////////////////////

// Create descriptor for Q matrix (M x K layout, row-major)
CUTE_DEVICE
uint64_t make_q_smem_desc(void const* smem_ptr, int M, int K) {
    // Q is stored in row-major: M rows × K cols
    // For FP4 packed: K/2 bytes per row (2 elements per byte)
    // LBO = K/2 (bytes per row)
    // SBO = 8 * K/2 (bytes for 8 rows)
    int bytes_per_row = K / 2;
    return SmemDescriptorSm100::make(smem_ptr, bytes_per_row, 8 * bytes_per_row, 2).get();
}

// Create descriptor for K matrix (N x K layout, row-major, used as B in QK^T)
CUTE_DEVICE
uint64_t make_k_smem_desc(void const* smem_ptr, int N, int K) {
    // K^T operation: K matrix is N rows × K cols
    // For FP4 packed: K/2 bytes per row
    int bytes_per_row = K / 2;
    return SmemDescriptorSm100::make(smem_ptr, bytes_per_row, 8 * bytes_per_row, 2).get();
}

// Create descriptor for V matrix (K x N layout, for P*V operation)
CUTE_DEVICE
uint64_t make_v_smem_desc(void const* smem_ptr, int K, int N) {
    // V is transposed: K rows × N cols
    // For FP4 packed: N/2 bytes per row
    int bytes_per_row = N / 2;
    return SmemDescriptorSm100::make(smem_ptr, bytes_per_row, 8 * bytes_per_row, 2).get();
}

// Create descriptor for P matrix (M x N, attention probabilities)
CUTE_DEVICE
uint64_t make_p_smem_desc(void const* smem_ptr, int M, int N) {
    // P after softmax and quantization: M rows × N cols
    // For FP4 packed: N/2 bytes per row
    int bytes_per_row = N / 2;
    return SmemDescriptorSm100::make(smem_ptr, bytes_per_row, 8 * bytes_per_row, 2).get();
}

} // namespace cute
