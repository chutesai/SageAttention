/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention Kernel Traits
 *
 * High-performance kernel using SM100's block-scaled tcgen05.mma instructions
 * with TMA for memory access and TMEM for accumulators.
 *
 * Key design principles:
 * - Use CUTLASS CollectiveBuilder for automatic MMA/TMA configuration
 * - Block-scaled FP4 (E2M1) with E4M3 scale factors
 * - Two-stage GEMM: QK (attention scores) -> softmax -> PV (output)
 * - Delta-S correction for smooth attention integrated into QK accumulator
 *
 * Tile shape constraints for SM100 FP4:
 * - MMA tile: (128, 128, 256) for block-scaled operations
 * - K dimension must be 256 for optimal tensor core utilization
 * - HeadDim = 128 or 256 (padded if needed)
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Attention Kernel Traits
///////////////////////////////////////////////////////////////////////////////

template<
    int kHeadDim_,          // Head dimension (128 or 256)
    int kBlockM_,           // Block size for Q/output rows (128)
    int kBlockN_,           // Block size for K/V sequence (128 or 256)
    int kStages_,           // Pipeline stages (2-4)
    class ElementOut_       // Output element type (bfloat16_t)
>
struct Flash_fwd_kernel_traits_sm100_fp4_blockscaled {

    // Compile-time constants
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kStages = kStages_;

    // FP4 element types (block-scaled)
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // Q, K FP4 data
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // K, V FP4 data
    using ElementSF = cutlass::float_e4m3_t;                        // Scale factors (E4M3)
    using ElementAccum = float;                                     // FP32 accumulator
    using ElementOut = ElementOut_;                                 // Output (BF16)

    // Scale factor vector size (16 elements per scale factor block)
    static constexpr int SFVecSize = 16;

    // Architecture tags
    using ArchTag = cutlass::arch::Sm100;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // Cluster shape (single CTA for now, can extend to 2-CTA)
    using ClusterShape = Shape<_1, _1, _1>;

    // MMA tile shape for block-scaled operations
    // SM100 FP4 MMA requires K=256 for optimal performance
    using MmaTileShapeQK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    // For PV GEMM, we transpose the tile shape
    // P: (BlockM, BlockN) @ V: (BlockN, HeadDim) -> O: (BlockM, HeadDim)
    using MmaTileShapePV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;

    // Alignment requirements (32 elements for FP4 = 16 bytes)
    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;
    static constexpr int AlignmentOut = 128 / cutlass::sizeof_bits<ElementOut>::value;

    // Stride types for Q, K, V tensors
    // Q: (seq, 1, ((head_group, head), batch))
    // K: (seq, 1, ((0, head), batch))
    // V: (seq, 1, ((0, head), batch))
    using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
    using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
    using StrideV = StrideK;
    using StrideO = StrideQ;

    // Block-scaled layout configuration
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

    // Build QK GEMM collective using CUTLASS builder
    // Q @ K^T -> S (attention scores)
    using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, cutlass::layout::RowMajor, AlignmentA,    // Q: row-major
        ElementB, cutlass::layout::ColumnMajor, AlignmentB, // K^T: col-major (K is row-major)
        ElementAccum,
        MmaTileShapeQK, ClusterShape,
        cutlass::gemm::collective::StageCount<kStages>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // Build PV GEMM collective using CUTLASS builder
    // P @ V -> O (output)
    using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, cutlass::layout::RowMajor, AlignmentA,    // P: row-major (from softmax)
        ElementB, cutlass::layout::RowMajor, AlignmentB,    // V: row-major
        ElementAccum,
        MmaTileShapePV, ClusterShape,
        cutlass::gemm::collective::StageCount<kStages>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // Extract types from collectives
    using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
    using TiledMmaPV = typename CollectiveMmaPV::TiledMma;

    using SmemLayoutQ = typename CollectiveMmaQK::SmemLayoutA;
    using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
    using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;

    // Scale factor layouts from block-scaled config
    using LayoutSFQ = typename CollectiveMmaQK::LayoutSFA;
    using LayoutSFK = typename CollectiveMmaQK::LayoutSFB;
    using LayoutSFV = typename CollectiveMmaPV::LayoutSFB;

    // Thread configuration
    static constexpr int kNThreads = 256;  // 8 warps for warp-specialized execution

    // TMEM allocation for SM100
    // Accumulators and softmax stats stored in TMEM
    enum class TmemAllocation : uint32_t {
        kSizeS = 128,    // QK accumulator
        kSizeO = 128,    // Output accumulator
        kSizeP = 32,     // Softmax probabilities (quantized)
        S0 = 0,
        S1 = S0 + kSizeS,
        V0 = S0,         // Stats storage for softmax
        V1 = S1,
        P0 = S0 + kSizeP,
        P1 = S1 + kSizeP,
        O0 = S1 + kSizeS,
        O1 = O0 + kSizeO,
        kEnd = O1 + kSizeO
    };

    // Shared memory structure
    struct SharedStorage {
        cute::array_aligned<typename ElementA::DataType, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<typename ElementSF,
            kBlockM * kHeadDim / SFVecSize> smem_sfq;
        union {
            struct {
                cute::array_aligned<typename ElementB::DataType, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<typename ElementSF,
                    kBlockN * kHeadDim / SFVecSize> smem_sfk;
            };
            struct {
                cute::array_aligned<typename ElementB::DataType, cute::cosize_v<SmemLayoutV>> smem_v;
                cute::array_aligned<typename ElementSF,
                    kBlockN * kHeadDim / SFVecSize> smem_sfv;
            };
        };
        // Delta-S for smooth attention
        cute::array_aligned<float, kBlockM / 128 * kBlockN> smem_delta_s;
    };
};

///////////////////////////////////////////////////////////////////////////////
// Default configurations for common head dimensions
///////////////////////////////////////////////////////////////////////////////

// HeadDim=128, BlockM=128, BlockN=128 (standard attention)
using Ktraits_FP4_128_128_128 = Flash_fwd_kernel_traits_sm100_fp4_blockscaled<
    128,    // kHeadDim
    128,    // kBlockM
    128,    // kBlockN
    3,      // kStages
    cutlass::bfloat16_t
>;

// HeadDim=256, BlockM=128, BlockN=256 (for large head dim)
using Ktraits_FP4_256_128_256 = Flash_fwd_kernel_traits_sm100_fp4_blockscaled<
    256,    // kHeadDim
    128,    // kBlockM
    256,    // kBlockN
    2,      // kStages (fewer stages due to larger tile)
    cutlass::bfloat16_t
>;

} // namespace flash
