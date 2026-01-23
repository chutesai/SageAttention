/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled kernel traits for SageAttention3.
 *
 * This uses CUTLASS's SM100 block-scaled UMMA support with tcgen05.mma
 * instructions for FP4 quantized attention on datacenter Blackwell.
 *
 * IMPORTANT: SM100 FP4 requires K dimension = 256 for MMA.
 * Valid 1SM tile shapes: 128x128x256, 128x192x256, 128x256x256
 * Valid 2SM tile shapes: 256x128x256, 256x192x256, 256x256x256
 *
 * For Flash Attention with HeadDim=128 (Wan2.2):
 * - QK matmul: Q(128×128) @ K^T(128×128) - K dimension is HeadDim
 *   PROBLEM: HeadDim=128 but FP4 MMA requires K=256
 *   SOLUTION: Pad HeadDim to 256, or use 2 K-blocks of 128 each (not supported)
 *
 * For Flash Attention with HeadDim=256:
 * - QK matmul: Q(BlockM×256) @ K^T(256×BlockN) - K=256 ✓
 * - PV matmul: P(BlockM×BlockN) @ V(BlockN×256) - K must be 256
 *   CONSTRAINT: BlockN must be 256 for FP4 PV matmul
 *
 * Tile configuration:
 * - QK: (128, 256, 256) - 1SM tile, outputs S(128×256)
 * - PV: (128, 256, 256) - 1SM tile, outputs O(128×256)
 */

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

// SM100 block-scaled support
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

// CUTE SM100 MMA support
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Forward Kernel Traits
//
// Uses CUTLASS's SM100 block-scaled UMMA for FP4 attention with tcgen05.mma
// Both QK and PV matmuls use tile shape (128, 256, 256) to satisfy K=256
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,    // Must be 256 for FP4 (or data padded to 256)
    int kBlockM_,     // 128 for 1SM
    int kBlockN_,     // Must be 256 for FP4 PV matmul K dimension
    int kStages_,
    int kClusterM_,
    bool BlockMean_,
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4 {

    // Basic configuration
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;

    // SM100 FP4 requirements - K must be 256 for both matmuls
    static_assert(kHeadDim == 256, "SM100 FP4 requires HeadDim = 256 (pad if needed)");
    static_assert(kBlockN == 256, "SM100 FP4 requires BlockN = 256 for PV matmul K dimension");
    static_assert(kBlockM == 128, "SM100 FP4 1SM requires BlockM = 128");

    // Pipeline stages
    static constexpr int kStageCountQ = 2;
    static constexpr int kStageCountKV = kStages_;

    // Element types for FP4 block-scaled attention
    // NVF4 format: FP4 data (e2m1) with FP8 UE4M3 scale factors, vector size 16
    using ElementData = cutlass::float_e2m1_t;           // FP4 data type
    using ElementSF = cutlass::float_ue4m3_t;            // FP8 E4M3 unsigned scale factors
    using Element = cutlass::nv_float4_t<ElementData>;   // NVF4 packed type with scales
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Scale factor configuration (NVF4 uses vector size 16)
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFPerHeadDim = kHeadDim / SFVectorSize;  // 256/16 = 16 scale factors
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;    // 256/16 = 16 scale factors

    // Tile shapes for SM100 block-scaled GEMM
    // CRITICAL: Must use valid SM100 FP4 tile shapes with K=256
    //
    // QK matmul: Q(BlockM x HeadDim) @ K^T(HeadDim x BlockN) -> S(BlockM x BlockN)
    //   Tile: (128, 256, 256) - M=BlockM=128, N=BlockN=256, K=HeadDim=256
    //
    // PV matmul: P(BlockM x BlockN) @ V(BlockN x HeadDim) -> O(BlockM x HeadDim)
    //   Tile: (128, 256, 256) - M=BlockM=128, N=HeadDim=256, K=BlockN=256
    using TileShapeQK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;  // (128, 256, 256)
    using TileShapePV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;  // (128, 256, 256)
    using ClusterShape_MNK = Shape<Int<kClusterM_>, _1, _1>;

    // Architecture tags
    using ArchTag = cutlass::arch::Sm100;
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // Layout tags for CUTLASS - TN layout (A row-major, B col-major)
    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;

    // Alignment for FP4 (32 elements = 16 bytes for packed FP4)
    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;

    // Strides for GMEM tensors
    using StrideQ = Stride<int64_t, _1, int64_t>;
    using StrideK = Stride<int64_t, _1, int64_t>;
    using StrideV = Stride<_1, int64_t, int64_t>;  // V transposed
    using StrideO = Stride<int64_t, _1, int64_t>;

    ///////////////////////////////////////////////////////////////////////////
    // Use CollectiveBuilder for SM100 block-scaled configuration
    ///////////////////////////////////////////////////////////////////////////

    // QK matmul Collective - uses tile (128, 256, 256)
    using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        Element, LayoutATag, AlignmentA,  // A = Q (FP4 with scales)
        Element, LayoutBTag, AlignmentB,  // B = K^T (FP4 with scales)
        ElementAccum,
        TileShapeQK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // PV matmul Collective - uses tile (128, 256, 256)
    using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        Element, LayoutATag, AlignmentA,  // A = P (quantized softmax)
        Element, LayoutBTag, AlignmentB,  // B = V (FP4 with scales)
        ElementAccum,
        TileShapePV, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // Extract types from CollectiveBuilder
    using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
    using TiledMmaPV = typename CollectiveMmaPV::TiledMma;

    // SMEM layouts from CollectiveBuilder (includes data + scale factor layouts)
    using SmemLayoutQ = typename CollectiveMmaQK::SmemLayoutA;
    using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
    using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;
    using SmemLayoutSFA = typename CollectiveMmaQK::SmemLayoutSFA;
    using SmemLayoutSFB_QK = typename CollectiveMmaQK::SmemLayoutSFB;
    using SmemLayoutSFB_PV = typename CollectiveMmaPV::SmemLayoutSFB;

    // Scale factor layout types
    using LayoutSFA = typename CollectiveMmaQK::LayoutSFA;
    using LayoutSFB = typename CollectiveMmaQK::LayoutSFB;

    // TMA descriptors from CollectiveBuilder
    using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
    using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
    using TMA_V = typename CollectiveMmaPV::Params::TMA_B;
    using TMA_SFA = typename CollectiveMmaQK::Params::TMA_SFA;
    using TMA_SFB = typename CollectiveMmaQK::Params::TMA_SFB;
    using TMA_SFV = typename CollectiveMmaPV::Params::TMA_SFB;

    ///////////////////////////////////////////////////////////////////////////
    // Warp Configuration - simplified for initial implementation
    ///////////////////////////////////////////////////////////////////////////

    // For simplified direct GMEM version: 128 threads = 128 rows (one thread per row)
    // Each thread needs HeadDim floats = 256 * 4 = 1024 bytes = 256 registers
    // With 128 threads per block, this should fit in SM100's register file
    static constexpr int kNWarps = 4;  // 128 threads
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;

    ///////////////////////////////////////////////////////////////////////////
    // Pipeline Types - use CUTLASS's SM100 pipelines
    ///////////////////////////////////////////////////////////////////////////

    using AtomThrShape = typename CollectiveMmaQK::AtomThrShapeMNK;

    // TMA-UMMA async pipelines
    using PipelineQ = cutlass::PipelineTmaUmmaAsync<kStageCountQ, ClusterShape_MNK, AtomThrShape>;
    using PipelineKV = cutlass::PipelineTmaUmmaAsync<kStageCountKV, ClusterShape_MNK, AtomThrShape>;

    // Inter-warp communication pipelines
    using PipelineS = cutlass::PipelineUmmaAsync<1, AtomThrShape>;
    using PipelineC = cutlass::PipelineAsync<1>;
    using PipelineO = cutlass::PipelineUmmaAsync<2, AtomThrShape>;
    using PipelineE = cutlass::PipelineAsync<2>;
    using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<1, 2>;

    ///////////////////////////////////////////////////////////////////////////
    // TMEM Allocation for SM100 FP4
    ///////////////////////////////////////////////////////////////////////////

    enum class TmemAlloc : uint32_t {
        kSizeS = 128,  // S matrix (softmax scores)
        kSizeO = 128,  // O matrix (output accumulator)
        kSizeP = 32,   // P matrix (quantized softmax for PV)
        S0 = 0,
        S1 = S0 + kSizeS,
        P0 = S0 + kSizeP,
        P1 = S1 + kSizeP,
        O0 = S1 + kSizeS,
        O1 = O0 + kSizeO,
        kEnd = O1 + kSizeO
    };

    using TmemAllocator = cute::TMEM::Allocator1Sm;

    ///////////////////////////////////////////////////////////////////////////
    // Output Layout
    ///////////////////////////////////////////////////////////////////////////

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        cute::UMMA::Major::K, ElementOut,
        Int<kBlockM>, Int<kHeadDim>
    >());
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>, _2>{},
        Step<_2, _1, _3>{}
    ));

    ///////////////////////////////////////////////////////////////////////////
    // Shared Storage - Simplified for direct GMEM implementation
    // Full SMEM/pipeline storage will be needed for TMA-based version
    ///////////////////////////////////////////////////////////////////////////

    struct SharedStorage : cute::aligned_struct<128, _0> {
        // Minimal shared storage for synchronization
        // The simplified implementation uses direct GMEM access
        alignas(128) uint32_t sync_flag;
    };

    ///////////////////////////////////////////////////////////////////////////
    // TMA Transaction Bytes
    ///////////////////////////////////////////////////////////////////////////

    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(SmemLayoutQ{}) * cute::sizeof_bits_v<ElementData>) +
        cosize(SmemLayoutSFA{}) * sizeof(ElementSF));

    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(SmemLayoutK{}) * cute::sizeof_bits_v<ElementData>) +
        cosize(SmemLayoutSFB_QK{}) * sizeof(ElementSF));

    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(SmemLayoutV{}) * cute::sizeof_bits_v<ElementData>) +
        cosize(SmemLayoutSFB_PV{}) * sizeof(ElementSF));

    ///////////////////////////////////////////////////////////////////////////
    // Legacy compatibility
    ///////////////////////////////////////////////////////////////////////////

    using TileShape_MNK = TileShapeQK;  // Primary tile shape
    using MainloopPipeline = PipelineKV;
    using PipelineState = typename PipelineKV::PipelineState;
    static constexpr int kStages = kStageCountKV;
};

} // namespace flash
