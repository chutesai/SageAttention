/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled kernel traits for SageAttention3.
 *
 * This uses CUTLASS's SM100 block-scaled UMMA support with tcgen05.mma
 * instructions for FP4 quantized attention on datacenter Blackwell.
 *
 * Key Design Decisions:
 * ---------------------
 * 1. SM100 FP4 block-scaled MMA requires K=256 for optimal tensor core usage
 * 2. For HeadDim=128, we pad to 256 or use K-loop with K=128 tiles
 * 3. Block-scaled format: E2M1 data with E4M3 scale factors, 16 elements per SF
 *
 * For Flash Attention:
 * - QK GEMM: Q(BlockM×HeadDim) @ K^T(HeadDim×BlockN) → S(BlockM×BlockN)
 * - PV GEMM: P(BlockM×BlockN) @ V(BlockN×HeadDim) → O(BlockM×HeadDim)
 *
 * With HeadDim=128, BlockN=128:
 * - We use TileShape (128, 128, 128) and rely on K-loop for larger K
 * - SM100 block-scaled handles the scale factor application automatically
 */

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/pipeline/sm100_pipeline.hpp"

// SM100 block-scaled support
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 Warp Roles for FP4 FMHA
///////////////////////////////////////////////////////////////////////////////

struct Sm100FP4WarpSpecializedSchedule {
    enum class WarpRole {
        Softmax0,      // warps 0-3: softmax processing for S0
        Softmax1,      // warps 4-7: softmax processing for S1
        Correction,    // warps 8-11: online softmax correction
        MMA,           // warp 12: MMA warp
        Load,          // warp 13: TMA load warp
        Epilogue,      // warp 14: epilogue warp
        Empty          // warp 15: unused
    };

    static constexpr WarpRole warp_idx_to_role(int warp_idx) {
        int wg_idx = warp_idx / 4;
        if (wg_idx == 0) return WarpRole::Softmax0;
        if (wg_idx == 1) return WarpRole::Softmax1;
        if (wg_idx == 2) return WarpRole::Correction;
        if (warp_idx == 12) return WarpRole::MMA;
        if (warp_idx == 13) return WarpRole::Load;
        if (warp_idx == 14) return WarpRole::Epilogue;
        return WarpRole::Empty;
    }

    static constexpr int kNumWarps = 16;
    static constexpr int kNumRegsSoftmax = 192;
    static constexpr int kNumRegsCorrection = 96;
    static constexpr int kNumRegsOther = 32;
};

///////////////////////////////////////////////////////////////////////////////
// TMEM Allocation for SM100 FP4 FMHA
///////////////////////////////////////////////////////////////////////////////

enum class Sm100FP4TmemAlloc : uint32_t {
    kSizeS = 128,  // S matrix (QK scores)
    kSizeO = 128,  // O matrix (output accumulator)
    kSizeP = 32,   // P matrix (quantized softmax)
    S0 = 0,
    S1 = S0 + kSizeS,
    V0 = S0,       // Reuse S0 for stats storage
    V1 = S1,       // Reuse S1 for stats storage
    P0 = S0 + kSizeP,
    P1 = S1 + kSizeP,
    O0 = S1 + kSizeS,
    O1 = O0 + kSizeO,
    kEnd = O1 + kSizeO
};

// Stats indices for online softmax
enum {
    kIdxOldRowMax = 0,
    kIdxNewRowMax = 1,
    kIdxFinalRowSum = 0,
    kIdxFinalRowMax = 1
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Forward Kernel Traits
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,    // Head dimension (128 or 256, padded if needed)
    int kBlockM_,     // Block size for Q rows (128 or 256)
    int kBlockN_,     // Block size for K/V sequence (128)
    int kStages_,     // Pipeline stages (2-4)
    int kClusterM_,   // Cluster shape in M (1 or 2)
    bool BlockMean_,  // Whether to use block mean subtraction
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4 {

    // Basic configuration
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr int kClusterM = kClusterM_;

    // Validate dimensions
    static_assert(kHeadDim >= 128 && kHeadDim % 64 == 0,
                  "HeadDim must be >= 128 and multiple of 64");
    static_assert(kBlockM >= 128 && kBlockM % 128 == 0,
                  "BlockM must be >= 128 and multiple of 128");
    static_assert(kBlockN >= 64 && kBlockN % 64 == 0,
                  "BlockN must be >= 64 and multiple of 64");

    // Pipeline stages
    static constexpr int kStageCountQ = 2;
    static constexpr int kStageCountKV = kStages_;

    //=========================================================================
    // Element Types for FP4 Block-Scaled Attention
    //=========================================================================

    // FP4 format: E2M1 data with E4M3 scale factors
    // CUTLASS nv_float4_t wraps FP4 data for block-scaled operations
    using ElementData = cutlass::float_e2m1_t;       // FP4 data (E2M1)
    using ElementSF = cutlass::float_e4m3_t;         // Scale factor (E4M3)
    using Element = cutlass::nv_float4_t<ElementData>; // Packed type for MMA
    using ElementAccum = float;                       // FP32 accumulator
    using ElementOut = ElementOut_;                   // Output type (BF16)
    using index_t = int64_t;

    // Scale factor configuration - 16 elements per scale factor block
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFPerHeadDim = kHeadDim / SFVectorSize;
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;

    //=========================================================================
    // Tile Shapes
    //=========================================================================

    // For QK GEMM: Q(M×K) @ K^T(K×N) → S(M×N) where K=HeadDim
    // For PV GEMM: P(M×K) @ V(K×N) → O(M×N) where K=BlockN, N=HeadDim
    using TileShapeQK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShapePV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
    using TileShape_MNK = TileShapeQK;  // Primary tile shape
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    // Thread shape for warp-specialized execution
    using ThreadShape = Shape<_2, _1, _1>;  // 2 Q groups processed
    using TileShapeQK_perWG = decltype(shape_div(TileShapeQK{}, ThreadShape{}));
    using TileShapePV_perWG = decltype(select<0,2,1>(TileShapeQK_perWG{}));

    //=========================================================================
    // Architecture and Layout Configuration
    //=========================================================================

    using ArchTag = cutlass::arch::Sm100;
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // GMEM layouts - row-major for Q/K, column-major for V
    using LayoutATag = cutlass::layout::RowMajor;     // Q: row-major
    using LayoutBTag = cutlass::layout::ColumnMajor;  // K^T: col-major (K is row-major, transposed)
    using LayoutVTag = cutlass::layout::RowMajor;     // V: row-major

    // Alignment for FP4 (32 elements = 16 bytes for packed FP4 pair)
    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;
    static constexpr int AlignmentOut = 128 / cutlass::sizeof_bits<ElementOut>::value;

    // Strides for GMEM tensors
    // Layout: (seq, dim, batch*head) with dim stride = 1
    using StrideQ = Stride<int64_t, _1, int64_t>;
    using StrideK = Stride<int64_t, _1, int64_t>;
    using StrideV = Stride<_1, int64_t, int64_t>;  // V is (dim, seq, batch*head)
    using StrideO = Stride<int64_t, _1, int64_t>;

    //=========================================================================
    // CUTLASS CollectiveBuilder for QK GEMM
    //=========================================================================

    // The builder automatically configures:
    // - TMA descriptors for data and scale factors
    // - SMEM layouts with proper swizzling
    // - Block-scaled MMA atoms
    using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        Element, LayoutATag, AlignmentA,    // A = Q (FP4)
        Element, LayoutBTag, AlignmentB,    // B = K^T (FP4)
        ElementAccum,
        TileShapeQK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    //=========================================================================
    // CUTLASS CollectiveBuilder for PV GEMM
    //=========================================================================

    // Note: P (softmax output) needs to be quantized to FP4 for MMA
    // V is pre-quantized to FP4
    using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        Element, LayoutATag, AlignmentA,    // A = P (quantized softmax)
        Element, LayoutVTag, AlignmentB,    // B = V (FP4)
        ElementAccum,
        TileShapePV, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    //=========================================================================
    // Types from CollectiveBuilder
    //=========================================================================

    using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
    using TiledMmaPV = typename CollectiveMmaPV::TiledMma;
    using AtomThrShapeMNK = typename CollectiveMmaQK::AtomThrShapeMNK;

    // SMEM layouts for data and scale factors
    using SmemLayoutQ = typename CollectiveMmaQK::SmemLayoutA;
    using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
    using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;
    using SmemLayoutSFQ = typename CollectiveMmaQK::SmemLayoutSFA;
    using SmemLayoutSFK = typename CollectiveMmaQK::SmemLayoutSFB;
    using SmemLayoutSFV = typename CollectiveMmaPV::SmemLayoutSFB;

    // GMEM layouts for scale factors
    using LayoutSFA = typename CollectiveMmaQK::LayoutSFA;
    using LayoutSFB = typename CollectiveMmaQK::LayoutSFB;

    //=========================================================================
    // Warp Configuration
    //=========================================================================

    using Schedule = Sm100FP4WarpSpecializedSchedule;
    static constexpr int kNWarps = Schedule::kNumWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

    //=========================================================================
    // Pipeline Types
    //=========================================================================

    // TMA-UMMA async pipelines for data loading
    using PipelineQ = cutlass::PipelineTmaUmmaAsync<kStageCountQ, ClusterShape_MNK, AtomThrShapeMNK>;
    using PipelineKV = cutlass::PipelineTmaUmmaAsync<kStageCountKV, ClusterShape_MNK, AtomThrShapeMNK>;

    // Inter-warp communication pipelines
    using PipelineS = cutlass::PipelineUmmaAsync<1, AtomThrShapeMNK>;  // MMA → Softmax
    using PipelineC = cutlass::PipelineAsync<1>;                        // Softmax → Correction
    using PipelineO = cutlass::PipelineUmmaAsync<2, AtomThrShapeMNK>;  // MMA → Correction
    using PipelineE = cutlass::PipelineAsync<2>;                        // Correction → Epilogue
    using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<1, 2>;

    //=========================================================================
    // TMEM Allocation
    //=========================================================================

    using TmemAlloc = Sm100FP4TmemAlloc;
    using TmemAllocator = cute::TMEM::Allocator1Sm;

    //=========================================================================
    // Output Layout
    //=========================================================================

    using EpilogueTileShape = Shape<
        decltype(get<0>(TileShapeQK_perWG{})),  // M
        decltype(get<2>(TileShapeQK_perWG{})),  // HeadDim
        _1
    >;

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        cute::UMMA::Major::K, ElementOut,
        decltype(get<0>(EpilogueTileShape{})),
        decltype(get<1>(EpilogueTileShape{}))
    >());

    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        replace<2>(EpilogueTileShape{}, _2{}),  // 2 stages for double buffering
        Step<_2, _1, _3>{}
    ));

    //=========================================================================
    // Shared Storage
    //=========================================================================

    struct SharedStorage : cute::aligned_struct<128, _0> {
        // Q data + scale factors
        cute::array_aligned<typename Element::DataType, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_sfq;

        // K/V data + scale factors (union since they're used at different times)
        union {
            struct {
                cute::array_aligned<typename Element::DataType, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_sfk;
            };
            struct {
                cute::array_aligned<typename Element::DataType, cute::cosize_v<SmemLayoutV>> smem_v;
                cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFV>> smem_sfv;
            };
        };

        // Output buffer
        cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutO>> smem_o;

        // Pipeline storage
        struct {
            alignas(16) typename PipelineQ::SharedStorage pipeline_q;
            alignas(16) typename PipelineKV::SharedStorage pipeline_kv;
            alignas(16) typename PipelineS::SharedStorage pipeline_s0;
            alignas(16) typename PipelineS::SharedStorage pipeline_s1;
            alignas(16) typename PipelineC::SharedStorage pipeline_c0;
            alignas(16) typename PipelineC::SharedStorage pipeline_c1;
            alignas(16) typename PipelineO::SharedStorage pipeline_o;
            alignas(16) typename PipelineE::SharedStorage pipeline_epi;
            alignas(16) typename OrderBarrierSoftmax::SharedStorage order_s01;
        } pipelines;

        // TMEM base pointer
        uint32_t tmem_base_ptr;
    };

    //=========================================================================
    // TMA Transaction Sizes
    //=========================================================================

    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
        size(AtomThrShapeMNK{}) * (
            cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<ElementData>) +
            cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutSFQ{})) * cute::sizeof_bits_v<ElementSF>)
        ));

    static constexpr uint32_t TmaTransactionBytesKV = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<ElementData>) +
        cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutSFK{})) * cute::sizeof_bits_v<ElementSF>));

    //=========================================================================
    // Legacy Compatibility Aliases
    //=========================================================================

    using MainloopPipeline = PipelineKV;
    using PipelineState = typename PipelineKV::PipelineState;
    static constexpr int kStages = kStageCountKV;
    static constexpr int EpiStages = 2;
};

} // namespace flash
