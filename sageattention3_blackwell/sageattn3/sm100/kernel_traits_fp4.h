/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled kernel traits for SageAttention3.
 *
 * This uses CUTLASS's SM100 block-scaled UMMA support with tcgen05.mma
 * instructions for FP4 quantized attention on datacenter Blackwell.
 *
 * Key differences from SM120 (RTX 5090):
 * - SM100: tcgen05.mma (UMMA) with TMEM (Tensor Memory)
 * - SM120: mma.sync.aligned with SMEM
 *
 * IMPORTANT: SM100 FP4 requires K dimension = 256 for MMA.
 * Supported tile shapes:
 * - 1SM: 128x128x256, 128x192x256, 128x256x256
 * - 2SM: 256x128x256, 256x192x256, 256x256x256
 *
 * This means:
 * - QK matmul: HeadDim must be 256 (or padded to 256)
 * - PV matmul: BlockN must be 256
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

#include "../blackwell/blockscaled_layout.h"
#include "../blackwell/named_barrier.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 Warp Roles for FP4 Attention
///////////////////////////////////////////////////////////////////////////////

struct Sm100FP4WarpSpecializedSchedule {
    enum class WarpRole {
        Softmax0,
        Softmax1,
        Correction,
        MMA,
        Load,
        Epilogue,
        Empty
    };

    static constexpr WarpRole warp_idx_to_role(int warp_idx) {
        int wg_idx = warp_idx / 4;
        if (wg_idx == 0) return WarpRole::Softmax0;   // warps 0-3
        if (wg_idx == 1) return WarpRole::Softmax1;   // warps 4-7
        if (wg_idx == 2) return WarpRole::Correction; // warps 8-11
        if (warp_idx == 12) return WarpRole::MMA;
        if (warp_idx == 13) return WarpRole::Load;
        if (warp_idx == 14) return WarpRole::Epilogue;
        return WarpRole::Empty;                       // warp 15
    }

    static constexpr int kNumWarpsSoftmax = 4;
    static constexpr int kNumWarpsCorrection = 4;
    static constexpr int kNumWarpsEpilogue = 1;
    static constexpr int kNumWarpsLoad = 1;
    static constexpr int kNumWarpsMMA = 1;
    static constexpr int kNumWarps = 16;

    static constexpr int kNumRegsSoftmax = 192;
    static constexpr int kNumRegsCorrection = 96;
    static constexpr int kNumRegsOther = 32;
    static constexpr int kNumRegsEmpty = 24;
};

///////////////////////////////////////////////////////////////////////////////
// TMEM Allocation for SM100 FP4
///////////////////////////////////////////////////////////////////////////////

enum class Sm100FP4TmemAlloc : uint32_t {
    kSizeS = 128,
    kSizeO = 128,
    kSizeP = 32,
    S0 = 0,
    S1 = S0 + kSizeS,
    V0 = S0,  // stats storage overlaps with S
    V1 = S1,
    P0 = S0 + kSizeP,
    P1 = S1 + kSizeP,
    O0 = S1 + kSizeS,
    O1 = O0 + kSizeO,
    kEnd = O1 + kSizeO
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Forward Kernel Traits
//
// Uses CUTLASS's SM100 block-scaled UMMA for FP4 attention with tcgen05.mma
// K dimension MUST be 256 for FP4 MMA.
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,    // Must be 256 for FP4 (or data padded to 256)
    int kBlockM_,     // 128 for 1SM, 256 for 2SM
    int kBlockN_,     // Must be 256 for FP4 PV matmul
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
    static constexpr bool SmoothQ = true;

    // SM100 FP4 requirements - K must be 256
    static_assert(kHeadDim == 256, "SM100 FP4 requires HeadDim = 256 (pad if needed)");
    static_assert(kBlockN == 256, "SM100 FP4 requires BlockN = 256 for PV matmul");
    static_assert(kBlockM == 128 || kBlockM == 256, "SM100 FP4 supports BlockM=128 (1SM) or 256 (2SM)");

    // Determine if using 1SM or 2SM
    static constexpr bool Is2SM = (kBlockM == 256);

    // Warp configuration for SM100
    using Schedule = Sm100FP4WarpSpecializedSchedule;
    static constexpr int kNWarps = Schedule::kNumWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;

    // Pipeline stages
    static constexpr int kStageCountQ = 2;
    static constexpr int kStageCountKV = kStages_;

    // Element types for FP4 block-scaled attention
    // FP4 = 4-bit floating point (e2m1: 2-bit exponent, 1-bit mantissa, 1-bit sign)
    // NVF4 format: FP4 data with FP8 UE4M3 scale factors, vector size 16
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
    // MMA tile shape: (M, N, K) where K is the reduction dimension
    // For FP4, K = 256 is REQUIRED
    //
    // QK matmul: Q(BlockM x HeadDim) @ K^T(HeadDim x BlockN) -> S(BlockM x BlockN)
    //   Tile: (BlockM/2, BlockN, HeadDim) for ThreadShape=(2,1,1)
    //   K dimension = HeadDim = 256 ✓
    //
    // PV matmul: P(BlockM x BlockN) @ V(BlockN x HeadDim) -> O(BlockM x HeadDim)
    //   Tile: (BlockM/2, HeadDim, BlockN)
    //   K dimension = BlockN = 256 ✓
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;
    using ThreadShape = Shape<_2, _1, _1>;
    using TileShapeQK = decltype(shape_div(TileShape_MNK{}, ThreadShape{}));
    using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));

    // Architecture tags
    using ArchTag = cutlass::arch::Sm100;
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // Layout tags for CUTLASS
    // TN layout: A row-major (T), B column-major (N)
    using LayoutATag = cutlass::layout::RowMajor;      // Q: row-major (seqlen x headdim)
    using LayoutBTag = cutlass::layout::ColumnMajor;   // K^T: column-major for TN layout

    // Alignment for FP4 (32 elements = 16 bytes for packed FP4)
    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;

    // Strides for GMEM tensors
    // Q/K/O: (seq, dim, batch*head) - row major within head
    using StrideQ = Stride<int64_t, _1, int64_t>;
    using StrideK = Stride<int64_t, _1, int64_t>;
    using StrideV = Stride<_1, int64_t, int64_t>;  // V transposed: (dim, seq, batch*head)
    using StrideO = Stride<int64_t, _1, int64_t>;

    ///////////////////////////////////////////////////////////////////////////
    // Use CollectiveBuilder to get correct SM100 block-scaled configuration
    ///////////////////////////////////////////////////////////////////////////

    // For QK matmul: Q (M x K) @ K^T (K x N) -> S (M x N)
    // Using block-scaled FP4 inputs with FP32 accumulation
    using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        Element, LayoutATag, AlignmentA,  // A = Q (FP4 with scales)
        Element, LayoutBTag, AlignmentB,  // B = K (FP4 with scales)
        ElementAccum,
        TileShapeQK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::KernelScheduleAuto  // Auto-select best SM100 blockscaled schedule
    >::CollectiveOp;

    // For PV matmul: P (M x N) @ V (N x D) -> O (M x D)
    // P is softmax output - needs to be quantized to FP4 on-the-fly
    // V is pre-quantized FP4
    using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        Element, LayoutATag, AlignmentA,  // A = P (quantized on-the-fly)
        Element, LayoutBTag, AlignmentB,  // B = V (FP4 with scales)
        ElementAccum,
        TileShapePV, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::KernelScheduleAuto
    >::CollectiveOp;

    // Extract MMA types from CollectiveBuilder
    using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
    using TiledMmaPV = typename CollectiveMmaPV::TiledMma;

    // Extract SMEM layouts from CollectiveBuilder
    // These include both data and scale factor layouts
    using SmemLayoutAtomPairA_QK = typename CollectiveMmaQK::SmemLayoutAtomPairA;
    using SmemLayoutAtomPairB_QK = typename CollectiveMmaQK::SmemLayoutAtomPairB;
    using SmemLayoutAtomPairB_PV = typename CollectiveMmaPV::SmemLayoutAtomPairB;

    using SmemLayoutAtomA = decltype(get<0>(SmemLayoutAtomPairA_QK{}));
    using SmemLayoutAtomSFA = decltype(get<1>(SmemLayoutAtomPairA_QK{}));
    using SmemLayoutAtomB = decltype(get<0>(SmemLayoutAtomPairB_QK{}));
    using SmemLayoutAtomSFB = decltype(get<1>(SmemLayoutAtomPairB_QK{}));

    // Full staged layouts
    using SmemLayoutQ = typename CollectiveMmaQK::SmemLayoutA;
    using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
    using SmemLayoutSFA = typename CollectiveMmaQK::SmemLayoutSFA;
    using SmemLayoutSFB_QK = typename CollectiveMmaQK::SmemLayoutSFB;

    using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;
    using SmemLayoutSFB_PV = typename CollectiveMmaPV::SmemLayoutSFB;

    // Block-scaled configuration helper
    using Sm1xxBlkScaledConfig = typename CollectiveMmaQK::Sm1xxBlkScaledConfig;
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
    // Pipeline Types
    ///////////////////////////////////////////////////////////////////////////

    using AtomThrShape = typename CollectiveMmaQK::AtomThrShapeMNK;

    // From Load to MMA warp (TMA loads Q/K/V into SMEM)
    using PipelineQ = cutlass::PipelineTmaUmmaAsync<kStageCountQ, ClusterShape_MNK, AtomThrShape>;
    using PipelineKV = cutlass::PipelineTmaUmmaAsync<kStageCountKV, ClusterShape_MNK, AtomThrShape>;

    // From MMA to Softmax0/1 warps (protects S in TMEM) - uses UMMA peer signaling
    using PipelineS = cutlass::PipelineUmmaAsync<1, AtomThrShape>;

    // From Softmax0/1 to Correction warps (simple async barrier)
    using PipelineC = cutlass::PipelineAsync<1>;

    // From MMA to Correction warps (protects O in TMEM) - uses UMMA peer signaling
    using PipelineO = cutlass::PipelineUmmaAsync<2, AtomThrShape>;

    // From Correction to Epilogue warp (simple async barrier)
    using PipelineE = cutlass::PipelineAsync<2>;
    using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<1, 2>;

    ///////////////////////////////////////////////////////////////////////////
    // TMEM Allocator
    ///////////////////////////////////////////////////////////////////////////

    using TmemAlloc = Sm100FP4TmemAlloc;
    using TmemAllocator = cute::TMEM::Allocator1Sm;

    ///////////////////////////////////////////////////////////////////////////
    // Output Layout
    ///////////////////////////////////////////////////////////////////////////

    using EpilogueTileShape = Shape<
        decltype(get<0>(TileShapeQK{})),  // M = BlockM/2
        decltype(get<2>(TileShapeQK{})),  // K = HeadDim
        _1                                 // batch dimension (tiled by 1)
    >;
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        cute::UMMA::Major::K, ElementOut,
        decltype(get<0>(EpilogueTileShape{})),
        decltype(get<1>(EpilogueTileShape{}))
    >());
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        replace<2>(EpilogueTileShape{}, _2{}),
        Step<_2, _1, _3>{}
    ));

    ///////////////////////////////////////////////////////////////////////////
    // Shared Storage
    ///////////////////////////////////////////////////////////////////////////

    struct SharedStorage : cute::aligned_struct<128, _0> {
        // FP4 data tensors - staged for double buffering
        cute::array_aligned<ElementData, cute::cosize_v<SmemLayoutQ>> smem_q;
        union {
            cute::array_aligned<ElementData, cute::cosize_v<SmemLayoutK>> smem_k;
            cute::array_aligned<ElementData, cute::cosize_v<SmemLayoutV>> smem_v;
        };

        // Scale factor tensors
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_sfq;
        union {
            cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFB_QK>> smem_sfk;
            cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFB_PV>> smem_sfv;
        };

        // Output tensor
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

        // TMEM base pointer for this CTA
        uint32_t tmem_base_ptr;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Legacy compatibility types
    ///////////////////////////////////////////////////////////////////////////

    using MainloopPipeline = PipelineKV;
    using PipelineState = typename PipelineKV::PipelineState;
    using MainloopPipelineQ = PipelineQ;
    using PipelineParamsQ = typename PipelineQ::Params;
    using PipelineStateQ = typename PipelineQ::PipelineState;
    static constexpr int kStages = kStageCountKV;
    static constexpr int EpiStages = 2;
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;

    ///////////////////////////////////////////////////////////////////////////
    // TMA Transaction Bytes
    ///////////////////////////////////////////////////////////////////////////

    // Transaction bytes for FP4 with scale factors
    // Data: M*K * 4bits / 8 = M*K / 2 bytes
    // SF: (M/128) * (K/16) * 8bits = M*K / (128*16) bytes
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(SmemLayoutQ{}) * cute::sizeof_bits_v<ElementData>) +
        cosize(SmemLayoutSFA{}) * sizeof(ElementSF));

    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(SmemLayoutK{}) * cute::sizeof_bits_v<ElementData>) +
        cosize(SmemLayoutSFB_QK{}) * sizeof(ElementSF));

    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(SmemLayoutV{}) * cute::sizeof_bits_v<ElementData>) +
        cosize(SmemLayoutSFB_PV{}) * sizeof(ElementSF));
};

} // namespace flash
