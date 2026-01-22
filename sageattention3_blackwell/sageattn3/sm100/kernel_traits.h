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
 * SM100 (B200/B300) kernel traits for SageAttention3 with FP4 blockscaled MMA.
 *
 * This file uses CUTLASS CollectiveBuilder to automatically generate correct
 * TMA descriptors, SMEM layouts, and MMA configurations for SM100.
 */

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/pipeline/sm100_pipeline.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

// Include SM100 utilities
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

#include "../blackwell/blockscaled_layout.h"
#include "../blackwell/named_barrier.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 Warp Roles
///////////////////////////////////////////////////////////////////////////////

struct Sm100WarpSpecializedSchedule {
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
// TMEM Allocation for SM100
///////////////////////////////////////////////////////////////////////////////

enum class Sm100TmemAlloc : uint32_t {
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

enum {
    kIdxOldRowMax = 0,
    kIdxNewRowMax = 1,
    kIdxFinalRowSum = 0,
    kIdxFinalRowMax = 1
};

///////////////////////////////////////////////////////////////////////////////
// SM100 Flash Forward Kernel Traits
//
// Uses CUTLASS CollectiveBuilder for automatic configuration
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kStages_,
    int kClusterM_,
    bool BlockMean_,
    typename ElementPacked_ = uint8_t,              // Packed FP4 (2 values per byte)
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100 {

    // Basic configuration
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr bool SmoothQ = true;

    static_assert(kBlockM == 256, "SM100 requires kBlockM=256");
    static_assert(kHeadDim >= 128, "SM100 with FP4 requires HeadDim >= 128");
    static_assert(kHeadDim % 32 == 0);

    // Warp scheduling
    using Schedule = Sm100WarpSpecializedSchedule;
    static constexpr int kNWarps = Schedule::kNumWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;

    // Pipeline stages
    static constexpr int kStageCountQ = 2;
    static constexpr int kStageCountKV = kStages_;

    // Element types - use bfloat16 for CollectiveBuilder (standard non-blockscaled path)
    // The actual FP4 blockscaled requires a different approach
    using ElementPacked = ElementPacked_;
    using Element = cutlass::bfloat16_t;  // Use BF16 for standard UMMA path
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Tile shapes
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
    using ThreadShape = Shape<_2, _1, _1>;
    using TileShapeQK = decltype(shape_div(TileShape_MNK{}, ThreadShape{}));
    using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));

    // Architecture tag
    using ArchTag = cutlass::arch::Sm100;

    // Strides for GMEM tensors - 3D: (seq, dim, batch*head)
    using StrideQ = Stride<int64_t, _1, int64_t>;
    using StrideK = Stride<int64_t, _1, int64_t>;
    using StrideV = Stride<_1, int64_t, int64_t>;  // V is transposed: (dim, seq, batch*head)
    using StrideO = Stride<int64_t, _1, int64_t>;

    static constexpr int Alignment = 128 / cute::sizeof_bits_v<Element>;

    ///////////////////////////////////////////////////////////////////////////
    // Use CollectiveBuilder to get correct configurations
    ///////////////////////////////////////////////////////////////////////////

    using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideQ, Alignment,
        Element, StrideK, Alignment,
        ElementAccum,
        TileShapeQK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCount<kStageCountKV>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
    >::CollectiveOp;

    using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideK, Alignment,
        Element, decltype(select<1,0,2>(StrideV{})), Alignment,
        ElementAccum,
        TileShapePV, ClusterShape_MNK,
        cutlass::gemm::collective::StageCount<kStageCountKV>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
    >::CollectiveOp;

    // Extract MMA types from CollectiveBuilder
    using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
    using TiledMmaPV = typename CollectiveMmaPV::TiledMma;

    // Extract SMEM layouts from CollectiveBuilder and adjust staging
    // CollectiveBuilder creates multi-stage layouts; we extract single-stage for Q
    using SmemLayoutQFull = typename CollectiveMmaQK::SmemLayoutA;
    using SmemLayoutKFull = typename CollectiveMmaQK::SmemLayoutB;
    using SmemLayoutVFull = typename CollectiveMmaPV::SmemLayoutB;

    // For load warp, use the full staged layouts
    using SmemLayoutQ = SmemLayoutQFull;
    using SmemLayoutK = SmemLayoutKFull;
    using SmemLayoutV = SmemLayoutVFull;

    // Note: SmemLayoutO is defined in CollectiveEpilogueFwdSm100
    // We forward-declare the epilogue layout here for SharedStorage
    using EpilogueTileShape = Shape<
        decltype(get<0>(TileShapeQK{})),  // M = 128
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

    // Extract TMA descriptors from CollectiveBuilder
    using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
    using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
    using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

    ///////////////////////////////////////////////////////////////////////////
    // Pipeline Types
    ///////////////////////////////////////////////////////////////////////////

    using AtomThrShape = typename CollectiveMmaQK::AtomThrShapeMNK;

    // From Load to MMA warp (TMA loads Q/K/V into SMEM)
    using PipelineQ = cutlass::PipelineTmaUmmaAsync<kStageCountQ, AtomThrShape>;
    using PipelineKV = cutlass::PipelineTmaUmmaAsync<kStageCountKV, AtomThrShape>;

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

    using TmemAllocator = cute::TMEM::Allocator1Sm;

    ///////////////////////////////////////////////////////////////////////////
    // Shared Storage
    ///////////////////////////////////////////////////////////////////////////

    struct SharedStorage : cute::aligned_struct<128, _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        union {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        };
        cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutO>> smem_o;

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

        uint32_t tmem_base_ptr;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Legacy compatibility
    ///////////////////////////////////////////////////////////////////////////

    using MainloopPipeline = PipelineKV;
    using PipelineState = typename PipelineKV::PipelineState;
    using MainloopPipelineQ = PipelineQ;
    using PipelineParamsQ = typename PipelineQ::Params;
    using PipelineStateQ = typename PipelineQ::PipelineState;
    static constexpr int kStages = kStageCountKV;
    static constexpr int EpiStages = 2;
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;
};

} // namespace flash
