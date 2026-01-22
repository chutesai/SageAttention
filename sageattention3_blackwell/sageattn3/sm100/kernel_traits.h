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
 * KEY DIFFERENCES FROM SM120:
 *   - Uses CollectiveBuilder to generate SM100-specific MMA operations
 *   - Accumulators live in TMEM (256KB per SM), not registers
 *   - Warp-specialized execution: 16 warps with specific roles
 *   - Uses tcgen05.mma with UMMA atoms
 *   - M dimension MUST be 128 (hardware constraint)
 */

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/pipeline/sm100_pipeline.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

// Include SM100 utilities
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cute/arch/tmem_allocator_sm100.hpp"

#include "../blackwell/blockscaled_layout.h"
#include "../blackwell/named_barrier.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 Warp Roles
//
// SM100 uses warp-specialized execution with 16 warps total:
//   - Warps 0-3:   Softmax0 (compute softmax on S0)
//   - Warps 4-7:   Softmax1 (compute softmax on S1)
//   - Warps 8-11:  Correction (rescale O accumulators)
//   - Warp 12:     MMA (tensor core computations)
//   - Warp 13:     Load (TMA loads Q, K, V)
//   - Warp 14:     Epilogue (TMA store O)
//   - Warp 15:     Empty (donates registers)
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

    // Register allocation per warp role
    static constexpr int kNumRegsSoftmax = 192;
    static constexpr int kNumRegsCorrection = 96;
    static constexpr int kNumRegsOther = 32;
    static constexpr int kNumRegsEmpty = 24;
};

///////////////////////////////////////////////////////////////////////////////
// TMEM Allocation for SM100
//
// SM100 has 256KB TMEM per SM. For flash attention we allocate:
//   - S0, S1: Score matrices (128 columns each, overlaps with V0/V1 and P0/P1)
//   - O0, O1: Output accumulators (128 columns each)
//   - V0, V1: Stats storage (row max/sum) - overlaps with S
//   - P0, P1: Probabilities after softmax - overlaps with S
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

// Indices for V0/V1 (stats storage)
enum {
    kIdxOldRowMax = 0,
    kIdxNewRowMax = 1,
    kIdxFinalRowSum = 0,
    kIdxFinalRowMax = 1
};

///////////////////////////////////////////////////////////////////////////////
// Shared Storage for SM100
///////////////////////////////////////////////////////////////////////////////

template <
    typename Element,
    typename ElementOut,
    typename SmemLayoutQ,
    typename SmemLayoutKV,
    typename SmemLayoutO,
    int StageCountQ,
    int StageCountKV
>
struct SharedStorageSm100 : cute::aligned_struct<128, _0> {
    // Q tensor in SMEM
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;

    // K and V share SMEM (loaded sequentially)
    union {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutKV>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutKV>> smem_v;
    };

    // Output tensor in SMEM (for epilogue TMA store)
    cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutO>> smem_o;

    // Pipeline synchronization
    struct {
        alignas(16) typename cutlass::PipelineTmaUmmaAsync<StageCountQ, Shape<_1,_1,_1>>::SharedStorage pipeline_q;
        alignas(16) typename cutlass::PipelineTmaUmmaAsync<StageCountKV, Shape<_1,_1,_1>>::SharedStorage pipeline_kv;
        alignas(16) typename cutlass::PipelineUmmaAsync<1>::SharedStorage pipeline_s0;
        alignas(16) typename cutlass::PipelineUmmaAsync<1>::SharedStorage pipeline_s1;
        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipeline_c0;
        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipeline_c1;
        alignas(16) typename cutlass::PipelineUmmaAsync<2>::SharedStorage pipeline_o;
        alignas(16) typename cutlass::PipelineAsync<2>::SharedStorage pipeline_epi;
        alignas(16) typename cutlass::OrderedSequenceBarrier<1, 2>::SharedStorage order_s01;
    } pipelines;

    // TMEM base pointer (set by MMA warp after allocation)
    uint32_t tmem_base_ptr;
};

///////////////////////////////////////////////////////////////////////////////
// SM100 Flash Forward Kernel Traits
//
// Uses CollectiveBuilder pattern from CUTLASS to generate correct MMA ops
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kStages_,
    int kClusterM_,
    bool BlockMean_,
    typename ElementPairType_ = uint8_t,  // Packed FP4 bytes
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100 {

    // Basic configuration
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr bool SmoothQ = true;

    // SM100 requires M >= 128 AFTER ThreadShape division
    // With ThreadShape = (2,1,1), we need kBlockM >= 256 to get TileShapeQK.M >= 128
    static_assert(kBlockM == 256, "SM100 requires kBlockM=256 (TileShapeQK.M=128 after ThreadShape division)");
    // SM100 with FP4 requires TileShape_K >= 128 (hardware constraint for F4/F6 TMA loads)
    // Since TileShape_K = HeadDim, we require HeadDim >= 128
    static_assert(kHeadDim >= 128, "SM100 with FP4 requires HeadDim >= 128 (TMA load constraint)");
    static_assert(kHeadDim % 32 == 0);

    // Warp scheduling
    using Schedule = Sm100WarpSpecializedSchedule;
    static constexpr int kNWarps = Schedule::kNumWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;

    // Pipeline stages
    static constexpr int kStageCountQ = 2;
    static constexpr int kStageCountKV = 4;  // More stages for K/V

    // Scale factor configuration
    static constexpr int SFVectorSize = 16;
    static constexpr int kSFVecSize = SFVectorSize;
    static constexpr int NumSFQK = kHeadDim / SFVectorSize;
    static constexpr int NumSFPV = kBlockN / SFVectorSize;

    // Element types
    using ElementSF = cutlass::float_ue4m3_t;    // FP8 E4M3 scale factors
    using Element = cutlass::float_e2m1_t;       // FP4 E2M1 data
    using ElementAccum = float;                   // FP32 accumulators (in TMEM!)
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Tile shapes
    // For SM100 warp-specialized: tile processes 2 Q blocks alternating
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;

    // Thread shape: how Q tiles are distributed across softmax warps
    // (2, 1, 1) means two Q blocks stacked - best for large Q
    using ThreadShape = Shape<_2, _1, _1>;

    // TileShape for individual QK/PV computations
    using TileShapeQK = decltype(shape_div(TileShape_MNK{}, ThreadShape{}));
    using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));

    // Architecture tag
    using ArchTag = cutlass::arch::Sm100;

    // Alignment - For FP4 (4-bit), SM100 requires 512-bit (64-byte) alignment
    // This means 512 / 4 = 128 elements
    static constexpr int Alignment = 512 / sizeof_bits_v<Element>;

    ///////////////////////////////////////////////////////////////////////////
    // MMA Configuration using CollectiveBuilder
    //
    // CollectiveBuilder automatically generates correct SM100 MMA operations
    ///////////////////////////////////////////////////////////////////////////

    // Strides for Q, K, V
    // Q: [seqlen_q, head_dim, batch * num_heads] - row major
    // K: [seqlen_k, head_dim, batch * num_heads] - row major
    // V: [head_dim, seqlen_k, batch * num_heads] - transposed for PV gemm
    using StrideQ = cute::Stride<int64_t, _1, int64_t>;
    using StrideK = cute::Stride<int64_t, _1, int64_t>;
    using StrideV = cute::Stride<_1, int64_t, int64_t>;

    // Build QK collective MMA using CollectiveBuilder
    using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideQ, Alignment,
        Element, StrideK, Alignment,
        ElementAccum,
        TileShapeQK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCount<kStageCountKV>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
    >::CollectiveOp;

    // Build PV collective MMA using CollectiveBuilder
    using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideK, Alignment,  // P: reuse K stride (dummy, loaded from TMEM)
        Element, decltype(select<1,0,2>(StrideV{})), Alignment,  // V transposed
        ElementAccum,
        TileShapePV, ClusterShape_MNK,
        cutlass::gemm::collective::StageCount<kStageCountKV>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
    >::CollectiveOp;

    // Extract types from CollectiveBuilder
    using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
    using TiledMmaPV = typename CollectiveMmaPV::TiledMma;
    using AtomThrShapeMNK = typename CollectiveMmaQK::AtomThrShapeMNK;

    ///////////////////////////////////////////////////////////////////////////
    // SMEM Layouts (derived from CollectiveBuilder)
    ///////////////////////////////////////////////////////////////////////////

    using SmemLayoutQ = decltype(unstageSmemLayout(
        typename CollectiveMmaQK::SmemLayoutA{}, Int<kStageCountQ>{}));
    using SmemLayoutK = decltype(unstageSmemLayout(
        typename CollectiveMmaQK::SmemLayoutB{}, Int<kStageCountKV>{}));
    using SmemLayoutV = decltype(unstageSmemLayout(
        typename CollectiveMmaPV::SmemLayoutB{}, Int<kStageCountKV>{}));

    // Output SMEM layout
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        cute::UMMA::Major::K, ElementOut,
        decltype(cute::get<0>(TileShape_MNK{})),
        decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        replace<2>(TileShape_MNK{}, _2{}),  // 2 stages for double buffering
        Step<_2, _1, _3>{}));

    ///////////////////////////////////////////////////////////////////////////
    // Pipeline Types
    ///////////////////////////////////////////////////////////////////////////

    // TMA+UMMA async pipeline for Q loads
    using PipelineQ = cutlass::PipelineTmaUmmaAsync<kStageCountQ, AtomThrShapeMNK>;

    // TMA+UMMA async pipeline for K/V loads
    using PipelineKV = cutlass::PipelineTmaUmmaAsync<kStageCountKV, AtomThrShapeMNK>;

    // UMMA async pipeline for S (MMA -> Softmax)
    using PipelineS = cutlass::PipelineUmmaAsync<1>;

    // Async pipeline for correction (Softmax -> Correction)
    using PipelineC = cutlass::PipelineAsync<1>;

    // UMMA async pipeline for O (MMA -> Correction)
    using PipelineO = cutlass::PipelineUmmaAsync<2>;

    // Async pipeline for epilogue (Correction -> Epilogue)
    using PipelineE = cutlass::PipelineAsync<2>;

    // Ordered barrier for softmax warps
    using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<1, 2>;

    // Transaction bytes for TMA
    static constexpr int TransactionBytesQ = cutlass::bits_to_bytes(
        cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);
    static constexpr int TransactionBytesKV = cutlass::bits_to_bytes(
        cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);

    ///////////////////////////////////////////////////////////////////////////
    // TMEM Allocator
    ///////////////////////////////////////////////////////////////////////////

    using TmemAllocator = cute::TMEM::Allocator1Sm;

    ///////////////////////////////////////////////////////////////////////////
    // Shared Storage
    ///////////////////////////////////////////////////////////////////////////

    using SharedStorage = SharedStorageSm100<
        Element, ElementOut, SmemLayoutQ, SmemLayoutK, SmemLayoutO,
        kStageCountQ, kStageCountKV>;

    ///////////////////////////////////////////////////////////////////////////
    // TMA descriptors (for launch params)
    ///////////////////////////////////////////////////////////////////////////

    using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
    using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
    using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

    // Legacy compatibility (for params.h)
    using MainloopPipeline = PipelineKV;
    using PipelineState = typename PipelineKV::PipelineState;
    using MainloopPipelineQ = PipelineQ;
    using PipelineParamsQ = typename PipelineQ::Params;
    using PipelineStateQ = typename PipelineQ::PipelineState;
    static constexpr int kStages = kStageCountKV;
    static constexpr int EpiStages = 2;
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;
};

///////////////////////////////////////////////////////////////////////////////
// Helper to unstage SMEM layout (extract single stage from multi-stage layout)
///////////////////////////////////////////////////////////////////////////////

template <class SmemLayout, class StageCount>
CUTE_HOST_DEVICE constexpr auto unstageSmemLayout(SmemLayout, StageCount) {
    return SmemLayout{}(_, _, _, cute::Int<0>{});
}

} // namespace flash
