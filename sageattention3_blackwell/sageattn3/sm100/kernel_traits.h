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
 *   - Accumulators live in TMEM (256KB per SM), not registers
 *   - Warp-specialized execution: 16 warps with specific roles
 *   - Uses UMMA atoms instead of GMMA
 *   - M dimension MUST be 128 (hardware constraint for UMMA)
 *   - Manually constructed MMA and SMEM layouts (similar to SM120)
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
#include "cute/arch/mma_sm100_umma.hpp"

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
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> smem_q;

    // K and V share SMEM (loaded sequentially)
    union {
        alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> smem_k;
        alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> smem_v;
    };

    // Output tensor in SMEM (for epilogue TMA store)
    alignas(1024) cute::ArrayEngine<ElementOut, cute::cosize_v<SmemLayoutO>> smem_o;

    // Pipeline synchronization
    struct {
        alignas(16) typename cutlass::PipelineTmaAsync<StageCountQ>::SharedStorage pipeline_q;
        alignas(16) typename cutlass::PipelineTmaAsync<StageCountKV>::SharedStorage pipeline_kv;
        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipeline_s0;
        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipeline_s1;
        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipeline_c0;
        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipeline_c1;
        alignas(16) typename cutlass::PipelineAsync<2>::SharedStorage pipeline_o;
        alignas(16) typename cutlass::PipelineAsync<2>::SharedStorage pipeline_epi;
        alignas(16) typename cutlass::OrderedSequenceBarrier<1, 2>::SharedStorage order_s01;
    } pipelines;

    // TMEM base pointer (set by MMA warp after allocation)
    uint32_t tmem_base_ptr;
};

///////////////////////////////////////////////////////////////////////////////
// SM100 Flash Forward Kernel Traits
//
// Uses manual MMA/SMEM construction similar to SM120
// FP4 with E4M3 scale factors, SFVectorSize=16 (NVIDIA FP4 format)
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kStages_,
    int kClusterM_,
    bool BlockMean_,
    typename ElementPairType_ = cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100 {

    // Basic configuration
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr bool SmoothQ = true;

    // SM100 UMMA requires M=128 (single tile)
    // With ThreadShape = (2,1,1), we need kBlockM = 256 so TileShapeQK.M = 128
    static_assert(kBlockM == 256, "SM100 requires kBlockM=256 (TileShapeQK.M=128 after ThreadShape division)");
    // SM100 with FP4 requires HeadDim >= 128 (TMA load constraint for 4-bit data)
    static_assert(kHeadDim >= 128, "SM100 with FP4 requires HeadDim >= 128 (TMA load constraint)");
    static_assert(kHeadDim % 32 == 0);

    // Warp scheduling
    using Schedule = Sm100WarpSpecializedSchedule;
    static constexpr int kNWarps = Schedule::kNumWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;

    // Pipeline stages
    static constexpr int kStageCountQ = 1;  // Q is loaded once
    static constexpr int kStageCountKV = kStages_;

    // Scale factor configuration (NVIDIA FP4 format: SFVectorSize=16, E4M3 SF)
    static constexpr int SFVectorSize = 16;
    static constexpr int kSFVecSize = SFVectorSize;
    static constexpr int NumSFQK = kHeadDim / SFVectorSize;
    static constexpr int NumSFPV = kBlockN / SFVectorSize;

    // Element types
    using ElementSF = cutlass::float_ue4m3_t;    // FP8 E4M3 unsigned scale factors
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

    // TileShape for individual QK/PV computations (after ThreadShape division)
    // TileShapeQK.M = 256/2 = 128, which matches UMMA's M requirement
    using TileShapeQK = decltype(shape_div(TileShape_MNK{}, ThreadShape{}));
    using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));

    // Architecture tag
    using ArchTag = cutlass::arch::Sm100;

    // Permutation tile sizes (same as SM120)
    using PermTileM = decltype(cute::min(size<0>(TileShapeQK{}), _128{}));
    using PermTileN = _32;
    using PermTileK = Int<kHeadDim>;

    ///////////////////////////////////////////////////////////////////////////
    // MMA Configuration (manual, similar to SM120)
    //
    // Use SM100 UMMA blockscaled MMA atoms for FP4
    // For MXF4_NVF4 instruction (FP4 x FP4), IsF8F6F4 = false
    ///////////////////////////////////////////////////////////////////////////

    // MMA input element type conversion
    // For MXF4 instruction: IsF8F6F4 = false, so element type stays as-is
    using ElementQMma = decltype(cutlass::gemm::collective::detail::sm100_kernel_input_element_to_mma_input_element<Element, false>());
    using ElementKMma = decltype(cutlass::gemm::collective::detail::sm100_kernel_input_element_to_mma_input_element<Element, false>());

    // Atom layout for 128 rows
    using AtomLayoutMNK = Layout<Shape<_8, _1, _1>>;

    // TiledMMA for QK: using SM100 blockscaled atom for FP4
    // SM100_MMA_MXF4_SS: MXF4 instruction for FP4 with scale factors
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::SM100_MMA_MXF4_SS<
            ElementQMma, ElementKMma, ElementAccum, ElementSF,
            128,  // M - must be 128 for SM100 1-CTA cluster
            128,  // N
            SFVectorSize,
            UMMA::Major::K,  // A is row-major (K-major)
            UMMA::Major::K   // B is col-major (K-major for B)
        >{}
    ));

    // TiledMMA for PV (same structure)
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        cute::SM100_MMA_MXF4_SS<
            ElementQMma, ElementKMma, ElementAccum, ElementSF,
            128,  // M
            128,  // N
            SFVectorSize,
            UMMA::Major::K,
            UMMA::Major::K
        >{}
    ));

    static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVectorSize;

    ///////////////////////////////////////////////////////////////////////////
    // TMA Copy operations
    ///////////////////////////////////////////////////////////////////////////

    using GmemTiledCopy = SM90_TMA_LOAD;
    using GmemTiledCopySF = SM90_TMA_LOAD;

    ///////////////////////////////////////////////////////////////////////////
    // SMEM Layouts (manual, using SM100 selectors)
    ///////////////////////////////////////////////////////////////////////////

    // For blockscaled FP4 (MXF4_NVF4), use ElementQMma for SMEM allocation
    // This gives proper swizzling for the 4-bit data type
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        UMMA::Major::K, ElementQMma,
        decltype(cute::get<0>(TileShapeQK{})),
        decltype(cute::get<2>(TileShapeQK{}))>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        UMMA::Major::K, ElementKMma,
        decltype(cute::get<1>(TileShapeQK{})),
        decltype(cute::get<2>(TileShapeQK{}))>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        UMMA::Major::K, ElementKMma,
        decltype(cute::get<1>(TileShapePV{})),
        decltype(cute::get<2>(TileShapePV{}))>());

    // Q: single stage, 2 tiles (for ThreadShape=(2,1,1))
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        make_shape(get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}))));

    // K: multiple stages
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        make_shape(get<1>(TileShapeQK{}), get<2>(TileShapeQK{}), Int<kStageCountKV>{})));

    // V: multiple stages (same layout as K for now)
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        make_shape(get<1>(TileShapePV{}), get<2>(TileShapePV{}), Int<kStageCountKV>{})));

    // Output SMEM layout
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, ElementOut,
        decltype(cute::get<0>(TileShape_MNK{})),
        decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        select<0, 2>(TileShape_MNK{}),
        Step<_1, _2>{}));

    // SMEM copy atoms
    using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

    ///////////////////////////////////////////////////////////////////////////
    // Scale Factor Layouts (same as SM120)
    ///////////////////////////////////////////////////////////////////////////

    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;
    using SfAtom = typename BlkScaledConfig::SfAtom;
    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));

    ///////////////////////////////////////////////////////////////////////////
    // Strides for GMEM tensors
    ///////////////////////////////////////////////////////////////////////////

    // Row-major strides: (batch, head, seq, dim)
    using StrideQ = Stride<int64_t, int64_t, int64_t, _1>;
    using StrideK = Stride<int64_t, int64_t, int64_t, _1>;
    using StrideV = Stride<int64_t, int64_t, _1, int64_t>;  // V is transposed
    using StrideO = Stride<int64_t, int64_t, int64_t, _1>;

    ///////////////////////////////////////////////////////////////////////////
    // TMA descriptors
    ///////////////////////////////////////////////////////////////////////////

    using TMA_Q = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<Element const*>(nullptr), make_shape(1, 1, 1), StrideQ{}),
        SmemLayoutAtomQ{},
        select<0, 2>(TileShapeQK{}),
        _1{}));

    using TMA_K = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<Element const*>(nullptr), make_shape(1, 1, 1), StrideK{}),
        SmemLayoutAtomK{},
        select<1, 2>(TileShapeQK{}),
        _1{}));

    using TMA_V = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<Element const*>(nullptr), make_shape(1, 1, 1), StrideV{}),
        SmemLayoutAtomV{},
        select<1, 2>(TileShapePV{}),
        _1{}));

    ///////////////////////////////////////////////////////////////////////////
    // Pipeline Types (using simpler PipelineAsync for SM100)
    ///////////////////////////////////////////////////////////////////////////

    // TMA async pipeline for Q loads
    using PipelineQ = cutlass::PipelineTmaAsync<kStageCountQ>;

    // TMA async pipeline for K/V loads
    using PipelineKV = cutlass::PipelineTmaAsync<kStageCountKV>;

    // Async pipeline for S (MMA -> Softmax)
    using PipelineS = cutlass::PipelineAsync<1>;

    // Async pipeline for correction (Softmax -> Correction)
    using PipelineC = cutlass::PipelineAsync<1>;

    // Async pipeline for O (MMA -> Correction)
    using PipelineO = cutlass::PipelineAsync<2>;

    // Async pipeline for epilogue (Correction -> Epilogue)
    using PipelineE = cutlass::PipelineAsync<2>;

    // Ordered barrier for softmax warps
    using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<1, 2>;

    ///////////////////////////////////////////////////////////////////////////
    // TMEM Allocator
    ///////////////////////////////////////////////////////////////////////////

    using TmemAllocator = cute::TMEM::Allocator1Sm;

    ///////////////////////////////////////////////////////////////////////////
    // Shared Storage
    ///////////////////////////////////////////////////////////////////////////

    // K and V share SMEM - use whichever layout is larger
    // For now, use K's layout (they should be similar size)
    using SmemLayoutKV = SmemLayoutK;

    using SharedStorage = SharedStorageSm100<
        Element, ElementOut, SmemLayoutQ, SmemLayoutKV, SmemLayoutO,
        kStageCountQ, kStageCountKV>;

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
