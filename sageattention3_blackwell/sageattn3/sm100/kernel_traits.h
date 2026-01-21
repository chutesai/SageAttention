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
 * SM100 (B200/B300) kernel traits for FlashAttention with FP4 blockscaled MMA.
 *
 * KEY DIFFERENCES FROM SM120:
 *   - Uses SM100_MMA_MXF4_SS atom (not SM120_16x32x64_TN_VS_NVFP4)
 *   - Accumulators live in TMEM (256KB per SM), not registers
 *   - Scale factors also stored in TMEM
 *   - Uses tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale instruction
 *   - M dimension MUST be 128 (hardware constraint)
 *   - K dimension is 64 (256 bits / 4 bits per element)
 */

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

// Include SM100 MMA atoms and TMEM support
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

#include "../blackwell/blockscaled_layout.h"
#include "../blackwell/named_barrier.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// TMEM Allocation for SM100 Flash Attention
//
// SM100 has 256KB TMEM per SM. We need to allocate space for:
//   - S (QK^T scores): 128 x 128 x sizeof(float) = 64KB
//   - O (output accumulator): 128 x headdim x sizeof(float)
//   - P (softmax probabilities, FP4): 128 x 128 / 2 = 8KB
//   - Scale factors for P: 128 x (128/16) = 1KB
///////////////////////////////////////////////////////////////////////////////

enum class Sm100TmemAlloc : uint32_t {
    // For 128x128 tiles with headdim=128
    kSizeS = 128 * 128,     // 16K floats = 64KB (S = Q*K^T)
    kSizeO = 128 * 128,     // 16K floats = 64KB (O accumulator)
    kSizeSF = 128 * 8,      // Scale factors (128 rows, headdim/16 columns)

    S_offset = 0,
    O_offset = S_offset + kSizeS,
    SF_offset = O_offset + kSizeO,
    kEnd = SF_offset + kSizeSF
};

///////////////////////////////////////////////////////////////////////////////
// Shared Storage for SM100
// Same structure as SM120 but pipelines use different types for TMEM
///////////////////////////////////////////////////////////////////////////////

template <
    int kStages,
    int EpiStages,
    typename Element,
    typename ElementSF,
    typename OutputType,
    typename SmemLayoutQ,
    typename SmemLayoutK,
    typename SmemLayoutV,
    typename SmemLayoutDS,
    typename SmemLayoutO,
    typename SmemLayoutSFQ,
    typename SmemLayoutSFK,
    typename SmemLayoutSFV
>
struct SharedStorageSm100 : cute::aligned_struct<128, _0> {

    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_SFQ;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_SFK;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> smem_SFV;
    alignas(1024) cute::ArrayEngine<float, cute::cosize_v<SmemLayoutDS>> smem_ds;
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    alignas(1024) cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;

    struct {
        // SM100 uses PipelineTmaUmmaAsync for TMEM-based MMA
        alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        alignas(16) typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>::SharedStorage barrier_o;
        int tile_count_semaphore;
    };
};

///////////////////////////////////////////////////////////////////////////////
// SM100 Flash Forward Kernel Traits
//
// Uses SM100_MMA_MXF4_SS atoms with TMEM accumulators
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
    // SM100 MXF4 MMA requires M=128
    static constexpr int kBlockM = 128;  // MUST be 128 for SM100_MMA_MXF4_SS
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr bool SmoothQ = true;

    static_assert(kBlockM_ == 128, "SM100 FP4 MMA requires M=128");
    static_assert(kHeadDim % 32 == 0);

    // Thread/warp configuration
    // SM100 uses different warp counts for TMEM management
    static constexpr int kNWarps = 12;  // SM100 typically uses 12 warps
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 1;

    // Scale factor configuration
    // SM100 uses vector size 16 for NV FP4 scale factors
    static constexpr int SFVectorSize = 16;
    static constexpr int kSFVecSize = SFVectorSize;
    static constexpr int NumSFQK = kHeadDim / SFVectorSize;
    static constexpr int NumSFPV = kBlockN / SFVectorSize;

    // Element types
    using ElementSF = cutlass::float_ue4m3_t;   // FP8 E4M3 scale factors (NV format)
    using Element = cutlass::float_e2m1_t;       // FP4 E2M1 data
    using ElementAccum = float;                   // FP32 accumulators (in TMEM!)
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Tile shapes
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;

    // Architecture tag
    using ArchTag = cutlass::arch::Sm100;

    ///////////////////////////////////////////////////////////////////////////
    // MMA Configuration for SM100
    //
    // SM100_MMA_MXF4_SS: M=128, N=8-256, K=64, VS=16
    // Uses tcgen05.mma with TMEM accumulators
    ///////////////////////////////////////////////////////////////////////////

    using PermTileM = Int<128>;  // Must be 128 for SM100
    using PermTileN = _32;       // Match SM120 pattern
    using PermTileK = Int<kHeadDim>;

    // Element types for MMA (same as input, CUTLASS handles conversion)
    using ElementQMma = Element;
    using ElementKMma = Element;

    // SM100 MMA atom: 128 x N x 64 with VS=16 scale factors
    // For QK: M=128, N=kBlockN (128), K=kHeadDim (64 or 128)
    // For PV: M=128, N=kHeadDim (64 or 128), K=kBlockN (128)

    // Create tiled MMA using SM100_MMA_MXF4_SS
    // Note: SM100_MMA_MXF4_SS has fixed M=128, so we need to tile appropriately
    using MmaAtomQK = cute::SM100_MMA_MXF4_SS<
        Element, Element, ElementAccum, ElementSF,
        128,           // M = 128 (required)
        kBlockN,       // N = tile N dimension
        SFVectorSize,  // VS = 16 for NV FP4
        UMMA::Major::K,  // A major (row major Q)
        UMMA::Major::K   // B major (col major K^T)
    >;

    using MmaAtomPV = cute::SM100_MMA_MXF4_SS<
        Element, Element, ElementAccum, ElementSF,
        128,           // M = 128 (required)
        kHeadDim,      // N = head dimension
        SFVectorSize,  // VS = 16
        UMMA::Major::K,  // A major
        UMMA::Major::K   // B major
    >;

    // Create tiled MMA from atoms
    using TiledMmaQK = decltype(cute::make_tiled_mma(MmaAtomQK{}));
    using TiledMmaPV = decltype(cute::make_tiled_mma(MmaAtomPV{}));

    static constexpr int MMA_NSF = 64 / SFVectorSize;  // K=64 / VS=16 = 4

    ///////////////////////////////////////////////////////////////////////////
    // Copy Atoms
    ///////////////////////////////////////////////////////////////////////////

    using GmemTiledCopy = SM90_TMA_LOAD;
    using GmemTiledCopySF = SM90_TMA_LOAD;

    ///////////////////////////////////////////////////////////////////////////
    // SMEM Layouts
    // Use SM100 SMEM selectors for optimal memory access patterns
    ///////////////////////////////////////////////////////////////////////////

    // Use SM100 smem selector (falls back to SM90 patterns which work for SM100)
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm90_smem_selector<
        GMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm90_smem_selector<
        GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm90_smem_selector<
        GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm90_smem_selector<
        GMMA::Major::K, Element, Int<kHeadDim>, Int<kBlockN>>());

    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<_0, _1>>;
    using SmemLayoutDS = decltype(tile_to_shape(SmemLayoutAtomDS{},
            make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    // SMEM copy atoms
    using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
    using SmemCopyAtomDS = Copy_Atom<UniversalCopy<float>, float>;

    ///////////////////////////////////////////////////////////////////////////
    // Scale Factor Layouts
    ///////////////////////////////////////////////////////////////////////////

    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;
    using SfAtom = typename BlkScaledConfig::SfAtom;

    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));

    using LayoutSFP = decltype(
      make_layout(
          make_shape(make_shape(_16{}, _4{}), _1{}, Int<kBlockN / 64>{}),
          make_stride(make_stride(_0{}, _1{}), _0{}, _4{})
      )
    );
    using LayoutP = decltype(
      make_layout(
        make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBlockN / 64>{}),
        make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})
      )
    );

    using SmemLayoutSFQ = decltype(make_layout(
        shape(SmemLayoutAtomSFQ{}),
        stride(SmemLayoutAtomSFQ{})
    ));
    using SmemLayoutSFK = decltype(make_layout(
        append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))
    ));
    using SmemLayoutSFV = decltype(make_layout(
        append(shape(SmemLayoutAtomSFV{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFV{}), size(filter_zeros(SmemLayoutAtomSFV{})))
    ));
    using SmemLayoutSFVt = decltype(make_layout(
        append(shape(SmemLayoutAtomSFVt{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFVt{}), size(filter_zeros(SmemLayoutAtomSFVt{})))
    ));

    ///////////////////////////////////////////////////////////////////////////
    // Output Layout
    ///////////////////////////////////////////////////////////////////////////

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementOut,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{}), Step<_1, _2>{}));

    ///////////////////////////////////////////////////////////////////////////
    // Shared Storage
    ///////////////////////////////////////////////////////////////////////////

    using SharedStorage = SharedStorageSm100<kStages, EpiStages, Element, ElementSF, ElementOut,
        SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutDS,
        SmemLayoutO, SmemLayoutSFQ, SmemLayoutSFK, SmemLayoutSFVt>;

    ///////////////////////////////////////////////////////////////////////////
    // Pipeline Types
    // SM100 uses different pipeline types for TMEM-based MMA
    ///////////////////////////////////////////////////////////////////////////

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename cutlass::PipelineState<1>;
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;
};

} // namespace flash
