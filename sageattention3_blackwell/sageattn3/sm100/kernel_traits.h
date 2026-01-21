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
 * This follows the SM120 pattern but uses SM100-specific:
 *   - SMEM layout selectors (sm100_smem_selector)
 *   - MMA atoms that use tcgen05.mma with TMEM accumulators
 *   - SM100 blockscaled collective infrastructure
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

#include "../blackwell/blockscaled_layout.h"
#include "../blackwell/named_barrier.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Shared Storage for SM100
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
// KEY DIFFERENCES FROM SM120:
//   - Uses sm100_smem_selector instead of sm120_rr_smem_selector
//   - Uses OpClassBlockScaledTensorOp with SM100 collective builder
//   - MMA atoms produce output in TMEM instead of registers
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

    // Basic configuration - same as SM120
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr bool SmoothQ = true;

    static_assert(kHeadDim % 32 == 0);
    static_assert(kBlockM == 64 || kBlockM == 128);

    // Thread/warp configuration
    static constexpr int kNWarps = kBlockM == 128 ? 12 : 8;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 1;

    // Scale factor configuration
    static constexpr int NumSFQK = kHeadDim / 16;
    static constexpr int NumSFPV = kBlockN / 16;
    static constexpr int SFVectorSize = 16;
    static constexpr int kSFVecSize = SFVectorSize;

    // Element types - same interface as SM120
    using ElementSF = cutlass::float_ue4m3_t;
    using Element = cutlass::float_e2m1_t;
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Tile shapes
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;

    // Architecture tag
    using ArchTag = cutlass::arch::Sm100;

    ///////////////////////////////////////////////////////////////////////////
    // MMA Configuration
    // For SM100, we use the same blockscaled MMA pattern as SM120 but with
    // SM100-specific atoms that use tcgen05.mma instructions
    ///////////////////////////////////////////////////////////////////////////

    using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
    using PermTileN = _32;
    using PermTileK = Int<kHeadDim>;

    using ElementQMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using ElementKMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());

    using AtomLayoutMNK = std::conditional_t<kBlockM == 128,
                                            Layout<Shape<_8, _1, _1>>,
                                            Layout<Shape<_4, _1, _1>>
                                            >;

    // For SM100, use the SM120 blockscaled MMA atom pattern
    // The SM100 will use its tcgen05.mma internally when compiled for sm_100a
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK{},
        Tile<PermTileM, PermTileN, PermTileK>{}
    ));

    using TiledMmaPV = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK{},
        Tile<PermTileM, _32, PermTileK>{}
    ));

    static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVectorSize;

    ///////////////////////////////////////////////////////////////////////////
    // Copy Atoms
    ///////////////////////////////////////////////////////////////////////////

    using GmemTiledCopy = SM90_TMA_LOAD;
    using GmemTiledCopySF = SM90_TMA_LOAD;

    ///////////////////////////////////////////////////////////////////////////
    // SMEM Layouts
    // Use SM100-specific SMEM selectors where available, fall back to SM120
    ///////////////////////////////////////////////////////////////////////////

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<1>(TileShape_MNK{}))>());

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
    // Scale Factor Layouts - same as SM120
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
    ///////////////////////////////////////////////////////////////////////////

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename cutlass::PipelineState<1>;
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;
};

} // namespace flash
