/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled kernel traits for SageAttention3.
 *
 * This uses SM100's tcgen05.mma block-scaled FP4 instructions (SM100_MMA_MXF4_SS).
 *
 * Key Design Decisions:
 * ---------------------
 * 1. SM100 FP4 MMA uses K=64 elements (256 bits) per MMA operation
 * 2. We do NOT use CollectiveBuilder since it's designed for GEMM, not FMHA
 * 3. Instead, we follow the SM120 pattern of directly constructing TiledMma
 * 4. Block-scaled format: E2M1 data with E4M3 scale factors, 16 elements per SF
 *
 * For Flash Attention:
 * - QK GEMM: Q(BlockM×HeadDim) @ K^T(HeadDim×BlockN) → S(BlockM×BlockN)
 * - PV GEMM: P(BlockM×BlockN) @ V(BlockN×HeadDim) → O(BlockM×HeadDim)
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
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

// Include SM100 MMA atoms
#include "cute/arch/mma_sm100.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Forward Kernel Traits
//
// This follows the pattern from SM120 kernel_traits.h but adapts for SM100
// by using SM100_MMA_MXF4_SS instead of SM120's mma.sync atoms
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,    // Head dimension (128 or 256)
    int kBlockM_,     // Block size for Q rows (128 or 256)
    int kBlockN_,     // Block size for K/V sequence (128 or 256)
    int kStages_,     // Pipeline stages (2-4)
    int kClusterM_,   // Cluster shape in M (1)
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
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 2;
    static constexpr int kNWarps = 16;
    static constexpr int kNThreads = kNWarps * 32;

    //=========================================================================
    // Element Types for FP4 Block-Scaled Attention
    //=========================================================================

    // FP4 format: E2M1 data with E4M3 scale factors
    using Element = cutlass::float_e2m1_t;           // FP4 data (E2M1)
    using ElementSF = cutlass::float_e4m3_t;         // Scale factor (E4M3)
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

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    //=========================================================================
    // MMA Tile Permissions
    //=========================================================================

    // For kBlockM=128: 8 atoms along M, for kBlockM=256: 4 atoms (2SM mode)
    using PermTileM = decltype(cute::min(Int<kBlockM>{}, _128{}));
    using PermTileN = _32;
    using PermTileK = Int<kHeadDim>;

    using AtomLayoutMNK = std::conditional_t<kBlockM == 128,
                                            Layout<Shape<_8, _1, _1>>,
                                            Layout<Shape<_4, _1, _1>>>;

    //=========================================================================
    // TiledMma Definitions - Direct construction like SM120
    //
    // SM100 uses SM100_MMA_MXF4_SS for FP4 block-scaled MMA
    // The MMA atom shape is (M, N, K) where K=64 for FP4 (256 bits / 4 bits)
    //=========================================================================

    // For QK GEMM: A=Q (row-major), B=K^T (col-major, i.e., K transposed)
    // SM100_MMA_MXF4_SS<a_type, b_type, c_type, sf_type, M, N, VS, a_major, b_major, a_neg, b_neg>
    // where:
    //   - M, N are the MMA tile dimensions
    //   - VS = SFVectorSize (16)
    //   - a_major = UMMA::Major::K (row-major A)
    //   - b_major = UMMA::Major::K (col-major B)
    //   - a_neg, b_neg = UMMA::ScaleIn::One (no negation)

    // MMA atom for FP4 block-scaled: 64x128x64 (M x N x K) - K=64 elements for FP4
    using MMA_Atom_QK = SM100_MMA_MXF4_SS<
        Element,                    // a_type (FP4 E2M1)
        Element,                    // b_type (FP4 E2M1)
        ElementAccum,               // c_type (FP32)
        ElementSF,                  // sf_type (E4M3)
        64,                         // M
        128,                        // N
        SFVectorSize,               // VS = 16
        UMMA::Major::K,             // a_major (row-major A = Q)
        UMMA::Major::K,             // b_major (col-major B = K^T)
        UMMA::ScaleIn::One,         // a_neg (no negation)
        UMMA::ScaleIn::One          // b_neg (no negation)
    >;

    using TiledMmaQK = decltype(cute::make_tiled_mma(
        MMA_Atom_QK{},
        AtomLayoutMNK{},
        Tile<PermTileM, PermTileN, PermTileK>{}
    ));

    // For PV GEMM: A=P (row-major), B=V (col-major after transpose)
    // Note: V needs to be stored transposed for col-major access
    using MMA_Atom_PV = SM100_MMA_MXF4_SS<
        Element,                    // a_type (FP4)
        Element,                    // b_type (FP4)
        ElementAccum,               // c_type (FP32)
        ElementSF,                  // sf_type (E4M3)
        64,                         // M
        128,                        // N
        SFVectorSize,               // VS = 16
        UMMA::Major::K,             // a_major (row-major A = P)
        UMMA::Major::K,             // b_major (col-major B = Vt)
        UMMA::ScaleIn::One,
        UMMA::ScaleIn::One
    >;

    using TiledMmaPV = decltype(cute::make_tiled_mma(
        MMA_Atom_PV{},
        AtomLayoutMNK{},
        Tile<PermTileM, _32, PermTileK>{}
    ));

    //=========================================================================
    // Scale Factor Configuration
    //=========================================================================

    using BlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;
    using LayoutSF = decltype(BlkScaledConfig::deduce_layoutSFA());

    static constexpr int MMA_NSF = 64 / SFVectorSize;  // MMA K=64, SF covers 16 elements

    //=========================================================================
    // TMA Copy Atoms
    //=========================================================================

    using GmemTiledCopy = SM90_TMA_LOAD;
    using GmemTiledCopySF = SM90_TMA_LOAD;

    //=========================================================================
    // SMEM Layouts - following SM120 pattern
    //=========================================================================

    // Use CUTLASS SM100 SMEM selector for FP4 data
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        UMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        UMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        UMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        UMMA::Major::K, Element, Int<kHeadDim>, Int<kBlockN>>());

    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomVt{},
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    // Delta-S layout (for smooth attention correction)
    using SmemLayoutAtomDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<_0, _1>>;
    using SmemLayoutDS = decltype(tile_to_shape(SmemLayoutAtomDS{},
        make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    //=========================================================================
    // Scale Factor SMEM Layouts
    //=========================================================================

    // Scale factors are stored separately from data
    // Each SF covers 16 elements, so we have head_dim/16 SFs per row
    using SmemLayoutAtomSFQ = Layout<
        Shape<Int<kBlockM>, Int<NumSFPerHeadDim>>,
        Stride<Int<NumSFPerHeadDim>, _1>
    >;
    using SmemLayoutAtomSFK = Layout<
        Shape<Int<kBlockN>, Int<NumSFPerHeadDim>>,
        Stride<Int<NumSFPerHeadDim>, _1>
    >;
    using SmemLayoutAtomSFV = SmemLayoutAtomSFK;
    using SmemLayoutAtomSFVt = Layout<
        Shape<Int<NumSFPerHeadDim>, Int<kBlockN>>,
        Stride<_1, Int<NumSFPerHeadDim>>
    >;

    using SmemLayoutSFQ = decltype(make_layout(
        shape(SmemLayoutAtomSFQ{}),
        stride(SmemLayoutAtomSFQ{})
    ));
    using SmemLayoutSFK = decltype(make_layout(
        append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFK{}), Int<kBlockN * NumSFPerHeadDim>{})
    ));
    using SmemLayoutSFV = SmemLayoutSFK;
    using SmemLayoutSFVt = decltype(make_layout(
        append(shape(SmemLayoutAtomSFVt{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFVt{}), Int<NumSFPerHeadDim * kBlockN>{})
    ));

    //=========================================================================
    // Output SMEM Layout
    //=========================================================================

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        UMMA::Major::K, ElementOut, Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{}), Step<_1, _2>{}));

    //=========================================================================
    // Copy Atoms
    //=========================================================================

    using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
    using SmemCopyAtomDS = Copy_Atom<UniversalCopy<float>, float>;

    //=========================================================================
    // Shared Storage
    //=========================================================================

    struct SharedStorage : cute::aligned_struct<128, _0> {
        // Q data and scale factors
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_sfq;

        // K data and scale factors (pipelined)
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_sfk;

        // V data (transposed) and scale factors (pipelined)
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>> smem_v;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFVt>> smem_sfvt;

        // Delta-S for smooth attention (optional)
        cute::array_aligned<float, cute::cosize_v<SmemLayoutDS>> smem_ds;

        // Output buffer
        cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutO>> smem_o;

        // Pipeline storage
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_kv;
        alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
    };

    //=========================================================================
    // Pipeline Types
    //=========================================================================

    using MainloopPipeline = cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename cutlass::PipelineState<1>;

    //=========================================================================
    // TMA Transaction Sizes
    //=========================================================================

    static constexpr uint32_t TmaTransactionBytesQ =
        cutlass::bits_to_bytes(cosize(SmemLayoutQ{}) * cute::sizeof_bits_v<Element>) +
        cutlass::bits_to_bytes(cosize(SmemLayoutSFQ{}) * cute::sizeof_bits_v<ElementSF>);

    static constexpr uint32_t TmaTransactionBytesK =
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>) +
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFK{})) * cute::sizeof_bits_v<ElementSF>);

    static constexpr uint32_t TmaTransactionBytesVt =
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutVt{})) * cute::sizeof_bits_v<Element>) +
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFVt{})) * cute::sizeof_bits_v<ElementSF>);

    //=========================================================================
    // Strides
    //=========================================================================

    using StrideQ = Stride<int64_t, _1, int64_t>;
    using StrideK = Stride<int64_t, _1, int64_t>;
    using StrideV = Stride<_1, int64_t, int64_t>;  // V is transposed (dim, seq, batch*head)
    using StrideO = Stride<int64_t, _1, int64_t>;
};

} // namespace flash
