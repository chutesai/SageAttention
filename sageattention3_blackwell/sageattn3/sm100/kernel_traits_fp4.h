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

#include "../blackwell/blockscaled_layout.h"
#include "../blackwell/named_barrier.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Forward Kernel Traits
//
// Uses CUTLASS's SM100 block-scaled UMMA for FP4 attention with tcgen05.mma
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
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

    // SM100 FP4 requirements
    static_assert(kHeadDim >= 128, "SM100 with FP4 requires HeadDim >= 128");
    static_assert(kHeadDim % 32 == 0, "HeadDim must be multiple of 32");
    static_assert(kBlockM == 128 || kBlockM == 256, "SM100 FP4 supports BlockM=128 or 256");

    // Warp configuration for SM100
    // SM100 uses warp-specialized design with UMMA
    static constexpr int kNWarps = kBlockM == 256 ? 16 : 12;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;

    // Pipeline stages
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 1;

    // Element types for FP4 block-scaled attention
    // FP4 = 4-bit floating point (e2m1: 2-bit exponent, 1-bit mantissa, 1-bit sign)
    // NVF4 format: FP4 data with FP8 E4M3 scale factors, vector size 16
    using ElementData = cutlass::float_e2m1_t;           // FP4 data type
    using ElementSF = cutlass::float_ue4m3_t;            // FP8 E4M3 unsigned scale factors
    using Element = cutlass::nv_float4_t<ElementData>;   // NVF4 packed type with scales
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Scale factor configuration (NVF4 uses vector size 16)
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFQK = kHeadDim / SFVectorSize;  // One scale factor per 16 elements
    static constexpr int NumSFPV = kBlockN / SFVectorSize;

    // Tile shapes for SM100 block-scaled GEMM
    // MMA tile shape: (M, N, K) where K is the reduction dimension
    // For FP4, K must be 256 (CUTLASS constraint)
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    // Architecture tags
    using ArchTag = cutlass::arch::Sm100;
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // Strides for GMEM tensors
    // For FP4 packed data with interleaved scales
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;

    // Block-scaled configuration
    using BlkScaledConfig = cutlass::detail::Sm100BlockScaledConfig<
        ElementData, ElementData, ElementSF, ElementSF, ElementAccum>;

    // Layout tags for CUTLASS
    using LayoutATag = cutlass::layout::RowMajor;     // Q: row-major (seqlen x headdim)
    using LayoutBTag = cutlass::layout::ColumnMajor; // K^T: column-major for TN layout

    // Alignment for FP4 (32 elements = 16 bytes)
    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;

    ///////////////////////////////////////////////////////////////////////////
    // Use CollectiveBuilder to get correct SM100 block-scaled configuration
    ///////////////////////////////////////////////////////////////////////////

    // For QK matmul: Q (M x K) @ K^T (K x N) -> S (M x N)
    // Using block-scaled FP4 inputs with FP32 accumulation
    using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        Element, LayoutATag, AlignmentA,  // A = Q
        Element, LayoutBTag, AlignmentB,  // B = K
        ElementAccum,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::KernelScheduleAuto  // Auto-select best SM100 blockscaled schedule
    >::CollectiveOp;

    // Extract types from CollectiveBuilder
    using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
    using SmemLayoutQ = typename CollectiveMmaQK::SmemLayoutA;
    using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
    using SmemLayoutSFA = typename CollectiveMmaQK::SmemLayoutSFA;
    using SmemLayoutSFB = typename CollectiveMmaQK::SmemLayoutSFB;

    // For PV matmul: P (M x N) @ V (N x D) -> O (M x D)
    // P is softmax output (FP32), V is FP4
    // This requires converting P to FP4 for the MMA
    using TileShapePV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;

    using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        Element, LayoutATag, AlignmentA,  // A = P (quantized)
        Element, LayoutBTag, AlignmentB,  // B = V
        ElementAccum,
        TileShapePV, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::KernelScheduleAuto
    >::CollectiveOp;

    using TiledMmaPV = typename CollectiveMmaPV::TiledMma;
    using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;
    using SmemLayoutSFV = typename CollectiveMmaPV::SmemLayoutSFB;

    // Pipeline types for SM100 with UMMA
    // SM100 uses PipelineTmaUmmaAsync instead of PipelineTmaAsync
    using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
        kStages,
        typename CollectiveMmaQK::AtomThrShapeMNK
    >;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    using MainloopPipelineQ = cutlass::PipelineTmaUmmaAsync<
        1,
        typename CollectiveMmaQK::AtomThrShapeMNK
    >;
    using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename MainloopPipelineQ::PipelineState;

    // Epilogue barrier
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;

    // TMA transaction bytes for FP4 with scale factors
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(SmemLayoutQ{}) * cute::sizeof_bits_v<ElementData>) +
        cutlass::bits_to_bytes(cosize(SmemLayoutSFA{}) * cute::sizeof_bits_v<ElementSF>));

    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(SmemLayoutK{}) * cute::sizeof_bits_v<ElementData>) +
        cutlass::bits_to_bytes(cosize(SmemLayoutSFB{}) * cute::sizeof_bits_v<ElementSF>));

    ///////////////////////////////////////////////////////////////////////////
    // Shared Storage
    ///////////////////////////////////////////////////////////////////////////

    struct SharedStorage : cute::aligned_struct<128, _0> {
        // FP4 data tensors
        cute::array_aligned<ElementData, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<ElementData, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<ElementData, cute::cosize_v<SmemLayoutV>> smem_v;

        // Scale factor tensors
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_sfq;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_sfk;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFV>> smem_sfv;

        // Output tensor
        cute::array_aligned<ElementOut, kBlockM * kHeadDim> smem_o;

        // Pipeline storage
        struct {
            alignas(16) typename MainloopPipelineQ::SharedStorage pipeline_q;
            alignas(16) typename MainloopPipeline::SharedStorage pipeline_k;
            alignas(16) typename MainloopPipeline::SharedStorage pipeline_v;
            alignas(16) typename EpilogueBarrier::SharedStorage barrier_o;
            int tile_count_semaphore;
        };
    };
};

} // namespace flash
