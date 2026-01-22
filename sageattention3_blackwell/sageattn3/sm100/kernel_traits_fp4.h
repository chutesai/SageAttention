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
 * SM100 (B200/B300) FP4 Block-Scaled kernel traits for SageAttention3.
 *
 * This uses CUTLASS's SM100 block-scaled UMMA support for maximum performance
 * with FP4 quantized attention.
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

#include "../blackwell/blockscaled_layout.h"
#include "../blackwell/named_barrier.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Forward Kernel Traits
//
// Uses CUTLASS's SM100 block-scaled UMMA CollectiveBuilder for FP4 attention
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

    static_assert(kHeadDim >= 128, "SM100 with FP4 requires HeadDim >= 128");
    static_assert(kHeadDim % 32 == 0);

    // Warp configuration for SM100
    static constexpr int kNWarps = 12;  // Match SM120 configuration
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;

    // Pipeline stages
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 1;

    // Element types for FP4 block-scaled attention
    // FP4 = 4-bit floating point (2-bit exponent, 1-bit mantissa, 1-bit sign)
    using Element = cutlass::float_e2m1_t;  // FP4 element type
    using ElementSF = cutlass::float_ue4m3_t;  // FP8 E4M3 for scale factors
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;

    // Scale factor configuration
    static constexpr int NumSFQK = kHeadDim / 16;  // One scale factor per 16 elements
    static constexpr int NumSFPV = kBlockN / 16;
    static constexpr auto SFVectorSize = 16;

    // Tile shapes
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;

    // Architecture tag
    using ArchTag = cutlass::arch::Sm100;

    // Strides for GMEM tensors - 4D: (seqlen, dim, head, batch)
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeSF = cute::Shape<int32_t, int32_t, int32_t, int32_t>;

    // Block-scaled configuration
    using BlkScaledConfig = cutlass::detail::Sm100BlockScaledConfig<
        Element, Element, ElementSF, ElementSF, ElementAccum>;

    // MMA element types (may differ from storage types)
    using ElementQMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using ElementKMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());

    // Use SM100 block-scaled kernel schedule
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledSm100;

    ///////////////////////////////////////////////////////////////////////////
    // Use CollectiveBuilder to get correct configurations for block-scaled GEMM
    ///////////////////////////////////////////////////////////////////////////

    // For QK matmul: Q (M x K) @ K^T (K x N) -> S (M x N)
    // Input: FP4 packed with FP8 scale factors
    using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
        Element, StrideQKV, 16,  // A = Q
        Element, StrideQKV, 16,  // B = K
        ElementAccum,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        KernelSchedule
    >::CollectiveOp;

    // For PV matmul: P (M x N) @ V (N x D) -> O (M x D)
    using TileShapePV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
    using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
        Element, StrideQKV, 16,  // A = P (softmax output)
        Element, StrideQKV, 16,  // B = V
        ElementAccum,
        TileShapePV, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        KernelSchedule
    >::CollectiveOp;

    // Extract types from CollectiveBuilder
    using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
    using TiledMmaPV = typename CollectiveMmaPV::TiledMma;

    using SmemLayoutQ = typename CollectiveMmaQK::SmemLayoutA;
    using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
    using SmemLayoutSFQ = typename CollectiveMmaQK::SmemLayoutSFA;
    using SmemLayoutSFK = typename CollectiveMmaQK::SmemLayoutSFB;

    using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;
    using SmemLayoutSFV = typename CollectiveMmaPV::SmemLayoutSFB;

    // Pipeline types from CollectiveBuilder
    using MainloopPipeline = cutlass::PipelineTmaAsync<kStages>;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename MainloopPipelineQ::PipelineState;

    // Epilogue barrier
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;

    // TMA transaction bytes
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(SmemLayoutSFQ{}) * cute::sizeof_bits_v<ElementSF>) +
        cutlass::bits_to_bytes(size(SmemLayoutQ{}) * cute::sizeof_bits_v<Element>));

    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(SmemLayoutSFK{}) * cute::sizeof_bits_v<ElementSF>) +
        cutlass::bits_to_bytes(size(SmemLayoutK{}) * cute::sizeof_bits_v<Element>));

    ///////////////////////////////////////////////////////////////////////////
    // Shared Storage
    ///////////////////////////////////////////////////////////////////////////

    struct SharedStorage : cute::aligned_struct<128, _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_sfq;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_sfk;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFV>> smem_sfv;
        cute::array_aligned<ElementOut, kBlockM * kHeadDim> smem_o;

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
