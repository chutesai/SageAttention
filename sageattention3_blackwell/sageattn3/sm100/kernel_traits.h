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
 * Key differences from SM120:
 *   - Uses TMEM for accumulator storage instead of registers
 *   - Uses tcgen05.mma instructions instead of mma.sync.aligned
 *   - Supports flexible cluster shapes (not fixed to 1x1x1)
 */

#pragma once

#include "cute/tensor.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "cute_extension.h"
#include "blockscaled_layout.h"
#include "../blackwell/named_barrier.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 Flash Forward Kernel Traits
// Template parameters:
//   HeadDim: Attention head dimension (64 or 128)
//   kBlockM: Tile size for M dimension (query sequence)
//   kBlockN: Tile size for N dimension (key/value sequence)
//   kStages: Pipeline stages for K/V loading
//   ClusterM: Cluster dimension M (can be >1 on SM100)
//   per_block_mean: Whether to apply per-block mean subtraction
//   Element_: Packed FP4 element type (uint8_t)
//   ElementOut_: Output element type (bfloat16 or float16)
///////////////////////////////////////////////////////////////////////////////

template <int HeadDim,
          int kBlockM_,
          int kBlockN_,
          int kStages_,
          int ClusterM_ = 1,
          bool per_block_mean_ = true,
          typename Element_ = uint8_t,
          typename ElementOut_ = cutlass::bfloat16_t>
struct Flash_fwd_kernel_traits_sm100 {

    // Basic configuration
    static constexpr int kHeadDim = HeadDim;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kStages = kStages_;
    static constexpr int ClusterM = ClusterM_;
    static constexpr bool per_block_mean = per_block_mean_;

    // Element types
    using Element = Element_;  // Packed FP4 (uint8_t holds 2x e2m1)
    using ElementOut = ElementOut_;
    using ElementAccum = float;
    using ElementSF = cutlass::float_ue4m3_t;  // Scale factor type (FP8 E4M3)

    // Architecture tag for SM100
    using ArchTag = cutlass::arch::Sm100;

    // Tile shapes for attention computation
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    // Cluster shape - SM100 supports multi-SM clusters unlike SM120
    using ClusterShape_MNK = Shape<Int<ClusterM>, _1, _1>;

    // Number of warps depends on tile size
    // SM100 uses warp groups (128 threads = 4 warps) for tcgen05 instructions
    static constexpr int kNWarps = kBlockM == 64 ? 8 : 12;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

    // Scale factor configuration
    static constexpr int kSFVecSize = 16;  // 16 elements per scale factor for .scale_vec::4X
    static constexpr int kNumSFPerRowQ = kHeadDim / kSFVecSize;
    static constexpr int kNumSFPerRowK = kHeadDim / kSFVecSize;
    static constexpr int kNumSFPerRowV = kHeadDim / kSFVecSize;

    // Instruction descriptor for blockscaled FP4 MMA
    static constexpr uint32_t kInstrDescQK = InstrDescriptorSm100::make_mxf4nvf4(kBlockM, kBlockN).get();
    static constexpr uint32_t kInstrDescPV = InstrDescriptorSm100::make_mxf4nvf4(kBlockM, kHeadDim).get();

    ///////////////////////////////////////////////////////////////////////////
    // SMEM Layouts - using 128B swizzle for optimal bandwidth
    ///////////////////////////////////////////////////////////////////////////

    // Q layout: (kBlockM, kHeadDim/2) - FP4 packed, so half the dimension
    using SmemLayoutAtomQ = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}
    ));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim / 2>>{}
    ));

    // K layout: (kBlockN, kHeadDim/2) with stages
    using SmemLayoutAtomK = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}
    ));
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim / 2>, Int<kStages>>{}
    ));

    // V layout: (kHeadDim/2, kBlockN) transposed with stages
    using SmemLayoutAtomV = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_64, _8>, Stride<_1, _64>>{}
    ));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kHeadDim / 2>, Int<kBlockN>, Int<kStages>>{}
    ));

    // Output layout (BF16/FP16)
    using SmemLayoutAtomO = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}
    ));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}
    ));

    // P matrix (quantized attention weights) layout
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kBlockN / 2>>{}
    ));

    ///////////////////////////////////////////////////////////////////////////
    // Scale factor SMEM layouts
    ///////////////////////////////////////////////////////////////////////////

    using SmemLayoutSFQ = Layout<
        Shape<Int<kBlockM>, Int<kNumSFPerRowQ>>,
        Stride<Int<kNumSFPerRowQ>, _1>
    >;

    using SmemLayoutSFK = Layout<
        Shape<Int<kBlockN>, Int<kNumSFPerRowK>, Int<kStages>>,
        Stride<Int<kNumSFPerRowK>, _1, Int<kBlockN * kNumSFPerRowK>>
    >;

    using SmemLayoutSFV = Layout<
        Shape<Int<kBlockN>, Int<kNumSFPerRowV>, Int<kStages>>,
        Stride<Int<kNumSFPerRowV>, _1, Int<kBlockN * kNumSFPerRowV>>
    >;

    // Scale factors for P matrix (attention probabilities)
    using SmemLayoutSFP = Layout<
        Shape<Int<kBlockM>, Int<kBlockN / kSFVecSize>>,
        Stride<Int<kBlockN / kSFVecSize>, _1>
    >;

    ///////////////////////////////////////////////////////////////////////////
    // Delta_S layout (per-block mean for numerical stability)
    ///////////////////////////////////////////////////////////////////////////

    using SmemLayoutDS = Layout<
        Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>,
        Stride<Int<kBlockN>, _1, Int<kBlockM * kBlockN>>
    >;

    ///////////////////////////////////////////////////////////////////////////
    // TMEM Configuration (SM100 specific)
    // TMEM: 128 rows × 512 columns × 4 bytes = 256KB per SM
    ///////////////////////////////////////////////////////////////////////////

    static constexpr uint32_t kTmemRows = 128;
    static constexpr uint32_t kTmemCols = 512;

    // TMEM columns needed for various tensors
    // QK accumulator: kBlockM × kBlockN floats
    static constexpr uint32_t kTmemColsAccQK =
        (kBlockM * kBlockN * sizeof(float) + kTmemRows * 4 - 1) / (kTmemRows * 4);
    // PV accumulator: kBlockM × kHeadDim floats
    static constexpr uint32_t kTmemColsAccPV =
        (kBlockM * kHeadDim * sizeof(float) + kTmemRows * 4 - 1) / (kTmemRows * 4);
    // Scale factors (much smaller)
    static constexpr uint32_t kTmemColsSF = 8;  // Enough for all scale factors

    // Verify TMEM fits
    static_assert(kTmemColsAccQK + kTmemColsAccPV + kTmemColsSF * 4 <= kTmemCols,
                  "TMEM allocation exceeds capacity");

    ///////////////////////////////////////////////////////////////////////////
    // Shared Memory Storage
    ///////////////////////////////////////////////////////////////////////////

    struct SharedStorage {
        // Main operand storage
        union {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 128> smem_q;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>, 128> smem_p;
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 128> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, 128> smem_v;
        cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutO>, 128> smem_o;

        // Scale factors
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFQ>, 64> smem_sfq;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFK>, 64> smem_sfk;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFV>, 64> smem_sfv;
        cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFP>, 64> smem_sfp;

        // Delta_S for per-block mean
        cute::array_aligned<float, cute::cosize_v<SmemLayoutDS>, 128> smem_ds;

        // Pipeline state storage
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;

        // Epilogue barrier
        cutlass::arch::ClusterBarrier barrier_o;

        // mbarriers for tcgen05.mma synchronization (double-buffered)
        alignas(8) uint64_t mbar_qk[2];
        alignas(8) uint64_t mbar_pv[2];

        // TMEM addresses (populated by thread 0 after allocation)
        uint32_t tmem_acc_qk;      // QK accumulator
        uint32_t tmem_acc_pv;      // PV accumulator
        uint32_t tmem_sf_q;        // Q scale factors
        uint32_t tmem_sf_k;        // K scale factors
        uint32_t tmem_sf_v;        // V scale factors
        uint32_t tmem_sf_p;        // P scale factors (computed)

        // Phase tracking for mbarrier ping-pong
        uint32_t mma_phase_qk;
        uint32_t mma_phase_pv;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Pipeline types
    ///////////////////////////////////////////////////////////////////////////

    using MainloopPipeline = cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using PipelineStateQ = typename MainloopPipelineQ::PipelineState;
    using PipelineParamsQ = typename MainloopPipelineQ::Params;

    // Epilogue barrier
    using EpilogueBarrier = flash::OrderedSequenceBarrierVarGroupSize<2, 2>;

    ///////////////////////////////////////////////////////////////////////////
    // TMA copy operations
    ///////////////////////////////////////////////////////////////////////////

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopySF = cute::SM90_TMA_LOAD;

    ///////////////////////////////////////////////////////////////////////////
    // Dummy TiledMma types for interface compatibility
    // Actual MMA is done via tcgen05 inline PTX
    ///////////////////////////////////////////////////////////////////////////

    struct TiledMmaQK {
        CUTE_HOST_DEVICE static constexpr int size() { return kNThreads - 128; }
    };

    struct TiledMmaPV {
        CUTE_HOST_DEVICE static constexpr int size() { return kNThreads - 128; }
    };
};

///////////////////////////////////////////////////////////////////////////////
// Type aliases for common configurations
///////////////////////////////////////////////////////////////////////////////

template <bool per_block_mean = true, typename ElementOut = cutlass::bfloat16_t>
using Flash_fwd_kernel_traits_sm100_d64 =
    Flash_fwd_kernel_traits_sm100<64, 128, 128, 3, 1, per_block_mean, uint8_t, ElementOut>;

template <bool per_block_mean = true, typename ElementOut = cutlass::bfloat16_t>
using Flash_fwd_kernel_traits_sm100_d128 =
    Flash_fwd_kernel_traits_sm100<128, 128, 128, 3, 1, per_block_mean, uint8_t, ElementOut>;

} // namespace flash
