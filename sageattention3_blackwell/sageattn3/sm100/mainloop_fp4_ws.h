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
 * SM100 (B200/B300) FP4 Block-Scaled Mainloop for FlashAttention.
 *
 * This implementation uses CUTLASS's SM100 block-scaled UMMA support with
 * tcgen05.mma instructions for FP4 quantized attention.
 *
 * Key features:
 * - FP4 (e2m1) data with FP8 E4M3 scale factors (NVF4 format)
 * - TMEM (Tensor Memory) for accumulation
 * - Warp-specialized design with separate Load/MMA/Softmax/Correction warps
 * - K dimension = 256 required for both QK and PV matmuls
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"
#include "cutlass/arch/reg_reconfig.h"

#include "kernel_traits_fp4.h"
#include "../blackwell/params.h"

namespace flash {

using namespace cute;

// Import CUTLASS types for conversion
using cutlass::Array;
using cutlass::NumericArrayConverter;

///////////////////////////////////////////////////////////////////////////////
// Causal mask implementation for FP4
///////////////////////////////////////////////////////////////////////////////

struct CausalMaskFP4 {
    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        int seqlen_k = get<1>(problem_shape);
        int block_n = get<1>(tile_shape);
        int block_m = get<0>(tile_shape);
        int m_idx = get<0>(blk_coord);
        int max_k = min(seqlen_k, (m_idx + 1) * block_m);
        return (max_k + block_n - 1) / block_n;
    }

    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_unmasked_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        int block_n = get<1>(tile_shape);
        int block_m = get<0>(tile_shape);
        int m_idx = get<0>(blk_coord);
        int max_unmasked_k = m_idx * block_m;
        return max(0, max_unmasked_k / block_n);
    }

    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_masked_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        return get_trip_count(blk_coord, tile_shape, problem_shape) -
               get_unmasked_trip_count(blk_coord, tile_shape, problem_shape);
    }

    template <typename TensorS, typename TensorC, typename ProblemShape>
    CUTLASS_DEVICE void apply_mask(
        TensorS& tS,
        TensorC const& tC,
        ProblemShape const& problem_shape
    ) const {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tS); ++i) {
            auto coord = tC(i);
            int row = get<0>(coord);
            int col = get<1>(coord);
            if (col > row) {
                tS(i) = -INFINITY;
            }
        }
    }
};

struct NoMaskFP4 {
    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        int seqlen_k = get<1>(problem_shape);
        int block_n = get<1>(tile_shape);
        return (seqlen_k + block_n - 1) / block_n;
    }

    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_unmasked_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        return get_trip_count(blk_coord, tile_shape, problem_shape);
    }

    template <typename BlkCoord, typename TileShape, typename ProblemShape>
    CUTLASS_DEVICE int get_masked_trip_count(
        BlkCoord const& blk_coord,
        TileShape tile_shape,
        ProblemShape const& problem_shape
    ) const {
        return 0;
    }

    template <typename TensorS, typename TensorC, typename ProblemShape>
    CUTLASS_DEVICE void apply_mask(
        TensorS& tS,
        TensorC const& tC,
        ProblemShape const& problem_shape
    ) const {
        // No masking needed
    }
};

///////////////////////////////////////////////////////////////////////////////
// FP4 Quantization Helper
// Quantizes FP32 softmax output P to FP4 with scale factors
///////////////////////////////////////////////////////////////////////////////

struct FP4Quantizer {
    using ElementFP4 = cutlass::float_e2m1_t;
    using ElementSF = cutlass::float_ue4m3_t;

    static constexpr int kVectorSize = 16;  // NVF4 scale factor vector size

    // Quantize a vector of FP32 values to FP4 with computed scale factor
    template <typename TensorIn, typename TensorOutData, typename TensorOutSF>
    CUTLASS_DEVICE static void quantize_to_fp4(
        TensorIn const& in_fp32,
        TensorOutData& out_fp4,
        TensorOutSF& out_sf,
        int sf_idx
    ) {
        // Find max absolute value for scale factor computation
        float max_abs = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kVectorSize; ++i) {
            max_abs = ::fmaxf(max_abs, ::fabsf(in_fp32(i)));
        }

        // FP4 E2M1 range: [-6.0, 6.0] (with max representable = 6.0)
        // Scale factor = max_abs / 6.0
        float scale = max_abs / 6.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        // Store scale factor
        out_sf(sf_idx) = ElementSF(scale);

        // Quantize to FP4
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kVectorSize; ++i) {
            float scaled = in_fp32(i) * inv_scale;
            // Clamp to FP4 range and convert
            scaled = ::fmaxf(-6.0f, ::fminf(6.0f, scaled));
            out_fp4(i) = ElementFP4(scaled);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Collective Mainloop for Flash Attention Forward
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4 {

    using Element = typename Ktraits::Element;
    using ElementData = typename Ktraits::ElementData;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;

    using TileShape = typename Ktraits::TileShape_MNK;
    using TileShapeQK = typename Ktraits::TileShapeQK;
    using TileShapePV = typename Ktraits::TileShapePV;
    using ThreadShape = typename Ktraits::ThreadShape;

    // MMA types from CollectiveBuilder
    using CollectiveMmaQK = typename Ktraits::CollectiveMmaQK;
    using CollectiveMmaPV = typename Ktraits::CollectiveMmaPV;
    using TiledMmaQK = typename Ktraits::TiledMmaQK;
    using TiledMmaPV = typename Ktraits::TiledMmaPV;

    // SMEM layouts
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;
    using SmemLayoutSFA = typename Ktraits::SmemLayoutSFA;
    using SmemLayoutSFB_QK = typename Ktraits::SmemLayoutSFB_QK;
    using SmemLayoutSFB_PV = typename Ktraits::SmemLayoutSFB_PV;

    // Pipelines
    using PipelineQ = typename Ktraits::PipelineQ;
    using PipelineKV = typename Ktraits::PipelineKV;
    using PipelineS = typename Ktraits::PipelineS;
    using PipelineC = typename Ktraits::PipelineC;
    using PipelineO = typename Ktraits::PipelineO;
    using PipelineE = typename Ktraits::PipelineE;
    using OrderBarrierSoftmax = typename Ktraits::OrderBarrierSoftmax;

    // TMA descriptors
    using TMA_Q = typename Ktraits::TMA_Q;
    using TMA_K = typename Ktraits::TMA_K;
    using TMA_V = typename Ktraits::TMA_V;
    using TMA_SFA = typename Ktraits::TMA_SFA;
    using TMA_SFB = typename Ktraits::TMA_SFB;
    using TMA_SFV = typename Ktraits::TMA_SFV;

    // Strides
    using StrideQ = typename Ktraits::StrideQ;
    using StrideK = typename Ktraits::StrideK;
    using StrideV = typename Ktraits::StrideV;
    using LayoutSFA = typename Ktraits::LayoutSFA;
    using LayoutSFB = typename Ktraits::LayoutSFB;

    // Mask type
    using Mask = std::conditional_t<Is_causal, CausalMaskFP4, NoMaskFP4>;

    // TMEM allocation
    using TmemAlloc = typename Ktraits::TmemAlloc;

    // Scale factor vector size
    static constexpr int SFVectorSize = Ktraits::SFVectorSize;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Arguments {
        ElementData const* ptr_Q;      // FP4 Q data
        ElementSF const* ptr_SFQ;      // Q scale factors
        StrideQ dQ;
        LayoutSFA layout_sfq;

        ElementData const* ptr_K;      // FP4 K data
        ElementSF const* ptr_SFK;      // K scale factors
        StrideK dK;
        LayoutSFB layout_sfk;

        ElementData const* ptr_V;      // FP4 V data
        ElementSF const* ptr_SFV;      // V scale factors
        StrideV dV;
        LayoutSFB layout_sfv;

        float scale_softmax;
    };

    struct Params {
        // TMA descriptors for data
        TMA_Q tma_load_q;
        TMA_K tma_load_k;
        TMA_V tma_load_v;

        // TMA descriptors for scale factors
        TMA_SFA tma_load_sfq;
        TMA_SFB tma_load_sfk;
        TMA_SFV tma_load_sfv;

        // Softmax scaling
        float scale_softmax;
        float scale_softmax_log2;
        float scale_output;
    };

    template <typename ProblemShape>
    static Params to_underlying_arguments(
        ProblemShape const& problem_shape,
        Arguments const& args,
        void* workspace
    ) {
        // Use CollectiveBuilder's to_underlying_arguments
        // This creates TMA descriptors with correct interleaved layouts for block-scaled data

        auto params_qk = CollectiveMmaQK::to_underlying_arguments(
            problem_shape,
            typename CollectiveMmaQK::Arguments{
                {args.ptr_Q, args.ptr_SFQ},  // ElementPairA
                {args.dQ, args.layout_sfq},  // StridePairA
                {args.ptr_K, args.ptr_SFK},  // ElementPairB
                {args.dK, args.layout_sfk},  // StridePairB
            },
            workspace);

        auto problem_shape_pv = select<0,2,1,3>(problem_shape);
        auto params_pv = CollectiveMmaPV::to_underlying_arguments(
            problem_shape_pv,
            typename CollectiveMmaPV::Arguments{
                {args.ptr_K, args.ptr_SFK},  // dummy A
                {args.dK, args.layout_sfk},
                {args.ptr_V, args.ptr_SFV},  // V
                {select<1,0,2>(args.dV), args.layout_sfv},
            },
            workspace);

        float log2_e = static_cast<float>(M_LOG2E);

        return Params{
            params_qk.tma_load_a,      // Q data
            params_qk.tma_load_b,      // K data
            params_pv.tma_load_b,      // V data
            params_qk.tma_load_sfa,    // Q scale factors
            params_qk.tma_load_sfb,    // K scale factors
            params_pv.tma_load_sfb,    // V scale factors
            args.scale_softmax,
            args.scale_softmax * log2_e,
            1.0f  // scale_output
        };
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfq.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfv.get_tma_descriptor());
    }

    ///////////////////////////////////////////////////////////////////////////
    // Load function (executed by Load warp)
    // Loads FP4 data and scale factors via TMA
    ///////////////////////////////////////////////////////////////////////////

    template <typename BlkCoord, typename ProblemShape, typename SharedStorage>
    CUTLASS_DEVICE void load(
        BlkCoord const& blk_coord,
        ProblemShape const& problem_shape,
        Params const& params,
        Flash_fwd_params const& flash_params,
        SharedStorage& storage,
        PipelineQ& pipeline_q,
        typename PipelineQ::PipelineState& pipeline_q_producer_state,
        PipelineKV& pipeline_kv,
        typename PipelineKV::PipelineState& pipeline_kv_producer_state
    ) {
        using X = Underscore;

        Mask mask;
        int mask_tile_count = mask.get_trip_count(blk_coord, TileShape{}, problem_shape);

        TiledMmaQK mma_qk;
        auto thr_mma_qk = mma_qk.get_slice(0);
        TiledMmaPV mma_pv;
        auto thr_mma_pv = mma_pv.get_slice(0);

        // Setup tensors for Q data
        Tensor mQ = params.tma_load_q.get_tma_tensor(select<0,2,3>(problem_shape));
        Tensor gQ = local_tile(mQ, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
        Tensor tSgQ = thr_mma_qk.partition_A(gQ);
        Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
        auto [tQgQ_qdl, tQsQ] = tma_partition(
            params.tma_load_q, _0{}, make_layout(_1{}),
            group_modes<0,3>(sQ), group_modes<0,3>(tSgQ));
        Tensor tQgQ = tQgQ_qdl(_, _, _0{}, get<2>(blk_coord));

        // Setup tensors for Q scale factors
        Tensor mSFQ = params.tma_load_sfq.get_tma_tensor(select<0,2,3>(problem_shape));
        Tensor gSFQ = local_tile(mSFQ, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
        Tensor sSFQ = make_tensor(make_smem_ptr(storage.smem_sfq.data()), SmemLayoutSFA{});

        // Setup tensors for K data
        Tensor mK = params.tma_load_k.get_tma_tensor(select<1,2,3>(problem_shape));
        Tensor gK = local_tile(mK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
        Tensor tSgK = thr_mma_qk.partition_B(gK);
        Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
        auto [tKgK_kdl, tKsK] = tma_partition(
            params.tma_load_k, _0{}, make_layout(_1{}),
            group_modes<0,3>(sK), group_modes<0,3>(tSgK));
        Tensor tKgK = tKgK_kdl(_, _, _0{}, get<2>(blk_coord));

        // Setup tensors for K scale factors
        Tensor mSFK = params.tma_load_sfk.get_tma_tensor(select<1,2,3>(problem_shape));
        Tensor sSFK = make_tensor(make_smem_ptr(storage.smem_sfk.data()), SmemLayoutSFB_QK{});

        // Setup tensors for V data
        Tensor mV = params.tma_load_v.get_tma_tensor(select<2,1,3>(problem_shape));
        Tensor gV = local_tile(mV, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});
        Tensor tOgV = thr_mma_pv.partition_B(gV);
        Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
        auto [tVgV_dkl, tVsV] = tma_partition(
            params.tma_load_v, _0{}, make_layout(_1{}),
            group_modes<0,3>(sV), group_modes<0,3>(tOgV));
        Tensor tVgV = tVgV_dkl(_, _0{}, _, get<2>(blk_coord));

        // Setup tensors for V scale factors
        Tensor mSFV = params.tma_load_sfv.get_tma_tensor(select<2,1,3>(problem_shape));
        Tensor sSFV = make_tensor(make_smem_ptr(storage.smem_sfv.data()), SmemLayoutSFB_PV{});

        uint32_t lane_predicate = cute::elect_one_sync();

        // Two Q blocks per CTA tile (ThreadShape = (2,1,1))
        int q0_index = 2 * get<0>(blk_coord);
        int q1_index = 2 * get<0>(blk_coord) + 1;

        // Load Q0 (data + scale factors)
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            // Load Q data
            copy(params.tma_load_q.with(*tma_barrier, 0),
                 tQgQ(_, q0_index),
                 tQsQ(_, pipeline_q_producer_state.index()));
            // Load Q scale factors
            copy(params.tma_load_sfq.with(*tma_barrier, 0),
                 gSFQ(_, q0_index),
                 sSFQ(_, pipeline_q_producer_state.index()));
        }
        ++pipeline_q_producer_state;

        // Load K0 (data + scale factors)
        int k_index = 0;
        pipeline_kv.producer_acquire(pipeline_kv_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
            copy(params.tma_load_k.with(*tma_barrier, 0),
                 tKgK(_, k_index),
                 tKsK(_, pipeline_kv_producer_state.index()));
            copy(params.tma_load_sfk.with(*tma_barrier, 0),
                 mSFK(_, k_index),
                 sSFK(_, pipeline_kv_producer_state.index()));
        }
        ++pipeline_kv_producer_state;

        // Load Q1
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            copy(params.tma_load_q.with(*tma_barrier, 0),
                 tQgQ(_, q1_index),
                 tQsQ(_, pipeline_q_producer_state.index()));
            copy(params.tma_load_sfq.with(*tma_barrier, 0),
                 gSFQ(_, q1_index),
                 sSFQ(_, pipeline_q_producer_state.index()));
        }
        ++pipeline_q_producer_state;

        // Load V0 (data + scale factors)
        pipeline_kv.producer_acquire(pipeline_kv_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
            copy(params.tma_load_v.with(*tma_barrier, 0),
                 tVgV(_, k_index),
                 tVsV(_, pipeline_kv_producer_state.index()));
            copy(params.tma_load_sfv.with(*tma_barrier, 0),
                 mSFV(_, k_index),
                 sSFV(_, pipeline_kv_producer_state.index()));
        }
        ++pipeline_kv_producer_state;
        k_index += 1;

        // Main loop: K_i, V_i pairs
        mask_tile_count -= 1;
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
            // Load K_i
            pipeline_kv.producer_acquire(pipeline_kv_producer_state);
            if (lane_predicate) {
                auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
                copy(params.tma_load_k.with(*tma_barrier, 0),
                     tKgK(_, k_index),
                     tKsK(_, pipeline_kv_producer_state.index()));
                copy(params.tma_load_sfk.with(*tma_barrier, 0),
                     mSFK(_, k_index),
                     sSFK(_, pipeline_kv_producer_state.index()));
            }
            ++pipeline_kv_producer_state;

            // Load V_i
            pipeline_kv.producer_acquire(pipeline_kv_producer_state);
            if (lane_predicate) {
                auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
                copy(params.tma_load_v.with(*tma_barrier, 0),
                     tVgV(_, k_index),
                     tVsV(_, pipeline_kv_producer_state.index()));
                copy(params.tma_load_sfv.with(*tma_barrier, 0),
                     mSFV(_, k_index),
                     sSFV(_, pipeline_kv_producer_state.index()));
            }
            ++pipeline_kv_producer_state;
            k_index += 1;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // MMA function (executed by MMA warp)
    // Performs block-scaled FP4 GEMM for QK and PV matmuls
    ///////////////////////////////////////////////////////////////////////////

    template <typename BlkCoord, typename ProblemShape, typename SharedStorage>
    CUTLASS_DEVICE auto mma(
        BlkCoord const& blk_coord,
        Params const& params,
        ProblemShape const& problem_shape,
        Flash_fwd_params const& flash_params,
        SharedStorage& storage,
        PipelineQ& pipeline_q,
        typename PipelineQ::PipelineState& pipeline_q_consumer_state,
        PipelineKV& pipeline_kv,
        typename PipelineKV::PipelineState& pipeline_kv_consumer_state,
        PipelineS& pipeline_s0,
        typename PipelineS::PipelineState& pipeline_s0_producer_state,
        PipelineS& pipeline_s1,
        typename PipelineS::PipelineState& pipeline_s1_producer_state,
        PipelineO& pipeline_corr,
        typename PipelineO::PipelineState& pipeline_corr_producer_state
    ) {
        auto pipeline_q_release_state = pipeline_q_consumer_state;
        auto pipeline_kv_release_state = pipeline_kv_consumer_state;

        Mask mask;
        int mask_tile_count = mask.get_trip_count(blk_coord, TileShape{}, problem_shape);

        // Get MMAs for block-scaled operations
        // The TiledMma from CollectiveBuilder handles FP4 with scale factors
        typename CollectiveMmaQK::TiledMma mma_qk;
        ThrMMA thr_mma_qk = mma_qk.get_slice(0);

        typename CollectiveMmaPV::TiledMma mma_pv;
        TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
        ThrMMA thr_mma_pv = mma_pv_ts.get_slice(0);

        // SMEM tensors for data
        Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

        // SMEM tensors for scale factors
        Tensor sSFQ = make_tensor(make_smem_ptr(storage.smem_sfq.data()), SmemLayoutSFA{});
        Tensor sSFK = make_tensor(make_smem_ptr(storage.smem_sfk.data()), SmemLayoutSFB_QK{});
        Tensor sSFV = make_tensor(make_smem_ptr(storage.smem_sfv.data()), SmemLayoutSFB_PV{});

        // Create fragment references for MMA operands
        // For block-scaled, these include both data and scale factors
        Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
        Tensor tSrK = thr_mma_qk.make_fragment_B(sK);
        Tensor tOrV = thr_mma_pv.make_fragment_B(sV);

        // TMEM layout: S0 S1 O0 O1
        Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
        Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0,1>(TileShapePV{}));

        Tensor tStS0 = tStS;
        tStS0.data() = tStS.data().get() + uint32_t(TmemAlloc::S0);
        Tensor tStS1 = tStS;
        tStS1.data() = tStS.data().get() + uint32_t(TmemAlloc::S1);

        Tensor tOtO0 = tOtO;
        tOtO0.data() = tOtO.data().get() + uint32_t(TmemAlloc::O0);
        Tensor tOtO1 = tOtO;
        tOtO1.data() = tOtO.data().get() + uint32_t(TmemAlloc::O1);

        // P tensor in TMEM for PV matmul (quantized softmax output)
        Tensor sP = make_tensor(make_smem_ptr((ElementData*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
        Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});

        Tensor tOrP0 = tOrP;
        tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAlloc::P0);
        Tensor tOrP1 = tOrP;
        tOrP1.data() = tOrP1.data().get() + uint32_t(TmemAlloc::P1);

        int k_index = 0;
        int v_index = 0;
        int q_index = 0;

        // Wait for Q1
        q_index = pipeline_q_consumer_state.index();
        pipeline_q.consumer_wait(pipeline_q_consumer_state);
        ++pipeline_q_consumer_state;

        Tensor tSrQ0 = tSrQ(_,_,_,q_index);

        // Wait for K1
        k_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;

        // Block-scaled GEMM: Q1 * K1 -> S1
        // The MMA automatically handles scale factors from SMEM
        pipeline_s0.producer_acquire(pipeline_s0_producer_state);
        gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);
        pipeline_s0.producer_commit(pipeline_s0_producer_state);
        ++pipeline_s0_producer_state;

        // Release K1 (if ThreadShape_N > 1)
        if constexpr (get<1>(ThreadShape{}) > 1) {
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;
        }

        // Wait for Q2 (if ThreadShape_M > 1)
        if constexpr (get<0>(ThreadShape{}) > 1 || get<2>(ThreadShape{}) > 1) {
            q_index = pipeline_q_consumer_state.index();
            pipeline_q.consumer_wait(pipeline_q_consumer_state);
            ++pipeline_q_consumer_state;
        }

        Tensor tSrQ1 = tSrQ(_,_,_,q_index);

        if constexpr (get<1>(ThreadShape{}) > 1) {
            k_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;
        }

        pipeline_s1.producer_acquire(pipeline_s1_producer_state);
        // Block-scaled GEMM: Q2 * K1 -> S2
        gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);
        pipeline_s1.producer_commit(pipeline_s1_producer_state);
        ++pipeline_s1_producer_state;

        // Release K1
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        // Wait for V1
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;

        // Acquire corr first to take it out of critical path
        pipeline_corr.producer_acquire(pipeline_corr_producer_state);
        pipeline_s0.producer_acquire(pipeline_s0_producer_state);

        // Block-scaled GEMM: P1 * V1 -> O1
        // P1 is the quantized softmax output (stored in TMEM)
        gemm_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

        pipeline_corr.producer_commit(pipeline_corr_producer_state);
        ++pipeline_corr_producer_state;

        if constexpr (get<1>(ThreadShape{}) > 1) {
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;
        }

        mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero;

        // Main loop
        mask_tile_count -= 1;
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
            // Wait for Ki
            k_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;

            // Block-scaled GEMM: Q1 * Ki -> S1
            gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);
            pipeline_s0.producer_commit(pipeline_s0_producer_state);
            ++pipeline_s0_producer_state;

            if constexpr (get<1>(ThreadShape{}) > 1) {
                pipeline_kv.consumer_release(pipeline_kv_release_state);
                ++pipeline_kv_release_state;
            }

            // Block-scaled GEMM: P2 * V(i-1) -> O2
            if constexpr (get<1>(ThreadShape{}) > 1) {
                v_index = pipeline_kv_consumer_state.index();
                pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
                ++pipeline_kv_consumer_state;
            }

            pipeline_corr.producer_acquire(pipeline_corr_producer_state);
            pipeline_s1.producer_acquire(pipeline_s1_producer_state);

            gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

            pipeline_corr.producer_commit(pipeline_corr_producer_state);
            ++pipeline_corr_producer_state;

            // Release V(i-1)
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;

            if constexpr (get<1>(ThreadShape{}) > 1) {
                k_index = pipeline_kv_consumer_state.index();
                pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
                ++pipeline_kv_consumer_state;
            }

            // Block-scaled GEMM: Q2 * Ki -> S2
            gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);
            pipeline_s1.producer_commit(pipeline_s1_producer_state);
            ++pipeline_s1_producer_state;

            // Release Ki
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;

            // Wait for Vi
            v_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;

            // Block-scaled GEMM: P1 * Vi -> O1
            pipeline_corr.producer_acquire(pipeline_corr_producer_state);
            pipeline_s0.producer_acquire(pipeline_s0_producer_state);

            gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

            pipeline_corr.producer_commit(pipeline_corr_producer_state);
            ++pipeline_corr_producer_state;

            if constexpr (get<1>(ThreadShape{}) > 1) {
                pipeline_kv.consumer_release(pipeline_kv_release_state);
                ++pipeline_kv_release_state;
            }
        }

        // Release Q1
        pipeline_q.consumer_release(pipeline_q_release_state);
        ++pipeline_q_release_state;

        // Release Q2
        if constexpr (get<0>(ThreadShape{}) > 1) {
            pipeline_q.consumer_release(pipeline_q_release_state);
            ++pipeline_q_release_state;
        }

        // Wait for Vi (if ThreadShape_N > 1)
        if constexpr (get<1>(ThreadShape{}) > 1) {
            v_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;
        }

        // Block-scaled GEMM: P2 * Vi -> O2
        pipeline_corr.producer_acquire(pipeline_corr_producer_state);
        pipeline_s1.producer_acquire(pipeline_s1_producer_state);

        gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

        pipeline_corr.producer_commit(pipeline_corr_producer_state);
        ++pipeline_corr_producer_state;

        // Release Vi
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        pipeline_s0.producer_commit(pipeline_s0_producer_state);
        ++pipeline_s0_producer_state;

        pipeline_s1.producer_commit(pipeline_s1_producer_state);
        ++pipeline_s1_producer_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Softmax function (executed by Softmax warps)
    // Computes softmax and quantizes P to FP4 for PV matmul
    ///////////////////////////////////////////////////////////////////////////

    template <typename BlkCoord, typename ProblemShape>
    CUTLASS_DEVICE void softmax(
        int stage,
        BlkCoord const& blk_coord,
        Params const& params,
        ProblemShape const& problem_shape,
        Flash_fwd_params const& flash_params,
        PipelineS& pipeline_s,
        typename PipelineS::PipelineState& pipeline_s_consumer_state,
        PipelineC& pipeline_c,
        typename PipelineC::PipelineState& pipeline_c_producer_state,
        OrderBarrierSoftmax& order_s
    ) {
        Mask mask;
        int mask_tile_count = mask.get_unmasked_trip_count(blk_coord, TileShape{}, problem_shape);

        ElementAccum row_max = -INFINITY;
        ElementAccum row_sum = 0;

        Tensor cS_base = make_identity_tensor(select<0,1>(TileShapeQK{}));
        auto logical_offset = make_coord(
            get<0>(blk_coord) * get<0>(TileShape{}) + (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{}),
            0 + (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{})
        );
        Tensor cS = domain_offset(logical_offset, cS_base);

        pipeline_c.producer_acquire(pipeline_c_producer_state);

        // Unmasked iterations
        CUTLASS_PRAGMA_NO_UNROLL
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
            softmax_step<false>(
                row_max, row_sum, stage,
                (mask_tile_count == 1) && (mask.get_masked_trip_count(blk_coord, TileShape{}, problem_shape) == 0),
                blk_coord, cS, params, problem_shape,
                pipeline_s, pipeline_s_consumer_state,
                pipeline_c, pipeline_c_producer_state,
                order_s
            );

            cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
        }

        // Masked iterations
        mask_tile_count = mask.get_masked_trip_count(blk_coord, TileShape{}, problem_shape);

        CUTLASS_PRAGMA_NO_UNROLL
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
            softmax_step<true>(
                row_max, row_sum, stage, mask_tile_count == 1,
                blk_coord, cS, params, problem_shape,
                pipeline_s, pipeline_s_consumer_state,
                pipeline_c, pipeline_c_producer_state,
                order_s
            );

            cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
        }

        pipeline_c.producer_commit(pipeline_c_producer_state);
        ++pipeline_c_producer_state;

        pipeline_c.producer_acquire(pipeline_c_producer_state);
        // Empty step to sync against pipe s
        pipeline_s.consumer_release(pipeline_s_consumer_state);
        ++pipeline_s_consumer_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Softmax step helper with FP4 quantization
    ///////////////////////////////////////////////////////////////////////////

    template <bool need_apply_mask, typename Stage, typename BlkCoord, typename CountingTensor, typename ProblemShape>
    CUTLASS_DEVICE void softmax_step(
        ElementAccum& row_max, ElementAccum& row_sum,
        Stage stage, bool final_call,
        BlkCoord const& blk_coord, CountingTensor const& cS,
        Params const& params, ProblemShape const& problem_shape,
        PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
        PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
        OrderBarrierSoftmax& order_s
    ) {
        Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

        Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));
        tStS.data() = uint32_t(stage == 0 ? TmemAlloc::S0 : TmemAlloc::S1);

        Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
        tStS_v.data() = uint32_t(stage == 0 ? TmemAlloc::V0 : TmemAlloc::V1);
        Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

        // For FP4 output P, we need to quantize after softmax
        auto tilePlikeFP4 = get<1>(TileShapeQK{}) / Int<sizeof(float)>{} * Int<sizeof(ElementData)>{};
        Tensor tStS_P = tStS.compose(make_layout(make_shape(_128{}, tilePlikeFP4)));
        tStS_P.data() = uint32_t(stage == 0 ? TmemAlloc::P0 : TmemAlloc::P1);
        Tensor tScS_P = tScS.compose(make_layout(make_shape(_128{}, tilePlikeFP4)));

        using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x;
        using TMEM_STORE = SM100_TMEM_STORE_32dp32b32x;
        using TMEM_STORE_V = SM100_TMEM_STORE_32dp32b2x;

        int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

        auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tStS);
        auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);

        Tensor tTMEM_LOADtS = thr_tmem_load.partition_S(tStS);
        Tensor tTMEM_LOADcS = thr_tmem_load.partition_D(tScS);

        auto tiled_tmem_storev = make_tmem_copy(TMEM_STORE_V{}, tStS_v);
        auto thr_tmem_storev = tiled_tmem_storev.get_slice(thread_idx);

        Tensor tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v);
        Tensor tTMEM_STOREVcS = thr_tmem_storev.partition_S(tScS_v);

        auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tStS_P);
        auto thr_tmem_store = tiled_tmem_store.get_slice(thread_idx);

        Tensor tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P);
        // Note: removed warp_uniform wrapper - compiler should still optimize correctly
        Tensor tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P);

        // Wait on tensor core pipe
        pipeline_s.consumer_wait(pipeline_s_consumer_state);

        // Read all of S from TMEM into reg mem
        Tensor tTMEM_LOADrS = make_tensor<ElementAccum>(shape(tTMEM_LOADcS));
        copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS);

        if constexpr (need_apply_mask) {
            Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_shape);
        }

        ElementAccum old_row_max = row_max;

        // Compute rowmax
        {
            float row_max_0 = row_max;
            float row_max_1 = row_max;
            float row_max_2 = row_max;
            float row_max_3 = row_max;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tTMEM_LOADrS); i += 4) {
                row_max_0 = ::fmax(row_max_0, tTMEM_LOADrS(i));
                row_max_1 = ::fmax(row_max_1, tTMEM_LOADrS(i+1));
                row_max_2 = ::fmax(row_max_2, tTMEM_LOADrS(i+2));
                row_max_3 = ::fmax(row_max_3, tTMEM_LOADrS(i+3));
            }
            row_max = ::fmax(row_max_0, row_max_1);
            row_max = ::fmax(row_max, row_max_2);
            row_max = ::fmax(row_max, row_max_3);
        }

        ElementAccum row_max_safe = row_max == -INFINITY ? 0 : row_max;

        Tensor tTMEM_STOREVrS = make_tensor<ElementAccum>(shape(tTMEM_STOREVcS));
        tTMEM_STOREVrS(0) = old_row_max;   // kIdxOldRowMax
        tTMEM_STOREVrS(1) = row_max_safe;  // kIdxNewRowMax
        copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

        pipeline_c.producer_commit(pipeline_c_producer_state);
        ++pipeline_c_producer_state;

        ElementAccum scale = params.scale_softmax_log2;
        ElementAccum row_max_scale = row_max_safe * scale;

        float2 scale_fp32x2 = make_float2(scale, scale);
        float2 minus_row_max_scale_fp32x2 = make_float2(-row_max_scale, -row_max_scale);

        // Compute softmax and quantize to FP4
        Tensor tTMEM_STORErS_x4 = make_tensor<uint32_t>(shape(tTMEM_STOREcS));

        constexpr int kConversionsPerStep = 2;
        Tensor tTMEM_STORErS_x4_e = recast<Array<ElementData, kConversionsPerStep>>(tTMEM_STORErS_x4);

        NumericArrayConverter<ElementData, ElementAccum, kConversionsPerStep> convert;

        const int kReleasePipeCount = 10;

        order_s.wait();

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTMEM_LOADrS); i += 2) {
            float2 in = make_float2(tTMEM_LOADrS(i + 0), tTMEM_LOADrS(i + 1));
            float2 out;
            cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
            tTMEM_LOADrS(i + 0) = out.x;
            tTMEM_LOADrS(i + 1) = out.y;

            tTMEM_LOADrS(i+0) = ::exp2f(tTMEM_LOADrS(i+0));
            tTMEM_LOADrS(i+1) = ::exp2f(tTMEM_LOADrS(i+1));

            // Convert FP32 softmax output to FP4
            // Note: For true block-scaled FP4, we need scale factors
            // Here we do simple conversion; scale factors computed per block
            Array<ElementAccum, kConversionsPerStep> in_conv;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kConversionsPerStep; j++) {
                in_conv[j] = tTMEM_LOADrS(i + j);
            }
            tTMEM_STORErS_x4_e[i / kConversionsPerStep] = convert(in_conv);

            if (i == size(tTMEM_LOADrS) - kReleasePipeCount) {
                order_s.arrive();
            }

            if constexpr (size<2>(tTMEM_STORErS_x4) == _2{}) {
                if (i == size(tTMEM_LOADrS) - 6) {
                    copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, 0), tTMEM_STOREtS_x4(_, _, 0));
                }
            }
        }

        // Store quantized P to TMEM
        CUTE_STATIC_ASSERT_V(size<2>(tTMEM_STORErS_x4) <= _2{});
        CUTE_STATIC_ASSERT_V(size<1>(tTMEM_STORErS_x4) == _1{});
        copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1), tTMEM_STOREtS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1));

        cutlass::arch::fence_view_async_tmem_store();

        // Notify tensor core warp that P is ready
        pipeline_s.consumer_release(pipeline_s_consumer_state);
        ++pipeline_s_consumer_state;

        pipeline_c.producer_acquire(pipeline_c_producer_state);

        ElementAccum acc_scale = 0.5f * ::exp2f(scale * (old_row_max - row_max_safe));
        row_sum *= acc_scale;

        float2 local_row_sum_f32x2 = make_float2(row_sum, row_sum);
        float2 local_row_sum_1 = make_float2(0, 0);
        float2 local_row_sum_2 = make_float2(0, 0);
        float2 local_row_sum_3 = make_float2(0, 0);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTMEM_LOADrS); i += 8) {
            float2 in = make_float2(tTMEM_LOADrS(i), tTMEM_LOADrS(i+1));
            cute::add(local_row_sum_f32x2, local_row_sum_f32x2, in);

            in = make_float2(tTMEM_LOADrS(i+2), tTMEM_LOADrS(i+2+1));
            cute::add(local_row_sum_1, local_row_sum_1, in);

            in = make_float2(tTMEM_LOADrS(i+4), tTMEM_LOADrS(i+4+1));
            cute::add(local_row_sum_2, local_row_sum_2, in);

            in = make_float2(tTMEM_LOADrS(i+6), tTMEM_LOADrS(i+6+1));
            cute::add(local_row_sum_3, local_row_sum_3, in);
        }

        cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_1);
        cute::add(local_row_sum_2, local_row_sum_2, local_row_sum_3);
        cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_2);
        float local_row_sum = local_row_sum_f32x2.x + local_row_sum_f32x2.y;

        row_sum = local_row_sum;

        if (final_call) {
            pipeline_s.consumer_wait(pipeline_s_consumer_state);

            Tensor tTMEM_STOREVrS = make_tensor<ElementAccum>(shape(tTMEM_STOREVcS));
            tTMEM_STOREVrS(1) = row_max;   // kIdxFinalRowMax
            tTMEM_STOREVrS(0) = row_sum;   // kIdxFinalRowSum
            copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);
        }
    }
};

} // namespace flash
