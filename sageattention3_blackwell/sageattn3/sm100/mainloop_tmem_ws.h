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
 * SM100 (B200/B300) Mainloop for FlashAttention with TMEM accumulators.
 *
 * This implementation is adapted from CUTLASS example 77 (Blackwell FMHA)
 * with modifications for SageAttention's FP4 blockscaled format.
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

#include "kernel_traits.h"
#include "../blackwell/params.h"

// Include CUTLASS FMHA common helpers for SM100
#include "../../../csrc/cutlass/examples/77_blackwell_fmha/collective/fmha_common.hpp"

namespace flash {

using namespace cute;
using namespace cutlass::fmha::collective;

///////////////////////////////////////////////////////////////////////////////
// Causal mask implementation
///////////////////////////////////////////////////////////////////////////////

struct CausalMask {
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
        // Fully unmasked tiles: k_idx * block_n + block_n <= m_idx * block_m
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
        // Apply causal mask: set S[i,j] = -inf where j > i
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

struct NoMask {
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
// SM100 Collective Mainloop for Flash Attention Forward
// Adapted from CUTLASS example 77
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100 {

    using Element = typename Ktraits::Element;
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

    // SMEM layouts from CollectiveBuilder
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;

    // Pipelines
    using PipelineQ = typename Ktraits::PipelineQ;
    using PipelineKV = typename Ktraits::PipelineKV;
    using PipelineS = typename Ktraits::PipelineS;
    using PipelineC = typename Ktraits::PipelineC;
    using PipelineO = typename Ktraits::PipelineO;
    using PipelineE = typename Ktraits::PipelineE;
    using OrderBarrierSoftmax = typename Ktraits::OrderBarrierSoftmax;

    // TMA descriptors from CollectiveBuilder
    using TMA_Q = typename Ktraits::TMA_Q;
    using TMA_K = typename Ktraits::TMA_K;
    using TMA_V = typename Ktraits::TMA_V;

    // Strides
    using StrideQ = typename Ktraits::StrideQ;
    using StrideK = typename Ktraits::StrideK;
    using StrideV = typename Ktraits::StrideV;

    // Mask type
    using Mask = std::conditional_t<Is_causal, CausalMask, NoMask>;

    // TMEM allocation
    using TmemAlloc = flash::Sm100TmemAlloc;

    // Transaction bytes
    static constexpr int TransactionBytesLoadQ =
        cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);
    static constexpr int TransactionBytesLoadKV =
        cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Arguments {
        Element const* ptr_Q;
        StrideQ dQ;
        Element const* ptr_K;
        StrideK dK;
        Element const* ptr_V;
        StrideV dV;
        float scale_softmax;
    };

    struct Params {
        TMA_Q tma_load_q;
        TMA_K tma_load_k;
        TMA_V tma_load_v;
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
        // Use CollectiveBuilder's to_underlying_arguments for QK
        auto params_qk = CollectiveMmaQK::to_underlying_arguments(
            problem_shape,
            typename CollectiveMmaQK::Arguments{
                args.ptr_Q, args.dQ,
                args.ptr_K, args.dK,
            },
            workspace);

        // Use CollectiveBuilder's to_underlying_arguments for PV
        auto problem_shape_pv = select<0,2,1,3>(problem_shape);
        auto params_pv = CollectiveMmaPV::to_underlying_arguments(
            problem_shape_pv,
            typename CollectiveMmaPV::Arguments{
                args.ptr_K, args.dK,  // dummy, not used
                args.ptr_V, select<1,0,2>(args.dV),
            },
            workspace);

        float log2_e = static_cast<float>(M_LOG2E);

        return Params{
            params_qk.tma_load_a,
            params_qk.tma_load_b,
            params_pv.tma_load_b,
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
    }

    ///////////////////////////////////////////////////////////////////////////
    // Load function (executed by Load warp)
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

        // Setup TMA tensors for Q
        Tensor mQ = params.tma_load_q.get_tma_tensor(select<0,2,3>(problem_shape));
        Tensor gQ = local_tile(mQ, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
        Tensor tSgQ = thr_mma_qk.partition_A(gQ);
        Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
        auto [tQgQ_qdl, tQsQ] = tma_partition(
            params.tma_load_q, _0{}, make_layout(_1{}),
            group_modes<0,3>(sQ), group_modes<0,3>(tSgQ));
        Tensor tQgQ = tQgQ_qdl(_, _, _0{}, get<2>(blk_coord));

        // Setup TMA tensors for K
        Tensor mK = params.tma_load_k.get_tma_tensor(select<1,2,3>(problem_shape));
        Tensor gK = local_tile(mK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
        Tensor tSgK = thr_mma_qk.partition_B(gK);
        Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
        auto [tKgK_kdl, tKsK] = tma_partition(
            params.tma_load_k, _0{}, make_layout(_1{}),
            group_modes<0,3>(sK), group_modes<0,3>(tSgK));
        Tensor tKgK = tKgK_kdl(_, _, _0{}, get<2>(blk_coord));

        // Setup TMA tensors for V
        Tensor mV = params.tma_load_v.get_tma_tensor(select<2,1,3>(problem_shape));
        Tensor gV = local_tile(mV, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});
        Tensor tOgV = thr_mma_pv.partition_B(gV);
        Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
        auto [tVgV_dkl, tVsV] = tma_partition(
            params.tma_load_v, _0{}, make_layout(_1{}),
            group_modes<0,3>(sV), group_modes<0,3>(tOgV));
        Tensor tVgV = tVgV_dkl(_, _0{}, _, get<2>(blk_coord));

        uint32_t lane_predicate = cute::elect_one_sync();

        // Two Q blocks per CTA tile (ThreadShape = (2,1,1))
        int q0_index = 2 * get<0>(blk_coord);
        int q1_index = 2 * get<0>(blk_coord) + 1;

        // Q0
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            copy(params.tma_load_q.with(*tma_barrier, 0),
                 tQgQ(_, q0_index),
                 tQsQ(_, pipeline_q_producer_state.index()));
        }
        ++pipeline_q_producer_state;

        // K0
        int k_index = 0;
        pipeline_kv.producer_acquire(pipeline_kv_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
            copy(params.tma_load_k.with(*tma_barrier, 0),
                 tKgK(_, k_index),
                 tKsK(_, pipeline_kv_producer_state.index()));
        }
        ++pipeline_kv_producer_state;

        // Q1
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            copy(params.tma_load_q.with(*tma_barrier, 0),
                 tQgQ(_, q1_index),
                 tQsQ(_, pipeline_q_producer_state.index()));
        }
        ++pipeline_q_producer_state;

        // V0
        pipeline_kv.producer_acquire(pipeline_kv_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
            copy(params.tma_load_v.with(*tma_barrier, 0),
                 tVgV(_, k_index),
                 tVsV(_, pipeline_kv_producer_state.index()));
        }
        ++pipeline_kv_producer_state;
        k_index += 1;

        // Main loop: K_i, V_i pairs
        mask_tile_count -= 1;
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
            // K_i
            pipeline_kv.producer_acquire(pipeline_kv_producer_state);
            if (lane_predicate) {
                auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
                copy(params.tma_load_k.with(*tma_barrier, 0),
                     tKgK(_, k_index),
                     tKsK(_, pipeline_kv_producer_state.index()));
            }
            ++pipeline_kv_producer_state;

            // V_i
            pipeline_kv.producer_acquire(pipeline_kv_producer_state);
            if (lane_predicate) {
                auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
                copy(params.tma_load_v.with(*tma_barrier, 0),
                     tVgV(_, k_index),
                     tVsV(_, pipeline_kv_producer_state.index()));
            }
            ++pipeline_kv_producer_state;
            k_index += 1;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // MMA function (executed by MMA warp)
    // Adapted from CUTLASS example 77
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

        typename CollectiveMmaQK::TiledMma mma_qk;
        ThrMMA thr_mma_qk = mma_qk.get_slice(0);

        typename CollectiveMmaPV::TiledMma mma_pv;
        TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
        ThrMMA thr_mma_pv = mma_pv_ts.get_slice(0);

        Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

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

        Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
        Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});

        Tensor tOrP0 = tOrP;
        tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAlloc::P0);
        Tensor tOrP1 = tOrP;
        tOrP1.data() = tOrP1.data().get() + uint32_t(TmemAlloc::P1);

        int k_index = 0;
        int v_index = 0;
        int q_index = 0;

        // wait for Q1
        q_index = pipeline_q_consumer_state.index();
        pipeline_q.consumer_wait(pipeline_q_consumer_state);
        ++pipeline_q_consumer_state;

        Tensor tSrQ0 = tSrQ(_,_,_,q_index);

        // wait for K1
        k_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;

        // gemm Q1 * K1 -> S1
        pipeline_s0.producer_acquire(pipeline_s0_producer_state);
        gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);
        pipeline_s0.producer_commit(pipeline_s0_producer_state);
        ++pipeline_s0_producer_state;

        // release K1 (if ThreadShape_N > 1)
        if constexpr (get<1>(ThreadShape{}) > 1) {
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;
        }

        // wait for Q2 (if ThreadShape_M > 1)
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
        // gemm Q2 * K1 -> S2
        gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);
        pipeline_s1.producer_commit(pipeline_s1_producer_state);
        ++pipeline_s1_producer_state;

        // release K1
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        // wait for V1
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;

        // acquire corr first to take it out of critical path
        pipeline_corr.producer_acquire(pipeline_corr_producer_state);
        pipeline_s0.producer_acquire(pipeline_s0_producer_state);

        // gemm P1 * V1 -> O1
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
            // wait for Ki
            k_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;

            // gemm Q1 * Ki -> S1
            gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);
            pipeline_s0.producer_commit(pipeline_s0_producer_state);
            ++pipeline_s0_producer_state;

            if constexpr (get<1>(ThreadShape{}) > 1) {
                pipeline_kv.consumer_release(pipeline_kv_release_state);
                ++pipeline_kv_release_state;
            }

            // gemm P2 * V(i-1) -> O2
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

            // release V(i-1)
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;

            if constexpr (get<1>(ThreadShape{}) > 1) {
                k_index = pipeline_kv_consumer_state.index();
                pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
                ++pipeline_kv_consumer_state;
            }

            // gemm Q2 * Ki -> S2
            gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);
            pipeline_s1.producer_commit(pipeline_s1_producer_state);
            ++pipeline_s1_producer_state;

            // release Ki
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;

            // wait for Vi
            v_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;

            // gemm P1 * Vi -> O1
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

        // release Q1
        pipeline_q.consumer_release(pipeline_q_release_state);
        ++pipeline_q_release_state;

        // release Q2
        if constexpr (get<0>(ThreadShape{}) > 1) {
            pipeline_q.consumer_release(pipeline_q_release_state);
            ++pipeline_q_release_state;
        }

        // wait for Vi (if ThreadShape_N > 1)
        if constexpr (get<1>(ThreadShape{}) > 1) {
            v_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;
        }

        // gemm P2 * Vi -> O2
        pipeline_corr.producer_acquire(pipeline_corr_producer_state);
        pipeline_s1.producer_acquire(pipeline_s1_producer_state);

        gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

        pipeline_corr.producer_commit(pipeline_corr_producer_state);
        ++pipeline_corr_producer_state;

        // release Vi
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        pipeline_s0.producer_commit(pipeline_s0_producer_state);
        ++pipeline_s0_producer_state;

        pipeline_s1.producer_commit(pipeline_s1_producer_state);
        ++pipeline_s1_producer_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Softmax function (executed by Softmax warps)
    // Adapted from CUTLASS example 77
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
        // empty step to sync against pipe s
        pipeline_s.consumer_release(pipeline_s_consumer_state);
        ++pipeline_s_consumer_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Softmax step helper
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

        auto tilePlikeFP32 = get<1>(TileShapeQK{}) / Int<sizeof(float)>{} * Int<sizeof(Element)>{};
        Tensor tStS_P = tStS.compose(make_layout(make_shape(_128{}, tilePlikeFP32)));
        tStS_P.data() = warp_uniform(uint32_t(stage == 0 ? TmemAlloc::P0 : TmemAlloc::P1));
        Tensor tScS_P = tScS.compose(make_layout(make_shape(_128{}, tilePlikeFP32)));

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
        tTMEM_STOREtS_x4.data() = warp_uniform(tTMEM_STOREtS_x4.data().get());
        Tensor tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P);

        // wait on tensor core pipe
        pipeline_s.consumer_wait(pipeline_s_consumer_state);

        // read all of S from tmem into reg mem
        Tensor tTMEM_LOADrS = make_tensor<ElementAccum>(shape(tTMEM_LOADcS));
        copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS);

        if constexpr (need_apply_mask) {
            Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_shape);
        }

        ElementAccum old_row_max = row_max;

        // compute rowmax
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
        tTMEM_STOREVrS(0) = old_row_max;  // kIdxOldRowMax
        tTMEM_STOREVrS(1) = row_max_safe;  // kIdxNewRowMax
        copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

        pipeline_c.producer_commit(pipeline_c_producer_state);
        ++pipeline_c_producer_state;

        ElementAccum scale = params.scale_softmax_log2;
        ElementAccum row_max_scale = row_max_safe * scale;

        float2 scale_fp32x2 = make_float2(scale, scale);
        float2 minus_row_max_scale_fp32x2 = make_float2(-row_max_scale, -row_max_scale);

        Tensor tTMEM_STORErS_x4 = make_tensor<uint32_t>(shape(tTMEM_STOREcS));

        constexpr int kConversionsPerStep = 2;
        Tensor tTMEM_STORErS_x4_e = recast<Array<Element, kConversionsPerStep>>(tTMEM_STORErS_x4);

        NumericArrayConverter<Element, ElementAccum, kConversionsPerStep> convert;

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

        // tmem_store(reg_S8) -> op_P
        CUTE_STATIC_ASSERT_V(size<2>(tTMEM_STORErS_x4) <= _2{});
        CUTE_STATIC_ASSERT_V(size<1>(tTMEM_STORErS_x4) == _1{});
        copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1), tTMEM_STOREtS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1));

        cutlass::arch::fence_view_async_tmem_store();

        // notify tensor core warp that P is ready
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
            tTMEM_STOREVrS(1) = row_max;  // kIdxFinalRowMax
            tTMEM_STOREVrS(0) = row_sum;  // kIdxFinalRowSum
            copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Correction function (executed by Correction warps)
    // Adapted from CUTLASS example 77
    ///////////////////////////////////////////////////////////////////////////

    template <typename BlkCoord, typename ProblemShape, typename SharedStorage>
    CUTLASS_DEVICE void correction(
        BlkCoord const& blk_coord,
        Params const& params,
        ProblemShape const& problem_shape,
        Flash_fwd_params const& flash_params,
        SharedStorage& storage,
        PipelineC& pipeline_c0,
        typename PipelineC::PipelineState& pipeline_c0_consumer_state,
        PipelineC& pipeline_c1,
        typename PipelineC::PipelineState& pipeline_c1_consumer_state,
        PipelineO& pipeline_o,
        typename PipelineO::PipelineState& pipeline_o_consumer_state,
        PipelineE& pipeline_epi,
        typename PipelineE::PipelineState& pipeline_epi_producer_state
    ) {
        Mask mask;
        int mask_tile_count = mask.get_trip_count(blk_coord, TileShape{}, problem_shape);

        int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

        Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));

        Tensor cS = make_identity_tensor(select<0,1>(TileShapeQK{}));
        Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

        Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
        Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

        using TMEM_LOAD_V = SM100_TMEM_LOAD_32dp32b2x;

        auto tiled_tmem_loadv = make_tmem_copy(TMEM_LOAD_V{}, tStS_v);
        auto thr_tmem_loadv = tiled_tmem_loadv.get_slice(thread_idx);

        Tensor tTMEM_LOADVtS = thr_tmem_loadv.partition_S(tStS_v);
        Tensor tTMEM_LOADVcS = thr_tmem_loadv.partition_D(tScS_v);

        Tensor tTMEM_LOADVtS0 = tTMEM_LOADVtS;
        tTMEM_LOADVtS0.data() = tTMEM_LOADVtS0.data().get() + uint32_t(TmemAlloc::V0);
        Tensor tTMEM_LOADVtS1 = tTMEM_LOADVtS;
        tTMEM_LOADVtS1.data() = tTMEM_LOADVtS1.data().get() + uint32_t(TmemAlloc::V1);

        // ignore first signal from softmax as no correction is required
        pipeline_c0.consumer_wait(pipeline_c0_consumer_state);
        pipeline_c0.consumer_release(pipeline_c0_consumer_state);
        ++pipeline_c0_consumer_state;

        pipeline_c1.consumer_wait(pipeline_c1_consumer_state);

        mask_tile_count -= 1;

        CUTLASS_PRAGMA_NO_UNROLL
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
            pipeline_c0.consumer_wait(pipeline_c0_consumer_state);

            Tensor tTMEM_LOADVrS = make_tensor<ElementAccum>(shape(tTMEM_LOADVcS));
            copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

            float scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(0) - tTMEM_LOADVrS(1)));

            pipeline_o.consumer_wait(pipeline_o_consumer_state);

            correction_rescale(scale, uint32_t(TmemAlloc::O0));

            pipeline_c1.consumer_release(pipeline_c1_consumer_state);
            ++pipeline_c1_consumer_state;

            cutlass::arch::fence_view_async_tmem_store();

            pipeline_o.consumer_release(pipeline_o_consumer_state);
            ++pipeline_o_consumer_state;

            pipeline_c1.consumer_wait(pipeline_c1_consumer_state);

            copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

            scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(0) - tTMEM_LOADVrS(1)));

            pipeline_o.consumer_wait(pipeline_o_consumer_state);

            correction_rescale(scale, uint32_t(TmemAlloc::O1));

            pipeline_c0.consumer_release(pipeline_c0_consumer_state);
            ++pipeline_c0_consumer_state;

            cutlass::arch::fence_view_async_tmem_store();

            pipeline_o.consumer_release(pipeline_o_consumer_state);
            ++pipeline_o_consumer_state;
        }

        pipeline_c1.consumer_release(pipeline_c1_consumer_state);
        ++pipeline_c1_consumer_state;

        // Final correction to O1
        pipeline_c0.consumer_wait(pipeline_c0_consumer_state);

        Tensor tTMEM_LOADVrS = make_tensor<ElementAccum>(shape(tTMEM_LOADVcS));
        copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

        pipeline_c0.consumer_release(pipeline_c0_consumer_state);
        ++pipeline_c0_consumer_state;

        pipeline_o.consumer_wait(pipeline_o_consumer_state);
        pipeline_epi.producer_acquire(pipeline_epi_producer_state);

        // Store to epilogue smem
        Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), typename Ktraits::SmemLayoutO{});
        correction_epilogue(params.scale_output / tTMEM_LOADVrS(0), _0{}, sO);

        cutlass::arch::fence_view_async_tmem_load();

        pipeline_o.consumer_release(pipeline_o_consumer_state);
        ++pipeline_o_consumer_state;

        pipeline_epi.producer_commit(pipeline_epi_producer_state);
        ++pipeline_epi_producer_state;

        pipeline_c1.consumer_wait(pipeline_c1_consumer_state);

        copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

        pipeline_c1.consumer_release(pipeline_c1_consumer_state);
        ++pipeline_c1_consumer_state;

        pipeline_o.consumer_wait(pipeline_o_consumer_state);
        pipeline_epi.producer_acquire(pipeline_epi_producer_state);

        correction_epilogue(params.scale_output / tTMEM_LOADVrS(0), _1{}, sO);

        cutlass::arch::fence_view_async_tmem_load();

        pipeline_o.consumer_release(pipeline_o_consumer_state);
        ++pipeline_o_consumer_state;

        pipeline_epi.producer_commit(pipeline_epi_producer_state);
        ++pipeline_epi_producer_state;
    }

private:
    ///////////////////////////////////////////////////////////////////////////
    // Correction helpers
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE void correction_rescale(float scale, uint32_t tmem_O) {
        int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

        const int kCorrectionTileSize = 16;

        using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b16x;
        using TMEM_STORE = SM100_TMEM_STORE_32dp32b16x;

        typename CollectiveMmaPV::TiledMma mma;
        Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
        Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
        Tensor tOcO = mma.get_slice(0).partition_C(cO);

        Tensor tOtO_i = tOtO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
        Tensor tOcO_i = tOcO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

        tOtO_i.data() = tOtO_i.data().get() + tmem_O;

        auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i);
        auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);
        auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tOtO_i);
        auto thr_tmem_store = tiled_tmem_store.get_slice(thread_idx);

        Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i);
        Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i);
        Tensor tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i);

        float2 scale_f32x2 = make_float2(scale, scale);

        Tensor tTMrO = make_tensor<ElementAccum>(make_shape(shape(tTMEM_LOADcO), Int<128 / kCorrectionTileSize>{}));

        auto copy_in = [&](int i) {
            Tensor tTMEM_LOADtO_i = tTMEM_LOADtO;
            tTMEM_LOADtO_i.data() = tTMEM_LOADtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
            Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
            copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i);
        };

        auto copy_out = [&](int i) {
            Tensor tTMEM_STOREtO_i = tTMEM_STOREtO;
            tTMEM_STOREtO_i.data() = tTMEM_STOREtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
            Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
            copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i);
        };

        copy_in(0);

        int count = get<2>(TileShape{}) / kCorrectionTileSize;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < count; i++) {
            if (i != count - 1) {
                copy_in(i+1);
            }

            Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size(tTMrO_i); j += 2) {
                float2 in = make_float2(tTMrO_i(j), tTMrO_i(j+1));
                float2 out;
                cute::mul(out, scale_f32x2, in);
                tTMrO_i(j) = out.x;
                tTMrO_i(j+1) = out.y;
            }

            copy_out(i);
        }
    }

    template <typename Stage, typename TensorO>
    CUTLASS_DEVICE void correction_epilogue(float scale, Stage stage, TensorO const& sO_01) {
        int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

        Tensor sO = sO_01(_,_,stage);

        const int kCorrectionTileSize = 32 / sizeof(ElementOut);

        using TMEM_LOAD = std::conditional_t<kCorrectionTileSize == 32,
            SM100_TMEM_LOAD_32dp32b32x, SM100_TMEM_LOAD_32dp32b16x>;

        typename CollectiveMmaPV::TiledMma mma;
        Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
        Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
        Tensor tOcO = mma.get_slice(0).partition_C(cO);
        Tensor tOsO = mma.get_slice(0).partition_C(sO);

        Tensor tOtO_i = logical_divide(tOtO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
        Tensor tOcO_i = logical_divide(tOcO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
        Tensor tOsO_i = logical_divide(tOsO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

        if constexpr (decltype(stage == _0{})::value) {
            tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAlloc::O0);
        } else {
            static_assert(decltype(stage == _1{})::value, "stage is either 0 or 1");
            tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAlloc::O1);
        }

        auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i(make_coord(_, _), _0{}));
        auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);

        Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i(make_coord(_, _), _));
        Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i(make_coord(_, _), _));
        Tensor tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i(make_coord(_, _), _));

        float2 scale_f32x2 = make_float2(scale, scale);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < get<2>(TileShape{}) / kCorrectionTileSize; i++) {
            Tensor tTMEM_LOADtO_i = tTMEM_LOADtO(_, _0{}, _0{}, i);
            Tensor tTMEM_LOADsO_i = tTMEM_LOADsO(_, _0{}, _0{}, i);

            Tensor tTMrO = make_tensor<ElementAccum>(shape(tTMEM_LOADcO(_, _0{}, _0{}, i)));

            copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size(tTMrO); j += 2) {
                float2 in = make_float2(tTMrO(j), tTMrO(j+1));
                float2 out;
                cute::mul(out, scale_f32x2, in);
                tTMrO(j) = out.x;
                tTMrO(j+1) = out.y;
            }

            constexpr int N = 4 / sizeof(ElementOut);
            NumericArrayConverter<ElementOut, ElementAccum, N> convert;

            Tensor tSMrO = make_tensor_like<ElementOut>(tTMrO);

            Tensor tCs = recast<typename decltype(convert)::source_type>(tTMrO);
            Tensor tCd = recast<typename decltype(convert)::result_type>(tSMrO);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size(tCs); j++) {
                tCd(j) = convert.convert(tCs(j));
            }

            Tensor tSMsO_i = recast<uint32_t>(tTMEM_LOADsO_i);
            Tensor tSMrO_i = recast<uint32_t>(tSMrO);

            copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSMrO_i, tSMsO_i);
        }

        cutlass::arch::fence_view_async_shared();
    }
};

} // namespace flash
