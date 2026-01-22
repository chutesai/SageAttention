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
 * This implements the warp-specialized mainloop for SM100:
 *   - MMA warp: Q*K^T -> S (TMEM), P*V -> O (TMEM)
 *   - Softmax warps: Read S from TMEM, compute softmax, write P back
 *   - Correction warps: Rescale O in TMEM based on running max
 *   - Load warp: TMA loads for Q, K, V
 */

#pragma once

#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/arch/reg_reconfig.h"

#include "kernel_traits.h"
#include "../blackwell/params.h"

// Include CUTLASS FMHA common helpers for SM100
#include "../../../csrc/cutlass/examples/77_blackwell_fmha/collective/fmha_common.hpp"

namespace flash {

using namespace cute;
using namespace cutlass::fmha::collective;

///////////////////////////////////////////////////////////////////////////////
// SM100 Collective Mainloop for Flash Attention Forward
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

    using CollectiveMmaQK = typename Ktraits::CollectiveMmaQK;
    using CollectiveMmaPV = typename Ktraits::CollectiveMmaPV;
    using TiledMmaQK = typename Ktraits::TiledMmaQK;
    using TiledMmaPV = typename Ktraits::TiledMmaPV;

    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;

    using PipelineQ = typename Ktraits::PipelineQ;
    using PipelineKV = typename Ktraits::PipelineKV;
    using PipelineS = typename Ktraits::PipelineS;
    using PipelineC = typename Ktraits::PipelineC;
    using PipelineO = typename Ktraits::PipelineO;
    using OrderBarrierSoftmax = typename Ktraits::OrderBarrierSoftmax;

    using TMA_Q = typename Ktraits::TMA_Q;
    using TMA_K = typename Ktraits::TMA_K;
    using TMA_V = typename Ktraits::TMA_V;

    using TmemAlloc = Sm100TmemAlloc;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Arguments {
        Element const* ptr_Q;
        Element const* ptr_K;
        Element const* ptr_V;
        float scale_softmax;
    };

    struct Params {
        TMA_Q tma_load_q;
        TMA_K tma_load_k;
        TMA_V tma_load_v;
        float scale_softmax;
        float scale_softmax_log2;
    };

    template <typename ProblemShape>
    static Params to_underlying_arguments(
        ProblemShape const& problem_shape,
        Arguments const& args,
        void* workspace
    ) {
        auto problem_shape_qk = problem_shape;

        auto params_qk = CollectiveMmaQK::to_underlying_arguments(
            problem_shape_qk,
            typename CollectiveMmaQK::Arguments{
                args.ptr_Q, typename Ktraits::StrideQ{},
                args.ptr_K, typename Ktraits::StrideK{}
            }, nullptr);

        auto problem_shape_pv = select<0,2,1,3>(problem_shape_qk);
        auto params_pv = CollectiveMmaPV::to_underlying_arguments(
            problem_shape_pv,
            typename CollectiveMmaPV::Arguments{
                args.ptr_K, typename Ktraits::StrideK{},  // dummy
                args.ptr_V, select<1,0,2>(typename Ktraits::StrideV{})
            }, nullptr);

        float log2_e = static_cast<float>(std::log2(std::exp(1.0)));

        return Params{
            params_qk.tma_load_a,
            params_qk.tma_load_b,
            params_pv.tma_load_b,
            args.scale_softmax,
            args.scale_softmax * log2_e
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

        int mask_tile_count = get_tile_count<Is_causal>(blk_coord, problem_shape);

        auto mma_qk = TiledMmaQK{}.get_slice(0);
        auto mma_pv = TiledMmaPV{}.get_slice(0);

        // Setup TMA tensors for Q
        Tensor mQ = params.tma_load_q.get_tma_tensor(select<0,2,3>(problem_shape));
        Tensor gQ = local_tile(mQ, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
        Tensor tSgQ = mma_qk.partition_A(gQ);
        Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
        auto [tQgQ, tQsQ] = tma_partition(
            params.tma_load_q, _0{}, make_layout(_1{}),
            group_modes<0,3>(sQ), group_modes<0,3>(tSgQ));

        // Setup TMA tensors for K
        Tensor mK = params.tma_load_k.get_tma_tensor(select<1,2,3>(problem_shape));
        Tensor gK = local_tile(mK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
        Tensor tSgK = mma_qk.partition_B(gK);
        Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
        auto [tKgK, tKsK] = tma_partition(
            params.tma_load_k, _0{}, make_layout(_1{}),
            group_modes<0,3>(sK), group_modes<0,3>(tSgK));

        // Setup TMA tensors for V
        Tensor mV = params.tma_load_v.get_tma_tensor(select<2,1,3>(problem_shape));
        Tensor gV = local_tile(mV, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});
        Tensor tOgV = mma_pv.partition_B(gV);
        Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
        auto [tVgV, tVsV] = tma_partition(
            params.tma_load_v, _0{}, make_layout(_1{}),
            group_modes<0,3>(sV), group_modes<0,3>(tOgV));

        uint32_t lane_predicate = cute::elect_one_sync();

        // Load Q0 and Q1 (two Q tiles for alternating softmax warps)
        int q0_index = 2 * get<0>(blk_coord);
        int q1_index = 2 * get<0>(blk_coord) + 1;

        // Q0
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            copy(params.tma_load_q.with(*tma_barrier, 0),
                 tQgQ(_, q0_index, get<2>(blk_coord)),
                 tQsQ(_, pipeline_q_producer_state.index()));
        }
        ++pipeline_q_producer_state;

        // K0
        int k_index = 0;
        pipeline_kv.producer_acquire(pipeline_kv_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
            copy(params.tma_load_k.with(*tma_barrier, 0),
                 tKgK(_, k_index, get<2>(blk_coord)),
                 tKsK(_, pipeline_kv_producer_state.index()));
        }
        ++pipeline_kv_producer_state;

        // Q1
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            copy(params.tma_load_q.with(*tma_barrier, 0),
                 tQgQ(_, q1_index, get<2>(blk_coord)),
                 tQsQ(_, pipeline_q_producer_state.index()));
        }
        ++pipeline_q_producer_state;

        // V0
        pipeline_kv.producer_acquire(pipeline_kv_producer_state);
        if (lane_predicate) {
            auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
            copy(params.tma_load_v.with(*tma_barrier, 0),
                 tVgV(_, k_index, get<2>(blk_coord)),
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
                     tKgK(_, k_index, get<2>(blk_coord)),
                     tKsK(_, pipeline_kv_producer_state.index()));
            }
            ++pipeline_kv_producer_state;

            // V_i
            pipeline_kv.producer_acquire(pipeline_kv_producer_state);
            if (lane_predicate) {
                auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
                copy(params.tma_load_v.with(*tma_barrier, 0),
                     tVgV(_, k_index, get<2>(blk_coord)),
                     tVsV(_, pipeline_kv_producer_state.index()));
            }
            ++pipeline_kv_producer_state;
            k_index += 1;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // MMA function (executed by MMA warp)
    // Computes Q*K^T -> S (TMEM), then P*V -> O (TMEM)
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
        PipelineO& pipeline_o,
        typename PipelineO::PipelineState& pipeline_o_producer_state
    ) {
        auto pipeline_q_release_state = pipeline_q_consumer_state;
        auto pipeline_kv_release_state = pipeline_kv_consumer_state;

        int mask_tile_count = get_tile_count<Is_causal>(blk_coord, problem_shape);

        TiledMmaQK mma_qk;
        auto thr_mma_qk = mma_qk.get_slice(0);

        TiledMmaPV mma_pv;
        auto mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
        auto thr_mma_pv = mma_pv_ts.get_slice(0);

        // Setup SMEM tensors
        Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

        Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
        Tensor tSrK = thr_mma_qk.make_fragment_B(sK);
        Tensor tOrV = thr_mma_pv.make_fragment_B(sV);

        // Setup TMEM tensors for S (scores) and O (output)
        Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
        Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0,1>(TileShapePV{}));

        // S0 and S1 in TMEM (alternating for two Q blocks)
        Tensor tStS0 = tStS;
        tStS0.data() = tStS.data().get() + uint32_t(TmemAlloc::S0);
        Tensor tStS1 = tStS;
        tStS1.data() = tStS.data().get() + uint32_t(TmemAlloc::S1);

        // O0 and O1 in TMEM
        Tensor tOtO0 = tOtO;
        tOtO0.data() = tOtO.data().get() + uint32_t(TmemAlloc::O0);
        Tensor tOtO1 = tOtO;
        tOtO1.data() = tOtO.data().get() + uint32_t(TmemAlloc::O1);

        // P is stored in TMEM overlapping with S (after softmax)
        Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
        Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});

        Tensor tOrP0 = tOrP;
        tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAlloc::P0);
        Tensor tOrP1 = tOrP;
        tOrP1.data() = tOrP1.data().get() + uint32_t(TmemAlloc::P1);

        int k_index = 0;
        int v_index = 0;
        int q_index = 0;

        // Wait for Q0
        q_index = pipeline_q_consumer_state.index();
        pipeline_q.consumer_wait(pipeline_q_consumer_state);
        ++pipeline_q_consumer_state;

        Tensor tSrQ0 = tSrQ(_, _, _, q_index);

        // Wait for K0
        k_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;

        // Compute Q0 * K0 -> S0
        pipeline_s0.producer_acquire(pipeline_s0_producer_state);
        gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tStS0);
        pipeline_s0.producer_commit(pipeline_s0_producer_state);
        ++pipeline_s0_producer_state;

        // Release K0 (if thread shape allows)
        if constexpr (get<1>(ThreadShape{}) > 1) {
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;
        }

        // Wait for Q1 (if needed)
        if constexpr (get<0>(ThreadShape{}) > 1 || get<2>(ThreadShape{}) > 1) {
            q_index = pipeline_q_consumer_state.index();
            pipeline_q.consumer_wait(pipeline_q_consumer_state);
            ++pipeline_q_consumer_state;
        }

        Tensor tSrQ1 = tSrQ(_, _, _, q_index);

        if constexpr (get<1>(ThreadShape{}) > 1) {
            k_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;
        }

        // Compute Q1 * K0 -> S1
        pipeline_s1.producer_acquire(pipeline_s1_producer_state);
        gemm_zero_acc(mma_qk, tSrQ1, tSrK(_, _, _, k_index), tStS1);
        pipeline_s1.producer_commit(pipeline_s1_producer_state);
        ++pipeline_s1_producer_state;

        // Release K0
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        // Wait for V0
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;

        // Acquire for correction and softmax (P0 ready after softmax)
        pipeline_o.producer_acquire(pipeline_o_producer_state);
        pipeline_s0.producer_acquire(pipeline_s0_producer_state);

        // Compute P0 * V0 -> O0
        gemm_zero_acc(mma_pv_ts, tOrP0, tOrV(_, _, _, v_index), tOtO0);

        pipeline_o.producer_commit(pipeline_o_producer_state);
        ++pipeline_o_producer_state;

        if constexpr (get<1>(ThreadShape{}) > 1) {
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;
        }

        mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero;

        // Main loop
        mask_tile_count -= 1;
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
            // Wait for K_i
            k_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;

            // Q0 * K_i -> S0
            gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tStS0);

            pipeline_s0.producer_commit(pipeline_s0_producer_state);
            ++pipeline_s0_producer_state;

            if constexpr (get<1>(ThreadShape{}) > 1) {
                pipeline_kv.consumer_release(pipeline_kv_release_state);
                ++pipeline_kv_release_state;
            }

            // P1 * V_(i-1) -> O1
            if constexpr (get<1>(ThreadShape{}) > 1) {
                v_index = pipeline_kv_consumer_state.index();
                pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
                ++pipeline_kv_consumer_state;
            }

            pipeline_o.producer_acquire(pipeline_o_producer_state);
            pipeline_s1.producer_acquire(pipeline_s1_producer_state);

            gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_, _, _, v_index), tOtO1);

            pipeline_o.producer_commit(pipeline_o_producer_state);
            ++pipeline_o_producer_state;

            // Release V_(i-1)
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;

            if constexpr (get<1>(ThreadShape{}) > 1) {
                k_index = pipeline_kv_consumer_state.index();
                pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
                ++pipeline_kv_consumer_state;
            }

            // Q1 * K_i -> S1
            gemm_zero_acc(mma_qk, tSrQ1, tSrK(_, _, _, k_index), tStS1);

            pipeline_s1.producer_commit(pipeline_s1_producer_state);
            ++pipeline_s1_producer_state;

            // Release K_i
            pipeline_kv.consumer_release(pipeline_kv_release_state);
            ++pipeline_kv_release_state;

            // Wait for V_i
            v_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;

            // P0 * V_i -> O0
            pipeline_o.producer_acquire(pipeline_o_producer_state);
            pipeline_s0.producer_acquire(pipeline_s0_producer_state);

            gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_, _, _, v_index), tOtO0);

            pipeline_o.producer_commit(pipeline_o_producer_state);
            ++pipeline_o_producer_state;

            if constexpr (get<1>(ThreadShape{}) > 1) {
                pipeline_kv.consumer_release(pipeline_kv_release_state);
                ++pipeline_kv_release_state;
            }
        }

        // Release Q0 and Q1
        pipeline_q.consumer_release(pipeline_q_release_state);
        ++pipeline_q_release_state;

        if constexpr (get<0>(ThreadShape{}) > 1) {
            pipeline_q.consumer_release(pipeline_q_release_state);
            ++pipeline_q_release_state;
        }

        // Final P1 * V_last -> O1
        if constexpr (get<1>(ThreadShape{}) > 1) {
            v_index = pipeline_kv_consumer_state.index();
            pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
            ++pipeline_kv_consumer_state;
        }

        pipeline_o.producer_acquire(pipeline_o_producer_state);
        pipeline_s1.producer_acquire(pipeline_s1_producer_state);

        gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_, _, _, v_index), tOtO1);

        pipeline_o.producer_commit(pipeline_o_producer_state);
        ++pipeline_o_producer_state;

        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        pipeline_s0.producer_commit(pipeline_s0_producer_state);
        ++pipeline_s0_producer_state;

        pipeline_s1.producer_commit(pipeline_s1_producer_state);
        ++pipeline_s1_producer_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Softmax function (executed by Softmax0 and Softmax1 warps)
    // Reads S from TMEM, computes online softmax, writes P back to TMEM
    ///////////////////////////////////////////////////////////////////////////

    template <typename BlkCoord, typename ProblemShape>
    CUTLASS_DEVICE void softmax(
        int stage,  // 0 or 1
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
        int mask_tile_count = get_tile_count<Is_causal>(blk_coord, problem_shape);

        ElementAccum row_max = -INFINITY;
        ElementAccum row_sum = 0;

        Tensor cS_base = make_identity_tensor(select<0,1>(TileShapeQK{}));
        auto logical_offset = make_coord(
            get<0>(blk_coord) * get<0>(TileShape{}) + (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{}),
            0 + (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{}));
        Tensor cS = domain_offset(logical_offset, cS_base);

        pipeline_c.producer_acquire(pipeline_c_producer_state);

        // Unmasked iterations
        int unmasked_count = mask_tile_count;  // Simplified: all unmasked for now
        CUTLASS_PRAGMA_NO_UNROLL
        for (; unmasked_count > 0; unmasked_count -= 1) {
            softmax_step<false>(
                row_max, row_sum, stage,
                (unmasked_count == 1),
                blk_coord, cS, params, problem_shape, flash_params,
                pipeline_s, pipeline_s_consumer_state,
                pipeline_c, pipeline_c_producer_state,
                order_s);

            cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
        }

        pipeline_c.producer_commit(pipeline_c_producer_state);
        ++pipeline_c_producer_state;

        pipeline_c.producer_acquire(pipeline_c_producer_state);
        pipeline_s.consumer_release(pipeline_s_consumer_state);
        ++pipeline_s_consumer_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Correction function (executed by Correction warps)
    // Rescales O in TMEM based on running max from softmax warps
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
        typename Ktraits::PipelineE& pipeline_epi,
        typename Ktraits::PipelineE::PipelineState& pipeline_epi_producer_state
    ) {
        int mask_tile_count = get_tile_count<Is_causal>(blk_coord, problem_shape);

        int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

        TiledMmaQK mma_qk;
        Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
        Tensor cS = make_identity_tensor(select<0,1>(TileShapeQK{}));
        Tensor tScS = mma_qk.get_slice(0).partition_C(cS);

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

        // Skip first signal (no correction needed for first iteration)
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

            float scale = ::exp2f(params.scale_softmax_log2 *
                (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

            pipeline_o.consumer_wait(pipeline_o_consumer_state);

            correction_rescale(scale, uint32_t(TmemAlloc::O0));

            pipeline_c1.consumer_release(pipeline_c1_consumer_state);
            ++pipeline_c1_consumer_state;

            cutlass::arch::fence_view_async_tmem_store();

            pipeline_o.consumer_release(pipeline_o_consumer_state);
            ++pipeline_o_consumer_state;

            pipeline_c1.consumer_wait(pipeline_c1_consumer_state);

            copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

            scale = ::exp2f(params.scale_softmax_log2 *
                (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

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

        // Final correction and epilogue handoff
        pipeline_c0.consumer_wait(pipeline_c0_consumer_state);

        Tensor tTMEM_LOADVrS = make_tensor<ElementAccum>(shape(tTMEM_LOADVcS));
        copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

        pipeline_c0.consumer_release(pipeline_c0_consumer_state);
        ++pipeline_c0_consumer_state;

        pipeline_o.consumer_wait(pipeline_o_consumer_state);
        pipeline_epi.producer_acquire(pipeline_epi_producer_state);

        // Copy O from TMEM to SMEM for epilogue
        Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), typename Ktraits::SmemLayoutO{});
        correction_epilogue(1.0f / tTMEM_LOADVrS(kIdxFinalRowSum), _0{}, sO);

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

        correction_epilogue(1.0f / tTMEM_LOADVrS(kIdxFinalRowSum), _1{}, sO);

        cutlass::arch::fence_view_async_tmem_load();

        pipeline_o.consumer_release(pipeline_o_consumer_state);
        ++pipeline_o_consumer_state;

        pipeline_epi.producer_commit(pipeline_epi_producer_state);
        ++pipeline_epi_producer_state;
    }

private:
    ///////////////////////////////////////////////////////////////////////////
    // Helper: Get tile count for KV iteration
    ///////////////////////////////////////////////////////////////////////////

    template <bool Causal, typename BlkCoord, typename ProblemShape>
    CUTLASS_DEVICE static int get_tile_count(
        BlkCoord const& blk_coord,
        ProblemShape const& problem_shape
    ) {
        int seqlen_k = get<1>(problem_shape);
        int block_n = get<1>(typename Ktraits::TileShape_MNK{});

        if constexpr (Causal) {
            int block_m = get<0>(typename Ktraits::TileShape_MNK{});
            int m_idx = get<0>(blk_coord);
            int max_k = min(seqlen_k, (m_idx + 1) * block_m);
            return (max_k + block_n - 1) / block_n;
        } else {
            return (seqlen_k + block_n - 1) / block_n;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Helper: Softmax step (one K block)
    ///////////////////////////////////////////////////////////////////////////

    template <bool NeedApplyMask, typename BlkCoord, typename CountingTensor,
              typename ProblemShape>
    CUTLASS_DEVICE void softmax_step(
        ElementAccum& row_max,
        ElementAccum& row_sum,
        int stage,
        bool final_call,
        BlkCoord const& blk_coord,
        CountingTensor const& cS,
        Params const& params,
        ProblemShape const& problem_shape,
        Flash_fwd_params const& flash_params,
        PipelineS& pipeline_s,
        typename PipelineS::PipelineState& pipeline_s_consumer_state,
        PipelineC& pipeline_c,
        typename PipelineC::PipelineState& pipeline_c_producer_state,
        OrderBarrierSoftmax& order_s
    ) {
        TiledMmaQK mma_qk;
        Tensor tScS = mma_qk.get_slice(0).partition_C(cS);
        Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
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

        // Wait for S from MMA warp
        pipeline_s.consumer_wait(pipeline_s_consumer_state);

        // Load S from TMEM to registers
        Tensor tTMEM_LOADrS = make_tensor<ElementAccum>(shape(tTMEM_LOADcS));
        copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS);

        // Apply mask if needed
        if constexpr (NeedApplyMask) {
            // Causal masking would go here
        }

        // Compute row max
        ElementAccum old_row_max = row_max;
        {
            float row_max_0 = row_max, row_max_1 = row_max;
            float row_max_2 = row_max, row_max_3 = row_max;
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

        // Store old and new row max for correction
        Tensor tTMEM_STOREVrS = make_tensor<ElementAccum>(shape(tTMEM_STOREVcS));
        tTMEM_STOREVrS(kIdxOldRowMax) = old_row_max;
        tTMEM_STOREVrS(kIdxNewRowMax) = row_max_safe;
        copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

        pipeline_c.producer_commit(pipeline_c_producer_state);
        ++pipeline_c_producer_state;

        // Compute softmax: exp2(scale * (S - max)) and convert to FP4
        ElementAccum scale = params.scale_softmax_log2;
        ElementAccum row_max_scale = row_max_safe * scale;

        float2 scale_fp32x2 = make_float2(scale, scale);
        float2 minus_row_max_scale_fp32x2 = make_float2(-row_max_scale, -row_max_scale);

        Tensor tTMEM_STORErS_x4 = make_tensor<uint32_t>(shape(tTMEM_STOREcS));

        constexpr int kConversionsPerStep = 2;
        Tensor tTMEM_STORErS_x4_e = recast<Array<Element, kConversionsPerStep>>(tTMEM_STORErS_x4);

        NumericArrayConverter<Element, ElementAccum, kConversionsPerStep> convert;

        order_s.wait();

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTMEM_LOADrS); i += 2) {
            float2 in = make_float2(tTMEM_LOADrS(i), tTMEM_LOADrS(i+1));
            float2 out;
            cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
            tTMEM_LOADrS(i) = ::exp2f(out.x);
            tTMEM_LOADrS(i+1) = ::exp2f(out.y);

            Array<ElementAccum, kConversionsPerStep> in_conv;
            in_conv[0] = tTMEM_LOADrS(i);
            in_conv[1] = tTMEM_LOADrS(i+1);
            tTMEM_STORErS_x4_e[i / kConversionsPerStep] = convert(in_conv);

            if (i == size(tTMEM_LOADrS) - 10) {
                order_s.arrive();
            }
        }

        // Store P back to TMEM
        copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4);

        cutlass::arch::fence_view_async_tmem_store();

        pipeline_s.consumer_release(pipeline_s_consumer_state);
        ++pipeline_s_consumer_state;

        pipeline_c.producer_acquire(pipeline_c_producer_state);

        // Update row sum
        ElementAccum acc_scale = 0.5f * ::exp2f(scale * (old_row_max - row_max_safe));
        row_sum *= acc_scale;

        float local_row_sum = row_sum;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTMEM_LOADrS); i++) {
            local_row_sum += tTMEM_LOADrS(i);
        }
        row_sum = local_row_sum;

        // On final call, store row max and sum for epilogue scaling
        if (final_call) {
            pipeline_s.consumer_wait(pipeline_s_consumer_state);

            Tensor tTMEM_STOREVrS = make_tensor<ElementAccum>(shape(tTMEM_STOREVcS));
            tTMEM_STOREVrS(kIdxFinalRowMax) = row_max;
            tTMEM_STOREVrS(kIdxFinalRowSum) = row_sum;
            copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Helper: Rescale O in TMEM
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE void correction_rescale(float scale, uint32_t tmem_O) {
        int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

        const int kCorrectionTileSize = 16;

        using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b16x;
        using TMEM_STORE = SM100_TMEM_STORE_32dp32b16x;

        TiledMmaPV mma;
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

        Tensor tTMrO = make_tensor<ElementAccum>(make_shape(shape(tTMEM_LOADcO),
                                                            Int<Ktraits::kHeadDim / kCorrectionTileSize>{}));

        float2 scale_f32x2 = make_float2(scale, scale);

        // Load, scale, store loop
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

        int count = Ktraits::kHeadDim / kCorrectionTileSize;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < count; i++) {
            if (i != count - 1) {
                copy_in(i + 1);
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

    ///////////////////////////////////////////////////////////////////////////
    // Helper: Copy O from TMEM to SMEM for epilogue
    ///////////////////////////////////////////////////////////////////////////

    template <typename Stage, typename TensorO>
    CUTLASS_DEVICE void correction_epilogue(float scale, Stage stage, TensorO const& sO_01) {
        int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

        Tensor sO = sO_01(_, _, stage);

        const int kCorrectionTileSize = 32 / sizeof(ElementOut);

        using TMEM_LOAD = std::conditional_t<kCorrectionTileSize == 32,
            SM100_TMEM_LOAD_32dp32b32x, SM100_TMEM_LOAD_32dp32b16x>;

        TiledMmaPV mma;
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
            tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAlloc::O1);
        }

        auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i(make_coord(_, _), _0{}));
        auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);

        Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i(make_coord(_, _), _));
        Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i(make_coord(_, _), _));
        Tensor tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i(make_coord(_, _), _));

        float2 scale_f32x2 = make_float2(scale, scale);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Ktraits::kHeadDim / kCorrectionTileSize; i++) {
            Tensor tTMEM_LOADtO_i = tTMEM_LOADtO(_, _0{}, _0{}, i);
            Tensor tTMEM_LOADsO_i = tTMEM_LOADsO(_, _0{}, _0{}, i);

            Tensor tTMrO = make_tensor<ElementAccum>(shape(tTMEM_LOADcO(_, _0{}, _0{}, i)));

            copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO);

            // Scale and convert to output type
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
