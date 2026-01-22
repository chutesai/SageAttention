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

    // TMEM allocation
    using TmemAlloc = flash::Sm100TmemAlloc;

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
    // MMA function (executed by MMA warp) - placeholder
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
        // TODO: Implement SM100 TMEM-based MMA
        // This requires UMMA instructions with TMEM accumulators
    }

    ///////////////////////////////////////////////////////////////////////////
    // Softmax function (executed by Softmax warps) - placeholder
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
        // TODO: Implement SM100 softmax with TMEM reads/writes
    }

    ///////////////////////////////////////////////////////////////////////////
    // Correction function (executed by Correction warps) - placeholder
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
        // TODO: Implement SM100 correction with TMEM
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
        int block_n = get<1>(TileShape{});

        if constexpr (Causal) {
            int block_m = get<0>(TileShape{});
            int m_idx = get<0>(blk_coord);
            int max_k = min(seqlen_k, (m_idx + 1) * block_m);
            return (max_k + block_n - 1) / block_n;
        } else {
            return (seqlen_k + block_n - 1) / block_n;
        }
    }
};

} // namespace flash
