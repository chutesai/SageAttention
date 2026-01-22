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
 * SM100 (B200/B300) Warp-Specialized Flash Attention Kernel
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"

#include "kernel_traits.h"
#include "mainloop_tmem_ws.h"
#include "epilogue_tmem_ws.h"
#include "../blackwell/params.h"
#include "../blackwell/tile_scheduler.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 Flash Attention Forward Kernel (Warp-Specialized)
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal, typename TileScheduler>
struct Sm100FlashFwdKernel {

    using Element = typename Ktraits::Element;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;
    using TileShape = typename Ktraits::TileShape_MNK;
    using Schedule = typename Ktraits::Schedule;
    using WarpRole = typename Schedule::WarpRole;

    using CollectiveMainloop = CollectiveMainloopFwdSm100<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwdSm100<Ktraits>;

    using SharedStorage = typename Ktraits::SharedStorage;
    using TmemAllocator = typename Ktraits::TmemAllocator;

    // Pipeline types
    using PipelineQ = typename Ktraits::PipelineQ;
    using PipelineKV = typename Ktraits::PipelineKV;
    using PipelineS = typename Ktraits::PipelineS;
    using PipelineC = typename Ktraits::PipelineC;
    using PipelineO = typename Ktraits::PipelineO;
    using PipelineE = typename Ktraits::PipelineE;
    using OrderBarrierSoftmax = typename Ktraits::OrderBarrierSoftmax;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kNWarps = Schedule::kNumWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

    // Transaction bytes for TMA loads
    static constexpr int TransactionBytesLoadQ =
        cute::cosize_v<typename Ktraits::SmemLayoutQ> * sizeof(Element) / Ktraits::kStageCountQ;
    static constexpr int TransactionBytesLoadKV =
        cute::cosize_v<typename Ktraits::SmemLayoutK> * sizeof(Element) / Ktraits::kStageCountKV;

    ///////////////////////////////////////////////////////////////////////////
    // Kernel entry point
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE void operator()(
        Flash_fwd_params const& params,
        typename CollectiveMainloop::Params const& mainloop_params,
        typename CollectiveEpilogue::Params const& epilogue_params,
        typename TileScheduler::Params const& scheduler_params,
        char* smem
    ) {
        TileScheduler tile_scheduler;

        // Get work tile for this CTA
        auto work_tile_info = tile_scheduler.get_initial_work();
        auto blk_coord = work_tile_info.get_block_coord(scheduler_params);
        auto problem_shape = get_problem_shape(params, blk_coord);

        // Early exit if this tile is out of bounds
        if (!work_tile_info.is_valid(scheduler_params) ||
            get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(problem_shape)) {
            return;
        }

        int warp_idx = cutlass::canonical_warp_idx_sync();
        auto role = Schedule::warp_idx_to_role(warp_idx);
        uint32_t lane_predicate = cute::elect_one_sync();

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

        // Prefetch TMA descriptors
        if (role == WarpRole::Load && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        }
        if (role == WarpRole::Epilogue && lane_predicate) {
            CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
        }

        //
        // Initialize pipelines for SM100 UMMA execution
        //

        // Pipeline Q: Load -> MMA (PipelineTmaUmmaAsync)
        typename PipelineQ::Params pipeline_q_params;
        if (role == WarpRole::Load) {
            pipeline_q_params.role = PipelineQ::ThreadCategory::Producer;
        }
        if (role == WarpRole::MMA) {
            pipeline_q_params.role = PipelineQ::ThreadCategory::Consumer;
        }
        pipeline_q_params.is_leader = lane_predicate && (role == WarpRole::Load);
        pipeline_q_params.transaction_bytes = TransactionBytesLoadQ;
        PipelineQ pipeline_q(
            shared_storage.pipelines.pipeline_q,
            pipeline_q_params,
            ClusterShape{}, cute::true_type{}, cute::false_type{});

        // Pipeline KV: Load -> MMA (PipelineTmaUmmaAsync)
        typename PipelineKV::Params pipeline_kv_params;
        if (role == WarpRole::Load) {
            pipeline_kv_params.role = PipelineKV::ThreadCategory::Producer;
        }
        if (role == WarpRole::MMA) {
            pipeline_kv_params.role = PipelineKV::ThreadCategory::Consumer;
        }
        pipeline_kv_params.is_leader = lane_predicate && (role == WarpRole::Load);
        pipeline_kv_params.transaction_bytes = TransactionBytesLoadKV;
        PipelineKV pipeline_kv(
            shared_storage.pipelines.pipeline_kv,
            pipeline_kv_params,
            ClusterShape{}, cute::true_type{}, cute::false_type{});

        // Pipeline S0: MMA -> Softmax0 (PipelineAsync)
        typename PipelineS::Params pipeline_s0_params;
        if (role == WarpRole::MMA) {
            pipeline_s0_params.role = PipelineS::ThreadCategory::Producer;
        }
        if (role == WarpRole::Softmax0) {
            pipeline_s0_params.role = PipelineS::ThreadCategory::Consumer;
        }
        pipeline_s0_params.consumer_arv_count = Schedule::kNumWarpsSoftmax * cutlass::NumThreadsPerWarp;
        PipelineS pipeline_s0(
            shared_storage.pipelines.pipeline_s0,
            pipeline_s0_params,
            ClusterShape{}, cute::true_type{}, cute::false_type{});

        // Pipeline S1: MMA -> Softmax1 (PipelineAsync)
        typename PipelineS::Params pipeline_s1_params;
        if (role == WarpRole::MMA) {
            pipeline_s1_params.role = PipelineS::ThreadCategory::Producer;
        }
        if (role == WarpRole::Softmax1) {
            pipeline_s1_params.role = PipelineS::ThreadCategory::Consumer;
        }
        pipeline_s1_params.consumer_arv_count = Schedule::kNumWarpsSoftmax * cutlass::NumThreadsPerWarp;
        PipelineS pipeline_s1(
            shared_storage.pipelines.pipeline_s1,
            pipeline_s1_params,
            ClusterShape{}, cute::true_type{}, cute::false_type{});

        // Pipeline C0: Softmax0 -> Correction (PipelineAsync)
        typename PipelineC::Params pipeline_c0_params;
        if (role == WarpRole::Softmax0) {
            pipeline_c0_params.role = PipelineC::ThreadCategory::Producer;
        }
        if (role == WarpRole::Correction) {
            pipeline_c0_params.role = PipelineC::ThreadCategory::Consumer;
        }
        pipeline_c0_params.producer_arv_count = Schedule::kNumWarpsSoftmax * cutlass::NumThreadsPerWarp;
        pipeline_c0_params.consumer_arv_count = Schedule::kNumWarpsCorrection * cutlass::NumThreadsPerWarp;
        PipelineC pipeline_c0(
            shared_storage.pipelines.pipeline_c0,
            pipeline_c0_params,
            cute::true_type{});

        // Pipeline C1: Softmax1 -> Correction (PipelineAsync)
        typename PipelineC::Params pipeline_c1_params;
        if (role == WarpRole::Softmax1) {
            pipeline_c1_params.role = PipelineC::ThreadCategory::Producer;
        }
        if (role == WarpRole::Correction) {
            pipeline_c1_params.role = PipelineC::ThreadCategory::Consumer;
        }
        pipeline_c1_params.producer_arv_count = Schedule::kNumWarpsSoftmax * cutlass::NumThreadsPerWarp;
        pipeline_c1_params.consumer_arv_count = Schedule::kNumWarpsCorrection * cutlass::NumThreadsPerWarp;
        PipelineC pipeline_c1(
            shared_storage.pipelines.pipeline_c1,
            pipeline_c1_params,
            cute::true_type{});

        // Pipeline O: MMA -> Correction (PipelineAsync with cluster)
        typename PipelineO::Params pipeline_o_params;
        if (role == WarpRole::MMA) {
            pipeline_o_params.role = PipelineO::ThreadCategory::Producer;
        }
        if (role == WarpRole::Correction) {
            pipeline_o_params.role = PipelineO::ThreadCategory::Consumer;
        }
        pipeline_o_params.consumer_arv_count = Schedule::kNumWarpsCorrection * cutlass::NumThreadsPerWarp;
        PipelineO pipeline_o(
            shared_storage.pipelines.pipeline_o,
            pipeline_o_params,
            ClusterShape{}, cute::true_type{}, cute::false_type{});

        // Pipeline Epi: Correction -> Epilogue (PipelineAsync)
        typename PipelineE::Params pipeline_epi_params;
        if (role == WarpRole::Correction) {
            pipeline_epi_params.role = PipelineE::ThreadCategory::Producer;
        }
        if (role == WarpRole::Epilogue) {
            pipeline_epi_params.role = PipelineE::ThreadCategory::Consumer;
        }
        pipeline_epi_params.producer_arv_count = Schedule::kNumWarpsCorrection * cutlass::NumThreadsPerWarp;
        pipeline_epi_params.consumer_arv_count = Schedule::kNumWarpsEpilogue * cutlass::NumThreadsPerWarp;
        PipelineE pipeline_epi(
            shared_storage.pipelines.pipeline_epi,
            pipeline_epi_params,
            cute::true_type{});

        // Ordered barrier for softmax warps
        typename OrderBarrierSoftmax::Params order_s01_params;
        order_s01_params.group_id = (role == WarpRole::Softmax1) ? 1 : 0;
        order_s01_params.group_size = Schedule::kNumWarpsSoftmax * cutlass::NumThreadsPerWarp;
        OrderBarrierSoftmax order_s01(
            shared_storage.pipelines.order_s01, order_s01_params);

        // Initialize TMEM allocator
        TmemAllocator tmem_allocator;

        __syncthreads();

        // Initialize pipeline masks for UMMA pipelines
        pipeline_q.init_masks(ClusterShape{});
        pipeline_kv.init_masks(ClusterShape{});
        pipeline_s0.init_masks(ClusterShape{});
        pipeline_s1.init_masks(ClusterShape{});
        pipeline_o.init_masks(ClusterShape{});

        // Initialize pipeline states
        typename PipelineQ::PipelineState pipeline_q_consumer_state;
        typename PipelineQ::PipelineState pipeline_q_producer_state =
            cutlass::make_producer_start_state<PipelineQ>();

        typename PipelineKV::PipelineState pipeline_kv_consumer_state;
        typename PipelineKV::PipelineState pipeline_kv_producer_state =
            cutlass::make_producer_start_state<PipelineKV>();

        typename PipelineS::PipelineState pipeline_s0_consumer_state;
        typename PipelineS::PipelineState pipeline_s0_producer_state =
            cutlass::make_producer_start_state<PipelineS>();

        typename PipelineS::PipelineState pipeline_s1_consumer_state;
        typename PipelineS::PipelineState pipeline_s1_producer_state =
            cutlass::make_producer_start_state<PipelineS>();

        typename PipelineC::PipelineState pipeline_c0_consumer_state;
        typename PipelineC::PipelineState pipeline_c0_producer_state =
            cutlass::make_producer_start_state<PipelineC>();

        typename PipelineC::PipelineState pipeline_c1_consumer_state;
        typename PipelineC::PipelineState pipeline_c1_producer_state =
            cutlass::make_producer_start_state<PipelineC>();

        typename PipelineO::PipelineState pipeline_o_consumer_state;
        typename PipelineO::PipelineState pipeline_o_producer_state =
            cutlass::make_producer_start_state<PipelineO>();

        typename PipelineE::PipelineState pipeline_epi_consumer_state;
        typename PipelineE::PipelineState pipeline_epi_producer_state =
            cutlass::make_producer_start_state<PipelineE>();

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        // Dispatch based on warp role
        if (role == WarpRole::Softmax0 || role == WarpRole::Softmax1) {
            cutlass::arch::warpgroup_reg_alloc<Schedule::kNumRegsSoftmax>();

            bool is_softmax_0 = (role == WarpRole::Softmax0);

            mainloop.softmax(
                is_softmax_0 ? 0 : 1, blk_coord,
                mainloop_params, problem_shape, params,
                is_softmax_0 ? pipeline_s0 : pipeline_s1,
                is_softmax_0 ? pipeline_s0_consumer_state : pipeline_s1_consumer_state,
                is_softmax_0 ? pipeline_c0 : pipeline_c1,
                is_softmax_0 ? pipeline_c0_producer_state : pipeline_c1_producer_state,
                order_s01
            );
        }
        else if (role == WarpRole::Correction) {
            cutlass::arch::warpgroup_reg_dealloc<Schedule::kNumRegsCorrection>();

            mainloop.correction(
                blk_coord,
                mainloop_params, problem_shape, params,
                shared_storage,
                pipeline_c0, pipeline_c0_consumer_state,
                pipeline_c1, pipeline_c1_consumer_state,
                pipeline_o, pipeline_o_consumer_state,
                pipeline_epi, pipeline_epi_producer_state
            );

            if constexpr (Schedule::kNumWarpsEpilogue == 0) {
                uint32_t free_ptr = shared_storage.tmem_base_ptr;
                tmem_allocator.free(free_ptr, TmemAllocator::Sm100TmemCapacityColumns);
            }
        }
        else if (role == WarpRole::MMA) {
            cutlass::arch::warpgroup_reg_alloc<Schedule::kNumRegsOther>();

            tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                                    &shared_storage.tmem_base_ptr);
            __syncwarp();

            mainloop.mma(
                blk_coord,
                mainloop_params, problem_shape, params,
                shared_storage,
                pipeline_q, pipeline_q_consumer_state,
                pipeline_kv, pipeline_kv_consumer_state,
                pipeline_s0, pipeline_s0_producer_state,
                pipeline_s1, pipeline_s1_producer_state,
                pipeline_o, pipeline_o_producer_state
            );
        }
        else if (role == WarpRole::Load) {
            cutlass::arch::warpgroup_reg_alloc<Schedule::kNumRegsOther>();

            mainloop.load(
                blk_coord, problem_shape,
                mainloop_params, params,
                shared_storage,
                pipeline_q, pipeline_q_producer_state,
                pipeline_kv, pipeline_kv_producer_state
            );
        }
        else if (role == WarpRole::Epilogue) {
            cutlass::arch::warpgroup_reg_alloc<Schedule::kNumRegsOther>();

            epilogue.store(
                blk_coord, problem_shape,
                epilogue_params, params,
                shared_storage,
                pipeline_epi, pipeline_epi_consumer_state
            );

            if constexpr (Schedule::kNumWarpsEpilogue == 1) {
                uint32_t free_ptr = shared_storage.tmem_base_ptr;
                tmem_allocator.free(free_ptr, TmemAllocator::Sm100TmemCapacityColumns);
            }
        }
        else if (role == WarpRole::Empty) {
            cutlass::arch::warpgroup_reg_alloc<Schedule::kNumRegsEmpty>();
        }
    }

private:

    template <typename BlkCoord>
    CUTLASS_DEVICE auto get_problem_shape(
        Flash_fwd_params const& params,
        BlkCoord const& blk_coord
    ) {
        // Problem shape: (seqlen_q, seqlen_k, head_dim, batch*heads)
        return make_shape(
            params.seqlen_q,
            params.seqlen_k,
            params.d,
            params.b * params.h
        );
    }
};

///////////////////////////////////////////////////////////////////////////////
// Kernel launch wrapper
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNThreads, 1)
compute_attn_ws_sm100(
    CUTE_GRID_CONSTANT Flash_fwd_params const params,
    CUTE_GRID_CONSTANT typename CollectiveMainloopFwdSm100<Ktraits, Is_causal>::Params const mainloop_params,
    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwdSm100<Ktraits>::Params const epilogue_params,
    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
) {
    extern __shared__ char smem[];

    Sm100FlashFwdKernel<Ktraits, Is_causal, TileScheduler> kernel;
    kernel(params, mainloop_params, epilogue_params, scheduler_params, smem);
}

} // namespace flash
