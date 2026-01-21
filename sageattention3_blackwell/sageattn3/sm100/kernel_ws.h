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
 * SM100 (B200/B300) Warp-Specialized Attention Kernel
 *
 * This kernel uses tcgen05.mma instructions with TMEM for accumulator storage.
 * The warp specialization pattern is similar to SM120:
 *   - Producer warp group: Loads Q, K, V via TMA
 *   - Consumer warp groups: Compute attention using tcgen05.mma
 *   - Epilogue warp: Stores output via TMA
 *
 * KEY DIFFERENCES FROM SM120:
 * ==========================
 * 1. Must allocate TMEM at kernel start and deallocate at end
 * 2. Consumer thread 0 initializes mbarriers for async MMA synchronization
 * 3. Output accumulator lives in TMEM, must load to registers before storing
 * 4. Uses tcgen05.mma instructions instead of mma.sync.aligned
 */

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "../blackwell/params.h"
#include "../blackwell/utils.h"
#include "../blackwell/tile_scheduler.h"
#include "../blackwell/named_barrier.h"
#include "mainloop_tmem_ws.h"
#include "epilogue_tmem_ws.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 Warp-Specialized Attention Kernel
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNThreads, 1)
compute_attn_ws_sm100(
    CUTE_GRID_CONSTANT Flash_fwd_params const params,
    CUTE_GRID_CONSTANT typename CollectiveMainloopFwdSm100<Ktraits, Is_causal>::Params const mainloop_params,
    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwdSm100<Ktraits>::Params const epilogue_params,
    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
) {
    using Element = typename Ktraits::Element;
    using ElementAccum = typename Ktraits::ElementAccum;
    using SoftType = ElementAccum;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int NumMmaThreads = Ktraits::kNThreads - cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;

    using CollectiveMainloop = CollectiveMainloopFwdSm100<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwdSm100<Ktraits>;
    using Softmax = typename CollectiveMainloop::Softmax;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
    using PipelineParamsQ = typename Ktraits::PipelineParamsQ;
    using PipelineStateQ = typename Ktraits::PipelineStateQ;
    using EpilogueBarrier = typename Ktraits::EpilogueBarrier;

    // Warp group roles
    enum class WarpGroupRole {
        Producer = 0,
        Consumer0 = 1,
        Consumer1 = 2
    };
    enum class ProducerWarpRole {
        Mainloop = 0,
        Epilogue = 1,
        Warp2 = 2,
        Warp3 = 3
    };

    // Shared memory
    extern __shared__ char shared_memory[];
    auto& shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    // Thread identification
    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

    // Prefetch TMA descriptors (single thread)
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Pipeline setup
    PipelineParams pipeline_params_v;
    pipeline_params_v.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    pipeline_params_v.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_v.is_leader = warp_group_thread_idx == 0;
    pipeline_params_v.num_consumers = NumMmaThreads;

    PipelineParams pipeline_params_k;
    pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params_k.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_k.is_leader = warp_group_thread_idx == 0;
    pipeline_params_k.num_consumers = NumMmaThreads;

    PipelineParamsQ pipeline_params_q;
    pipeline_params_q.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ;
    pipeline_params_q.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipelineQ::ThreadCategory::Producer
        : MainloopPipelineQ::ThreadCategory::Consumer;
    pipeline_params_q.is_leader = warp_group_thread_idx == 0;
    pipeline_params_q.num_consumers = NumMmaThreads;

    // Initialize pipelines
    MainloopPipelineQ pipeline_q(shared_storage.pipeline_q, pipeline_params_q, ClusterShape{});
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params_k, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params_v, ClusterShape{});

    // Epilogue barrier
    uint32_t epilogue_barrier_group_size_list[2] = {cutlass::NumThreadsPerWarp, NumMmaThreads};
    typename EpilogueBarrier::Params params_epilogue_barrier;
    params_epilogue_barrier.group_id = (warp_group_role == WarpGroupRole::Producer);
    params_epilogue_barrier.group_size_list = epilogue_barrier_group_size_list;
    EpilogueBarrier barrier_o(shared_storage.barrier_o, params_epilogue_barrier);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    //=========================================================================
    // SM100-specific: Initialize TMEM and mbarriers (consumer thread 0)
    //=========================================================================
    if (warp_group_role == WarpGroupRole::Consumer0 && warp_group_thread_idx == 0) {
        CollectiveMainloop::init_tmem_and_mbarriers(shared_storage, 0);
    }
    __syncthreads();

    //=========================================================================
    // Producer warp group: Load Q, K, V and store output
    //=========================================================================
    if (warp_group_role == WarpGroupRole::Producer) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
        TileScheduler scheduler;

        if (producer_warp_role == ProducerWarpRole::Mainloop) {
            // Load Q, K, V tiles
            PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            int work_idx = 0;
            for (auto work_tile_info = scheduler.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {

                int tile_count_semaphore = 0;
                collective_mainloop.load(mainloop_params, scheduler_params,
                                         pipeline_q, pipeline_k, pipeline_v,
                                         smem_pipe_write_q, smem_pipe_write_k, smem_pipe_write_v,
                                         shared_storage, work_tile_info, work_idx, tile_count_semaphore);
            }
            collective_mainloop.load_tail(pipeline_q, pipeline_k, pipeline_v,
                                          smem_pipe_write_q, smem_pipe_write_k, smem_pipe_write_v);

        } else if (producer_warp_role == ProducerWarpRole::Epilogue) {
            // Store output tiles
            TileScheduler scheduler_epilogue;
            for (auto work_tile_info = scheduler_epilogue.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler_epilogue.get_next_work(scheduler_params, work_tile_info)) {

                barrier_o.wait();
                collective_epilogue.tma_store(shared_storage, epilogue_params,
                                              work_tile_info, scheduler_params, threadIdx.x);
                collective_epilogue.store_tail();
                barrier_o.arrive();
            }
        }
    }

    //=========================================================================
    // Consumer warp groups: Compute attention
    //=========================================================================
    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
        cutlass::arch::warpgroup_reg_alloc<232>();
        typename Ktraits::TiledMmaPV tiled_mma_pv;
        TileScheduler scheduler{};
        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        PipelineStateQ smem_pipe_read_q;

        int work_idx = 0;

        // Consumer thread index relative to consumer warp groups
        int const consumer_thread_idx = threadIdx.x - NumCopyThreads;

        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {

            // Output fragment (SM100 uses raw arrays internally, but we need tensor for epilogue interface)
            Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 2>(TileShape_MNK{}));
            Softmax softmax;
            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, bidh, bidb] = block_coord;

            int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block);

            // Early exit for causal masking
            if (Is_causal && n_block_max <= 0) {
                collective_epilogue.store_zero(epilogue_params, consumer_thread_idx, block_coord);
                continue;
            }

            // Compute attention
            collective_mainloop.mma(mainloop_params, pipeline_q, pipeline_k, pipeline_v,
                                    smem_pipe_read_q, smem_pipe_read_k, smem_pipe_read_v,
                                    tOrO, softmax, n_block_max, consumer_thread_idx,
                                    work_idx, m_block, shared_storage);

            // Store output to SMEM then signal for TMA store
            barrier_o.wait();
            collective_epilogue.mma_store(shared_storage, tiled_mma_pv, tOrO, consumer_thread_idx);
            barrier_o.arrive();
            ++work_idx;
        }

        //=====================================================================
        // SM100-specific: Cleanup TMEM (consumer thread 0)
        //=====================================================================
        __syncthreads();
        if (warp_group_role == WarpGroupRole::Consumer0 && warp_group_thread_idx == 0) {
            CollectiveMainloop::cleanup_tmem(shared_storage, 0);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Kernel Launch Helper
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
void run_attention_sm100(Flash_fwd_params const& params, cudaStream_t stream) {
    using Kernel = decltype(compute_attn_ws_sm100<Ktraits, Is_causal, flash::SingleTileScheduler>);
    using CollectiveMainloop = CollectiveMainloopFwdSm100<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwdSm100<Ktraits>;
    using TileScheduler = flash::SingleTileScheduler;

    // Compute grid dimensions
    int const seqlen_q = params.seqlen_q;
    int const seqlen_k = params.seqlen_k;
    int const num_heads = params.h;
    int const batch_size = params.b;

    int const num_m_blocks = cute::ceil_div(seqlen_q, Ktraits::kBlockM);
    int const num_n_blocks = cute::ceil_div(seqlen_k, Ktraits::kBlockN);

    dim3 grid(num_m_blocks, num_heads, batch_size);
    dim3 block(Ktraits::kNThreads);

    // Compute shared memory size
    int smem_size = sizeof(typename Ktraits::SharedStorage);

    // Set maximum dynamic shared memory
    cudaFuncSetAttribute(
        compute_attn_ws_sm100<Ktraits, Is_causal, TileScheduler>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    // Prepare kernel arguments
    typename CollectiveMainloop::Arguments mainloop_args{
        static_cast<typename Ktraits::Element const*>(params.q_ptr),
        {seqlen_q, Ktraits::kHeadDim, num_heads, batch_size},
        {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},
        static_cast<typename Ktraits::Element const*>(params.k_ptr),
        {seqlen_k, Ktraits::kHeadDim, num_heads, batch_size},
        {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},
        {seqlen_k, Ktraits::kHeadDim, num_heads, batch_size},  // unpadded_shape_K
        static_cast<typename Ktraits::Element const*>(params.v_ptr),
        {Ktraits::kHeadDim, seqlen_k, num_heads, batch_size},  // V is transposed
        {_1{}, params.v_row_stride, params.v_head_stride, params.v_batch_stride},
        static_cast<typename Ktraits::ElementSF const*>(params.sfq_ptr),
        {seqlen_q, Ktraits::kHeadDim / Ktraits::kSFVecSize, num_heads, batch_size},
        static_cast<typename Ktraits::ElementSF const*>(params.sfk_ptr),
        {seqlen_k, Ktraits::kHeadDim / Ktraits::kSFVecSize, num_heads, batch_size},
        static_cast<typename Ktraits::ElementSF const*>(params.sfv_ptr),
        {seqlen_k, Ktraits::kHeadDim / Ktraits::kSFVecSize, num_heads, batch_size},
        static_cast<float const*>(params.delta_s_ptr),
        {num_m_blocks * Ktraits::kBlockM, num_n_blocks * Ktraits::kBlockN, num_heads, batch_size},
        {num_n_blocks * Ktraits::kBlockN, _1{}, num_m_blocks * Ktraits::kBlockM * num_n_blocks * Ktraits::kBlockN,
         num_m_blocks * Ktraits::kBlockM * num_n_blocks * Ktraits::kBlockN * num_heads},
        params.scale_softmax_log2
    };

    typename CollectiveEpilogue::Arguments epilogue_args{
        static_cast<typename Ktraits::ElementOut*>(params.o_ptr),
        {seqlen_q, Ktraits::kHeadDim, num_heads, batch_size},
        {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, seqlen_q * num_heads}
    };

    typename TileScheduler::Arguments scheduler_args{
        num_m_blocks, num_heads, batch_size, 1  // num_splits = 1
    };

    // Convert to device params
    auto mainloop_params = CollectiveMainloop::to_underlying_arguments(mainloop_args);
    auto epilogue_device_params = CollectiveEpilogue::to_underlying_arguments(epilogue_args);
    auto scheduler_params = TileScheduler::to_underlying_arguments(scheduler_args);

    // Launch kernel
    compute_attn_ws_sm100<Ktraits, Is_causal, TileScheduler>
        <<<grid, block, smem_size, stream>>>(
            params,
            mainloop_params,
            epilogue_device_params,
            scheduler_params
        );
}

} // namespace flash
