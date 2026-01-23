/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Warp-Specialized Flash Attention Kernel
 *
 * This kernel implements FP4 quantized attention using CUTLASS's SM100
 * block-scaled UMMA support with tcgen05.mma instructions.
 *
 * Requirements:
 * - HeadDim = 256 (FP4 MMA K dimension constraint)
 * - BlockN = 256 (FP4 MMA K dimension constraint for PV matmul)
 * - Input Q, K, V must be pre-quantized to FP4 with scale factors
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cute/arch/cluster_sm90.hpp"

#include "kernel_traits_fp4.h"
#include "mainloop_fp4_ws.h"
#include "../blackwell/params.h"
#include "../blackwell/tile_scheduler.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Flash Attention Forward Kernel (Warp-Specialized)
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal, typename TileScheduler>
struct Sm100FlashFwdKernelFP4 {

    using Element = typename Ktraits::Element;
    using ElementData = typename Ktraits::ElementData;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;

    using TileShape = typename Ktraits::TileShape_MNK;
    using Schedule = typename Ktraits::Schedule;
    using WarpRole = typename Schedule::WarpRole;

    using CollectiveMainloop = CollectiveMainloopFwdSm100FP4<Ktraits, Is_causal>;

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

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kNWarps = Schedule::kNumWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

    // Transaction bytes for TMA loads (includes data + scale factors)
    static constexpr uint32_t TmaTransactionBytesQ = Ktraits::TmaTransactionBytesQ;
    static constexpr uint32_t TmaTransactionBytesK = Ktraits::TmaTransactionBytesK;
    static constexpr uint32_t TmaTransactionBytesV = Ktraits::TmaTransactionBytesV;

    ///////////////////////////////////////////////////////////////////////////
    // Arguments and Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Arguments {
        // Problem shape: (seqlen_q, seqlen_k, head_dim, num_heads, batch)
        int seqlen_q;
        int seqlen_k;
        int head_dim;
        int num_heads;
        int batch_size;

        // FP4 Q tensor and scale factors
        ElementData const* ptr_Q;
        ElementSF const* ptr_SFQ;
        int64_t stride_Q_seq;
        int64_t stride_Q_head;
        int64_t stride_Q_batch;

        // FP4 K tensor and scale factors
        ElementData const* ptr_K;
        ElementSF const* ptr_SFK;
        int64_t stride_K_seq;
        int64_t stride_K_head;
        int64_t stride_K_batch;

        // FP4 V tensor and scale factors
        ElementData const* ptr_V;
        ElementSF const* ptr_SFV;
        int64_t stride_V_seq;
        int64_t stride_V_head;
        int64_t stride_V_batch;

        // Output tensor (BF16 or FP16)
        ElementOut* ptr_O;
        int64_t stride_O_seq;
        int64_t stride_O_head;
        int64_t stride_O_batch;

        // Softmax scale
        float scale_softmax;
    };

    struct Params {
        typename CollectiveMainloop::Params mainloop;
        typename TileScheduler::Params scheduler;
        int seqlen_q;
        int seqlen_k;
        int head_dim;
        int num_heads;
        int batch_size;
        ElementOut* ptr_O;
        int64_t stride_O_seq;
        int64_t stride_O_head;
        int64_t stride_O_batch;
    };

    static Params to_underlying_arguments(Arguments const& args, void* workspace) {
        auto problem_shape = make_tuple(
            args.seqlen_q, args.seqlen_k, args.head_dim,
            make_tuple(args.num_heads, args.batch_size)
        );

        typename CollectiveMainloop::Arguments mainloop_args{
            args.ptr_Q,
            args.ptr_SFQ,
            make_tuple(args.stride_Q_seq, _1{}, args.stride_Q_head),
            typename Ktraits::LayoutSFA{},  // Scale factor layout
            args.ptr_K,
            args.ptr_SFK,
            make_tuple(args.stride_K_seq, _1{}, args.stride_K_head),
            typename Ktraits::LayoutSFB{},
            args.ptr_V,
            args.ptr_SFV,
            make_tuple(_1{}, args.stride_V_seq, args.stride_V_head),
            typename Ktraits::LayoutSFB{},
            args.scale_softmax
        };

        auto mainloop_params = CollectiveMainloop::to_underlying_arguments(
            problem_shape, mainloop_args, workspace);

        typename TileScheduler::Arguments scheduler_args{};
        auto scheduler_params = TileScheduler::to_underlying_arguments(
            problem_shape, TileShape{}, scheduler_args, workspace);

        return Params{
            mainloop_params,
            scheduler_params,
            args.seqlen_q,
            args.seqlen_k,
            args.head_dim,
            args.num_heads,
            args.batch_size,
            args.ptr_O,
            args.stride_O_seq,
            args.stride_O_head,
            args.stride_O_batch
        };
    }

    static dim3 get_grid_dim(Arguments const& args, int sm_count) {
        int num_m_blocks = (args.seqlen_q + kBlockM - 1) / kBlockM;
        int num_tiles = num_m_blocks * args.num_heads * args.batch_size;
        return dim3(min(num_tiles, sm_count), 1, 1);
    }

    static dim3 get_block_dim() {
        return dim3(kNThreads, 1, 1);
    }

    static size_t get_smem_size() {
        return sizeof(SharedStorage);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Kernel Entry Point
    ///////////////////////////////////////////////////////////////////////////

    CUTLASS_DEVICE void operator()(Params const& params, char* smem) {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

        int warp_idx = cutlass::canonical_warp_idx_sync();
        auto role = Schedule::warp_idx_to_role(warp_idx);
        uint32_t lane_predicate = cute::elect_one_sync();

        // Configure register allocation based on warp role
        if (role == WarpRole::Softmax0 || role == WarpRole::Softmax1) {
            cutlass::arch::warpgroup_reg_alloc<Schedule::kNumRegsSoftmax>();
        } else if (role == WarpRole::Correction) {
            cutlass::arch::warpgroup_reg_alloc<Schedule::kNumRegsCorrection>();
        } else if (role == WarpRole::Empty) {
            cutlass::arch::warpgroup_reg_alloc<Schedule::kNumRegsEmpty>();
        } else {
            cutlass::arch::warpgroup_reg_alloc<Schedule::kNumRegsOther>();
        }

        // Allocate TMEM for this CTA
        TmemAllocator tmem_allocator;
        if (role == WarpRole::MMA && lane_predicate) {
            shared_storage.tmem_base_ptr = tmem_allocator.allocate(
                static_cast<uint32_t>(Ktraits::TmemAlloc::kEnd));
        }
        __syncthreads();

        // Get work assignment from tile scheduler
        TileScheduler scheduler;
        auto work_tile = scheduler.get_initial_work(params.scheduler);

        if (!work_tile.is_valid()) {
            return;
        }

        auto [m_block, head_idx, batch_idx] = work_tile.get_block_coord();

        // Problem shape for this head/batch
        auto problem_shape = make_tuple(
            params.seqlen_q, params.seqlen_k, params.head_dim,
            make_tuple(params.num_heads, params.batch_size)
        );

        auto blk_coord = make_coord(m_block, _0{}, make_coord(head_idx, batch_idx));

        // Prefetch TMA descriptors
        if (role == WarpRole::Load && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
        }

        //
        // Initialize pipelines
        //

        // Pipeline Q: Load -> MMA
        typename PipelineQ::Params pipeline_q_params;
        pipeline_q_params.role = (role == WarpRole::Load) ?
            PipelineQ::ThreadCategory::Producer :
            PipelineQ::ThreadCategory::Consumer;
        pipeline_q_params.is_leader = lane_predicate && (role == WarpRole::Load);
        pipeline_q_params.transaction_bytes = TmaTransactionBytesQ;
        PipelineQ pipeline_q(
            shared_storage.pipelines.pipeline_q,
            pipeline_q_params,
            ClusterShape{});

        // Pipeline KV: Load -> MMA
        typename PipelineKV::Params pipeline_kv_params;
        pipeline_kv_params.role = (role == WarpRole::Load) ?
            PipelineKV::ThreadCategory::Producer :
            PipelineKV::ThreadCategory::Consumer;
        pipeline_kv_params.is_leader = lane_predicate && (role == WarpRole::Load);
        pipeline_kv_params.transaction_bytes = TmaTransactionBytesK;
        PipelineKV pipeline_kv(
            shared_storage.pipelines.pipeline_kv,
            pipeline_kv_params,
            ClusterShape{});

        // Pipeline S0: MMA -> Softmax0
        typename PipelineS::Params pipeline_s0_params;
        pipeline_s0_params.role = (role == WarpRole::MMA) ?
            PipelineS::ThreadCategory::Producer :
            PipelineS::ThreadCategory::Consumer;
        pipeline_s0_params.consumer_arv_count = Schedule::kNumWarpsSoftmax * 32;
        PipelineS pipeline_s0(
            shared_storage.pipelines.pipeline_s0,
            pipeline_s0_params,
            ClusterShape{});

        // Pipeline S1: MMA -> Softmax1
        typename PipelineS::Params pipeline_s1_params;
        pipeline_s1_params.role = (role == WarpRole::MMA) ?
            PipelineS::ThreadCategory::Producer :
            PipelineS::ThreadCategory::Consumer;
        pipeline_s1_params.consumer_arv_count = Schedule::kNumWarpsSoftmax * 32;
        PipelineS pipeline_s1(
            shared_storage.pipelines.pipeline_s1,
            pipeline_s1_params,
            ClusterShape{});

        // Pipeline C0: Softmax0 -> Correction
        typename PipelineC::Params pipeline_c0_params;
        pipeline_c0_params.role = (role == WarpRole::Softmax0) ?
            PipelineC::ThreadCategory::Producer :
            PipelineC::ThreadCategory::Consumer;
        pipeline_c0_params.producer_arv_count = Schedule::kNumWarpsSoftmax * 32;
        pipeline_c0_params.consumer_arv_count = Schedule::kNumWarpsCorrection * 32;
        PipelineC pipeline_c0(
            shared_storage.pipelines.pipeline_c0,
            pipeline_c0_params);

        // Pipeline C1: Softmax1 -> Correction
        typename PipelineC::Params pipeline_c1_params;
        pipeline_c1_params.role = (role == WarpRole::Softmax1) ?
            PipelineC::ThreadCategory::Producer :
            PipelineC::ThreadCategory::Consumer;
        pipeline_c1_params.producer_arv_count = Schedule::kNumWarpsSoftmax * 32;
        pipeline_c1_params.consumer_arv_count = Schedule::kNumWarpsCorrection * 32;
        PipelineC pipeline_c1(
            shared_storage.pipelines.pipeline_c1,
            pipeline_c1_params);

        // Pipeline O: MMA -> Correction
        typename PipelineO::Params pipeline_o_params;
        pipeline_o_params.role = (role == WarpRole::MMA) ?
            PipelineO::ThreadCategory::Producer :
            PipelineO::ThreadCategory::Consumer;
        pipeline_o_params.consumer_arv_count = Schedule::kNumWarpsCorrection * 32;
        PipelineO pipeline_o(
            shared_storage.pipelines.pipeline_o,
            pipeline_o_params,
            ClusterShape{});

        // Pipeline E: Correction -> Epilogue
        typename PipelineE::Params pipeline_e_params;
        pipeline_e_params.role = (role == WarpRole::Correction) ?
            PipelineE::ThreadCategory::Producer :
            PipelineE::ThreadCategory::Consumer;
        pipeline_e_params.producer_arv_count = Schedule::kNumWarpsCorrection * 32;
        pipeline_e_params.consumer_arv_count = Schedule::kNumWarpsEpilogue * 32;
        PipelineE pipeline_e(
            shared_storage.pipelines.pipeline_epi,
            pipeline_e_params);

        // Order barrier for softmax stages
        OrderBarrierSoftmax order_s(shared_storage.pipelines.order_s01);

        // Initialize pipeline states
        typename PipelineQ::PipelineState pipeline_q_producer_state = cutlass::make_producer_start_state<PipelineQ>();
        typename PipelineQ::PipelineState pipeline_q_consumer_state;
        typename PipelineKV::PipelineState pipeline_kv_producer_state = cutlass::make_producer_start_state<PipelineKV>();
        typename PipelineKV::PipelineState pipeline_kv_consumer_state;
        typename PipelineS::PipelineState pipeline_s0_producer_state = cutlass::make_producer_start_state<PipelineS>();
        typename PipelineS::PipelineState pipeline_s0_consumer_state;
        typename PipelineS::PipelineState pipeline_s1_producer_state = cutlass::make_producer_start_state<PipelineS>();
        typename PipelineS::PipelineState pipeline_s1_consumer_state;
        typename PipelineC::PipelineState pipeline_c0_producer_state = cutlass::make_producer_start_state<PipelineC>();
        typename PipelineC::PipelineState pipeline_c0_consumer_state;
        typename PipelineC::PipelineState pipeline_c1_producer_state = cutlass::make_producer_start_state<PipelineC>();
        typename PipelineC::PipelineState pipeline_c1_consumer_state;
        typename PipelineO::PipelineState pipeline_o_producer_state = cutlass::make_producer_start_state<PipelineO>();
        typename PipelineO::PipelineState pipeline_o_consumer_state;
        typename PipelineE::PipelineState pipeline_e_producer_state = cutlass::make_producer_start_state<PipelineE>();
        typename PipelineE::PipelineState pipeline_e_consumer_state;

        __syncthreads();

        // Create mainloop instance
        CollectiveMainloop mainloop;

        // Flash params (for compatibility)
        Flash_fwd_params flash_params;
        flash_params.scale_softmax = params.mainloop.scale_softmax;

        //
        // Dispatch based on warp role
        //

        if (role == WarpRole::Load) {
            mainloop.load(
                blk_coord, problem_shape,
                params.mainloop, flash_params,
                shared_storage,
                pipeline_q, pipeline_q_producer_state,
                pipeline_kv, pipeline_kv_producer_state
            );
        }
        else if (role == WarpRole::MMA) {
            mainloop.mma(
                blk_coord, params.mainloop, problem_shape, flash_params,
                shared_storage,
                pipeline_q, pipeline_q_consumer_state,
                pipeline_kv, pipeline_kv_consumer_state,
                pipeline_s0, pipeline_s0_producer_state,
                pipeline_s1, pipeline_s1_producer_state,
                pipeline_o, pipeline_o_producer_state
            );
        }
        else if (role == WarpRole::Softmax0) {
            mainloop.softmax(
                0, blk_coord, params.mainloop, problem_shape, flash_params,
                pipeline_s0, pipeline_s0_consumer_state,
                pipeline_c0, pipeline_c0_producer_state,
                order_s
            );
        }
        else if (role == WarpRole::Softmax1) {
            mainloop.softmax(
                1, blk_coord, params.mainloop, problem_shape, flash_params,
                pipeline_s1, pipeline_s1_consumer_state,
                pipeline_c1, pipeline_c1_producer_state,
                order_s
            );
        }
        else if (role == WarpRole::Correction) {
            // TODO: Implement correction and rescaling
            // This warp handles O = O * scale + correction
        }
        else if (role == WarpRole::Epilogue) {
            // TODO: Implement epilogue (write O to GMEM)
            // Wait for correction to finish, then store output
        }

        // Free TMEM
        if (role == WarpRole::MMA && lane_predicate) {
            tmem_allocator.free(shared_storage.tmem_base_ptr);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// Simple Tile Scheduler for FP4
///////////////////////////////////////////////////////////////////////////////

struct SimpleTileSchedulerFP4 {
    struct Arguments {};

    struct Params {
        int num_m_blocks;
        int num_heads;
        int batch_size;
    };

    struct WorkTileInfo {
        int m_block;
        int head_idx;
        int batch_idx;
        bool valid;

        CUTLASS_DEVICE bool is_valid() const { return valid; }

        CUTLASS_DEVICE auto get_block_coord() const {
            return cute::make_tuple(m_block, head_idx, batch_idx);
        }
    };

    template <typename ProblemShape, typename TileShape>
    static Params to_underlying_arguments(
        ProblemShape const& problem_shape,
        TileShape tile_shape,
        Arguments const& args,
        void* workspace
    ) {
        int seqlen_q = cute::get<0>(problem_shape);
        int num_heads = cute::get<0>(cute::get<3>(problem_shape));
        int batch_size = cute::get<1>(cute::get<3>(problem_shape));
        int block_m = cute::get<0>(tile_shape);
        int num_m_blocks = (seqlen_q + block_m - 1) / block_m;
        return Params{num_m_blocks, num_heads, batch_size};
    }

    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        int block_idx = blockIdx.x;
        int total_tiles = params.num_m_blocks * params.num_heads * params.batch_size;

        if (block_idx >= total_tiles) {
            return WorkTileInfo{0, 0, 0, false};
        }

        int m_block = block_idx % params.num_m_blocks;
        int remainder = block_idx / params.num_m_blocks;
        int head_idx = remainder % params.num_heads;
        int batch_idx = remainder / params.num_heads;

        return WorkTileInfo{m_block, head_idx, batch_idx, true};
    }
};

} // namespace flash
