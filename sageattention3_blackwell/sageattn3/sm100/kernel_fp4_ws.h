/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Flash Attention Kernel
 *
 * Simplified kernel wrapper for FP4 block-scaled attention.
 * This kernel uses CUTLASS's SM100 block-scaled GEMM infrastructure.
 *
 * Requirements:
 * - HeadDim = 256 (FP4 MMA K dimension constraint)
 * - BlockN = 256 (FP4 MMA K dimension constraint for PV matmul)
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "kernel_traits_fp4.h"
#include "mainloop_fp4_ws.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Flash Attention Forward Kernel
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits_, bool Is_causal, typename TileScheduler>
struct Sm100FlashFwdKernelFP4 {

    // Expose Ktraits for external access
    using Ktraits = Ktraits_;

    using ElementData = typename Ktraits::ElementData;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementOut = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;

    using TileShapeQK = typename Ktraits::TileShapeQK;
    using SharedStorage = typename Ktraits::SharedStorage;

    using CollectiveMainloop = CollectiveMainloopFwdSm100FP4<Ktraits, Is_causal>;

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kNThreads = Ktraits::kNThreads;

    ///////////////////////////////////////////////////////////////////////////
    // Arguments and Parameters
    // Layout matches fmha_sm100.cu initialization order
    ///////////////////////////////////////////////////////////////////////////

    struct Arguments {
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

        // Output tensor (BF16)
        ElementOut* ptr_O;
        int64_t stride_O_seq;
        int64_t stride_O_head;
        int64_t stride_O_batch;

        float scale_softmax;
    };

    struct Params {
        typename CollectiveMainloop::Params mainloop;
        typename TileScheduler::Params scheduler;
        int seqlen_q;
        int seqlen_k;
        int num_heads;
        int batch_size;
        ElementOut* ptr_O;
        int64_t stride_O_seq;
        int64_t stride_O_head;
        int64_t stride_O_batch;
    };

    static Params to_underlying_arguments(Arguments const& args, void* workspace) {
        // Problem shape: (seqlen_q, seqlen_k, head_dim, (num_heads, batch))
        auto problem_shape = make_tuple(
            args.seqlen_q, args.seqlen_k, args.head_dim,
            make_tuple(args.num_heads, args.batch_size)
        );

        // Mainloop now directly uses kernel arguments
        auto mainloop_params = CollectiveMainloop::to_underlying_arguments(args, workspace);

        typename TileScheduler::Arguments scheduler_args{};
        auto scheduler_params = TileScheduler::to_underlying_arguments(
            problem_shape, TileShapeQK{}, scheduler_args, workspace);

        return Params{
            mainloop_params,
            scheduler_params,
            args.seqlen_q,
            args.seqlen_k,
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

        // Get work assignment
        TileScheduler scheduler;
        auto work_tile = scheduler.get_initial_work(params.scheduler);

        if (!work_tile.is_valid()) {
            return;
        }

        auto [m_block, head_idx, batch_idx] = work_tile.get_block_coord();

        // Run mainloop
        CollectiveMainloop mainloop;
        mainloop(
            params.mainloop,
            shared_storage,
            m_block,
            head_idx,
            batch_idx,
            params.seqlen_q,
            params.seqlen_k
        );

        // TODO: Epilogue - write output to GMEM
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
