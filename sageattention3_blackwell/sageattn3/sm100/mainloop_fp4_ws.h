/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled Mainloop for FlashAttention.
 *
 * This implements FP4 attention using SM100's block-scaled tcgen05.mma
 * instructions. Uses CUTLASS CollectiveMma infrastructure.
 *
 * Key constraints:
 * - HeadDim = 256 (FP4 MMA K=256 requirement)
 * - BlockN = 256 (FP4 PV matmul K=256 requirement)
 */

#pragma once

#include <cmath>

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm100.hpp"
#include "cute/arch/tmem.hpp"
#include "cute/arch/copy_sm100_tma.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/arch/mma_sm100.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "kernel_traits_fp4.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Causal mask for FP4 attention
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
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Attention Mainloop
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100FP4 {

    using Element = typename Ktraits::Element;
    using ElementData = typename Ktraits::ElementData;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;

    using TileShapeQK = typename Ktraits::TileShapeQK;
    using TileShapePV = typename Ktraits::TileShapePV;

    // CollectiveMma types from kernel traits
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

    // TMA types from CollectiveBuilder
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

    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kNThreads = Ktraits::kNThreads;

    // Scale factor configuration
    static constexpr int kSFVectorSize = Ktraits::SFVectorSize;

    // Mask type
    using Mask = std::conditional_t<Is_causal, CausalMaskFP4, NoMaskFP4>;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Params {
        // TMA descriptors for data
        TMA_Q tma_load_q;
        TMA_K tma_load_k;
        TMA_V tma_load_v;

        // TMA descriptors for scale factors
        TMA_SFA tma_load_sfq;
        TMA_SFB tma_load_sfk;
        TMA_SFV tma_load_sfv;

        // Scale factor layouts
        LayoutSFA layout_sfq;
        LayoutSFB layout_sfk;
        LayoutSFB layout_sfv;

        float scale_softmax;
        float scale_softmax_log2;
    };

    template <typename KernelArgs>
    static Params to_underlying_arguments(
        KernelArgs const& args,
        void* workspace
    ) {
        float log2_e = static_cast<float>(M_LOG2E);

        // For now, return a minimal Params struct
        // The actual TMA descriptor construction requires problem shape info
        // which we don't have here. This will be set up properly when we
        // integrate with the full kernel infrastructure.

        return Params{
            TMA_Q{},
            TMA_K{},
            TMA_V{},
            TMA_SFA{},
            TMA_SFB{},
            TMA_SFV{},
            LayoutSFA{},
            LayoutSFB{},
            LayoutSFB{},
            args.scale_softmax,
            args.scale_softmax * log2_e
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Kernel Body
    ///////////////////////////////////////////////////////////////////////////

    template <typename SharedStorage>
    CUTLASS_DEVICE void operator()(
        Params const& params,
        SharedStorage& storage,
        int m_block,
        int head_idx,
        int batch_idx,
        int seqlen_q,
        int seqlen_k
    ) {
        int thread_idx = threadIdx.x;
        int warp_idx = thread_idx / 32;
        int lane_idx = thread_idx % 32;

        // Problem shape for this tile
        auto problem_shape = make_tuple(seqlen_q, seqlen_k, kHeadDim,
                                        make_tuple(1, 1));  // heads, batch handled externally

        // Calculate number of K/V tiles to process
        Mask mask;
        auto blk_coord = make_tuple(m_block, 0, make_tuple(head_idx, batch_idx));
        int num_kv_tiles = mask.get_trip_count(blk_coord, TileShapeQK{}, problem_shape);

        // Early exit if no tiles to process
        if (num_kv_tiles <= 0) return;

        // Create SMEM tensors
        Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

        // Create SMEM tensors for scale factors
        Tensor sSFQ = make_tensor(make_smem_ptr(storage.smem_sfq.data()), SmemLayoutSFA{});
        Tensor sSFK = make_tensor(make_smem_ptr(storage.smem_sfk.data()), SmemLayoutSFB_QK{});
        Tensor sSFV = make_tensor(make_smem_ptr(storage.smem_sfv.data()), SmemLayoutSFB_PV{});

        // The full implementation would:
        // 1. Set up TMA tensors using params.tma_load_*.get_tma_tensor()
        // 2. Partition with tma_partition()
        // 3. Load Q and its scale factors via TMA
        // 4. For each K/V tile:
        //    a. Load K, K scale factors via TMA
        //    b. Execute QK block-scaled GEMM
        //    c. Apply softmax scaling
        //    d. Compute online softmax
        //    e. Quantize P to FP4 (on-the-fly scale factor generation)
        //    f. Load V, V scale factors via TMA
        //    g. Execute PV block-scaled GEMM
        //    h. Correct output accumulator
        // 5. Normalize and write output

        // For now, this is a placeholder that ensures compilation
        __syncthreads();
    }
};

} // namespace flash
