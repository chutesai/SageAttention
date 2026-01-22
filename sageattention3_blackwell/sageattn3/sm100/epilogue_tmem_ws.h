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
 * SM100 (B200/B300) Epilogue for FlashAttention.
 *
 * The epilogue warp stores O from SMEM to GMEM via TMA.
 * The correction warps have already copied O from TMEM to SMEM.
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

#include "kernel_traits.h"
#include "../blackwell/params.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 Collective Epilogue for Flash Attention Forward
//
// Following CUTLASS example 77 pattern exactly.
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits>
struct CollectiveEpilogueFwdSm100 {

    using ElementOut = typename Ktraits::ElementOut;
    using TileShapeQK = typename Ktraits::TileShapeQK;
    using PipelineE = typename Ktraits::PipelineE;
    using SmemLayoutO = typename Ktraits::SmemLayoutO;

    // TileShape for epilogue: (M, K, _) where M and K come from TileShapeQK
    // This is a 3D shape like CUTLASS example 77 uses
    using TileShape = Shape<
        decltype(get<0>(TileShapeQK{})),  // M = 128
        decltype(get<2>(TileShapeQK{})),  // K = HeadDim
        _1                                 // batch dimension (tiled by 1)
    >;

    // Stride for O in GMEM: (seq_stride, dim_stride=1, batch_head_stride)
    using StrideO = cute::Stride<int64_t, _1, int64_t>;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    ///////////////////////////////////////////////////////////////////////////

    struct Arguments {
        ElementOut* ptr_O;
        int64_t stride_O_row;
        int64_t stride_O_head;
        int64_t stride_O_batch;
    };

    // TMA descriptor for storing O
    // SmemLayoutO is 3D (M, K, 2), slice with (_,_,_0{}) to get 2D for TMA
    using TMA_O = decltype(make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor((ElementOut*)nullptr, repeat_like(StrideO{}, 0), StrideO{}),
        SmemLayoutO{}(_, _, _0{})
    ));

    struct Params {
        TMA_O tma_store_o;
    };

    template <typename ProblemShape>
    static Params to_underlying_arguments(
        ProblemShape const& problem_shape,
        Arguments const& args,
        void* workspace
    ) {
        auto ptr_O = args.ptr_O;
        // Stride: (seq_stride, dim_stride=1, head_stride)
        // batch*heads are linearized in the 3rd dimension
        auto stride_O = make_stride(args.stride_O_row, _1{}, args.stride_O_head);
        // Problem shape for O: (seqlen_q, head_dim, batch*heads)
        auto problem_shape_O = select<0, 2, 3>(problem_shape);

        auto tma_store_o = make_tma_copy(
            SM90_TMA_STORE{},
            make_tensor(ptr_O, problem_shape_O, stride_O),
            SmemLayoutO{}(_, _, _0{})
        );

        return Params{tma_store_o};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
    }

    ///////////////////////////////////////////////////////////////////////////
    // Store function (executed by Epilogue warp)
    // Simplified approach: work with 2D slices after selecting batch dimension
    ///////////////////////////////////////////////////////////////////////////

    template <typename BlkCoord, typename ProblemShape, typename TensorStorage_>
    CUTLASS_DEVICE void store(
        BlkCoord const& blk_coord,
        ProblemShape const& problem_shape,
        Params const& params,
        Flash_fwd_params const& flash_params,
        TensorStorage_& shared_storage,
        PipelineE& pipeline,
        typename PipelineE::PipelineState& pipeline_consumer_state
    ) {
        using X = Underscore;

        uint32_t lane_predicate = cute::elect_one_sync();

        // Two Q blocks per CTA tile (ThreadShape = (2,1,1))
        int o0_index = 2 * get<0>(blk_coord);
        int o1_index = 2 * get<0>(blk_coord) + 1;
        int batch_head_idx = get<2>(blk_coord);

        // Get TMA tensor for O - problem_shape_O = (seqlen_q, head_dim, batch*heads)
        Tensor mO = params.tma_store_o.get_tma_tensor(select<0, 2, 3>(problem_shape));

        // First slice out the batch dimension to get a 2D tensor (seqlen_q, head_dim)
        Tensor mO_2d = mO(_, _, batch_head_idx);

        // Use 2D tile shape for 2D tensor
        using TileShape2D = Shape<
            decltype(get<0>(TileShapeQK{})),  // M = 128
            decltype(get<2>(TileShapeQK{}))   // K = HeadDim
        >;

        // local_tile with 2D TileShape on 2D tensor -> 4 modes: (TileM, TileK, num_tiles_m, num_tiles_k)
        // But num_tiles_k should be 1 since K = HeadDim
        Tensor gO_tiled = local_tile(mO_2d, TileShape2D{}, make_coord(_, _), Step<_1, X>{});
        // gO_tiled has shape: (TileM, TileK, num_tiles_m) since we only tile along M

        // Setup SMEM tensor for O - 3D layout (M, K, 2)
        Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});

        auto block_tma = params.tma_store_o.get_slice(0);
        Tensor tOsO = block_tma.partition_S(sO);
        Tensor tOgO = block_tma.partition_D(gO_tiled);

        auto pipeline_release_state = pipeline_consumer_state;

        // Wait for O0 from correction warps and store via TMA
        pipeline.consumer_wait(pipeline_consumer_state);
        ++pipeline_consumer_state;

        if (lane_predicate) {
            copy(params.tma_store_o, tOsO(_, _, _0{}), tOgO(_, _, o0_index));
        }
        tma_store_arrive();

        // Wait for O1 from correction warps and store via TMA
        pipeline.consumer_wait(pipeline_consumer_state);
        ++pipeline_consumer_state;

        if (lane_predicate) {
            copy(params.tma_store_o, tOsO(_, _, _1{}), tOgO(_, _, o1_index));
        }
        tma_store_arrive();

        // Wait for TMA stores to complete
        tma_store_wait<1>();

        pipeline.consumer_release(pipeline_release_state);
        ++pipeline_release_state;

        tma_store_wait<0>();

        pipeline.consumer_release(pipeline_release_state);
        ++pipeline_release_state;
    }
};

} // namespace flash
