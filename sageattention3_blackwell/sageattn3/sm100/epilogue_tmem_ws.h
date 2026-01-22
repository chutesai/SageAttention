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
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits>
struct CollectiveEpilogueFwdSm100 {

    using ElementOut = typename Ktraits::ElementOut;
    using TileShape = typename Ktraits::TileShape_MNK;
    using SmemLayoutO = typename Ktraits::SmemLayoutO;
    using PipelineE = typename Ktraits::PipelineE;

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
    using TMA_O = decltype(make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor((ElementOut*)nullptr, repeat_like(cute::Stride<int64_t, _1, int64_t>{}, 0),
                    cute::Stride<int64_t, _1, int64_t>{}),
        SmemLayoutO{}(_, _, _0{})));

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
        auto stride_O = make_stride(args.stride_O_row, _1{}, args.stride_O_head);
        auto problem_shape_O = select<0, 2, 3>(problem_shape);

        auto tma_store_o = make_tma_copy(
            SM90_TMA_STORE{},
            make_tensor(ptr_O, problem_shape_O, stride_O),
            SmemLayoutO{}(_, _, _0{}));

        return Params{tma_store_o};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
    }

    ///////////////////////////////////////////////////////////////////////////
    // Store function (executed by Epilogue warp)
    ///////////////////////////////////////////////////////////////////////////

    template <typename BlkCoord, typename ProblemShape, typename SharedStorage>
    CUTLASS_DEVICE void store(
        BlkCoord const& blk_coord,
        ProblemShape const& problem_shape,
        Params const& params,
        Flash_fwd_params const& flash_params,
        SharedStorage& storage,
        PipelineE& pipeline,
        typename PipelineE::PipelineState& pipeline_consumer_state
    ) {
        using X = Underscore;

        uint32_t lane_predicate = cute::elect_one_sync();

        int o0_index = 2 * get<0>(blk_coord);
        int o1_index = 2 * get<0>(blk_coord) + 1;

        // Get TMA tensor for O
        Tensor mO = params.tma_store_o.get_tma_tensor(select<0, 2, 3>(problem_shape));
        Tensor gO = local_tile(mO, TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{});

        // Setup SMEM tensor for O
        Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), SmemLayoutO{});

        auto block_tma = params.tma_store_o.get_slice(0);
        Tensor tOsO = block_tma.partition_S(sO);
        Tensor tOgO = block_tma.partition_D(gO(_, _, _, _0{}, get<2>(blk_coord)));

        auto pipeline_release_state = pipeline_consumer_state;

        // Wait for O0 from correction warps and store via TMA
        pipeline.consumer_wait(pipeline_consumer_state);
        ++pipeline_consumer_state;

        if (lane_predicate) {
            copy(params.tma_store_o, tOsO(_, _, _, _0{}), tOgO(_, _, _, o0_index));
        }
        tma_store_arrive();

        // Wait for O1 from correction warps and store via TMA
        pipeline.consumer_wait(pipeline_consumer_state);
        ++pipeline_consumer_state;

        if (lane_predicate) {
            copy(params.tma_store_o, tOsO(_, _, _, _1{}), tOgO(_, _, _, o1_index));
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
