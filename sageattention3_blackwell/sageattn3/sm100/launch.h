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
 * SM100 (B200/B300) Kernel Launch Infrastructure
 *
 * This file handles the kernel launch for SM100 warp-specialized attention.
 */

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"

#include "../blackwell/static_switch.h"
#include "../blackwell/params.h"
#include "../blackwell/tile_scheduler.h"
#include "kernel_traits.h"
#include "kernel_ws.h"
#include "mainloop_tmem_ws.h"
#include "epilogue_tmem_ws.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 Kernel Launch
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
void run_flash_fwd_sm100(Flash_fwd_params& params, cudaStream_t stream) {

    using Element = typename Ktraits::Element;
    using ElementOut = typename Ktraits::ElementOut;
    using TileShape = typename Ktraits::TileShape_MNK;

    using CollectiveMainloop = CollectiveMainloopFwdSm100<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwdSm100<Ktraits>;
    using TileScheduler = SingleTileScheduler;

    // Problem shape: (seqlen_q, seqlen_k, head_dim, batch * num_heads)
    auto problem_shape = make_shape(
        params.seqlen_q,
        params.seqlen_k,
        params.d,
        params.b * params.h
    );

    // Setup strides - 3D: (seq, dim, batch*head)
    // For FP4 packed data, q_row_stride is in bytes (dim/2)
    auto stride_Q = make_stride(
        static_cast<int64_t>(params.q_row_stride),
        _1{},
        static_cast<int64_t>(params.q_head_stride));
    auto stride_K = make_stride(
        static_cast<int64_t>(params.k_row_stride),
        _1{},
        static_cast<int64_t>(params.k_head_stride));
    // V is transposed: (dim, seq, batch*head)
    auto stride_V = make_stride(
        _1{},
        static_cast<int64_t>(params.v_row_stride),
        static_cast<int64_t>(params.v_head_stride));

    // Setup mainloop arguments
    typename CollectiveMainloop::Arguments mainloop_args{
        reinterpret_cast<Element const*>(params.q_ptr),
        stride_Q,
        reinterpret_cast<Element const*>(params.k_ptr),
        stride_K,
        reinterpret_cast<Element const*>(params.v_ptr),
        stride_V,
        params.scale_softmax
    };

    // Setup epilogue arguments
    typename CollectiveEpilogue::Arguments epilogue_args{
        reinterpret_cast<ElementOut*>(params.o_ptr),
        params.o_row_stride,
        params.o_head_stride,
        params.o_batch_stride
    };

    // Convert to underlying parameters
    auto mainloop_params = CollectiveMainloop::to_underlying_arguments(
        problem_shape, mainloop_args, nullptr);
    auto epilogue_params = CollectiveEpilogue::to_underlying_arguments(
        problem_shape, epilogue_args, nullptr);

    // Setup tile scheduler
    int num_m_blocks = cute::ceil_div(params.seqlen_q, get<0>(TileShape{}));

    typename TileScheduler::Arguments scheduler_args{
        num_m_blocks,
        params.h,
        params.b
    };

    auto scheduler_params = TileScheduler::to_underlying_arguments(scheduler_args);

    // Calculate shared memory size
    int smem_size = sizeof(typename Ktraits::SharedStorage);

    // Set max dynamic shared memory if needed
    auto kernel = compute_attn_ws_sm100<Ktraits, Is_causal, TileScheduler>;

    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    // Calculate grid dimensions
    dim3 grid(num_m_blocks, params.h, params.b);
    dim3 block(Ktraits::kNThreads, 1, 1);

    // Launch kernel
    kernel<<<grid, block, smem_size, stream>>>(
        params, mainloop_params, epilogue_params, scheduler_params);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "SM100 kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

///////////////////////////////////////////////////////////////////////////////
// Template dispatch for SM100
///////////////////////////////////////////////////////////////////////////////

template <typename T, int Headdim, typename O = cutlass::bfloat16_t>
void run_mha_fwd_sm100_(Flash_fwd_params& params, cudaStream_t stream) {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.per_block_mean, per_block, [&] {
            // SM100 with FP4:
            //   - kBlockM=256 because TileShapeQK = TileShape / ThreadShape<2,1,1>, giving M=128
            //   - HeadDim >= 128 required by FP4 TMA load constraint (TileShape_K % 128 == 0)
            if constexpr (Headdim == 128) {
                using Ktraits = Flash_fwd_kernel_traits_sm100<128, 256, 128, 3, 1, per_block, T, O>;
                run_flash_fwd_sm100<Ktraits, Is_causal>(params, stream);
            } else if constexpr (Headdim == 256) {
                using Ktraits = Flash_fwd_kernel_traits_sm100<256, 256, 128, 3, 1, per_block, T, O>;
                run_flash_fwd_sm100<Ktraits, Is_causal>(params, stream);
            } else {
                // HeadDim=64 not supported on SM100 with FP4 due to TMA load constraint
                static_assert(Headdim == 128 || Headdim == 256,
                    "SM100 with FP4 requires HeadDim >= 128 (TMA load constraint)");
            }
        });
    });
}

} // namespace flash
