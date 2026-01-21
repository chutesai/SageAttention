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
 * SM100 (B200/B300) Kernel Launch
 *
 * Uses the same launch infrastructure as SM120. The key insight is that
 * the SM120 blockscaled MMA atoms will work on SM100 when compiled with
 * -arch=sm_100a, as CUTLASS abstracts the hardware differences.
 */

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"

#include "../blackwell/static_switch.h"
#include "../blackwell/params.h"
#include "../blackwell/tile_scheduler.h"
#include "../blackwell/launch.h"  // SM120 launch infrastructure
#include "kernel_traits.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Template dispatch for SM100
//
// Uses SM100-specific kernel traits but launches via SM120 infrastructure.
// The SM100 kernel traits produce compatible types for the SM120 launch code.
///////////////////////////////////////////////////////////////////////////////

template <typename T, int Headdim, typename O = cutlass::bfloat16_t>
void run_mha_fwd_sm100_(Flash_fwd_params& params, cudaStream_t stream) {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.per_block_mean, per_block, [&] {
            if constexpr (Headdim == 64) {
                // Use the SM100 kernel traits with SM120 launch infrastructure
                using Ktraits = Flash_fwd_kernel_traits_sm100<64, 128, 128, 3, 1, per_block, T, O>;
                run_flash_fwd<Ktraits, Is_causal>(params, stream);
            } else if constexpr (Headdim == 128) {
                using Ktraits = Flash_fwd_kernel_traits_sm100<128, 128, 128, 3, 1, per_block, T, O>;
                run_flash_fwd<Ktraits, Is_causal>(params, stream);
            } else {
                static_assert(Headdim == 64 || Headdim == 128, "Unsupported Headdim for SM100");
            }
        });
    });
}

} // namespace flash
