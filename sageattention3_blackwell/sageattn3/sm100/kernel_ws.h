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
 * For now, this directly uses the SM120 kernel implementation.
 * The kernel code is architecture-agnostic and relies on CUTLASS
 * to emit the correct instructions for SM100 vs SM120.
 */

#pragma once

// Include the SM120 kernel directly
#include "../blackwell/kernel_ws.h"
#include "kernel_traits.h"
#include "mainloop_tmem_ws.h"
#include "epilogue_tmem_ws.h"

namespace flash {

// For SM100, we use the same kernel as SM120
// The hardware differences are handled by CUTLASS internally
template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNThreads, 1)
compute_attn_ws_sm100(
    CUTE_GRID_CONSTANT Flash_fwd_params const params,
    CUTE_GRID_CONSTANT typename CollectiveMainloopFwdSm100<Ktraits, Is_causal>::Params const mainloop_params,
    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwdSm100<Ktraits>::Params const epilogue_params,
    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
) {
    // Forward to the SM120 kernel implementation
    // The kernel code is architecture-agnostic
    compute_attn_ws<Ktraits, Is_causal, TileScheduler>(
        params, mainloop_params, epilogue_params, scheduler_params);
}

} // namespace flash
