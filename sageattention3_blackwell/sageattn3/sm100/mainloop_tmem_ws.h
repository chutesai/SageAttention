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
 * SM100 (B200/B300) Mainloop for FlashAttention.
 *
 * For now, this is a direct copy of the SM120 mainloop pattern.
 * The key insight is that the SM120 blockscaled MMA atoms will work on SM100
 * when compiled with -arch=sm_100a, as CUTLASS abstracts the hardware differences.
 */

#pragma once

// Include the SM120 mainloop directly
#include "../blackwell/mainloop_tma_ws.h"

namespace flash {

// For SM100, we use the same mainloop as SM120
// The hardware differences are handled by CUTLASS internally
template <typename Ktraits, bool Is_causal>
using CollectiveMainloopFwdSm100 = CollectiveMainloopFwd<Ktraits, Is_causal>;

} // namespace flash
