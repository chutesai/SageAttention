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
 * SM100 (B200/B300) CuTe extensions for FlashAttention.
 *
 * For now, this just includes the SM120 cute_extension since we're
 * reusing that infrastructure.
 */

#pragma once

// Include the SM120 cute_extension directly
#include "../blackwell/cute_extension.h"

namespace flash {

// Type aliases that match SM100 requirements
using ElementFP4 = cutlass::float_e2m1_t;      // FP4 E2M1
using ElementSF = cutlass::float_ue4m3_t;      // FP8 E4M3 scale factor
using ElementAccum = float;                     // FP32 accumulator
using ElementOut = cutlass::bfloat16_t;         // BF16 output

} // namespace flash
