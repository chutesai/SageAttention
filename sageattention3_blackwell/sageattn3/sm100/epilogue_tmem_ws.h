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
 * For now, this is a direct copy of the SM120 epilogue pattern.
 */

#pragma once

// Include the SM120 epilogue directly
#include "../blackwell/epilogue_tma_ws.h"

namespace flash {

// For SM100, we use the same epilogue as SM120
template <typename Ktraits>
using CollectiveEpilogueFwdSm100 = CollectiveEpilogueFwd<Ktraits>;

} // namespace flash
