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
 * SM100 (B200/B300) API for SageAttention3
 *
 * This file provides the Python binding for SM100 datacenter Blackwell GPUs.
 * It must be compiled with -arch=sm_100a flag.
 */

#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "../blackwell/params.h"
#include "../blackwell/static_switch.h"
#include "launch.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Set kernel parameters from PyTorch tensors
///////////////////////////////////////////////////////////////////////////////

void set_params_fprop_sm100(Flash_fwd_params& params,
                            const at::Tensor& q,
                            const at::Tensor& k,
                            const at::Tensor& v,
                            at::Tensor& out,
                            const at::Tensor& softmax_lse,
                            const at::Tensor& sfq,
                            const at::Tensor& sfk,
                            const at::Tensor& sfv,
                            const at::Tensor& delta_s,
                            float softmax_scale,
                            bool is_causal,
                            bool per_block_mean) {

    memset(&params, 0, sizeof(params));

    params.is_bf16 = q.dtype() == torch::kUInt8;  // Packed FP4
    params.is_causal = is_causal;
    params.per_block_mean = per_block_mean;

    // Q, K, V pointers and strides
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();

    params.q_row_stride = q.stride(2);
    params.k_row_stride = k.stride(2);
    params.v_row_stride = v.stride(1);
    params.q_head_stride = q.stride(1);
    params.k_head_stride = k.stride(1);
    params.v_head_stride = v.stride(1);
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);

    // Output pointer and strides
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(2);
    params.o_head_stride = out.stride(1);
    params.o_batch_stride = out.stride(0);

    // Softmax LSE
    params.softmax_lse_ptr = softmax_lse.data_ptr();

    // Scale factors (using the field names from params.h)
    params.sfq_ptr = sfq.data_ptr();
    params.sfk_ptr = sfk.data_ptr();
    params.sfv_ptr = sfv.data_ptr();

    // Scale factor strides
    params.sfq_row_stride = sfq.stride(2);
    params.sfk_row_stride = sfk.stride(2);
    params.sfv_row_stride = sfv.stride(1);  // V is transposed
    params.sfq_head_stride = sfq.stride(1);
    params.sfk_head_stride = sfk.stride(1);
    params.sfv_head_stride = sfv.stride(1);
    params.sfq_batch_stride = sfq.stride(0);
    params.sfk_batch_stride = sfk.stride(0);
    params.sfv_batch_stride = sfv.stride(0);

    // Delta_S (per-block means)
    params.delta_s_ptr = delta_s.data_ptr();
    params.ds_row_stride = delta_s.stride(2);
    params.ds_head_stride = delta_s.stride(1);
    params.ds_batch_stride = delta_s.stride(0);

    // Dimensions
    params.b = q.size(0);
    params.h = q.size(1);
    params.h_k = k.size(1);
    params.seqlen_q = q.size(2);
    params.seqlen_k = k.size(2);
    params.unpadded_seqlen_k = k.size(2);
    params.seqlen_s = delta_s.size(2);
    params.d = q.size(3) * 2;  // Packed FP4, actual dim is 2x

    // Softmax scale (log2 for faster exp2)
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
}

///////////////////////////////////////////////////////////////////////////////
// Forward pass entry point
///////////////////////////////////////////////////////////////////////////////

at::Tensor mha_fwd_sm100(at::Tensor& q,
                         const at::Tensor& k,
                         const at::Tensor& v,
                         at::Tensor& out,
                         const at::Tensor& sfq,
                         const at::Tensor& sfk,
                         const at::Tensor& sfv,
                         const at::Tensor& delta_s,
                         float softmax_scale,
                         bool is_causal,
                         bool per_block_mean) {

    auto dprops = at::cuda::getCurrentDeviceProperties();

    // Verify SM100 architecture
    bool is_sm100 = dprops->major == 10 && dprops->minor == 0;
    TORCH_CHECK(is_sm100, "SM100 kernel requires B200/B300 GPU (compute capability 10.0). "
                          "Got compute capability ", dprops->major, ".", dprops->minor);

    // Verify tensor properties
    TORCH_CHECK(q.dtype() == torch::kUInt8, "Q must be packed FP4 (uint8)");
    TORCH_CHECK(k.dtype() == torch::kUInt8, "K must be packed FP4 (uint8)");
    TORCH_CHECK(v.dtype() == torch::kUInt8, "V must be packed FP4 (uint8)");
    TORCH_CHECK(sfq.dtype() == torch::kFloat8_e4m3fn, "SFQ must be FP8 E4M3");
    TORCH_CHECK(sfk.dtype() == torch::kFloat8_e4m3fn, "SFK must be FP8 E4M3");
    TORCH_CHECK(sfv.dtype() == torch::kFloat8_e4m3fn, "SFV must be FP8 E4M3");

    TORCH_CHECK(q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "V must be on CUDA");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int num_heads = sizes[1];
    const int seqlen_q = sizes[2];
    const int head_dim_packed = sizes[3];
    const int head_dim = head_dim_packed * 2;  // FP4 is 4 bits, packed 2 per byte

    // SM100 with FP4 requires HeadDim >= 128 due to TMA load constraint
    // (TileShape_K must be divisible by 128 for FP4 data)
    TORCH_CHECK(head_dim == 128 || head_dim == 256,
                "SM100 kernel with FP4 requires head_dim >= 128 (TMA constraint). Got ", head_dim,
                ". For head_dim=64, use SM120 (RTX 5090) kernel or FP8 data format.");

    // Create output tensors if needed
    if (out.numel() == 0) {
        out = torch::empty({batch_size, num_heads, seqlen_q, head_dim},
                           q.options().dtype(torch::kBFloat16));
    }

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q},
                                    q.options().dtype(torch::kFloat32));

    // Set up kernel parameters
    Flash_fwd_params params;
    set_params_fprop_sm100(params, q, k, v, out, softmax_lse, sfq, sfk, sfv,
                           delta_s, softmax_scale, is_causal, per_block_mean);

    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Dispatch based on head dimension
    // Note: HeadDim=64 not supported on SM100 with FP4 due to TMA constraint
    if (head_dim == 128) {
        flash::run_mha_fwd_sm100_<uint8_t, 128, cutlass::bfloat16_t>(params, stream);
    } else if (head_dim == 256) {
        flash::run_mha_fwd_sm100_<uint8_t, 256, cutlass::bfloat16_t>(params, stream);
    } else {
        TORCH_CHECK(false, "Unsupported head_dim for SM100: ", head_dim);
    }

    return out;
}

///////////////////////////////////////////////////////////////////////////////
// Check if SM100 is supported
///////////////////////////////////////////////////////////////////////////////

bool is_sm100_supported() {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    return dprops->major == 10 && dprops->minor == 0;
}

///////////////////////////////////////////////////////////////////////////////
// Python bindings
///////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &mha_fwd_sm100, "SM100 Flash Attention Forward");
    m.def("is_supported", &is_sm100_supported, "Check if SM100 GPU is available");
}
