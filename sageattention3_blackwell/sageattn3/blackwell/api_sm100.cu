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
 */

#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cute/tensor.hpp>

#include <cutlass/float_subbyte.h>

#include <math.h>
#include <type_traits>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LASTDIM_CONTIGUOUS(x) TORCH_CHECK(x.stride(-1) == 1, #x " must have contiguous last dim")

__device__ __forceinline__ float fp4_to_float(uint8_t nibble) {
    cutlass::float_e2m1_t fp4 = cutlass::float_e2m1_t::bitcast(nibble);
    return float(fp4);
}

__device__ __forceinline__ float fp8_to_float(uint8_t byte) {
    __nv_fp8_e4m3 fp8 = reinterpret_cast<__nv_fp8_e4m3&>(byte);
    return float(fp8);
}

__device__ __forceinline__ float load_fp4_scaled(const uint8_t* packed, const uint8_t* sf, int d,
                                                 int token, int head_dim, int l,
                                                 int seqlen, int total_l) {
    uint8_t byte = packed[d >> 1];
    uint8_t nibble = (d & 1) ? (byte >> 4) : (byte & 0xF);
    float val = fp4_to_float(nibble);
    using Sm100Config = cutlass::detail::Sm1xxBlockScaledConfig<16>;
    auto layout_sfa = Sm100Config::tile_atom_to_shape_SFA(cute::make_shape(seqlen, head_dim / 16, total_l));
    int k_coord = d >> 4;
    int64_t offset = layout_sfa(cute::make_coord(token, k_coord, l));
    float scale = fp8_to_float(sf[offset]);
    return val * scale;
}

template <int HEAD_DIM, typename OutT>
__global__ void fp4_attn_sm100_kernel(
    const uint8_t* q,
    const uint8_t* k,
    const uint8_t* v,
    const uint8_t* sfq,
    const uint8_t* sfk,
    const uint8_t* sfv,
    const float* delta_s,
    OutT* out,
    float* softmax_lse,
    int seqlen_q,
    int seqlen_k,
    int unpadded_k,
    float softmax_scale,
    bool is_causal,
    bool per_block_mean,
    int num_heads,
    int total_l,
    int64_t stride_qb,
    int64_t stride_qh,
    int64_t stride_qq,
    int64_t stride_kb,
    int64_t stride_kh,
    int64_t stride_kk,
    int64_t stride_vb,
    int64_t stride_vh,
    int64_t stride_vk,
    int64_t stride_sfq_b,
    int64_t stride_sfq_h,
    int64_t stride_sfq_q,
    int64_t stride_sfk_b,
    int64_t stride_sfk_h,
    int64_t stride_sfk_k,
    int64_t stride_sfv_b,
    int64_t stride_sfv_h,
    int64_t stride_sfv_k,
    int64_t stride_out_b,
    int64_t stride_out_h,
    int64_t stride_out_q,
    int64_t stride_lse_b,
    int64_t stride_lse_h,
    int64_t stride_lse_q,
    int64_t stride_ds_b,
    int64_t stride_ds_h,
    int64_t stride_ds_m,
    int64_t stride_ds_k) {
    constexpr int kBlockM = 128;
    int q_idx = blockIdx.x * kBlockM + threadIdx.x;
    if (q_idx >= seqlen_q) {
        return;
    }

    int b = blockIdx.y;
    int h = blockIdx.z;
    int group_id = per_block_mean ? (q_idx / kBlockM) : 0;

    const uint8_t* q_ptr = q + b * stride_qb + h * stride_qh + q_idx * stride_qq;
    const uint8_t* sfq_ptr = sfq;
    const float* ds_ptr = delta_s + b * stride_ds_b + h * stride_ds_h + group_id * stride_ds_m;
    int l = b * num_heads + h;

    int max_k = is_causal ? (q_idx + 1 + seqlen_k - seqlen_q) : seqlen_k;
    int k_limit = unpadded_k < max_k ? unpadded_k : max_k;
    if (k_limit <= 0) {
        OutT* out_ptr = out + b * stride_out_b + h * stride_out_h + q_idx * stride_out_q;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            if constexpr (std::is_same<OutT, __half>::value) {
                out_ptr[d] = __float2half_rn(0.0f);
            } else {
                out_ptr[d] = __float2bfloat16_rn(0.0f);
            }
        }
        softmax_lse[b * stride_lse_b + h * stride_lse_h + q_idx * stride_lse_q] = INFINITY;
        return;
    }

    float max_logit = -INFINITY;
    for (int k_idx = 0; k_idx < k_limit; ++k_idx) {
        const uint8_t* k_ptr = k + b * stride_kb + h * stride_kh + k_idx * stride_kk;
        const uint8_t* sfk_ptr = sfk;
        float acc = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float q_val = load_fp4_scaled(q_ptr, sfq_ptr, d, q_idx, HEAD_DIM, l, seqlen_q, total_l);
            float k_val = load_fp4_scaled(k_ptr, sfk_ptr, d, k_idx, HEAD_DIM, l, seqlen_k, total_l);
            acc += q_val * k_val;
        }
        acc = (acc + ds_ptr[k_idx * stride_ds_k]) * softmax_scale;
        max_logit = acc > max_logit ? acc : max_logit;
    }

    float sum = 0.0f;
    float out_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        out_acc[d] = 0.0f;
    }

    for (int k_idx = 0; k_idx < k_limit; ++k_idx) {
        const uint8_t* k_ptr = k + b * stride_kb + h * stride_kh + k_idx * stride_kk;
        const uint8_t* sfk_ptr = sfk;
        const uint8_t* v_ptr = v + b * stride_vb + h * stride_vh + k_idx * stride_vk;
        const uint8_t* sfv_ptr = sfv;
        float acc = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float q_val = load_fp4_scaled(q_ptr, sfq_ptr, d, q_idx, HEAD_DIM, l, seqlen_q, total_l);
            float k_val = load_fp4_scaled(k_ptr, sfk_ptr, d, k_idx, HEAD_DIM, l, seqlen_k, total_l);
            acc += q_val * k_val;
        }
        acc = (acc + ds_ptr[k_idx * stride_ds_k]) * softmax_scale;
        float w = expf(acc - max_logit);
        sum += w;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v_val = load_fp4_scaled(v_ptr, sfv_ptr, d, k_idx, HEAD_DIM, l, seqlen_k, total_l);
            out_acc[d] += w * v_val;
        }
    }

    float inv_sum = (sum == 0.0f) ? 0.0f : 1.0f / sum;
    OutT* out_ptr = out + b * stride_out_b + h * stride_out_h + q_idx * stride_out_q;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        float out_val = out_acc[d] * inv_sum;
        if constexpr (std::is_same<OutT, __half>::value) {
            out_ptr[d] = __float2half_rn(out_val);
        } else {
            out_ptr[d] = __float2bfloat16_rn(out_val);
        }
    }

    float lse = (sum == 0.0f) ? INFINITY : max_logit + logf(sum);
    softmax_lse[b * stride_lse_b + h * stride_lse_h + q_idx * stride_lse_q] = lse;
}

template <typename OutT>
void run_fp4_attn_sm100(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& sfq,
    const at::Tensor& sfk,
    const at::Tensor& sfv,
    const at::Tensor& delta_s,
    at::Tensor& out,
    at::Tensor& softmax_lse,
    int unpadded_k,
    float softmax_scale,
    bool is_causal,
    bool per_block_mean) {
    const int seqlen_q = q.size(2);
    const int seqlen_k = k.size(2);
    constexpr int kBlockM = 128;
    dim3 block(kBlockM, 1, 1);
    dim3 grid((seqlen_q + kBlockM - 1) / kBlockM, q.size(0), q.size(1));

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    int64_t stride_qb = q.stride(0);
    int64_t stride_qh = q.stride(1);
    int64_t stride_qq = q.stride(2);
    int64_t stride_kb = k.stride(0);
    int64_t stride_kh = k.stride(1);
    int64_t stride_kk = k.stride(2);
    int64_t stride_vb = v.stride(0);
    int64_t stride_vh = v.stride(1);
    int64_t stride_vk = v.stride(2);
    int64_t stride_sfq_b = sfq.stride(0);
    int64_t stride_sfq_h = sfq.stride(1);
    int64_t stride_sfq_q = sfq.stride(2);
    int64_t stride_sfk_b = sfk.stride(0);
    int64_t stride_sfk_h = sfk.stride(1);
    int64_t stride_sfk_k = sfk.stride(2);
    int64_t stride_sfv_b = sfv.stride(0);
    int64_t stride_sfv_h = sfv.stride(1);
    int64_t stride_sfv_k = sfv.stride(2);
    int64_t stride_out_b = out.stride(0);
    int64_t stride_out_h = out.stride(1);
    int64_t stride_out_q = out.stride(2);
    int64_t stride_lse_b = softmax_lse.stride(0);
    int64_t stride_lse_h = softmax_lse.stride(1);
    int64_t stride_lse_q = softmax_lse.stride(2);
    int64_t stride_ds_b = delta_s.stride(0);
    int64_t stride_ds_h = delta_s.stride(1);
    int64_t stride_ds_m = delta_s.stride(2);
    int64_t stride_ds_k = delta_s.stride(3);

    int head_dim = q.size(3) * 2;
    if (head_dim == 64) {
        fp4_attn_sm100_kernel<64, OutT><<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(k.data_ptr()),
            reinterpret_cast<const uint8_t*>(v.data_ptr()),
            reinterpret_cast<const uint8_t*>(sfq.data_ptr()),
            reinterpret_cast<const uint8_t*>(sfk.data_ptr()),
            reinterpret_cast<const uint8_t*>(sfv.data_ptr()),
            reinterpret_cast<const float*>(delta_s.data_ptr()),
            reinterpret_cast<OutT*>(out.data_ptr()),
            reinterpret_cast<float*>(softmax_lse.data_ptr()),
            seqlen_q,
            seqlen_k,
            unpadded_k,
            softmax_scale,
            is_causal,
            per_block_mean,
            q.size(1),
            q.size(0) * q.size(1),
            stride_qb,
            stride_qh,
            stride_qq,
            stride_kb,
            stride_kh,
            stride_kk,
            stride_vb,
            stride_vh,
            stride_vk,
            stride_sfq_b,
            stride_sfq_h,
            stride_sfq_q,
            stride_sfk_b,
            stride_sfk_h,
            stride_sfk_k,
            stride_sfv_b,
            stride_sfv_h,
            stride_sfv_k,
            stride_out_b,
            stride_out_h,
            stride_out_q,
            stride_lse_b,
            stride_lse_h,
            stride_lse_q,
            stride_ds_b,
            stride_ds_h,
            stride_ds_m,
            stride_ds_k);
    } else if (head_dim == 128) {
        fp4_attn_sm100_kernel<128, OutT><<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(k.data_ptr()),
            reinterpret_cast<const uint8_t*>(v.data_ptr()),
            reinterpret_cast<const uint8_t*>(sfq.data_ptr()),
            reinterpret_cast<const uint8_t*>(sfk.data_ptr()),
            reinterpret_cast<const uint8_t*>(sfv.data_ptr()),
            reinterpret_cast<const float*>(delta_s.data_ptr()),
            reinterpret_cast<OutT*>(out.data_ptr()),
            reinterpret_cast<float*>(softmax_lse.data_ptr()),
            seqlen_q,
            seqlen_k,
            unpadded_k,
            softmax_scale,
            is_causal,
            per_block_mean,
            q.size(1),
            q.size(0) * q.size(1),
            stride_qb,
            stride_qh,
            stride_qq,
            stride_kb,
            stride_kh,
            stride_kk,
            stride_vb,
            stride_vh,
            stride_vk,
            stride_sfq_b,
            stride_sfq_h,
            stride_sfq_q,
            stride_sfk_b,
            stride_sfk_h,
            stride_sfk_k,
            stride_sfv_b,
            stride_sfv_h,
            stride_sfv_k,
            stride_out_b,
            stride_out_h,
            stride_out_q,
            stride_lse_b,
            stride_lse_h,
            stride_lse_q,
            stride_ds_b,
            stride_ds_h,
            stride_ds_m,
            stride_ds_k);
    } else {
        TORCH_CHECK(false, "Unsupported head dim: ", head_dim);
    }
}

std::vector<at::Tensor>
mha_fwd_sm100(
    at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &sfq,
    const at::Tensor &sfk,
    const at::Tensor &sfv,
    const at::Tensor &delta_s,
    int unpadded_k,
    c10::optional<at::Tensor> &out_,
    const float softmax_scale,
    bool is_causal,
    bool per_block_mean,
    bool is_bf16) {
    (void)out_;

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm100 = dprops->major == 10 && dprops->minor == 0;
    TORCH_CHECK(is_sm100, "only supports Blackwell SM100 GPUs.");

    TORCH_CHECK(q.dtype() == torch::kUInt8, "q dtype must be uint8");
    TORCH_CHECK(k.dtype() == q.dtype(), "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q.dtype(), "query and value must have the same dtype");
    TORCH_CHECK(sfq.dtype() == torch::kFloat8_e4m3fn, "sfq dtype must be float8_e4m3fn");
    TORCH_CHECK(sfk.dtype() == sfq.dtype(), "sfq and sfk must have the same dtype");
    TORCH_CHECK(sfv.dtype() == sfq.dtype(), "sfq and sfv must have the same dtype");
    TORCH_CHECK(delta_s.dtype() == torch::kFloat, "delta_s must be float32");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(sfq);
    CHECK_DEVICE(sfk);
    CHECK_DEVICE(sfv);
    CHECK_DEVICE(delta_s);

    CHECK_LASTDIM_CONTIGUOUS(q);
    CHECK_LASTDIM_CONTIGUOUS(k);
    CHECK_LASTDIM_CONTIGUOUS(v);
    CHECK_LASTDIM_CONTIGUOUS(sfq);
    CHECK_LASTDIM_CONTIGUOUS(sfk);
    CHECK_LASTDIM_CONTIGUOUS(sfv);
    CHECK_LASTDIM_CONTIGUOUS(delta_s);

    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(sfq);
    CHECK_CONTIGUOUS(sfk);
    CHECK_CONTIGUOUS(sfv);
    CHECK_CONTIGUOUS(delta_s);

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int num_heads = sizes[1];
    const int seqlen_q = sizes[2];
    const int head_size_og = sizes[3];
    const int unpacked_head_size = head_size_og * 2;
    const int seqlen_k = k.size(2);
    const int num_heads_k = k.size(1);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads == num_heads_k, "We do not support MQA/GQA yet");
    TORCH_CHECK(unpacked_head_size == 64 || unpacked_head_size == 128,
                "Only support head size 64 and 128 for SM100");

    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch size mismatch");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "head count mismatch");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "head dim mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "key/value sequence length mismatch");

    auto dtype = is_bf16 ? at::ScalarType::BFloat16 : at::ScalarType::Half;
    auto opts = q.options();
    at::Tensor out = torch::empty({batch_size, num_heads, seqlen_q, unpacked_head_size}, opts.dtype(dtype));
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    if (is_bf16) {
        run_fp4_attn_sm100<__nv_bfloat16>(
            q, k, v, sfq, sfk, sfv, delta_s,
            out, softmax_lse, unpadded_k, softmax_scale, is_causal, per_block_mean);
    } else {
        run_fp4_attn_sm100<__half>(
            q, k, v, sfq, sfk, sfv, delta_s,
            out, softmax_lse, unpadded_k, softmax_scale, is_causal, per_block_mean);
    }

    return {out, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FP4 Attention SM100";
    m.def("fwd", &mha_fwd_sm100, "Forward pass");
}
