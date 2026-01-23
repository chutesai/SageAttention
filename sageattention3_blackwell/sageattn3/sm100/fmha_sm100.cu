/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) Flash Multi-Head Attention using CUTLASS Example 77
 *
 * High-performance FMHA implementation for B200/B300 supporting:
 * - BF16: Standard precision for video generation (Wan2.2)
 * - FP8 E4M3: 2x tensor core throughput for maximum performance
 *
 * Uses NVIDIA's official CUTLASS example 77 warp-specialized kernels.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

// CUTLASS Example 77 includes
#include "device/fmha.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

using namespace cute;

// Bring in specific types from CUTLASS namespaces
namespace fmha_kernel = cutlass::fmha::kernel;
namespace fmha_collective = cutlass::fmha::collective;

///////////////////////////////////////////////////////////////////////////////
// Common Types
///////////////////////////////////////////////////////////////////////////////

// Problem shape format: (seqlen_q, seqlen_k, head_dim, ((h_groups, h_per_group), batch))
using ProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

// Stride format (matches CUTLASS example 77):
// Q/O: (seq_stride, 1, ((head_group_stride, head_stride), batch_stride))
// K/V: (seq_stride, 1, ((0, head_stride), batch_stride))
using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;

// Mask types
using MaskNone = fmha_collective::NoMask;
using MaskCausal = fmha_collective::CausalMask<true>;

///////////////////////////////////////////////////////////////////////////////
// BF16 FMHA Kernel Types
///////////////////////////////////////////////////////////////////////////////

using ElementBF16 = cutlass::bfloat16_t;
using TileShapeBF16 = Shape<_256, _128, _128>;

template <typename ActiveMask>
struct FmhaKernelBF16 {
    using TileScheduler = fmha_kernel::PersistentTileScheduler;

    using Mainloop = fmha_collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
        ElementBF16, float, float,
        TileShapeBF16, StrideQ, StrideK, StrideV,
        ActiveMask
    >;

    using Epilogue = fmha_collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementBF16, float,
        typename Mainloop::TileShapePV,
        StrideO, StrideLSE
    >;

    using Kernel = fmha_kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
        ProblemShape, Mainloop, Epilogue, TileScheduler
    >;

    using Operation = cutlass::fmha::device::FMHA<Kernel>;
};

using FmhaBF16NoMask = FmhaKernelBF16<MaskNone>;
using FmhaBF16Causal = FmhaKernelBF16<MaskCausal>;

///////////////////////////////////////////////////////////////////////////////
// FP8 E4M3 FMHA Kernel Types - 2x tensor core throughput
///////////////////////////////////////////////////////////////////////////////

using ElementFP8 = cutlass::float_e4m3_t;
// FP8 uses 256x256 tile shape for optimal performance
using TileShapeFP8 = Shape<_256, _256, _128>;

template <typename ActiveMask>
struct FmhaKernelFP8 {
    using TileScheduler = fmha_kernel::PersistentTileScheduler;

    using Mainloop = fmha_collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
        ElementFP8, float, float,
        TileShapeFP8, StrideQ, StrideK, StrideV,
        ActiveMask
    >;

    using Epilogue = fmha_collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementFP8, float,
        typename Mainloop::TileShapePV,
        StrideO, StrideLSE
    >;

    using Kernel = fmha_kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
        ProblemShape, Mainloop, Epilogue, TileScheduler
    >;

    using Operation = cutlass::fmha::device::FMHA<Kernel>;
};

using FmhaFP8NoMask = FmhaKernelFP8<MaskNone>;
using FmhaFP8Causal = FmhaKernelFP8<MaskCausal>;

// Backwards compat aliases
using Element = ElementBF16;
using ElementOut = ElementBF16;
using ElementAccumulatorQK = float;
using ElementAccumulatorPV = float;
using TileShape = TileShapeBF16;
using FmhaNoMask = FmhaBF16NoMask;
using FmhaCausal = FmhaBF16Causal;

///////////////////////////////////////////////////////////////////////////////
// Kernel Launch Functions
///////////////////////////////////////////////////////////////////////////////

// Generic launch for BF16
template <typename FmhaType>
torch::Tensor run_fmha_bf16_impl(
    torch::Tensor& q,      // [batch, heads, seqlen_q, head_dim]
    torch::Tensor& k,      // [batch, heads, seqlen_k, head_dim]
    torch::Tensor& v,      // [batch, heads, seqlen_k, head_dim]
    float scale
) {
    using Operation = typename FmhaType::Operation;

    const int batch = q.size(0);
    const int heads = q.size(1);
    const int seqlen_q = q.size(2);
    const int seqlen_k = k.size(2);
    const int head_dim = q.size(3);

    TORCH_CHECK(head_dim == 128, "SM100 FMHA currently requires head_dim=128");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "SM100 BF16 FMHA requires BF16 input");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");

    auto out = torch::empty_like(q);

    const int h_groups = 1;
    const int h_per_group = heads;

    const int q_seq_stride = head_dim;
    const int q_head_stride = seqlen_q * head_dim;
    const int q_batch_stride = heads * seqlen_q * head_dim;

    const int k_seq_stride = head_dim;
    const int k_head_stride = seqlen_k * head_dim;
    const int k_batch_stride = heads * seqlen_k * head_dim;

    auto stride_Q = make_tuple(
        q_seq_stride, _1{},
        make_tuple(make_tuple(q_head_stride, h_groups * q_head_stride), q_batch_stride)
    );
    auto stride_K = make_tuple(
        k_seq_stride, _1{},
        make_tuple(make_tuple(_0{}, k_head_stride), k_batch_stride)
    );
    auto stride_V = stride_K;
    auto stride_O = stride_Q;
    auto stride_LSE = make_tuple(
        _1{},
        make_tuple(make_tuple(seqlen_q, h_groups * seqlen_q), seqlen_q * heads)
    );

    auto problem_shape = make_tuple(
        seqlen_q, seqlen_k, head_dim,
        make_tuple(make_tuple(h_groups, h_per_group), batch)
    );

    auto lse = torch::empty({batch, heads, seqlen_q}, q.options().dtype(torch::kFloat32));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = q.device().index();
    cudaDeviceGetAttribute(&hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);

    // Mainloop arguments structure:
    // { load: {ptr_Q, stride_Q, ptr_K, stride_K, ptr_V, stride_V},
    //   scale_softmax, scale_q, scale_k, scale_v, inv_scale_o }
    typename Operation::Arguments arguments{
        problem_shape,
        {
            {  // Load arguments
                reinterpret_cast<ElementBF16 const*>(q.data_ptr()),
                stride_Q,
                reinterpret_cast<ElementBF16 const*>(k.data_ptr()),
                stride_K,
                reinterpret_cast<ElementBF16 const*>(v.data_ptr()),
                stride_V
            },
            scale,   // scale_softmax
            1.0f,    // scale_q (no dequantization for BF16)
            1.0f,    // scale_k
            1.0f,    // scale_v
            1.0f     // inv_scale_o
        },
        {
            reinterpret_cast<ElementBF16*>(out.data_ptr()),
            stride_O,
            reinterpret_cast<float*>(lse.data_ptr()),
            stride_LSE
        },
        hw_info
    };

    Operation op;
    auto status = op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SM100 BF16 FMHA kernel cannot implement this problem: ", int(status));

    size_t workspace_size = Operation::get_workspace_size(arguments);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
        q.options().dtype(torch::kUInt8));

    status = op.initialize(arguments, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SM100 BF16 FMHA kernel initialization failed: ", int(status));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    status = op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SM100 BF16 FMHA kernel launch failed: ", int(status));

    return out;
}

// Generic launch for FP8 E4M3
template <typename FmhaType>
torch::Tensor run_fmha_fp8_impl(
    torch::Tensor& q,      // [batch, heads, seqlen_q, head_dim] in FP8
    torch::Tensor& k,      // [batch, heads, seqlen_k, head_dim] in FP8
    torch::Tensor& v,      // [batch, heads, seqlen_k, head_dim] in FP8
    float scale,
    float scale_q,         // Dequantization scale for Q
    float scale_k,         // Dequantization scale for K
    float scale_v          // Dequantization scale for V
) {
    using Operation = typename FmhaType::Operation;

    const int batch = q.size(0);
    const int heads = q.size(1);
    const int seqlen_q = q.size(2);
    const int seqlen_k = k.size(2);
    const int head_dim = q.size(3);

    TORCH_CHECK(head_dim == 128, "SM100 FP8 FMHA currently requires head_dim=128");
    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn, "SM100 FP8 FMHA requires FP8 E4M3 input");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");

    // Output is FP8 as well
    auto out = torch::empty_like(q);

    const int h_groups = 1;
    const int h_per_group = heads;

    const int q_seq_stride = head_dim;
    const int q_head_stride = seqlen_q * head_dim;
    const int q_batch_stride = heads * seqlen_q * head_dim;

    const int k_seq_stride = head_dim;
    const int k_head_stride = seqlen_k * head_dim;
    const int k_batch_stride = heads * seqlen_k * head_dim;

    auto stride_Q = make_tuple(
        q_seq_stride, _1{},
        make_tuple(make_tuple(q_head_stride, h_groups * q_head_stride), q_batch_stride)
    );
    auto stride_K = make_tuple(
        k_seq_stride, _1{},
        make_tuple(make_tuple(_0{}, k_head_stride), k_batch_stride)
    );
    auto stride_V = stride_K;
    auto stride_O = stride_Q;
    auto stride_LSE = make_tuple(
        _1{},
        make_tuple(make_tuple(seqlen_q, h_groups * seqlen_q), seqlen_q * heads)
    );

    auto problem_shape = make_tuple(
        seqlen_q, seqlen_k, head_dim,
        make_tuple(make_tuple(h_groups, h_per_group), batch)
    );

    auto lse = torch::empty({batch, heads, seqlen_q}, q.options().dtype(torch::kFloat32));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = q.device().index();
    cudaDeviceGetAttribute(&hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);

    // Mainloop arguments structure:
    // { load: {ptr_Q, stride_Q, ptr_K, stride_K, ptr_V, stride_V},
    //   scale_softmax, scale_q, scale_k, scale_v, inv_scale_o }
    typename Operation::Arguments arguments{
        problem_shape,
        {
            {  // Load arguments
                reinterpret_cast<ElementFP8 const*>(q.data_ptr()),
                stride_Q,
                reinterpret_cast<ElementFP8 const*>(k.data_ptr()),
                stride_K,
                reinterpret_cast<ElementFP8 const*>(v.data_ptr()),
                stride_V
            },
            scale,      // scale_softmax
            scale_q,    // scale_q for dequantization
            scale_k,    // scale_k for dequantization
            scale_v,    // scale_v for dequantization
            1.0f        // inv_scale_o (output is FP8, will need requantization)
        },
        {
            reinterpret_cast<ElementFP8*>(out.data_ptr()),
            stride_O,
            reinterpret_cast<float*>(lse.data_ptr()),
            stride_LSE
        },
        hw_info
    };

    Operation op;
    auto status = op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SM100 FP8 FMHA kernel cannot implement this problem: ", int(status));

    size_t workspace_size = Operation::get_workspace_size(arguments);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
        q.options().dtype(torch::kUInt8));

    status = op.initialize(arguments, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SM100 FP8 FMHA kernel initialization failed: ", int(status));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    status = op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SM100 FP8 FMHA kernel launch failed: ", int(status));

    return out;
}

// Legacy wrapper for backward compatibility
template <typename FmhaType>
torch::Tensor run_fmha_impl(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    float scale
) {
    return run_fmha_bf16_impl<FmhaType>(q, k, v, scale);
}

///////////////////////////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////////////////////////

// BF16 forward pass
torch::Tensor fmha_fwd_bf16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    float scale
) {
    TORCH_CHECK(q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "V must be on CUDA");

    at::cuda::CUDAGuard device_guard(q.device());

    if (is_causal) {
        return run_fmha_bf16_impl<FmhaBF16Causal>(q, k, v, scale);
    }
    return run_fmha_bf16_impl<FmhaBF16NoMask>(q, k, v, scale);
}

// FP8 E4M3 forward pass - 2x faster tensor core throughput
torch::Tensor fmha_fwd_fp8(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    float scale,
    float scale_q,
    float scale_k,
    float scale_v
) {
    TORCH_CHECK(q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "V must be on CUDA");

    at::cuda::CUDAGuard device_guard(q.device());

    if (is_causal) {
        return run_fmha_fp8_impl<FmhaFP8Causal>(q, k, v, scale, scale_q, scale_k, scale_v);
    }
    return run_fmha_fp8_impl<FmhaFP8NoMask>(q, k, v, scale, scale_q, scale_k, scale_v);
}

// Auto-quantize BF16 to FP8 and run FP8 attention (convenience function)
torch::Tensor fmha_fwd_fp8_from_bf16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    float scale
) {
    TORCH_CHECK(q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "Input must be BF16 for auto-quantization");

    at::cuda::CUDAGuard device_guard(q.device());

    // Compute per-tensor scales for dynamic quantization
    float q_absmax = q.abs().max().item<float>();
    float k_absmax = k.abs().max().item<float>();
    float v_absmax = v.abs().max().item<float>();

    // FP8 E4M3 max value is 448
    const float fp8_max = 448.0f;
    float scale_q = q_absmax / fp8_max;
    float scale_k = k_absmax / fp8_max;
    float scale_v = v_absmax / fp8_max;

    // Quantize to FP8
    auto q_fp8 = (q / scale_q).to(torch::kFloat8_e4m3fn);
    auto k_fp8 = (k / scale_k).to(torch::kFloat8_e4m3fn);
    auto v_fp8 = (v / scale_v).to(torch::kFloat8_e4m3fn);

    // Run FP8 attention
    auto out_fp8 = fmha_fwd_fp8(q_fp8, k_fp8, v_fp8, is_causal, scale, scale_q, scale_k, scale_v);

    // Dequantize output back to BF16 (scale is already applied in kernel)
    return out_fp8.to(torch::kBFloat16);
}

// Legacy wrapper for backward compatibility
torch::Tensor fmha_fwd_sm100(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    float scale
) {
    return fmha_fwd_bf16(q, k, v, is_causal, scale);
}

torch::Tensor fmha_fwd_sm100_auto_scale(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(q.size(-1)));
    return fmha_fwd_bf16(q, k, v, is_causal, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // BF16 functions
    m.def("fwd", &fmha_fwd_sm100, "SM100 Flash Multi-Head Attention Forward (BF16)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("is_causal") = false, py::arg("scale") = 1.0f);
    m.def("fwd_auto_scale", &fmha_fwd_sm100_auto_scale,
          "SM100 Flash MHA Forward (BF16, auto-compute scale)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("is_causal") = false);
    m.def("fwd_bf16", &fmha_fwd_bf16, "SM100 Flash MHA Forward (BF16)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("is_causal") = false, py::arg("scale") = 1.0f);

    // FP8 functions - 2x tensor core throughput
    m.def("fwd_fp8", &fmha_fwd_fp8, "SM100 Flash MHA Forward (FP8 E4M3, 2x faster)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("is_causal") = false, py::arg("scale") = 1.0f,
          py::arg("scale_q") = 1.0f, py::arg("scale_k") = 1.0f, py::arg("scale_v") = 1.0f);
    m.def("fwd_fp8_from_bf16", &fmha_fwd_fp8_from_bf16,
          "SM100 Flash MHA Forward (auto-quantize BF16->FP8, 2x faster)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("is_causal") = false, py::arg("scale") = 1.0f);
}
