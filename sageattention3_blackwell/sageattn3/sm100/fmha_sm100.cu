/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) Flash Multi-Head Attention using CUTLASS Example 77
 *
 * This is a high-performance BF16 FMHA implementation for B200/B300
 * using NVIDIA's official CUTLASS example 77 warp-specialized kernels.
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
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;

///////////////////////////////////////////////////////////////////////////////
// SM100 FMHA Kernel Types (matching CUTLASS example 77 exactly)
///////////////////////////////////////////////////////////////////////////////

// Use BF16 for Wan2.2 and most video generation models
using Element = cutlass::bfloat16_t;
using ElementAccumulatorQK = float;
using ElementAccumulatorPV = float;
using ElementOut = cutlass::bfloat16_t;

// Tile shape: 256x128 for BF16
using TileShape = Shape<_256, _128, _128>;

// Problem shape format: (seqlen_q, seqlen_k, head_dim, ((h_groups, h_per_group), batch))
// For standard MHA: h_groups=1, h_per_group=num_heads
using ProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

// Stride format (matches CUTLASS example 77):
// Q/O: (seq_stride, 1, ((head_group_stride, head_stride), batch_stride))
// K/V: (seq_stride, 1, ((0, head_stride), batch_stride))  -- 0 for MQA broadcasting
using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;

// Build kernel types for causal and non-causal variants
template <typename Mask>
struct FmhaKernel {
    using TileScheduler = PersistentTileScheduler;

    using Mainloop = Sm100FmhaFwdMainloopTmaWarpspecialized<
        Element, ElementAccumulatorQK, ElementAccumulatorPV,
        TileShape, StrideQ, StrideK, StrideV,
        Mask
    >;

    using Epilogue = Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementOut, ElementAccumulatorPV,
        typename Mainloop::TileShapePV,
        StrideO, StrideLSE
    >;

    using Kernel = Sm100FmhaFwdKernelTmaWarpspecialized<
        ProblemShape,
        Mainloop,
        Epilogue,
        TileScheduler
    >;

    using Operation = cutlass::fmha::device::FMHA<Kernel>;
};

using FmhaNoMask = FmhaKernel<NoMask>;
using FmhaCausal = FmhaKernel<CausalMask>;

///////////////////////////////////////////////////////////////////////////////
// Kernel Launch Function
///////////////////////////////////////////////////////////////////////////////

template <typename FmhaType>
torch::Tensor run_fmha_impl(
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
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "SM100 FMHA requires BF16 input");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");

    // Allocate output
    auto out = torch::empty_like(q);

    // For standard MHA: h_groups=1, h_per_group=heads
    // This gives the standard attention behavior
    const int h_groups = 1;
    const int h_per_group = heads;

    // Compute strides for BHSD layout (batch, heads, seq, dim)
    // In BHSD layout with contiguous tensors:
    // - dim stride = 1
    // - seq stride = head_dim
    // - head stride = seqlen * head_dim
    // - batch stride = heads * seqlen * head_dim
    const int q_seq_stride = head_dim;
    const int q_head_stride = seqlen_q * head_dim;
    const int q_batch_stride = heads * seqlen_q * head_dim;

    const int k_seq_stride = head_dim;
    const int k_head_stride = seqlen_k * head_dim;
    const int k_batch_stride = heads * seqlen_k * head_dim;

    // Create strides (matching CUTLASS example 77 format)
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

    // Problem shape: (seqlen_q, seqlen_k, head_dim, ((h_groups, h_per_group), batch))
    auto problem_shape = make_tuple(
        seqlen_q, seqlen_k, head_dim,
        make_tuple(make_tuple(h_groups, h_per_group), batch)
    );

    // Allocate LSE buffer (log-sum-exp for numerical stability)
    auto lse = torch::empty({batch, heads, seqlen_q}, q.options().dtype(torch::kFloat32));

    // Get hardware info
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = q.device().index();
    cudaDeviceGetAttribute(&hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);

    // Build arguments
    typename Operation::Arguments arguments{
        problem_shape,
        {
            reinterpret_cast<Element const*>(q.data_ptr()),
            stride_Q,
            reinterpret_cast<Element const*>(k.data_ptr()),
            stride_K,
            reinterpret_cast<Element const*>(v.data_ptr()),
            stride_V,
            scale  // softmax scale
        },
        {
            reinterpret_cast<ElementOut*>(out.data_ptr()),
            stride_O,
            reinterpret_cast<float*>(lse.data_ptr()),
            stride_LSE
        },
        hw_info
    };

    // Create and run operation
    Operation op;

    auto status = op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SM100 FMHA kernel cannot implement this problem: ", int(status));

    size_t workspace_size = Operation::get_workspace_size(arguments);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
        q.options().dtype(torch::kUInt8));

    status = op.initialize(arguments, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SM100 FMHA kernel initialization failed: ", int(status));

    // Get the current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    status = op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SM100 FMHA kernel launch failed: ", int(status));

    return out;
}

///////////////////////////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////////////////////////

torch::Tensor fmha_fwd_sm100(
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
        return run_fmha_impl<FmhaCausal>(q, k, v, scale);
    } else {
        return run_fmha_impl<FmhaNoMask>(q, k, v, scale);
    }
}

torch::Tensor fmha_fwd_sm100_auto_scale(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(q.size(-1)));
    return fmha_fwd_sm100(q, k, v, is_causal, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fmha_fwd_sm100, "SM100 Flash Multi-Head Attention Forward",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("is_causal") = false, py::arg("scale") = 1.0f);
    m.def("fwd_auto_scale", &fmha_fwd_sm100_auto_scale,
          "SM100 Flash MHA Forward (auto-compute scale)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("is_causal") = false);
}
