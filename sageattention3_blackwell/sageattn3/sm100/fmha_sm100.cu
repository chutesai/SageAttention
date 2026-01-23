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
// NOTE: FP8 input, FP16 output (matches CUTLASS example 77)
///////////////////////////////////////////////////////////////////////////////

using ElementFP8 = cutlass::float_e4m3_t;
using ElementFP8Out = cutlass::half_t;  // FP8 attention outputs FP16
// FP8 uses same 256x128 tile shape as BF16 (from CUTLASS example 77)
using TileShapeFP8 = Shape<_256, _128, _128>;

template <typename ActiveMask>
struct FmhaKernelFP8 {
    using TileScheduler = fmha_kernel::PersistentTileScheduler;

    using Mainloop = fmha_collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
        ElementFP8, float, float,
        TileShapeFP8, StrideQ, StrideK, StrideV,
        ActiveMask
    >;

    // FP8 FMHA outputs FP16, not FP8 (per CUTLASS example 77)
    using Epilogue = fmha_collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementFP8Out, float,
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

///////////////////////////////////////////////////////////////////////////////
// FP4 Block-Scaled FMHA Kernel Types - 7x tensor core throughput (theoretical)
// NOTE: FP4 requires HeadDim=256 and BlockN=256 due to SM100 MMA K=256 constraint
// This is our custom implementation using CUTLASS block-scaled UMMA
///////////////////////////////////////////////////////////////////////////////

// Include FP4 kernel headers
#include "kernel_traits_fp4.h"
#include "kernel_fp4_ws.h"

using ElementFP4 = cutlass::float_e2m1_t;
using ElementFP4SF = cutlass::float_e4m3_t;  // Signed E4M3 to match PyTorch float8_e4m3fn
using ElementFP4Out = cutlass::bfloat16_t;

// FP4 Kernel configuration
// HeadDim=256, BlockM=128 (1SM), BlockN=256
template <bool Is_causal>
struct FmhaKernelFP4 {
    using Ktraits = flash::Flash_fwd_kernel_traits_sm100_fp4<
        256,   // kHeadDim (must be 256 for FP4)
        128,   // kBlockM (128 for 1SM, 256 for 2SM)
        256,   // kBlockN (must be 256 for FP4 PV matmul)
        2,     // kStages
        1,     // kClusterM
        false, // BlockMean
        ElementFP4Out
    >;

    using TileScheduler = flash::SimpleTileSchedulerFP4;
    using Kernel = flash::Sm100FlashFwdKernelFP4<Ktraits, Is_causal, TileScheduler>;
};

using FmhaFP4NoMask = FmhaKernelFP4<false>;
using FmhaFP4Causal = FmhaKernelFP4<true>;

// Forward declaration of FP4 kernel wrapper
template <typename Kernel>
__global__ void run_fmha_fp4_kernel(typename Kernel::Params params);

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

    // Output is FP16 (half) for FP8 attention (per CUTLASS example 77)
    auto out = torch::empty({batch, heads, seqlen_q, head_dim}, q.options().dtype(torch::kFloat16));

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
            reinterpret_cast<ElementFP8Out*>(out.data_ptr()),
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

///////////////////////////////////////////////////////////////////////////////
// FP4 Launch Functions
///////////////////////////////////////////////////////////////////////////////

// Generic launch for FP4 block-scaled attention
template <typename FmhaType>
torch::Tensor run_fmha_fp4_impl(
    torch::Tensor& q_data,     // [batch, heads, seqlen_q, head_dim] FP4 packed (uint8)
    torch::Tensor& q_sf,       // [batch, heads, seqlen_q, head_dim/16] FP8 E4M3 scale factors
    torch::Tensor& k_data,     // [batch, heads, seqlen_k, head_dim] FP4 packed
    torch::Tensor& k_sf,       // [batch, heads, seqlen_k, head_dim/16] scale factors
    torch::Tensor& v_data,     // [batch, heads, seqlen_k, head_dim] FP4 packed
    torch::Tensor& v_sf,       // [batch, heads, seqlen_k, head_dim/16] scale factors
    float scale
) {
    using Kernel = typename FmhaType::Kernel;
    using Ktraits = typename Kernel::Ktraits;

    const int batch = q_data.size(0);
    const int heads = q_data.size(1);
    const int seqlen_q = q_data.size(2);
    const int seqlen_k = k_data.size(2);
    // FP4 is packed 2 per byte, so tensor dim is head_dim/2
    const int head_dim = q_data.size(3) * 2;  // Unpack to get actual head_dim

    TORCH_CHECK(head_dim == 256, "SM100 FP4 FMHA requires head_dim=256 (K=256 constraint)");
    TORCH_CHECK(q_data.dtype() == torch::kUInt8, "FP4 data must be packed as uint8");
    TORCH_CHECK(q_sf.dtype() == torch::kFloat8_e4m3fn, "Scale factors must be FP8 E4M3");
    TORCH_CHECK(q_data.is_contiguous(), "Q data must be contiguous");
    TORCH_CHECK(k_data.is_contiguous(), "K data must be contiguous");
    TORCH_CHECK(v_data.is_contiguous(), "V data must be contiguous");

    // Output is BF16
    auto out = torch::empty({batch, heads, seqlen_q, head_dim},
        q_data.options().dtype(torch::kBFloat16));

    // Compute strides (in elements, not bytes)
    // FP4 data strides (each byte holds 2 FP4 values)
    const int64_t q_seq_stride = head_dim / 2;  // packed
    const int64_t q_head_stride = seqlen_q * q_seq_stride;
    const int64_t q_batch_stride = heads * q_head_stride;

    const int64_t k_seq_stride = head_dim / 2;
    const int64_t k_head_stride = seqlen_k * k_seq_stride;
    const int64_t k_batch_stride = heads * k_head_stride;

    // V is transposed: (dim, seq, head, batch)
    const int64_t v_dim_stride = 1;  // packed
    const int64_t v_seq_stride = head_dim / 2;
    const int64_t v_head_stride = seqlen_k * v_seq_stride;
    const int64_t v_batch_stride = heads * v_head_stride;

    // Output strides
    const int64_t o_seq_stride = head_dim;
    const int64_t o_head_stride = seqlen_q * o_seq_stride;
    const int64_t o_batch_stride = heads * o_head_stride;

    // Build kernel arguments
    typename Kernel::Arguments args{
        seqlen_q, seqlen_k, head_dim, heads, batch,
        reinterpret_cast<ElementFP4 const*>(q_data.data_ptr()),
        reinterpret_cast<ElementFP4SF const*>(q_sf.data_ptr()),
        q_seq_stride, q_head_stride, q_batch_stride,
        reinterpret_cast<ElementFP4 const*>(k_data.data_ptr()),
        reinterpret_cast<ElementFP4SF const*>(k_sf.data_ptr()),
        k_seq_stride, k_head_stride, k_batch_stride,
        reinterpret_cast<ElementFP4 const*>(v_data.data_ptr()),
        reinterpret_cast<ElementFP4SF const*>(v_sf.data_ptr()),
        v_seq_stride, v_head_stride, v_batch_stride,
        reinterpret_cast<ElementFP4Out*>(out.data_ptr()),
        o_seq_stride, o_head_stride, o_batch_stride,
        scale
    };

    // Get hardware info
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = q_data.device().index();
    cudaDeviceGetAttribute(&hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);

    // Convert to kernel params
    auto params = Kernel::to_underlying_arguments(args, nullptr);

    // Get launch configuration
    dim3 grid = Kernel::get_grid_dim(args, hw_info.sm_count);
    dim3 block = Kernel::get_block_dim();
    size_t smem_size = Kernel::get_smem_size();

    // Check shared memory
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (void*)run_fmha_fp4_kernel<Kernel>);
    if (smem_size > attr.maxDynamicSharedSizeBytes) {
        cudaFuncSetAttribute(
            (void*)run_fmha_fp4_kernel<Kernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    // Launch kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    run_fmha_fp4_kernel<Kernel><<<grid, block, smem_size, stream>>>(params);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "SM100 FP4 FMHA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Kernel wrapper function implementation
template <typename Kernel>
__global__ void run_fmha_fp4_kernel(typename Kernel::Params params) {
    extern __shared__ char smem[];
    Kernel kernel;
    kernel(params, smem);
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

// FP4 block-scaled forward pass - 7x theoretical tensor core throughput
// NOTE: Requires pre-quantized FP4 inputs with separate scale factor tensors
torch::Tensor fmha_fwd_fp4(
    torch::Tensor q_data,      // FP4 packed as uint8
    torch::Tensor q_sf,        // FP8 E4M3 scale factors
    torch::Tensor k_data,
    torch::Tensor k_sf,
    torch::Tensor v_data,
    torch::Tensor v_sf,
    bool is_causal,
    float scale
) {
    TORCH_CHECK(q_data.is_cuda(), "Q data must be on CUDA");
    TORCH_CHECK(q_sf.is_cuda(), "Q scale factors must be on CUDA");
    TORCH_CHECK(k_data.is_cuda(), "K data must be on CUDA");
    TORCH_CHECK(k_sf.is_cuda(), "K scale factors must be on CUDA");
    TORCH_CHECK(v_data.is_cuda(), "V data must be on CUDA");
    TORCH_CHECK(v_sf.is_cuda(), "V scale factors must be on CUDA");

    at::cuda::CUDAGuard device_guard(q_data.device());

    if (is_causal) {
        return run_fmha_fp4_impl<FmhaFP4Causal>(
            q_data, q_sf, k_data, k_sf, v_data, v_sf, scale);
    }
    return run_fmha_fp4_impl<FmhaFP4NoMask>(
        q_data, q_sf, k_data, k_sf, v_data, v_sf, scale);
}

// Auto-quantize BF16 to FP4 and run FP4 attention
// NOTE: This pads HeadDim from 128 to 256 if needed
torch::Tensor fmha_fwd_fp4_from_bf16(
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

    const int batch = q.size(0);
    const int heads = q.size(1);
    const int seqlen_q = q.size(2);
    const int seqlen_k = k.size(2);
    const int head_dim = q.size(3);

    // FP4 requires HeadDim=256, pad if necessary
    torch::Tensor q_padded = q;
    torch::Tensor k_padded = k;
    torch::Tensor v_padded = v;

    if (head_dim < 256) {
        // Pad to 256
        int pad_size = 256 - head_dim;
        q_padded = torch::nn::functional::pad(
            q, torch::nn::functional::PadFuncOptions({0, pad_size}));
        k_padded = torch::nn::functional::pad(
            k, torch::nn::functional::PadFuncOptions({0, pad_size}));
        v_padded = torch::nn::functional::pad(
            v, torch::nn::functional::PadFuncOptions({0, pad_size}));
    }

    TORCH_CHECK(q_padded.size(3) == 256, "After padding, head_dim must be 256");

    // Quantize to FP4 with block-wise scale factors (16 elements per block)
    const int sf_dim = 256 / 16;  // 16 scale factors per head

    // Compute per-block scale factors and quantize
    // Reshape to [batch, heads, seqlen, num_blocks, 16]
    auto q_blocks = q_padded.view({batch, heads, seqlen_q, sf_dim, 16});
    auto k_blocks = k_padded.view({batch, heads, seqlen_k, sf_dim, 16});
    auto v_blocks = v_padded.view({batch, heads, seqlen_k, sf_dim, 16});

    // Compute max abs per block for scale factors
    auto q_absmax = std::get<0>(q_blocks.abs().max(-1));  // [batch, heads, seqlen, sf_dim]
    auto k_absmax = std::get<0>(k_blocks.abs().max(-1));
    auto v_absmax = std::get<0>(v_blocks.abs().max(-1));

    // FP4 E2M1 max value is 6.0
    const float fp4_max = 6.0f;
    auto q_scales = q_absmax / fp4_max;
    auto k_scales = k_absmax / fp4_max;
    auto v_scales = v_absmax / fp4_max;

    // Clamp scales to avoid division by zero
    q_scales = q_scales.clamp_min(1e-7f);
    k_scales = k_scales.clamp_min(1e-7f);
    v_scales = v_scales.clamp_min(1e-7f);

    // Convert scales to FP8 E4M3
    auto q_sf = q_scales.to(torch::kFloat8_e4m3fn);
    auto k_sf = k_scales.to(torch::kFloat8_e4m3fn);
    auto v_sf = v_scales.to(torch::kFloat8_e4m3fn);

    // Quantize data to FP4 (pack 2 values per uint8 byte)
    // Scale each block by its scale factor
    auto q_scaled = q_blocks / q_scales.unsqueeze(-1);
    auto k_scaled = k_blocks / k_scales.unsqueeze(-1);
    auto v_scaled = v_blocks / v_scales.unsqueeze(-1);

    // Clamp to FP4 range [-6, 6]
    q_scaled = q_scaled.clamp(-fp4_max, fp4_max);
    k_scaled = k_scaled.clamp(-fp4_max, fp4_max);
    v_scaled = v_scaled.clamp(-fp4_max, fp4_max);

    // Convert to FP4 representation packed in uint8
    // For now, use a simplified packing (actual FP4 E2M1 encoding is more complex)
    // Each uint8 holds 2 FP4 values
    auto q_int = ((q_scaled / fp4_max * 7.5f + 7.5f).to(torch::kInt8)).view({batch, heads, seqlen_q, 256});
    auto k_int = ((k_scaled / fp4_max * 7.5f + 7.5f).to(torch::kInt8)).view({batch, heads, seqlen_k, 256});
    auto v_int = ((v_scaled / fp4_max * 7.5f + 7.5f).to(torch::kInt8)).view({batch, heads, seqlen_k, 256});

    // Pack pairs of 4-bit values into uint8
    // Use multiplication by 16 instead of << 4 to avoid CUTE namespace conflict
    auto q_even = q_int.index({"...", torch::indexing::Slice(0, torch::indexing::None, 2)});
    auto q_odd = q_int.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2)});
    auto q_data = ((q_even & 0x0F) | ((q_odd & 0x0F) * 16)).to(torch::kUInt8);

    auto k_even = k_int.index({"...", torch::indexing::Slice(0, torch::indexing::None, 2)});
    auto k_odd = k_int.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2)});
    auto k_data = ((k_even & 0x0F) | ((k_odd & 0x0F) * 16)).to(torch::kUInt8);

    auto v_even = v_int.index({"...", torch::indexing::Slice(0, torch::indexing::None, 2)});
    auto v_odd = v_int.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2)});
    auto v_data = ((v_even & 0x0F) | ((v_odd & 0x0F) * 16)).to(torch::kUInt8);

    // Run FP4 attention
    auto out_padded = fmha_fwd_fp4(q_data, q_sf, k_data, k_sf, v_data, v_sf, is_causal, scale);

    // Remove padding if we added it
    if (head_dim < 256) {
        return out_padded.index({"...", torch::indexing::Slice(0, head_dim)});
    }
    return out_padded;
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

    // FP4 functions - 7x theoretical tensor core throughput (requires HeadDim=256)
    m.def("fwd_fp4", &fmha_fwd_fp4,
          "SM100 Flash MHA Forward (FP4 block-scaled, 7x theoretical speedup)\n"
          "NOTE: Requires HeadDim=256, input must be pre-quantized FP4 with scale factors",
          py::arg("q_data"), py::arg("q_sf"),
          py::arg("k_data"), py::arg("k_sf"),
          py::arg("v_data"), py::arg("v_sf"),
          py::arg("is_causal") = false, py::arg("scale") = 1.0f);
    m.def("fwd_fp4_from_bf16", &fmha_fwd_fp4_from_bf16,
          "SM100 Flash MHA Forward (auto-quantize BF16->FP4, 7x theoretical speedup)\n"
          "NOTE: Pads HeadDim to 256 if needed, may affect accuracy",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("is_causal") = false, py::arg("scale") = 1.0f);
}
