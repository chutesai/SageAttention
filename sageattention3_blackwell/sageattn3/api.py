"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4quant_cuda

# Runtime detection of GPU architecture and appropriate kernel selection
_fp4attn_cuda = None
_gpu_arch = None

def _get_fp4attn_cuda():
    """Lazy load the appropriate attention CUDA module based on GPU architecture."""
    global _fp4attn_cuda, _gpu_arch

    if _fp4attn_cuda is not None:
        return _fp4attn_cuda

    # Detect GPU architecture
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device_props = torch.cuda.get_device_properties(0)
    cc_major, cc_minor = device_props.major, device_props.minor
    _gpu_arch = (cc_major, cc_minor)

    if cc_major == 10 and cc_minor == 0:
        # SM100 (B200/B300) - Datacenter Blackwell with tcgen05/TMEM
        try:
            import fp4attn_cuda_sm100 as _fp4attn_cuda
        except ImportError:
            raise RuntimeError(
                f"SM100 kernel (fp4attn_cuda_sm100) not found. "
                f"Detected GPU: {device_props.name} (compute capability {cc_major}.{cc_minor}). "
                f"Please rebuild with a B200/B300 GPU or use a pre-built wheel for SM100."
            )
    elif cc_major == 12 and cc_minor in (0, 1):
        # SM120/SM121 (RTX 5090/GB10) - Consumer Blackwell with mma.sync.aligned
        try:
            import fp4attn_cuda as _fp4attn_cuda
        except ImportError:
            raise RuntimeError(
                f"SM120 kernel (fp4attn_cuda) not found. "
                f"Detected GPU: {device_props.name} (compute capability {cc_major}.{cc_minor}). "
                f"Please rebuild with an RTX 5090/GB10 GPU or use a pre-built wheel for SM120."
            )
    else:
        raise RuntimeError(
            f"Unsupported GPU architecture: compute capability {cc_major}.{cc_minor}. "
            f"SageAttention3 requires Blackwell GPUs: "
            f"SM100 (B200/B300, compute 10.0) or SM120/SM121 (RTX 5090/GB10, compute 12.0/12.1)."
        )

    return _fp4attn_cuda

def get_gpu_arch():
    """Return the detected GPU architecture as (major, minor) tuple."""
    if _gpu_arch is None:
        _get_fp4attn_cuda()  # Force detection
    return _gpu_arch

def is_sm100():
    """Check if running on SM100 (B200/B300) datacenter Blackwell."""
    arch = get_gpu_arch()
    return arch == (10, 0)

def is_sm120():
    """Check if running on SM120/SM121 (RTX 5090/GB10) consumer Blackwell."""
    arch = get_gpu_arch()
    return arch[0] == 12 and arch[1] in (0, 1)


@triton.jit
def group_mean_kernel(
    q_ptr,          
    q_out_ptr,      
    qm_out_ptr,     
    B, H, L, D: tl.constexpr,    
    stride_qb, stride_qh, stride_ql, stride_qd,  
    stride_qmb, stride_qmh, stride_qml, stride_qmd,  
    GROUP_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    
    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets)
    
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE
    
    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE

    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)

    grid = (B, H, num_groups)

    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE
    )
    return q_out, qm


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Simple matmul kernel: C = A @ B^T where A is (M, K) and B is (N, K)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load A block: (BLOCK_M, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B block: (BLOCK_N, BLOCK_K) - B is (N, K), we want B^T
        b_ptrs = b_ptr + offs_n[:, None] * stride_bk + k_offs[None, :] * stride_bn
        b_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Compute: A @ B^T = A @ B.T, but B is loaded as (BLOCK_N, BLOCK_K)
        # So we do A (BLOCK_M, BLOCK_K) @ B.T (BLOCK_K, BLOCK_N)
        acc += tl.dot(a, tl.trans(b))

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _matmul_no_cublas(qm: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute qm @ k^T without using cuBLAS (which is broken on B200 with CUDA 13.0).

    Args:
        qm: (B, H, M, D) query means
        k: (B, H, N, D) keys

    Returns:
        delta_s: (B, H, M, N) in float32
    """
    B, H, M, D = qm.shape
    N = k.size(2)

    # Reshape to 3D for simpler kernel: (B*H, M, D) and (B*H, N, D)
    qm_3d = qm.reshape(B * H, M, D).contiguous()
    k_3d = k.reshape(B * H, N, D).contiguous()

    # Output: (B*H, M, N)
    out = torch.empty(B * H, M, N, device=qm.device, dtype=torch.float32)

    # Launch kernel for each batch*head
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = lambda meta: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B * H)

    # Process each batch*head separately
    for bh in range(B * H):
        qm_bh = qm_3d[bh]  # (M, D)
        k_bh = k_3d[bh]    # (N, D)
        out_bh = out[bh]   # (M, N)

        grid_2d = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_kernel[grid_2d](
            qm_bh, k_bh, out_bh,
            M, N, D,
            qm_bh.stride(0), qm_bh.stride(1),
            k_bh.stride(0), k_bh.stride(1),
            out_bh.stride(0), out_bh.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

    # Reshape back to 4D
    return out.reshape(B, H, M, N).contiguous()


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):

    def pad_128(x):
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
    
    k -= k.mean(dim=-2, keepdim=True)  
    q, k, v = map(lambda x: pad_128(x), [q, k, v])
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    # Compute delta_s = qm @ k^T without using cuBLAS (broken on B200 with CUDA 13.0)
    # Use a triton kernel or manual computation instead
    delta_s = _matmul_no_cublas(qm, k)
    return q, k, v, delta_s

def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def blockscaled_fp4_attn(qlist: Tuple,
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False,
                         per_block_mean: bool = True,
                         is_bf16: bool = True
                        ):
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    fp4attn = _get_fp4attn_cuda()

    # SM100 has a different API (no KL, out, is_bf16 parameters)
    if is_sm100():
        # SM100 kernel: fwd(q, k, v, out, sfq, sfk, sfv, delta_s, softmax_scale, is_causal, per_block_mean)
        out = torch.empty(
            qlist[0].shape[0],  # batch
            qlist[0].shape[1],  # heads
            qlist[0].shape[2],  # seqlen_q
            qlist[0].shape[3] * 2,  # head_dim (unpacked from FP4)
            device=qlist[0].device,
            dtype=torch.bfloat16 if is_bf16 else torch.float16
        )
        return (fp4attn.fwd(
            qlist[0], klist[0], vlist[0], out,
            qlist[1], klist[1], vlist[1],
            delta_s, softmax_scale, is_causal, per_block_mean
        ),)
    else:
        # SM120 kernel: fwd(q, k, v, sfq, sfk, sfv, delta_s, KL, out, softmax_scale, is_causal, per_block_mean, is_bf16)
        return fp4attn.fwd(
            qlist[0], klist[0], vlist[0],
            qlist[1], klist[1], vlist[1],
            delta_s, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16
        )


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, **kwargs):
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    o_fp4 = blockscaled_fp4_attn(
    qlist_from_cuda,
    klist_from_cuda, 
    vlist_from_cuda,
    delta_s,
    KL,
    is_causal,
    per_block_mean,
    is_bf16
    )[0][:, :, :QL, :].contiguous()
    return o_fp4