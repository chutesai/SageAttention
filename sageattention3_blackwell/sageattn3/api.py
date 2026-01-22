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


def _compute_delta_s(qm: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute delta_s = qm @ k^T using cuBLAS.

    Args:
        qm: (B, H, M, D) query means
        k: (B, H, N, D) keys

    Returns:
        delta_s: (B, H, M, N) in float32
    """
    # Use einsum which leverages cuBLAS - fast and correct
    return torch.einsum('bhmd,bhnd->bhmn', qm.float(), k.float())


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
    # Compute delta_s = qm @ k^T using cuBLAS
    delta_s = _compute_delta_s(qm, k)
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


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, use_fp4 = False, **kwargs):
    """SageAttention3 for Blackwell GPUs.

    For SM100 (B200/B300), uses BF16 CUTLASS FMHA by default (stable, ~90% of cuDNN perf).
    For SM120 (RTX 5090), uses the FP4 blockscaled attention for maximum performance.

    Args:
        q: Query tensor [batch, heads, seqlen_q, head_dim]
        k: Key tensor [batch, heads, seqlen_k, head_dim]
        v: Value tensor [batch, heads, seqlen_k, head_dim]
        attn_mask: Optional attention mask (not used in SM100 path)
        is_causal: Whether to use causal masking
        per_block_mean: Whether to use per-block mean subtraction (FP4 path only)
        use_fp4: Force FP4 path on SM100 (experimental, may crash)
    """
    # SM100 (B200/B300) path
    if is_sm100():
        if use_fp4:
            # Experimental FP4 path - has known issues with TMEM/tcgen05
            pass  # Fall through to FP4 path below
        else:
            # Default: Use stable BF16 CUTLASS FMHA
            return sageattn3_sm100_fast(q, k, v, is_causal=is_causal)

    # Fall back to FP4 path for SM120 or if explicitly requested
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


def sageattn3_sm100_fast(q, k, v, is_causal=False, scale=None):
    """High-performance BF16 FMHA for SM100 (B200/B300) using CUTLASS.

    Uses NVIDIA's official warp-specialized FMHA kernels. Performance is
    approximately 85-90% of PyTorch's cuDNN attention backend.

    For maximum performance, consider using PyTorch's native SDPA which
    automatically selects cuDNN on B200:
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    Args:
        q: Query tensor [batch, heads, seqlen_q, head_dim] in BF16
        k: Key tensor [batch, heads, seqlen_k, head_dim] in BF16
        v: Value tensor [batch, heads, seqlen_k, head_dim] in BF16
        is_causal: Whether to use causal masking
        scale: Softmax scale (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, heads, seqlen_q, head_dim] in BF16
    """
    try:
        import fmha_sm100
    except ImportError:
        raise RuntimeError(
            "SM100 FMHA (fmha_sm100) not found. "
            "Please rebuild with a B200/B300 GPU."
        )

    # Ensure contiguous BF16 tensors
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
    if k.dtype != torch.bfloat16:
        k = k.to(torch.bfloat16)
    if v.dtype != torch.bfloat16:
        v = v.to(torch.bfloat16)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if scale is None:
        return fmha_sm100.fwd_auto_scale(q, k, v, is_causal)
    else:
        return fmha_sm100.fwd(q, k, v, is_causal, scale)


def sageattn3_fp4_kernel_only(q_fp4, k_fp4, v_fp4, sfq, sfk, sfv, delta_s,
                               seqlen_k, is_causal=False, per_block_mean=True, is_bf16=True):
    """Low-level FP4 attention kernel for pre-quantized inputs (SM100/SM120).

    WARNING: The SM100 FP4 kernel has known issues (illegal instruction errors).
    Use at your own risk. For SM120 (RTX 5090), this should work correctly.

    This bypasses preprocessing overhead for maximum throughput when inputs
    are already quantized. Users are responsible for:
    1. Quantizing Q, K, V to FP4 with proper scaling
    2. Computing delta_s correction terms
    3. Ensuring correct tensor layouts

    Args:
        q_fp4: Packed FP4 query [batch, heads, seqlen_q, head_dim//2] uint8
        k_fp4: Packed FP4 key [batch, heads, seqlen_k, head_dim//2] uint8
        v_fp4: Packed FP4 value (transposed) [batch, heads, head_dim, seqlen_k//2] uint8
        sfq: Q scale factors [batch, heads, seqlen_q, head_dim//16] float8_e4m3fn
        sfk: K scale factors [batch, heads, seqlen_k, head_dim//16] float8_e4m3fn
        sfv: V scale factors [batch, heads, head_dim, seqlen_k//16] float8_e4m3fn
        delta_s: Correction terms [batch, heads, seqlen_q//128, seqlen_k] float32
        seqlen_k: Original K sequence length (before padding)
        is_causal: Whether to use causal masking
        per_block_mean: Whether delta_s uses per-block means
        is_bf16: Output in BF16 (True) or FP16 (False)

    Returns:
        Output tensor [batch, heads, seqlen_q, head_dim] in BF16/FP16
    """
    return blockscaled_fp4_attn(
        (q_fp4, sfq), (k_fp4, sfk), (v_fp4, sfv),
        delta_s, seqlen_k, is_causal, per_block_mean, is_bf16
    )