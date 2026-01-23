#!/usr/bin/env python3
"""
Test script for SM100 FP4 block-scaled attention with smooth attention mechanism.
Uses SageAttention3's mean subtraction + delta_s correction for better accuracy.
"""

import torch
import torch.nn.functional as F

# Import the attention module
import fmha_sm100

# Import smooth attention helpers from api.py
import triton
import triton.language as tl


@triton.jit
def group_mean_kernel(
    q_ptr, q_out_ptr, qm_out_ptr,
    B, H, L, D: tl.constexpr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_qmb, stride_qmh, stride_qml, stride_qmd,
    GROUP_SIZE: tl.constexpr
):
    """Per-block mean extraction kernel."""
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
    """Extract per-block (128 element) means from Q and subtract them."""
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE

    q_out = torch.empty_like(q)
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


def compute_delta_s(qm: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute delta_s = qm @ k^T correction term.

    Args:
        qm: Query means [batch, heads, num_q_groups, head_dim]
        k: Smoothed keys [batch, heads, seqlen_k, head_dim]

    Returns:
        delta_s: [batch, heads, num_q_groups, seqlen_k] correction tensor
    """
    # qm: [B, H, M, D] where M = num_q_groups
    # k:  [B, H, N, D] where N = seqlen_k
    # output: [B, H, M, N]
    B, H, M, D = qm.shape
    _, _, N, _ = k.shape

    # Do computation on CPU then move back to avoid CUBLAS issues
    qm_cpu = qm.float().cpu()
    k_cpu = k.float().cpu()
    result_cpu = torch.matmul(qm_cpu, k_cpu.transpose(-2, -1))
    return result_cpu.to(qm.device)


def preprocess_smooth(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):
    """Preprocess Q, K, V with smooth attention (mean subtraction).

    Returns:
        q_smooth: Q with mean subtracted
        k_smooth: K with global mean subtracted
        v: V unchanged
        delta_s: Correction term qm @ k^T [batch, heads, num_q_groups, seqlen_k]
    """
    # Smooth K: subtract global mean
    k_smooth = k - k.mean(dim=-2, keepdim=True)

    # Smooth Q: per-block mean subtraction
    if per_block_mean:
        q_smooth, qm = triton_group_mean(q.contiguous())
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q_smooth = q - qm

    print(f"  qm shape: {qm.shape}, k_smooth shape: {k_smooth.shape}")

    # Compute correction term: qm @ k_smooth^T
    delta_s = compute_delta_s(qm, k_smooth)

    return q_smooth, k_smooth, v, delta_s


def quantize_bf16_to_fp4(tensor, fp4_max=6.0):
    """Quantize BF16 tensor to FP4 with block-wise scale factors.

    Returns:
        data: uint8 tensor with packed FP4 values
        sf: FP8 E4M3 scale factors
    """
    batch, heads, seqlen, head_dim = tensor.shape
    sf_dim = head_dim // 16  # 16 elements per scale factor block

    # Reshape to [batch, heads, seqlen, num_blocks, 16]
    blocks = tensor.view(batch, heads, seqlen, sf_dim, 16)

    # Compute per-block max abs
    absmax = blocks.abs().max(dim=-1).values  # [batch, heads, seqlen, sf_dim]

    # Compute scale factors (clamp to avoid /0)
    scales = (absmax / fp4_max).clamp_min(1e-7)
    sf = scales.to(torch.float8_e4m3fn)

    # Scale and quantize
    scaled = blocks / scales.unsqueeze(-1)
    scaled = scaled.clamp(-fp4_max, fp4_max)

    # Linear encoding: [-6, 6] -> [0, 15]
    int_val = ((scaled / fp4_max * 7.5 + 7.5).to(torch.int8)).view(batch, heads, seqlen, head_dim)

    # Pack pairs into uint8
    even = int_val[..., 0::2]
    odd = int_val[..., 1::2]
    data = ((even & 0x0F) | ((odd & 0x0F) * 16)).to(torch.uint8)

    return data, sf


def test_smooth_with_delta_s(batch=1, heads=1, seqlen=256, head_dim=256, scale_factor=1.0, is_causal=False):
    """Test FP4 attention with smooth attention + delta_s correction in kernel."""

    print(f"\n{'='*60}")
    print(f"Testing SMOOTH+DELTA_S: batch={batch}, heads={heads}, seqlen={seqlen}, head_dim={head_dim}")
    print(f"                        scale_factor={scale_factor}, is_causal={is_causal}")
    print('='*60)

    # Create random inputs
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * scale_factor
    k = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * scale_factor
    v = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * scale_factor

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Reference: original SDPA (no smoothing)
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    # Apply smooth attention preprocessing
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    q_smooth, k_smooth, v_smooth, delta_s = preprocess_smooth(q, k, v, per_block_mean=True)
    print(f"After smooth: q_smooth={q_smooth.shape}, k_smooth={k_smooth.shape}, delta_s={delta_s.shape}")

    # Quantize smoothed inputs to FP4
    q_data, q_sf = quantize_bf16_to_fp4(q_smooth)
    k_data, k_sf = quantize_bf16_to_fp4(k_smooth)
    v_data, v_sf = quantize_bf16_to_fp4(v_smooth)

    # Ensure delta_s is contiguous float32
    delta_s = delta_s.contiguous().float()

    print(f"delta_s shape: {delta_s.shape}")  # [batch, heads, num_q_groups, seqlen_k]
    print(f"delta_s stats: min={delta_s.min().item():.4f}, max={delta_s.max().item():.4f}, mean={delta_s.mean().item():.4f}")

    # Debug: Show first few delta_s values
    print(f"delta_s[0,0,0,:5]: {delta_s[0,0,0,:5].tolist()}")

    # Run FP4 kernel with smooth inputs AND delta_s correction
    with torch.no_grad():
        out = fmha_sm100.fwd_fp4(
            q_data, q_sf,
            k_data, k_sf,
            v_data, v_sf,
            delta_s,  # Pass delta_s for smooth attention correction
            is_causal,
            softmax_scale
        )

    # Compare against original reference (the delta_s should compensate for mean subtraction)
    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    ref_abs = ref.float().abs()
    rel_err = (diff / (ref_abs + 1e-6)).mean().item()

    ref_mean = ref.float().abs().mean().item()
    out_mean = out.float().abs().mean().item()

    print(f"FP4+smooth+delta_s vs original:")
    print(f"  Max diff:     {max_diff:.4f}")
    print(f"  Mean diff:    {mean_diff:.4f}")
    print(f"  Mean rel err: {rel_err*100:.2f}%")
    print(f"  Ref mean:     {ref_mean:.4f}")
    print(f"  Out mean:     {out_mean:.4f}")

    return max_diff, rel_err


def test_without_smooth(batch=1, heads=1, seqlen=256, head_dim=256, scale_factor=1.0, is_causal=False):
    """Test FP4 attention WITHOUT smooth preprocessing (baseline)."""

    print(f"\n{'='*60}")
    print(f"Testing NO SMOOTH: batch={batch}, heads={heads}, seqlen={seqlen}, head_dim={head_dim}")
    print(f"                   scale_factor={scale_factor}, is_causal={is_causal}")
    print('='*60)

    torch.manual_seed(42)
    q = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * scale_factor
    k = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * scale_factor
    v = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * scale_factor

    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    softmax_scale = 1.0 / (head_dim ** 0.5)

    with torch.no_grad():
        out = fmha_sm100.fwd_fp4_from_bf16(q, k, v, is_causal, softmax_scale)

    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item()
    rel_err = (diff / (ref.float().abs() + 1e-6)).mean().item()

    print(f"  Max diff:     {max_diff:.4f}")
    print(f"  Mean rel err: {rel_err*100:.2f}%")

    return max_diff, rel_err


def verify_smooth_math():
    """Verify that smooth attention math is correct in pure PyTorch."""
    print("\n" + "="*60)
    print("VERIFICATION: Smooth attention math in pure PyTorch")
    print("="*60)

    batch, heads, seqlen, head_dim = 1, 1, 256, 256

    torch.manual_seed(42)
    q = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.float32)
    v = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.float32)

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Original attention scores: S = Q @ K^T * scale
    S_orig = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    # Smooth Q and K
    k_smooth = k - k.mean(dim=-2, keepdim=True)

    # Per-block (128) Q mean subtraction
    GROUP_SIZE = 128
    q_reshaped = q.view(batch, heads, seqlen // GROUP_SIZE, GROUP_SIZE, head_dim)
    qm = q_reshaped.mean(dim=3)  # [batch, heads, num_groups, head_dim]
    q_smooth = q - qm.unsqueeze(3).expand_as(q_reshaped).reshape(batch, heads, seqlen, head_dim)

    # Smooth attention scores: S_smooth = Q_smooth @ K_smooth^T * scale
    S_smooth = torch.matmul(q_smooth, k_smooth.transpose(-2, -1)) * softmax_scale

    # Delta-S correction: delta_s = Qm @ K_smooth^T * scale
    # Shape: [batch, heads, num_groups, seqlen_k]
    delta_s = torch.matmul(qm, k_smooth.transpose(-2, -1)) * softmax_scale

    # Expand delta_s to match S shape
    # Each delta_s[g] applies to rows [g*128 : (g+1)*128]
    delta_s_expanded = delta_s.unsqueeze(3).expand(batch, heads, seqlen // GROUP_SIZE, GROUP_SIZE, seqlen)
    delta_s_expanded = delta_s_expanded.reshape(batch, heads, seqlen, seqlen)

    # Reconstructed: S_reconstructed = S_smooth + delta_s
    S_reconstructed = S_smooth + delta_s_expanded

    # Compare
    diff = (S_orig - S_reconstructed).abs()
    print(f"S_orig vs S_smooth+delta_s:")
    print(f"  Max diff:  {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")

    if diff.max().item() < 1e-5:
        print("  PASS: Smooth attention math is correct!")
    else:
        print("  FAIL: There's an error in the smooth attention math!")

    return diff.max().item() < 1e-5


def main():
    print("SM100 FP4 Attention Test - Smooth+Delta_s vs No Smooth")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")

    # First verify the math is correct
    if not verify_smooth_math():
        print("ERROR: Smooth attention math is incorrect, fix before proceeding!")
        return

    # Compare smooth+delta_s vs no smooth for different scale factors
    results = []

    for scale in [0.1, 1.0, 2.0]:
        print(f"\n{'#'*60}")
        print(f"# Scale factor: {scale}")
        print(f"{'#'*60}")

        # Without smooth
        max_diff_no, rel_err_no = test_without_smooth(scale_factor=scale)

        # With smooth + delta_s
        max_diff_smooth, rel_err_smooth = test_smooth_with_delta_s(scale_factor=scale)

        results.append({
            'scale': scale,
            'no_smooth': (max_diff_no, rel_err_no),
            'smooth': (max_diff_smooth, rel_err_smooth)
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Smooth Attention Effect on FP4 Accuracy")
    print("=" * 70)
    print(f"{'Scale':<10} {'No Smooth Err%':<18} {'Smooth+DS Err%':<18} {'Improvement':<12}")
    print("-" * 70)
    for r in results:
        no_err = r['no_smooth'][1] * 100
        sm_err = r['smooth'][1] * 100
        improvement = (no_err - sm_err) / no_err * 100 if no_err > 0 else 0
        print(f"{r['scale']:<10} {no_err:<18.2f} {sm_err:<18.2f} {improvement:>+.1f}%")

    print("\nNOTE: Smooth attention reduces quantization error by centering values around zero")
    print("      before FP4 quantization. Delta_s compensates for the mean subtraction.")


if __name__ == "__main__":
    main()
