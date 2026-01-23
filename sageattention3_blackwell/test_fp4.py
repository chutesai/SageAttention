#!/usr/bin/env python3
"""
Test script for SM100 FP4 block-scaled attention kernel.
Compares against PyTorch SDPA reference.
"""

import torch
import torch.nn.functional as F

# Import the attention module
import fmha_sm100


def test_fp4_attention(batch=1, heads=1, seqlen=256, head_dim=256, scale_factor=1.0, is_causal=False):
    """Test FP4 attention against SDPA reference."""

    print(f"\n{'='*60}")
    print(f"Testing: batch={batch}, heads={heads}, seqlen={seqlen}, head_dim={head_dim}")
    print(f"         scale_factor={scale_factor}, is_causal={is_causal}")
    print('='*60)

    # Create random inputs
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * scale_factor
    k = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * scale_factor
    v = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * scale_factor

    # Compute reference with PyTorch SDPA
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    # Compute softmax scale
    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Run FP4 kernel using the built-in BF16->FP4 quantization path
    # This handles padding to head_dim=256 and quantization internally
    with torch.no_grad():
        out = fmha_sm100.fwd_fp4_from_bf16(
            q, k, v,
            is_causal,
            softmax_scale
        )

    # Compare results
    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative error (avoid division by zero)
    ref_abs = ref.float().abs()
    rel_err = (diff / (ref_abs + 1e-6)).mean().item()

    ref_mean = ref.float().abs().mean().item()
    out_mean = out.float().abs().mean().item()

    print(f"Max diff:     {max_diff:.4f}")
    print(f"Mean diff:    {mean_diff:.4f}")
    print(f"Mean rel err: {rel_err:.4f}")
    print(f"Ref mean:     {ref_mean:.4f}")
    print(f"Out mean:     {out_mean:.4f}")

    # Check if output is reasonable (not all zeros, not NaN)
    if torch.isnan(out).any():
        print("WARNING: Output contains NaN!")
    if out.abs().max() == 0:
        print("WARNING: Output is all zeros!")

    return max_diff, rel_err


def main():
    print("SM100 FP4 Attention Test")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")

    # Run tests with different configurations
    results = []

    # Basic test - small scale
    results.append(("small_scale", test_fp4_attention(
        batch=1, heads=1, seqlen=256, head_dim=256, scale_factor=0.1
    )))

    # Medium scale
    results.append(("medium_scale", test_fp4_attention(
        batch=1, heads=1, seqlen=256, head_dim=256, scale_factor=1.0
    )))

    # Larger scale
    results.append(("large_scale", test_fp4_attention(
        batch=1, heads=1, seqlen=256, head_dim=256, scale_factor=2.0
    )))

    # Multiple heads
    results.append(("multi_head", test_fp4_attention(
        batch=1, heads=4, seqlen=256, head_dim=256, scale_factor=1.0
    )))

    # Causal mask
    results.append(("causal", test_fp4_attention(
        batch=1, heads=1, seqlen=256, head_dim=256, scale_factor=1.0, is_causal=True
    )))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, (max_diff, rel_err) in results:
        status = "PASS" if max_diff < 1.0 else "FAIL"
        print(f"{name:15s}: max_diff={max_diff:.4f}, rel_err={rel_err:.4f} [{status}]")


if __name__ == "__main__":
    main()
