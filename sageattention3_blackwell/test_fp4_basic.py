#!/usr/bin/env python3
"""
Basic diagnostic test for SM100 FP4 attention kernel.
Tests with identity-like patterns to verify correctness.
"""

import torch
import torch.nn.functional as F
import fmha_sm100


def test_identity_attention():
    """Test with Q=K so attention is approximately identity on V."""
    print("\n" + "="*60)
    print("Test 1: Identity attention (Q=K)")
    print("="*60)

    batch, heads, seqlen, head_dim = 1, 1, 256, 256

    # Create Q=K so S = Q @ K^T has high diagonal values
    torch.manual_seed(42)
    qk = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16)
    qk = qk * 0.1  # Small values to avoid saturation
    v = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * 0.1

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Reference
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(qk, qk, v, is_causal=False)

    # FP4 kernel
    with torch.no_grad():
        out = fmha_sm100.fwd_fp4_from_bf16(qk, qk, v, False, softmax_scale)

    diff = (out.float() - ref.float()).abs()
    print(f"  Max diff:  {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Ref max:   {ref.abs().max().item():.6f}")
    print(f"  Out max:   {out.abs().max().item():.6f}")

    # Check correlation
    ref_flat = ref.flatten()
    out_flat = out.flatten()
    corr = torch.corrcoef(torch.stack([ref_flat.float(), out_flat.float()]))[0, 1].item()
    print(f"  Correlation: {corr:.4f}")


def test_uniform_attention():
    """Test with uniform Q and K so attention is uniform (1/N) on V."""
    print("\n" + "="*60)
    print("Test 2: Uniform attention (all Q and K the same)")
    print("="*60)

    batch, heads, seqlen, head_dim = 1, 1, 256, 256

    # Uniform Q and K - attention should be uniform across all V
    q = torch.ones(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * 0.1
    k = torch.ones(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * 0.1
    v = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * 0.5

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Reference
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # For uniform attention, output should be mean of V
    v_mean = v.mean(dim=-2, keepdim=True).expand_as(v)
    print(f"  V mean error from ref: {(ref - v_mean).abs().mean().item():.6f}")

    # FP4 kernel
    with torch.no_grad():
        out = fmha_sm100.fwd_fp4_from_bf16(q, k, v, False, softmax_scale)

    diff = (out.float() - ref.float()).abs()
    print(f"  Max diff:  {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Ref max:   {ref.abs().max().item():.6f}")
    print(f"  Out max:   {out.abs().max().item():.6f}")


def test_one_hot_attention():
    """Test with one-hot style K so attention picks specific V rows."""
    print("\n" + "="*60)
    print("Test 3: One-hot attention (Q picks specific K row)")
    print("="*60)

    batch, heads, seqlen, head_dim = 1, 1, 256, 256

    torch.manual_seed(42)
    # Make K have one "hot" row that Q will attend to
    k = torch.zeros(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16)
    k[:, :, 0, :] = 1.0  # First row is hot

    q = torch.ones(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16)  # Matches first K row

    v = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * 0.5

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Reference
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Should mostly attend to V[0]
    print(f"  Ref should be ~V[0]: error = {(ref - v[:,:,0:1,:].expand_as(ref)).abs().mean().item():.6f}")

    # FP4 kernel
    with torch.no_grad():
        out = fmha_sm100.fwd_fp4_from_bf16(q, k, v, False, softmax_scale)

    diff = (out.float() - ref.float()).abs()
    print(f"  Max diff:  {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")


def test_zeros():
    """Test with zero inputs."""
    print("\n" + "="*60)
    print("Test 4: Zero inputs")
    print("="*60)

    batch, heads, seqlen, head_dim = 1, 1, 256, 256

    q = torch.zeros(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.zeros(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.bfloat16) * 0.5

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Reference - with zero Q and K, all attention scores are equal (exp(0) = 1)
    # So output should be mean of V
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    v_mean = v.mean(dim=-2, keepdim=True).expand_as(v)
    print(f"  Ref should be V mean: error = {(ref - v_mean).abs().mean().item():.6f}")

    # FP4 kernel
    with torch.no_grad():
        out = fmha_sm100.fwd_fp4_from_bf16(q, k, v, False, softmax_scale)

    diff = (out.float() - ref.float()).abs()
    print(f"  Max diff:  {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Ref max:   {ref.abs().max().item():.6f}")
    print(f"  Out max:   {out.abs().max().item():.6f}")


def test_raw_fp4_quantization():
    """Test the FP4 quantization/dequantization path."""
    print("\n" + "="*60)
    print("Test 5: FP4 quantization round-trip")
    print("="*60)

    # Test values in FP4 range [-6, 6]
    values = torch.tensor([-6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0], device='cuda', dtype=torch.bfloat16)

    # Linear encoding: [-6, 6] -> [0, 15]
    fp4_max = 6.0
    encoded = ((values / fp4_max * 7.5 + 7.5).to(torch.int32).clamp(0, 15))

    # Linear decoding: [0, 15] -> [-6, 6]
    decoded = (encoded.float() - 7.5) * 0.8

    print("  Value      Encoded  Decoded  Error")
    for i, v in enumerate(values):
        err = abs(v.item() - decoded[i].item())
        print(f"  {v.item():+6.2f}     {encoded[i].item():2d}      {decoded[i].item():+6.2f}   {err:.4f}")


def main():
    print("SM100 FP4 Attention Diagnostic Tests")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")

    test_raw_fp4_quantization()
    test_zeros()
    test_uniform_attention()
    test_identity_attention()
    test_one_hot_attention()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("If outputs show high correlation with reference but large absolute")
    print("differences, the kernel is computing correctly but has scale issues.")
    print("If correlation is low, there's a fundamental algorithmic problem.")


if __name__ == "__main__":
    main()
