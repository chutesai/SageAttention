#!/usr/bin/env python3
"""
Test FP4 kernel correctness against PyTorch reference implementation.
"""

import torch
import numpy as np

# Import SM100 kernels
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmha_sm100

def quantize_to_fp4(tensor, block_size=16):
    """Quantize BF16 tensor to FP4 with block-wise scale factors."""
    batch, heads, seq, dim = tensor.shape
    num_blocks = dim // block_size
    blocks = tensor.view(batch, heads, seq, num_blocks, block_size)
    absmax = blocks.abs().amax(dim=-1)
    fp4_max = 6.0
    scales = absmax / fp4_max
    scales = scales.clamp(min=1e-7)
    sf = scales.to(torch.float8_e4m3fn)
    scaled = blocks / scales.unsqueeze(-1)
    scaled = scaled.clamp(-fp4_max, fp4_max)
    int_vals = (scaled * 1.25 + 7.5).round().clamp(0, 15).to(torch.int32)
    int_vals = int_vals.view(batch, heads, seq, dim)
    even = int_vals[..., 0::2]
    odd = int_vals[..., 1::2]
    packed = ((even & 0x0F) | ((odd & 0x0F) << 4)).to(torch.uint8)
    return packed, sf

def dequantize_fp4(packed, sf, block_size=16):
    """Dequantize FP4 back to float for reference computation."""
    batch, heads, seq, packed_dim = packed.shape
    dim = packed_dim * 2

    # Unpack
    even = (packed & 0x0F).to(torch.float32)
    odd = ((packed >> 4) & 0x0F).to(torch.float32)

    # Interleave
    result = torch.zeros(batch, heads, seq, dim, dtype=torch.float32, device=packed.device)
    result[..., 0::2] = even
    result[..., 1::2] = odd

    # Decode: value = (nibble - 7.5) * 0.8
    result = (result - 7.5) * 0.8

    # Apply scale factors
    num_blocks = dim // block_size
    result = result.view(batch, heads, seq, num_blocks, block_size)
    sf_float = sf.to(torch.float32).unsqueeze(-1)
    result = result * sf_float
    result = result.view(batch, heads, seq, dim)

    return result

def reference_attention(q, k, v, scale, is_causal=False):
    """PyTorch reference attention implementation."""
    # q, k, v: [batch, heads, seq, dim]
    # Use bfloat16 for matmul to avoid cuBLAS issues with float32
    q_bf = q.to(torch.bfloat16)
    k_bf = k.to(torch.bfloat16)
    v_bf = v.to(torch.bfloat16)

    scores = torch.matmul(q_bf, k_bf.transpose(-2, -1)).float() * scale

    if is_causal:
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn.to(torch.bfloat16), v_bf).float()
    return out

def test_fp4_correctness():
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    print("="*60)
    print("FP4 Attention Correctness Test")
    print("="*60)

    test_cases = [
        # (batch, heads, seqlen, head_dim)
        (1, 1, 128, 256),
        (1, 1, 256, 256),
        (1, 2, 128, 256),
        (2, 1, 128, 256),
    ]

    for batch, heads, seqlen, head_dim in test_cases:
        print(f"\nTest: batch={batch}, heads={heads}, seqlen={seqlen}, head_dim={head_dim}")

        # Create random inputs
        q_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device) * 0.5
        k_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device) * 0.5
        v_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device) * 0.5

        scale = 1.0 / (head_dim ** 0.5)

        # Quantize to FP4
        q_data, q_sf = quantize_to_fp4(q_bf16)
        k_data, k_sf = quantize_to_fp4(k_bf16)
        v_data, v_sf = quantize_to_fp4(v_bf16)

        # Dequantize for reference
        q_deq = dequantize_fp4(q_data, q_sf)
        k_deq = dequantize_fp4(k_data, k_sf)
        v_deq = dequantize_fp4(v_data, v_sf)

        # Reference attention with dequantized values
        ref_out = reference_attention(q_deq, k_deq, v_deq, scale)

        # Run FP4 kernel
        try:
            fp4_out = fmha_sm100.fwd_fp4(q_data, q_sf, k_data, k_sf, v_data, v_sf, None, False, scale)
        except Exception as e:
            print(f"  ERROR: Kernel failed: {e}")
            continue

        # Compare outputs
        fp4_out_float = fp4_out.float()
        ref_out_float = ref_out.float()

        # Check for NaN/Inf
        if torch.isnan(fp4_out_float).any():
            print("  ERROR: FP4 output contains NaN")
            continue
        if torch.isinf(fp4_out_float).any():
            print("  ERROR: FP4 output contains Inf")
            continue

        # Compute correlation
        fp4_flat = fp4_out_float.flatten()
        ref_flat = ref_out_float.flatten()

        correlation = torch.corrcoef(torch.stack([fp4_flat, ref_flat]))[0, 1].item()

        # Compute relative error
        abs_diff = (fp4_out_float - ref_out_float).abs()
        rel_error = abs_diff / (ref_out_float.abs().clamp(min=1e-6))
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()

        print(f"  Correlation: {correlation:.6f}")
        print(f"  Max rel error: {max_rel_error:.4f}")
        print(f"  Mean rel error: {mean_rel_error:.4f}")

        # Pass/Fail
        if correlation > 0.99:
            print("  STATUS: PASS")
        elif correlation > 0.95:
            print("  STATUS: MARGINAL (correlation < 0.99)")
        else:
            print("  STATUS: FAIL (correlation < 0.95)")

def test_causal_mask():
    """Test causal masking."""
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    print("\n" + "="*60)
    print("Causal Mask Test")
    print("="*60)

    batch, heads, seqlen, head_dim = 1, 1, 128, 256

    q_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device) * 0.5
    k_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device) * 0.5
    v_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device) * 0.5

    scale = 1.0 / (head_dim ** 0.5)

    # Quantize
    q_data, q_sf = quantize_to_fp4(q_bf16)
    k_data, k_sf = quantize_to_fp4(k_bf16)
    v_data, v_sf = quantize_to_fp4(v_bf16)

    # Dequantize for reference
    q_deq = dequantize_fp4(q_data, q_sf)
    k_deq = dequantize_fp4(k_data, k_sf)
    v_deq = dequantize_fp4(v_data, v_sf)

    # Reference with causal
    ref_out = reference_attention(q_deq, k_deq, v_deq, scale, is_causal=True)

    # FP4 kernel with causal
    try:
        fp4_out = fmha_sm100.fwd_fp4(q_data, q_sf, k_data, k_sf, v_data, v_sf, None, True, scale)
    except Exception as e:
        print(f"ERROR: Kernel failed: {e}")
        return

    # Compare
    correlation = torch.corrcoef(torch.stack([fp4_out.float().flatten(), ref_out.float().flatten()]))[0, 1].item()
    print(f"Causal correlation: {correlation:.6f}")

    if correlation > 0.99:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")

if __name__ == "__main__":
    test_fp4_correctness()
    test_causal_mask()
