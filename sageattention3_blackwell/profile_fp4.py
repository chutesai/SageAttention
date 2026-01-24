#!/usr/bin/env python3
"""
Profile FP4 kernel to understand bottlenecks.
"""

import torch
import time

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

def main():
    device = torch.device('cuda:0')

    # Small test first
    print("Testing small sizes:")
    for seqlen in [128, 256, 512, 1024]:
        batch, heads, head_dim = 1, 1, 256

        q = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)

        q_data, q_sf = quantize_to_fp4(q)
        k_data, k_sf = quantize_to_fp4(k)
        v_data, v_sf = quantize_to_fp4(v)

        scale = 1.0 / (head_dim ** 0.5)

        # Warmup
        for _ in range(3):
            out = fmha_sm100.fwd_fp4(q_data, q_sf, k_data, k_sf, v_data, v_sf, None, False, scale)
        torch.cuda.synchronize()

        # Time
        start = time.perf_counter()
        for _ in range(10):
            out = fmha_sm100.fwd_fp4(q_data, q_sf, k_data, k_sf, v_data, v_sf, None, False, scale)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 10 * 1000

        # Compute expected FLOPs
        flops = 4 * batch * heads * seqlen * seqlen * head_dim
        tflops = flops / (elapsed * 1e-3) / 1e12

        # Compute theoretical memory bandwidth usage
        # The kernel does seqlen * seqlen memory accesses per row
        # Each access reads from global memory (no caching in scalar impl)
        mem_accesses = seqlen * seqlen * head_dim  # rough estimate

        print(f"  seqlen={seqlen:4d}: {elapsed:8.3f} ms, {tflops:.3f} TFLOPS")

    print("\nBottleneck analysis:")
    print("The FP4 kernel is O(seqlen^2) in computation but the scalar")
    print("implementation has very poor memory access patterns:")
    print("- Each thread reads Q from SMEM once")
    print("- Each thread reads K from global memory for EVERY K position")
    print("- Each thread reads V from global memory for EVERY K position")
    print("- No tensor cores, just scalar FP4 decoding")
    print("\nTo fix: Need proper SMEM staging and tensor core MMA")

if __name__ == "__main__":
    main()
