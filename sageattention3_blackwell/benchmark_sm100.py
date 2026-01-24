#!/usr/bin/env python3
"""
Benchmark SM100 FMHA kernels (BF16, FP8, FP4).
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmha_sm100


def benchmark_kernel(fn, warmup=10, iterations=100):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iterations * 1000  # ms


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
    torch.manual_seed(42)

    print("=" * 70)
    print("SM100 FMHA Benchmark")
    print("=" * 70)

    # Test configurations
    configs = [
        # (batch, heads, seqlen, head_dim)
        (1, 32, 1024, 128),   # Standard config for BF16/FP8
        (1, 32, 2048, 128),
        (1, 32, 4096, 128),
        (1, 16, 1024, 256),   # HeadDim=256 for FP4
        (1, 16, 2048, 256),
        (1, 16, 4096, 256),
    ]

    for batch, heads, seqlen, head_dim in configs:
        print(f"\nConfig: batch={batch}, heads={heads}, seqlen={seqlen}, head_dim={head_dim}")
        print("-" * 60)

        # Create input tensors
        q_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        k_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        v_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        scale = 1.0 / (head_dim ** 0.5)

        # Compute TFLOPS (2 * batch * heads * seqlen^2 * head_dim for QK, same for PV)
        flops = 4 * batch * heads * seqlen * seqlen * head_dim

        # BF16 benchmark
        try:
            def run_bf16():
                return fmha_sm100.fwd_bf16(q_bf16, k_bf16, v_bf16, None, False, scale)

            time_bf16 = benchmark_kernel(run_bf16)
            tflops_bf16 = flops / (time_bf16 / 1000) / 1e12
            print(f"  BF16:  {time_bf16:8.3f} ms  ({tflops_bf16:7.1f} TFLOPS)")
        except Exception as e:
            print(f"  BF16:  ERROR - {e}")

        # FP8 benchmark
        try:
            q_fp8 = q_bf16.to(torch.float8_e4m3fn)
            k_fp8 = k_bf16.to(torch.float8_e4m3fn)
            v_fp8 = v_bf16.to(torch.float8_e4m3fn)

            def run_fp8():
                return fmha_sm100.fwd_fp8(q_fp8, k_fp8, v_fp8, None, False, scale)

            time_fp8 = benchmark_kernel(run_fp8)
            tflops_fp8 = flops / (time_fp8 / 1000) / 1e12
            print(f"  FP8:   {time_fp8:8.3f} ms  ({tflops_fp8:7.1f} TFLOPS)")
        except Exception as e:
            print(f"  FP8:   ERROR - {e}")

        # FP4 benchmark (only for head_dim=256)
        if head_dim == 256:
            try:
                # Quantize to FP4
                q_data, q_sf = quantize_to_fp4(q_bf16)
                k_data, k_sf = quantize_to_fp4(k_bf16)
                v_data, v_sf = quantize_to_fp4(v_bf16)

                def run_fp4():
                    return fmha_sm100.fwd_fp4(q_data, q_sf, k_data, k_sf, v_data, v_sf, None, False, scale)

                time_fp4 = benchmark_kernel(run_fp4)
                tflops_fp4 = flops / (time_fp4 / 1000) / 1e12
                print(f"  FP4:   {time_fp4:8.3f} ms  ({tflops_fp4:7.1f} TFLOPS)")

                # Show speedup/slowdown
                if 'time_bf16' in dir():
                    ratio = time_fp4 / time_bf16
                    if ratio > 1:
                        print(f"         (FP4 is {ratio:.1f}x SLOWER than BF16 - scalar fallback)")
                    else:
                        print(f"         (FP4 is {1/ratio:.1f}x FASTER than BF16)")
            except Exception as e:
                print(f"  FP4:   ERROR - {e}")

    print("\n" + "=" * 70)
    print("Notes:")
    print("- FP4 currently uses scalar fallback (no tensor cores yet)")
    print("- Target: FP4 should be faster than BF16 with tensor cores")
    print("- HeadDim=256 required for FP4 (MMA K=64 constraint)")
    print("=" * 70)


if __name__ == "__main__":
    main()
