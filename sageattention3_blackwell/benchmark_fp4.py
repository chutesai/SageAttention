#!/usr/bin/env python3
"""
SM100 FP4 Attention Performance Benchmark

Measures throughput of FP4 block-scaled attention vs BF16 and FP8 baselines.
"""

import torch
import time
import argparse
from typing import Optional, Tuple

# Check CUDA availability
assert torch.cuda.is_available(), "CUDA required"

# Import SM100 kernels
try:
    import sageattn3.sm100.fmha_sm100 as fmha_sm100
    HAS_SM100 = True
except ImportError as e:
    print(f"Warning: Could not import SM100 kernels: {e}")
    HAS_SM100 = False

def get_gpu_info():
    """Get GPU information."""
    props = torch.cuda.get_device_properties(0)
    return {
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'sm_count': props.multi_processor_count,
        'memory_gb': props.total_memory / (1024**3),
    }

def quantize_to_fp4(tensor: torch.Tensor, block_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize BF16 tensor to FP4 with block-wise scale factors.

    Returns:
        data: uint8 tensor with packed FP4 values (2 per byte)
        sf: FP8 E4M3 scale factors
    """
    batch, heads, seq, dim = tensor.shape
    assert dim % block_size == 0, f"dim {dim} must be divisible by block_size {block_size}"

    num_blocks = dim // block_size

    # Reshape to blocks
    blocks = tensor.view(batch, heads, seq, num_blocks, block_size)

    # Compute per-block scale factors
    absmax = blocks.abs().amax(dim=-1)  # [batch, heads, seq, num_blocks]

    # FP4 E2M1 max value is 6.0
    fp4_max = 6.0
    scales = absmax / fp4_max
    scales = scales.clamp(min=1e-7)

    # Convert scales to FP8 E4M3
    sf = scales.to(torch.float8_e4m3fn)

    # Quantize data
    scaled = blocks / scales.unsqueeze(-1)
    scaled = scaled.clamp(-fp4_max, fp4_max)

    # Convert to linear FP4 encoding: value = (nibble - 7.5) * 0.8
    # So: nibble = value / 0.8 + 7.5 = value * 1.25 + 7.5
    int_vals = (scaled * 1.25 + 7.5).round().clamp(0, 15).to(torch.int32)
    int_vals = int_vals.view(batch, heads, seq, dim)

    # Pack pairs of 4-bit values into uint8
    even = int_vals[..., 0::2]
    odd = int_vals[..., 1::2]
    packed = ((even & 0x0F) | ((odd & 0x0F) << 4)).to(torch.uint8)

    return packed, sf

def benchmark_kernel(
    kernel_fn,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    **kwargs
) -> Tuple[float, float]:
    """
    Benchmark a kernel function.

    Returns:
        mean_ms: Mean execution time in milliseconds
        std_ms: Standard deviation in milliseconds
    """
    # Warmup
    for _ in range(warmup_iters):
        _ = kernel_fn(**kwargs)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = kernel_fn(**kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return sum(times) / len(times), (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5

def compute_tflops(batch: int, heads: int, seqlen_q: int, seqlen_k: int, head_dim: int, time_ms: float) -> float:
    """Compute TFLOPS for attention."""
    # FLOPs = 2 * batch * heads * seqlen_q * seqlen_k * head_dim (for QK^T)
    #       + 2 * batch * heads * seqlen_q * seqlen_k * head_dim (for PV)
    #       = 4 * batch * heads * seqlen_q * seqlen_k * head_dim
    flops = 4 * batch * heads * seqlen_q * seqlen_k * head_dim
    return flops / (time_ms * 1e-3) / 1e12

def run_benchmarks(
    batch: int = 1,
    heads: int = 32,
    seqlen: int = 4096,
    head_dim: int = 128,
    is_causal: bool = False,
    warmup: int = 10,
    iters: int = 100,
):
    """Run benchmarks for all kernel variants."""

    print(f"\n{'='*70}")
    print(f"SM100 Attention Benchmark")
    print(f"{'='*70}")

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']}")
    print(f"Compute: {gpu_info['compute_capability']}, SMs: {gpu_info['sm_count']}, Memory: {gpu_info['memory_gb']:.1f} GB")
    print(f"\nConfig: batch={batch}, heads={heads}, seqlen={seqlen}, head_dim={head_dim}, causal={is_causal}")
    print(f"Warmup: {warmup} iters, Benchmark: {iters} iters")

    device = torch.device('cuda:0')
    scale = 1.0 / (head_dim ** 0.5)

    results = {}

    # =========================================================================
    # BF16 Baseline (if head_dim=128)
    # =========================================================================
    if head_dim == 128 and HAS_SM100:
        print(f"\n{'-'*70}")
        print("BF16 Attention (head_dim=128)")
        print(f"{'-'*70}")

        q_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        k_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        v_bf16 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)

        try:
            mean_ms, std_ms = benchmark_kernel(
                fmha_sm100.fwd_bf16,
                warmup_iters=warmup,
                bench_iters=iters,
                q=q_bf16, k=k_bf16, v=v_bf16,
                is_causal=is_causal, scale=scale
            )
            tflops = compute_tflops(batch, heads, seqlen, seqlen, head_dim, mean_ms)
            results['bf16'] = {'time_ms': mean_ms, 'std_ms': std_ms, 'tflops': tflops}
            print(f"  Time: {mean_ms:.3f} ± {std_ms:.3f} ms")
            print(f"  TFLOPS: {tflops:.2f}")
        except Exception as e:
            print(f"  ERROR: {e}")

        del q_bf16, k_bf16, v_bf16
        torch.cuda.empty_cache()

    # =========================================================================
    # FP8 (if head_dim=128)
    # =========================================================================
    if head_dim == 128 and HAS_SM100:
        print(f"\n{'-'*70}")
        print("FP8 E4M3 Attention (head_dim=128)")
        print(f"{'-'*70}")

        # Create FP8 inputs
        q_fp8 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        k_fp8 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        v_fp8 = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)

        # Quantize to FP8
        q_fp8 = q_fp8.to(torch.float8_e4m3fn)
        k_fp8 = k_fp8.to(torch.float8_e4m3fn)
        v_fp8 = v_fp8.to(torch.float8_e4m3fn)

        try:
            mean_ms, std_ms = benchmark_kernel(
                fmha_sm100.fwd_fp8,
                warmup_iters=warmup,
                bench_iters=iters,
                q=q_fp8, k=k_fp8, v=v_fp8,
                is_causal=is_causal, scale=scale,
                scale_q=1.0, scale_k=1.0, scale_v=1.0
            )
            tflops = compute_tflops(batch, heads, seqlen, seqlen, head_dim, mean_ms)
            results['fp8'] = {'time_ms': mean_ms, 'std_ms': std_ms, 'tflops': tflops}
            print(f"  Time: {mean_ms:.3f} ± {std_ms:.3f} ms")
            print(f"  TFLOPS: {tflops:.2f}")
            if 'bf16' in results:
                speedup = results['bf16']['time_ms'] / mean_ms
                print(f"  Speedup vs BF16: {speedup:.2f}x")
        except Exception as e:
            print(f"  ERROR: {e}")

        del q_fp8, k_fp8, v_fp8
        torch.cuda.empty_cache()

    # =========================================================================
    # FP4 Block-Scaled (head_dim=256 required)
    # =========================================================================
    print(f"\n{'-'*70}")
    print(f"FP4 Block-Scaled Attention (head_dim=256)")
    print(f"{'-'*70}")

    # FP4 requires head_dim=256
    fp4_head_dim = 256
    fp4_seqlen = seqlen

    # Adjust heads to keep similar total compute
    fp4_heads = max(1, heads * head_dim // fp4_head_dim) if head_dim != 256 else heads

    print(f"  Adjusted config: heads={fp4_heads}, head_dim={fp4_head_dim}")

    # Create inputs
    q_raw = torch.randn(batch, fp4_heads, fp4_seqlen, fp4_head_dim, dtype=torch.bfloat16, device=device)
    k_raw = torch.randn(batch, fp4_heads, fp4_seqlen, fp4_head_dim, dtype=torch.bfloat16, device=device)
    v_raw = torch.randn(batch, fp4_heads, fp4_seqlen, fp4_head_dim, dtype=torch.bfloat16, device=device)

    # Quantize to FP4
    q_data, q_sf = quantize_to_fp4(q_raw)
    k_data, k_sf = quantize_to_fp4(k_raw)
    v_data, v_sf = quantize_to_fp4(v_raw)

    fp4_scale = 1.0 / (fp4_head_dim ** 0.5)

    if HAS_SM100:
        try:
            mean_ms, std_ms = benchmark_kernel(
                fmha_sm100.fwd_fp4,
                warmup_iters=warmup,
                bench_iters=iters,
                q_data=q_data, q_sf=q_sf,
                k_data=k_data, k_sf=k_sf,
                v_data=v_data, v_sf=v_sf,
                delta_s=None,
                is_causal=is_causal, scale=fp4_scale
            )
            tflops = compute_tflops(batch, fp4_heads, fp4_seqlen, fp4_seqlen, fp4_head_dim, mean_ms)
            results['fp4'] = {'time_ms': mean_ms, 'std_ms': std_ms, 'tflops': tflops}
            print(f"  Time: {mean_ms:.3f} ± {std_ms:.3f} ms")
            print(f"  TFLOPS: {tflops:.2f}")
            if 'bf16' in results:
                # Normalize by FLOPs ratio
                bf16_flops = 4 * batch * heads * seqlen * seqlen * head_dim
                fp4_flops = 4 * batch * fp4_heads * fp4_seqlen * fp4_seqlen * fp4_head_dim
                normalized_speedup = (results['bf16']['time_ms'] / mean_ms) * (fp4_flops / bf16_flops)
                print(f"  Normalized speedup vs BF16: {normalized_speedup:.2f}x")
        except Exception as e:
            print(f"  ERROR: {e}")

    del q_raw, k_raw, v_raw, q_data, q_sf, k_data, k_sf, v_data, v_sf
    torch.cuda.empty_cache()

    # =========================================================================
    # PyTorch SDPA Baseline
    # =========================================================================
    print(f"\n{'-'*70}")
    print(f"PyTorch SDPA (BF16, head_dim={head_dim})")
    print(f"{'-'*70}")

    q_sdpa = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
    k_sdpa = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
    v_sdpa = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)

    def sdpa_fn(q, k, v, is_causal, scale):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=scale
        )

    try:
        mean_ms, std_ms = benchmark_kernel(
            sdpa_fn,
            warmup_iters=warmup,
            bench_iters=iters,
            q=q_sdpa, k=k_sdpa, v=v_sdpa,
            is_causal=is_causal, scale=scale
        )
        tflops = compute_tflops(batch, heads, seqlen, seqlen, head_dim, mean_ms)
        results['sdpa'] = {'time_ms': mean_ms, 'std_ms': std_ms, 'tflops': tflops}
        print(f"  Time: {mean_ms:.3f} ± {std_ms:.3f} ms")
        print(f"  TFLOPS: {tflops:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    del q_sdpa, k_sdpa, v_sdpa
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Kernel':<20} {'Time (ms)':<15} {'TFLOPS':<12} {'vs BF16':<12}")
    print(f"{'-'*70}")

    bf16_time = results.get('bf16', {}).get('time_ms', None)

    for name, data in results.items():
        speedup = ""
        if bf16_time and name != 'bf16':
            speedup = f"{bf16_time / data['time_ms']:.2f}x"
        print(f"{name:<20} {data['time_ms']:<15.3f} {data['tflops']:<12.2f} {speedup:<12}")

    return results

def run_scaling_benchmark(
    batch: int = 1,
    heads: int = 32,
    head_dim: int = 256,
    is_causal: bool = False,
    warmup: int = 5,
    iters: int = 50,
):
    """Run benchmarks across different sequence lengths."""

    print(f"\n{'='*70}")
    print(f"FP4 Scaling Benchmark (batch={batch}, heads={heads}, head_dim={head_dim})")
    print(f"{'='*70}")

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']}")

    device = torch.device('cuda:0')
    scale = 1.0 / (head_dim ** 0.5)

    seqlens = [256, 512, 1024, 2048, 4096, 8192]

    print(f"\n{'SeqLen':<10} {'Time (ms)':<15} {'TFLOPS':<12} {'GB/s':<12}")
    print(f"{'-'*50}")

    for seqlen in seqlens:
        try:
            # Create and quantize inputs
            q_raw = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
            k_raw = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
            v_raw = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)

            q_data, q_sf = quantize_to_fp4(q_raw)
            k_data, k_sf = quantize_to_fp4(k_raw)
            v_data, v_sf = quantize_to_fp4(v_raw)

            mean_ms, std_ms = benchmark_kernel(
                fmha_sm100.fwd_fp4,
                warmup_iters=warmup,
                bench_iters=iters,
                q_data=q_data, q_sf=q_sf,
                k_data=k_data, k_sf=k_sf,
                v_data=v_data, v_sf=v_sf,
                delta_s=None,
                is_causal=is_causal, scale=scale
            )

            tflops = compute_tflops(batch, heads, seqlen, seqlen, head_dim, mean_ms)

            # Compute memory bandwidth
            # Read: Q, K, V data (packed FP4) + scale factors
            # Write: Output (BF16)
            bytes_read = batch * heads * seqlen * head_dim // 2 * 3  # Q, K, V packed
            bytes_read += batch * heads * seqlen * (head_dim // 16) * 3  # scale factors
            bytes_write = batch * heads * seqlen * head_dim * 2  # BF16 output
            total_bytes = bytes_read + bytes_write
            bandwidth_gbs = total_bytes / (mean_ms * 1e-3) / 1e9

            print(f"{seqlen:<10} {mean_ms:<15.3f} {tflops:<12.2f} {bandwidth_gbs:<12.1f}")

            del q_raw, k_raw, v_raw, q_data, q_sf, k_data, k_sf, v_data, v_sf
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{seqlen:<10} ERROR: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SM100 FP4 Attention Benchmark")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--seqlen", type=int, default=4096, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (128 for BF16/FP8, 256 for FP4)")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark across sequence lengths")

    args = parser.parse_args()

    if args.scaling:
        run_scaling_benchmark(
            batch=args.batch,
            heads=args.heads,
            head_dim=256,  # FP4 requires 256
            is_causal=args.causal,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        run_benchmarks(
            batch=args.batch,
            heads=args.heads,
            seqlen=args.seqlen,
            head_dim=args.head_dim,
            is_causal=args.causal,
            warmup=args.warmup,
            iters=args.iters,
        )
