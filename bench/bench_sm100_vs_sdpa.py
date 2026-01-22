"""
Copyright (c) 2025 by SageAttention team.

Benchmark SM100 SageAttention3 vs PyTorch SDPA.

Compares performance of FP4 blockscaled attention on B200/B300 GPUs
against PyTorch's scaled_dot_product_attention (Flash Attention backend).
"""
import torch
import torch.nn.functional as F
import argparse
import sys
import time

# Import SageAttention3
try:
    from sageattn3 import sageattn3_blackwell, get_gpu_arch, is_sm100
except ImportError:
    print("Error: sageattn3 not found. Please run 'python setup.py install' first.")
    sys.exit(1)


def benchmark_kernel(fn, *args, warmup=10, repeats=100, **kwargs):
    """Benchmark a kernel function with warmup and repeated runs."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5
    }


def compute_flops(batch, heads, seq_q, seq_k, head_dim, is_causal):
    """Compute total FLOPs for attention operation."""
    # QK^T: batch * heads * seq_q * seq_k * head_dim * 2
    # Softmax: batch * heads * seq_q * seq_k * 5 (exp, sum, div, etc.)
    # PV: batch * heads * seq_q * seq_k * head_dim * 2
    # For causal, roughly half the elements are computed
    factor = 2 if is_causal else 1
    qk_flops = batch * heads * seq_q * seq_k * head_dim * 2 // factor
    pv_flops = batch * heads * seq_q * seq_k * head_dim * 2 // factor
    return qk_flops + pv_flops


def run_benchmark(batch, heads, seq_len, head_dim, is_causal, dtype, warmup=10, repeats=100):
    """Run benchmark comparing SageAttention3 vs SDPA."""
    # Create inputs
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")

    flops = compute_flops(batch, heads, seq_len, seq_len, head_dim, is_causal)

    results = {}

    # Benchmark SDPA
    try:
        sdpa_fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        results['sdpa'] = benchmark_kernel(sdpa_fn, warmup=warmup, repeats=repeats)
        results['sdpa']['tflops'] = flops / (results['sdpa']['mean_ms'] * 1e-3) / 1e12
    except Exception as e:
        results['sdpa'] = {'error': str(e)}

    # Benchmark SageAttention3
    try:
        sage_fn = lambda: sageattn3_blackwell(q, k, v, is_causal=is_causal, per_block_mean=True)
        results['sage3'] = benchmark_kernel(sage_fn, warmup=warmup, repeats=repeats)
        results['sage3']['tflops'] = flops / (results['sage3']['mean_ms'] * 1e-3) / 1e12
    except Exception as e:
        results['sage3'] = {'error': str(e)}

    # Compute speedup
    if 'mean_ms' in results.get('sdpa', {}) and 'mean_ms' in results.get('sage3', {}):
        results['speedup'] = results['sdpa']['mean_ms'] / results['sage3']['mean_ms']
    else:
        results['speedup'] = None

    return results


def print_header():
    """Print benchmark table header."""
    print(f"{'Config':<35} {'SDPA (ms)':<12} {'Sage3 (ms)':<12} {'SDPA TF/s':<12} {'Sage3 TF/s':<12} {'Speedup':<8}")
    print("-" * 95)


def print_result(config_str, results):
    """Print a single benchmark result row."""
    sdpa_ms = f"{results['sdpa']['mean_ms']:.3f}" if 'mean_ms' in results.get('sdpa', {}) else "ERROR"
    sage_ms = f"{results['sage3']['mean_ms']:.3f}" if 'mean_ms' in results.get('sage3', {}) else "ERROR"
    sdpa_tf = f"{results['sdpa']['tflops']:.2f}" if 'tflops' in results.get('sdpa', {}) else "N/A"
    sage_tf = f"{results['sage3']['tflops']:.2f}" if 'tflops' in results.get('sage3', {}) else "N/A"
    speedup = f"{results['speedup']:.2f}x" if results.get('speedup') else "N/A"

    print(f"{config_str:<35} {sdpa_ms:<12} {sage_ms:<12} {sdpa_tf:<12} {sage_tf:<12} {speedup:<8}")


def main():
    parser = argparse.ArgumentParser(description='SM100 SageAttention3 vs SDPA Benchmark')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--repeats', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--seq_lens', type=str, default='1024,2048,4096,8192,16384',
                        help='Comma-separated sequence lengths')
    parser.add_argument('--causal', action='store_true', help='Run causal attention only')
    parser.add_argument('--non_causal', action='store_true', help='Run non-causal attention only')
    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    arch = get_gpu_arch()
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"Compute Capability: {arch[0]}.{arch[1]}")
    print(f"SM100 (B200/B300): {is_sm100()}")
    print()

    batch = args.batch_size
    heads = args.num_heads
    head_dim = args.head_dim
    seq_lens = [int(s) for s in args.seq_lens.split(',')]

    # Determine which attention modes to test
    if args.causal and args.non_causal:
        causals = [False, True]
    elif args.causal:
        causals = [True]
    elif args.non_causal:
        causals = [False]
    else:
        causals = [False, True]

    dtype = torch.bfloat16

    print(f"Configuration: batch={batch}, heads={heads}, head_dim={head_dim}, dtype={dtype}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    print()

    for is_causal in causals:
        print(f"{'Causal' if is_causal else 'Non-causal'} Attention")
        print("=" * 95)
        print_header()

        for seq_len in seq_lens:
            config_str = f"B={batch} H={heads} S={seq_len} D={head_dim}"
            try:
                results = run_benchmark(
                    batch, heads, seq_len, head_dim, is_causal, dtype,
                    warmup=args.warmup, repeats=args.repeats
                )
                print_result(config_str, results)
            except Exception as e:
                print(f"{config_str:<35} ERROR: {e}")

        print()

    # Summary comparison at different head dimensions
    print("Head Dimension Comparison (seq=4096)")
    print("=" * 95)
    print_header()

    for hd in [128, 256]:
        if hd > 256:  # SM100 FP4 max supported
            continue
        config_str = f"B={batch} H={heads} S=4096 D={hd}"
        try:
            results = run_benchmark(
                batch, heads, 4096, hd, False, dtype,
                warmup=args.warmup, repeats=args.repeats
            )
            print_result(config_str, results)
        except Exception as e:
            print(f"{config_str:<35} ERROR: {e}")


if __name__ == "__main__":
    main()
