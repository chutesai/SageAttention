"""
Test SM100 FMHA implementation against PyTorch SDPA.

This tests the high-performance CUTLASS example 77 based FMHA
on B200/B300 GPUs.
"""
import torch
import torch.nn.functional as F
import time
import sys

def reference_attention(q, k, v, is_causal=False):
    """Reference implementation using PyTorch SDPA."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def compute_error(output, reference):
    """Compute error metrics."""
    out_f32 = output.float()
    ref_f32 = reference.float()

    abs_error = (out_f32 - ref_f32).abs()
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    cos_sim = F.cosine_similarity(
        out_f32.reshape(-1, output.size(-1)),
        ref_f32.reshape(-1, reference.size(-1)),
        dim=-1
    )
    min_cos_sim = cos_sim.min().item()
    mean_cos_sim = cos_sim.mean().item()

    return {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'min_cos_sim': min_cos_sim,
        'mean_cos_sim': mean_cos_sim,
    }


def test_correctness():
    """Test correctness against SDPA."""
    print("=" * 60)
    print("SM100 FMHA Correctness Test")
    print("=" * 60)

    try:
        import fmha_sm100
        print("Successfully imported fmha_sm100 module")
    except ImportError as e:
        print(f"Failed to import fmha_sm100: {e}")
        print("Please rebuild with: python setup.py install")
        return False

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Test configurations
    configs = [
        (1, 32, 512, 512, 128, False),   # Small, non-causal
        (1, 32, 512, 512, 128, True),    # Small, causal
        (2, 16, 1024, 1024, 128, False), # Medium
        (2, 16, 1024, 1024, 128, True),  # Medium, causal
        (1, 32, 2048, 2048, 128, False), # Large
        (4, 8, 4096, 4096, 128, False),  # Very large (video generation size)
    ]

    all_passed = True

    for batch, heads, seq_q, seq_k, head_dim, is_causal in configs:
        print(f"\nTest: B={batch}, H={heads}, Sq={seq_q}, Sk={seq_k}, D={head_dim}, causal={is_causal}")

        # Create inputs
        q = torch.randn(batch, heads, seq_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)

        # Run reference (SDPA)
        with torch.no_grad():
            ref_output = reference_attention(q, k, v, is_causal=is_causal)

        # Run SM100 FMHA
        try:
            with torch.no_grad():
                fmha_output = fmha_sm100.fwd_auto_scale(q, k, v, is_causal)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  FAIL: {e}")
            all_passed = False
            continue

        # Compute errors
        metrics = compute_error(fmha_output, ref_output)

        # Check pass/fail (BF16 should have high accuracy)
        passed = metrics['mean_cos_sim'] > 0.99 and metrics['min_cos_sim'] > 0.95

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]")
        print(f"    Max Abs Error: {metrics['max_abs_error']:.6f}")
        print(f"    Mean Abs Error: {metrics['mean_abs_error']:.6f}")
        print(f"    Min Cos Sim: {metrics['min_cos_sim']:.6f}")
        print(f"    Mean Cos Sim: {metrics['mean_cos_sim']:.6f}")

        if not passed:
            all_passed = False

    return all_passed


def benchmark():
    """Benchmark SM100 FMHA vs SDPA."""
    print("\n" + "=" * 60)
    print("SM100 FMHA Benchmark vs SDPA")
    print("=" * 60)

    try:
        import fmha_sm100
    except ImportError as e:
        print(f"Failed to import fmha_sm100: {e}")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16
    warmup = 10
    iterations = 100

    # Benchmark configurations (typical video generation sizes)
    configs = [
        (1, 24, 4096, 4096, 128),   # Wan2.2 style
        (4, 24, 4096, 4096, 128),   # Batched
        (1, 32, 8192, 8192, 128),   # Large sequence
        (2, 16, 16384, 16384, 128), # Very large
    ]

    print(f"\n{'Config':<40} {'SDPA (ms)':<12} {'SM100 (ms)':<12} {'Speedup':<10}")
    print("-" * 74)

    for batch, heads, seq_q, seq_k, head_dim in configs:
        config_str = f"B={batch}, H={heads}, S={seq_q}, D={head_dim}"

        q = torch.randn(batch, heads, seq_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)

        # Warmup and benchmark SDPA
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        sdpa_time = (time.perf_counter() - start) / iterations * 1000

        # Warmup and benchmark SM100 FMHA
        try:
            for _ in range(warmup):
                _ = fmha_sm100.fwd_auto_scale(q, k, v, False)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(iterations):
                _ = fmha_sm100.fwd_auto_scale(q, k, v, False)
            torch.cuda.synchronize()
            fmha_time = (time.perf_counter() - start) / iterations * 1000

            speedup = sdpa_time / fmha_time
            print(f"{config_str:<40} {sdpa_time:<12.3f} {fmha_time:<12.3f} {speedup:<10.2f}x")
        except Exception as e:
            print(f"{config_str:<40} {sdpa_time:<12.3f} {'ERROR':<12} - {e}")


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Compute Capability: {props.major}.{props.minor}")

    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        sys.exit(1)

    # Run correctness tests
    passed = test_correctness()

    # Run benchmarks if correctness passed
    if passed:
        benchmark()
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
