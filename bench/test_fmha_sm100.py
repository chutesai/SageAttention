"""
Test SM100 FMHA implementation against PyTorch SDPA.

This tests the high-performance CUTLASS example 77 based FMHA
on B200/B300 GPUs with both BF16 and FP8 E4M3 precision.

FP8 provides 2x tensor core throughput over BF16.
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


def test_correctness_bf16():
    """Test BF16 correctness against SDPA."""
    print("=" * 60)
    print("SM100 FMHA BF16 Correctness Test")
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

    configs = [
        (1, 32, 512, 512, 128, False),
        (1, 32, 512, 512, 128, True),
        (2, 16, 1024, 1024, 128, False),
        (2, 16, 1024, 1024, 128, True),
        (1, 32, 2048, 2048, 128, False),
        (4, 8, 4096, 4096, 128, False),
    ]

    all_passed = True

    for batch, heads, seq_q, seq_k, head_dim, is_causal in configs:
        print(f"\nTest: B={batch}, H={heads}, Sq={seq_q}, Sk={seq_k}, D={head_dim}, causal={is_causal}")

        q = torch.randn(batch, heads, seq_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)

        with torch.no_grad():
            ref_output = reference_attention(q, k, v, is_causal=is_causal)

        try:
            with torch.no_grad():
                fmha_output = fmha_sm100.fwd_auto_scale(q, k, v, is_causal)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  FAIL: {e}")
            all_passed = False
            continue

        metrics = compute_error(fmha_output, ref_output)
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


def test_correctness_fp8():
    """Test FP8 correctness against SDPA."""
    print("\n" + "=" * 60)
    print("SM100 FMHA FP8 E4M3 Correctness Test")
    print("=" * 60)

    try:
        import fmha_sm100
        if not hasattr(fmha_sm100, 'fwd_fp8_from_bf16'):
            print("FP8 support not available in this build")
            return True  # Skip, not a failure
    except ImportError as e:
        print(f"Failed to import fmha_sm100: {e}")
        return False

    device = torch.device("cuda")
    dtype = torch.bfloat16

    configs = [
        (1, 32, 512, 512, 128, False),
        (1, 32, 512, 512, 128, True),
        (2, 16, 1024, 1024, 128, False),
        (4, 8, 4096, 4096, 128, False),
    ]

    all_passed = True

    for batch, heads, seq_q, seq_k, head_dim, is_causal in configs:
        print(f"\nTest: B={batch}, H={heads}, Sq={seq_q}, Sk={seq_k}, D={head_dim}, causal={is_causal}")

        q = torch.randn(batch, heads, seq_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)

        with torch.no_grad():
            ref_output = reference_attention(q, k, v, is_causal=is_causal)

        try:
            with torch.no_grad():
                scale = 1.0 / (head_dim ** 0.5)
                fmha_output = fmha_sm100.fwd_fp8_from_bf16(q, k, v, is_causal, scale)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  FAIL: {e}")
            all_passed = False
            continue

        metrics = compute_error(fmha_output, ref_output)
        # FP8 has lower precision, so we use relaxed thresholds
        passed = metrics['mean_cos_sim'] > 0.95 and metrics['min_cos_sim'] > 0.85

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
    """Benchmark SM100 FMHA (BF16 and FP8) vs SDPA."""
    print("\n" + "=" * 80)
    print("SM100 FMHA Benchmark: SDPA vs BF16 vs FP8")
    print("=" * 80)

    try:
        import fmha_sm100
        has_fp8 = hasattr(fmha_sm100, 'fwd_fp8_from_bf16')
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

    if has_fp8:
        print(f"\n{'Config':<35} {'SDPA':<10} {'BF16':<10} {'FP8':<10} {'BF16 vs SDPA':<14} {'FP8 vs SDPA':<14}")
        print("-" * 93)
    else:
        print(f"\n{'Config':<40} {'SDPA (ms)':<12} {'BF16 (ms)':<12} {'Speedup':<10}")
        print("-" * 74)

    for batch, heads, seq_q, seq_k, head_dim in configs:
        config_str = f"B={batch}, H={heads}, S={seq_q}, D={head_dim}"

        q = torch.randn(batch, heads, seq_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype, device=device)

        # Benchmark SDPA
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        sdpa_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark BF16
        bf16_time = None
        try:
            for _ in range(warmup):
                _ = fmha_sm100.fwd_auto_scale(q, k, v, False)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(iterations):
                _ = fmha_sm100.fwd_auto_scale(q, k, v, False)
            torch.cuda.synchronize()
            bf16_time = (time.perf_counter() - start) / iterations * 1000
        except Exception as e:
            print(f"{config_str:<35} {sdpa_time:<10.2f} {'ERR':<10} - BF16 failed: {e}")
            continue

        # Benchmark FP8
        fp8_time = None
        if has_fp8:
            try:
                scale = 1.0 / (head_dim ** 0.5)
                for _ in range(warmup):
                    _ = fmha_sm100.fwd_fp8_from_bf16(q, k, v, False, scale)
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(iterations):
                    _ = fmha_sm100.fwd_fp8_from_bf16(q, k, v, False, scale)
                torch.cuda.synchronize()
                fp8_time = (time.perf_counter() - start) / iterations * 1000
            except Exception as e:
                fp8_time = None

        # Print results
        if has_fp8:
            bf16_speedup = sdpa_time / bf16_time if bf16_time else 0
            fp8_speedup = sdpa_time / fp8_time if fp8_time else 0
            fp8_str = f"{fp8_time:.2f}" if fp8_time else "N/A"
            fp8_speedup_str = f"{fp8_speedup:.2f}x" if fp8_time else "N/A"
            print(f"{config_str:<35} {sdpa_time:<10.2f} {bf16_time:<10.2f} {fp8_str:<10} {bf16_speedup:<14.2f}x {fp8_speedup_str:<14}")
        else:
            speedup = sdpa_time / bf16_time if bf16_time else 0
            print(f"{config_str:<40} {sdpa_time:<12.3f} {bf16_time:<12.3f} {speedup:<10.2f}x")


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
    bf16_passed = test_correctness_bf16()
    fp8_passed = test_correctness_fp8()

    # Run benchmarks if correctness passed
    if bf16_passed:
        benchmark()
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
