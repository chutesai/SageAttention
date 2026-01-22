"""
Copyright (c) 2025 by SageAttention team.

Correctness tests for SM100 SageAttention3 vs PyTorch SDPA.

Tests the FP4 blockscaled attention implementation on B200/B300 GPUs
against the reference PyTorch scaled_dot_product_attention.
"""
import torch
import torch.nn.functional as F
import argparse
import sys
import os

# Import SageAttention3
try:
    from sageattn3 import sageattn3_blackwell, get_gpu_arch, is_sm100
except ImportError:
    print("Error: sageattn3 not found. Please run 'python setup.py install' first.")
    sys.exit(1)


def reference_attention(q, k, v, is_causal=False):
    """Reference implementation using PyTorch SDPA.

    SDPA works on B200 (uses Flash Attention backend, not cuBLAS).
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def test_preprocessing_only():
    """Test just the preprocessing and quantization steps (no kernel)."""
    from sageattn3.api import (
        preprocess_qkv, scale_and_quant_fp4, scale_and_quant_fp4_permute,
        scale_and_quant_fp4_transpose
    )

    print("Testing preprocessing pipeline (no kernel)...")

    B, H, S, D = 1, 32, 512, 128
    dtype = torch.bfloat16

    q = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, S, D, dtype=dtype, device="cuda")

    try:
        # Test preprocess
        q_p, k_p, v_p, delta_s = preprocess_qkv(q, k, v, per_block_mean=True)
        print(f"  preprocess_qkv: PASS")
        print(f"    q: {q_p.shape}, k: {k_p.shape}, v: {v_p.shape}, delta_s: {delta_s.shape}")

        # Test quantization
        q_fp4, q_scale = scale_and_quant_fp4(q_p)
        print(f"  scale_and_quant_fp4 (Q): PASS - {q_fp4.shape}, {q_scale.shape}")

        k_fp4, k_scale = scale_and_quant_fp4_permute(k_p)
        print(f"  scale_and_quant_fp4_permute (K): PASS - {k_fp4.shape}, {k_scale.shape}")

        v_fp4, v_scale = scale_and_quant_fp4_transpose(v_p)
        print(f"  scale_and_quant_fp4_transpose (V): PASS - {v_fp4.shape}, {v_scale.shape}")

        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def compute_error_metrics(output, reference):
    """Compute various error metrics between output and reference."""
    # Convert to float32 for accurate error computation
    out_f32 = output.float()
    ref_f32 = reference.float()

    # Absolute error
    abs_error = (out_f32 - ref_f32).abs()
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    # Relative error (avoid division by zero)
    ref_abs = ref_f32.abs()
    rel_error = abs_error / (ref_abs + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    # Cosine similarity (per sequence position)
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
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'min_cos_sim': min_cos_sim,
        'mean_cos_sim': mean_cos_sim,
    }


def run_test(batch, heads, seq_len, head_dim, is_causal, per_block_mean, dtype, verbose=True):
    """Run a single correctness test."""
    # Create random inputs
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")

    # Run reference
    with torch.no_grad():
        ref_output = reference_attention(q, k, v, is_causal=is_causal)

    # Run SageAttention3
    with torch.no_grad():
        sage_output = sageattn3_blackwell(q, k, v, is_causal=is_causal, per_block_mean=per_block_mean)

    # Compute metrics
    metrics = compute_error_metrics(sage_output, ref_output)

    # Determine pass/fail based on reasonable FP4 quantization error thresholds
    # FP4 has limited precision, so we expect larger errors than typical FP16/BF16
    passed = (
        metrics['mean_cos_sim'] > 0.95 and  # At least 95% average similarity
        metrics['min_cos_sim'] > 0.85       # No position drops below 85% similarity
    )

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] B={batch}, H={heads}, S={seq_len}, D={head_dim}, "
              f"causal={is_causal}, per_block_mean={per_block_mean}, dtype={dtype}")
        print(f"       Max Abs Error: {metrics['max_abs_error']:.6f}, "
              f"Mean Abs Error: {metrics['mean_abs_error']:.6f}")
        print(f"       Max Rel Error: {metrics['max_rel_error']:.4f}, "
              f"Mean Rel Error: {metrics['mean_rel_error']:.4f}")
        print(f"       Min Cos Sim: {metrics['min_cos_sim']:.6f}, "
              f"Mean Cos Sim: {metrics['mean_cos_sim']:.6f}")

    return passed, metrics


def main():
    parser = argparse.ArgumentParser(description='SM100 SageAttention3 Correctness Tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer configs)')
    parser.add_argument('--preprocess-only', action='store_true',
                        help='Only test preprocessing (skip kernel)')
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

    # Test preprocessing only mode
    if args.preprocess_only:
        success = test_preprocessing_only()
        sys.exit(0 if success else 1)

    # Test configurations
    if args.quick:
        batch_sizes = [1]
        head_counts = [32]
        seq_lens = [512, 2048]
        head_dims = [128]
        causals = [False]
        per_block_means = [True]
        dtypes = [torch.bfloat16]
    else:
        batch_sizes = [1, 2, 4]
        head_counts = [8, 16, 32]
        seq_lens = [256, 512, 1024, 2048, 4096]
        head_dims = [128, 256]  # SM100 with FP4 requires HeadDim >= 128
        causals = [False, True]
        per_block_means = [False, True]
        dtypes = [torch.bfloat16, torch.float16]

    print("Running correctness tests...")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_configs = []

    for dtype in dtypes:
        print(f"\nDtype: {dtype}")
        print("-" * 60)

        for head_dim in head_dims:
            for batch in batch_sizes:
                for heads in head_counts:
                    for seq_len in seq_lens:
                        for is_causal in causals:
                            for per_block_mean in per_block_means:
                                try:
                                    passed, metrics = run_test(
                                        batch, heads, seq_len, head_dim,
                                        is_causal, per_block_mean, dtype,
                                        verbose=args.verbose
                                    )
                                    total_tests += 1
                                    if passed:
                                        passed_tests += 1
                                    else:
                                        failed_configs.append({
                                            'batch': batch,
                                            'heads': heads,
                                            'seq_len': seq_len,
                                            'head_dim': head_dim,
                                            'is_causal': is_causal,
                                            'per_block_mean': per_block_mean,
                                            'dtype': str(dtype),
                                            'metrics': metrics
                                        })
                                except Exception as e:
                                    total_tests += 1
                                    print(f"  [ERROR] B={batch}, H={heads}, S={seq_len}, "
                                          f"D={head_dim}: {e}")
                                    failed_configs.append({
                                        'batch': batch,
                                        'heads': heads,
                                        'seq_len': seq_len,
                                        'head_dim': head_dim,
                                        'is_causal': is_causal,
                                        'per_block_mean': per_block_mean,
                                        'dtype': str(dtype),
                                        'error': str(e)
                                    })

    print("\n" + "=" * 80)
    print(f"Results: {passed_tests}/{total_tests} tests passed")

    if failed_configs:
        print(f"\nFailed configurations ({len(failed_configs)}):")
        for cfg in failed_configs:
            if 'error' in cfg:
                print(f"  Error: B={cfg['batch']}, H={cfg['heads']}, S={cfg['seq_len']}, "
                      f"D={cfg['head_dim']}, causal={cfg['is_causal']}, "
                      f"pbm={cfg['per_block_mean']}, dtype={cfg['dtype']}: {cfg['error']}")
            else:
                print(f"  Failed: B={cfg['batch']}, H={cfg['heads']}, S={cfg['seq_len']}, "
                      f"D={cfg['head_dim']}, causal={cfg['is_causal']}, "
                      f"pbm={cfg['per_block_mean']}, dtype={cfg['dtype']}, "
                      f"cos_sim={cfg['metrics']['mean_cos_sim']:.4f}")

    sys.exit(0 if passed_tests == total_tests else 1)


if __name__ == "__main__":
    main()
