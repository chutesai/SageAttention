import argparse
import time

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

from sageattn3.api import sageattn3_blackwell


def is_sm100():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 10 and minor == 0


def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    return p50, p90


def tflops_ms(ms, b, h, q, k, d):
    flops = 4.0 * b * h * q * k * d
    return (flops / 1e12) / (ms / 1e3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--qlen", type=int, default=2048)
    parser.add_argument("--klen", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--per-block-mean", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not is_sm100():
        raise SystemExit("SM100 GPU required for this benchmark.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = "cuda"

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.heads, args.qlen, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.heads, args.klen, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.heads, args.klen, args.head_dim, device=device, dtype=dtype)

    def run_sage():
        sageattn3_blackwell(q, k, v, is_causal=args.causal, per_block_mean=args.per_block_mean)

    def run_sdpa():
        sdpa(q, k, v, is_causal=args.causal)

    sage_p50, sage_p90 = time_ms(run_sage, args.warmup, args.iters)
    sdpa_p50, sdpa_p90 = time_ms(run_sdpa, args.warmup, args.iters)

    sage_tflops = tflops_ms(sage_p50, args.batch, args.heads, args.qlen, args.klen, args.head_dim)
    sdpa_tflops = tflops_ms(sdpa_p50, args.batch, args.heads, args.qlen, args.klen, args.head_dim)

    print("SM100 FP4 attention benchmark")
    print(f"shape: B={args.batch} H={args.heads} Q={args.qlen} K={args.klen} D={args.head_dim}")
    print(f"dtype: {args.dtype} causal={args.causal} per_block_mean={args.per_block_mean}")
    print(f"sageattn3_blackwell: p50={sage_p50:.3f} ms p90={sage_p90:.3f} ms ~{sage_tflops:.2f} TFLOP/s")
    print(f"sdpa:               p50={sdpa_p50:.3f} ms p90={sdpa_p90:.3f} ms ~{sdpa_tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
