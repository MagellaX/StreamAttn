"""Probe certified attention skip rates and error bounds.

This script is intentionally small and dependency-light. It is meant to answer
one question before kernel work: do the block summaries produce useful skip
rates for a given activation pattern and error budget?
"""

import argparse
import time

import torch

from stream_attention.certified import certified_attention


def _make_inputs(args, device):
    dtype = torch.float32
    q = torch.randn(args.batch, args.seq_q, args.heads, args.dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.seq_k, args.heads, args.dim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.seq_k, args.heads, args.dim, device=device, dtype=dtype)

    if args.pattern == "peaked":
        q.zero_()
        k.zero_()
        q[..., 0] = args.peak
        first = min(args.block_size, args.seq_k)
        k[:, :first, :, 0] = args.peak
        k[:, first:, :, 0] = -args.peak
    elif args.pattern == "local":
        q.zero_()
        k.zero_()
        q[..., 0] = args.peak
        for start in range(0, args.seq_k, args.block_size):
            end = min(start + args.block_size, args.seq_k)
            sign = 1.0 if (start // args.block_size) % 4 == 0 else -1.0
            k[:, start:end, :, 0] = sign * args.peak

    return q, k, v


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_call(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    result = None
    for _ in range(iters):
        result = fn()
    _sync(device)
    return result, (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-q", type=int, default=512)
    parser.add_argument("--seq-k", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--skip-predicate", choices=["mass", "value_bound"], default="value_bound")
    parser.add_argument("--block-order", choices=["sequential", "reverse", "sink_local", "summary_desc"], default="sequential")
    parser.add_argument("--num-summary-outliers", type=int, default=0)
    parser.add_argument("--post-qk-threshold", type=float, default=0.0)
    parser.add_argument("--disable-summary-gate", dest="enable_summary_gate", action="store_false")
    parser.add_argument("--disable-post-qk-gate", dest="enable_post_qk_gate", action="store_false")
    parser.add_argument("--pattern", choices=["random", "peaked", "local"], default="peaked")
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q, k, v = _make_inputs(args, device)

    exact_fn = lambda: certified_attention(
        q,
        k,
        v,
        causal=args.causal,
        error_budget=0.0,
        block_size=args.block_size,
        skip_predicate=args.skip_predicate,
        block_order=args.block_order,
        num_summary_outliers=args.num_summary_outliers,
        enable_summary_gate=args.enable_summary_gate,
        enable_post_qk_gate=args.enable_post_qk_gate,
        post_qk_threshold=args.post_qk_threshold,
        return_stats=True,
    )
    cert_fn = lambda: certified_attention(
        q,
        k,
        v,
        causal=args.causal,
        error_budget=args.error_budget,
        block_size=args.block_size,
        skip_predicate=args.skip_predicate,
        block_order=args.block_order,
        num_summary_outliers=args.num_summary_outliers,
        enable_summary_gate=args.enable_summary_gate,
        enable_post_qk_gate=args.enable_post_qk_gate,
        post_qk_threshold=args.post_qk_threshold,
        return_stats=True,
    )

    exact, exact_s = _time_call(exact_fn, device, args.warmup, args.iters)
    cert, cert_s = _time_call(cert_fn, device, args.warmup, args.iters)

    err = torch.linalg.vector_norm(cert.output - exact.output, dim=-1)
    print(f"device={device.type} pattern={args.pattern} causal={args.causal}")
    print(f"shape=batch:{args.batch} seq_q:{args.seq_q} seq_k:{args.seq_k} heads:{args.heads} dim:{args.dim}")
    print(f"block_size={args.block_size} error_budget={args.error_budget:g}")
    print(f"skip_predicate={args.skip_predicate} block_order={args.block_order} outliers={args.num_summary_outliers}")
    print(f"exact_time_ms={exact_s * 1000:.3f}")
    print(f"certified_time_ms={cert_s * 1000:.3f}")
    print(f"skip_fraction={cert.stats.skip_fraction:.4f}")
    print(f"skipped_row_blocks={cert.stats.skipped_row_blocks}")
    print(f"skipped_pre_k_row_blocks={cert.stats.skipped_pre_k_row_blocks}")
    print(f"skipped_post_qk_row_blocks={cert.stats.skipped_post_qk_row_blocks}")
    print(f"computed_row_blocks={cert.stats.computed_row_blocks}")
    print(f"max_observed_error={err.max().item():.6g}")
    print(f"max_error_bound={cert.stats.max_error_bound:.6g}")
    print(f"mean_error_bound={cert.stats.mean_error_bound:.6g}")


if __name__ == "__main__":
    main()
