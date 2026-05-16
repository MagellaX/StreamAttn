"""Profile the Gate-1 Triton diagnostic kernel.

This script keeps allocations and optional value-norm-bound construction out of
the timed region when requested. It is intended for CUDA machines or Modal runs,
not CPU-only CI.
"""

import argparse
import json
from typing import Optional

import torch

from stream_attention.kernels.gate1_fwd_triton import (
    build_value_norm_bounds,
    gate1_attention_triton_forward,
)


def _make_tensors(args):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda")
    shape_q = (args.batch, args.seq_q, args.heads, args.dim)
    shape_kv = (args.batch, args.seq_k, args.heads, args.dim)

    if args.pattern == "random":
        query = torch.randn(shape_q, device=device, dtype=dtype)
        key = torch.randn(shape_kv, device=device, dtype=dtype)
        value = torch.randn(shape_kv, device=device, dtype=dtype)
    elif args.pattern == "peaked":
        query = torch.zeros(shape_q, device=device, dtype=dtype)
        key = torch.zeros(shape_kv, device=device, dtype=dtype)
        value = torch.randn(shape_kv, device=device, dtype=dtype)
        query[..., 0] = args.peak
        key[:, : args.block_size, :, 0] = args.peak
        key[:, args.block_size :, :, 0] = -args.peak
    else:
        raise ValueError(f"unknown pattern: {args.pattern}")

    return query, key, value


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _summarize_stats(raw_stats: torch.Tensor):
    totals = raw_stats.detach().sum(dim=(0, 1, 2)).cpu()
    return {
        "row_skips": int(totals[0].item()),
        "row_computes": int(totals[1].item()),
        "cta_tiles_total": int(totals[2].item()),
        "cta_pv_skipped": int(totals[3].item()),
        "cta_pv_executed": int(totals[4].item()),
        "force_mode_sum": int(totals[5].item()),
    }


def _maybe_build_bounds(args, value) -> Optional[torch.Tensor]:
    if not args.precompute_bounds:
        return None
    if args.skip_predicate != "value_bound":
        return None
    return build_value_norm_bounds(value, block_size=args.block_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-q", type=int, default=1024)
    parser.add_argument("--seq-k", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--pattern", choices=["random", "peaked"], default="peaked")
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--skip-predicate", choices=["mass", "value_bound"], default="value_bound")
    parser.add_argument("--force-mode", type=int, default=0)
    parser.add_argument("--precompute-bounds", action="store_true")
    parser.add_argument("--return-stats", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    query, key, value = _make_tensors(args)

    bounds_build_ms = None
    if args.skip_predicate == "value_bound":
        bounds_build_ms = _time_cuda(
            lambda: build_value_norm_bounds(value, block_size=args.block_size),
            warmup=max(1, args.warmup // 10),
            iters=max(1, args.iters // 10),
        )

    value_norm_bounds = _maybe_build_bounds(args, value)
    if value_norm_bounds is not None:
        torch.cuda.synchronize()

    def run_once(return_raw_stats=False):
        return gate1_attention_triton_forward(
            query,
            key,
            value,
            causal=args.causal,
            error_budget=args.error_budget,
            block_size=args.block_size,
            tile_size_q=args.tile_size_q,
            value_norm_bounds=value_norm_bounds,
            skip_predicate=args.skip_predicate,
            force_mode=args.force_mode,
            return_raw_stats=return_raw_stats,
        )

    kernel_ms = _time_cuda(
        lambda: run_once(return_raw_stats=False),
        warmup=args.warmup,
        iters=args.iters,
    )

    stats = None
    if args.return_stats:
        _, raw_stats = run_once(return_raw_stats=True)
        torch.cuda.synchronize()
        stats = _summarize_stats(raw_stats)

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "shape": {
                    "batch": args.batch,
                    "seq_q": args.seq_q,
                    "seq_k": args.seq_k,
                    "heads": args.heads,
                    "dim": args.dim,
                    "dtype": args.dtype,
                },
                "pattern": args.pattern,
                "block_size": args.block_size,
                "tile_size_q": args.tile_size_q,
                "skip_predicate": args.skip_predicate,
                "force_mode": args.force_mode,
                "precompute_bounds": args.precompute_bounds,
                "bounds_build_ms": bounds_build_ms,
                "kernel_ms": kernel_ms,
                "stats": stats,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
