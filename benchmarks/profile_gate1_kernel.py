"""Profile the Gate-1 Triton diagnostic kernel.

This script keeps allocations and optional value-norm-bound construction out of
the timed region when requested. It is intended for CUDA machines or Modal runs,
not CPU-only CI.
"""

import argparse
import copy
import json
from typing import Optional

import torch

from stream_attention.kernels.gate1_fwd_triton import (
    build_value_norm_bounds,
    gate1_attention_triton_forward,
)
from stream_attention.kernels.metadata_triton import build_value_norm_bounds_triton
from stream_attention.router import AttentionRouteRequest, CostEntry, CostKey, Gate1CostModel


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
        num_blocks = (args.seq_k + args.block_size - 1) // args.block_size
        if args.active_blocks is not None:
            active_blocks = args.active_blocks
        elif args.active_fraction is not None:
            active_blocks = round(args.active_fraction * num_blocks)
        else:
            active_blocks = 1
        active_blocks = max(0, min(num_blocks, active_blocks))
        active_tokens = min(args.seq_k, active_blocks * args.block_size)

        query = torch.zeros(shape_q, device=device, dtype=dtype)
        key = torch.zeros(shape_kv, device=device, dtype=dtype)
        value = torch.randn(shape_kv, device=device, dtype=dtype)
        query[..., 0] = args.peak
        key[:, :active_tokens, :, 0] = args.peak
        key[:, active_tokens:, :, 0] = -args.peak
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
    return _build_bounds(args, value)


def _build_bounds(args, value) -> torch.Tensor:
    if args.bounds_builder == "triton":
        return build_value_norm_bounds_triton(value, block_size=args.block_size)
    return build_value_norm_bounds(value, block_size=args.block_size)


def _run_gate1(
    args,
    query,
    key,
    value,
    *,
    force_mode: int,
    skip_predicate: Optional[str] = None,
    value_norm_bounds: Optional[torch.Tensor] = None,
    return_raw_stats: bool = False,
):
    return gate1_attention_triton_forward(
        query,
        key,
        value,
        causal=args.causal,
        error_budget=args.error_budget,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        value_norm_bounds=value_norm_bounds,
        skip_predicate=skip_predicate or args.skip_predicate,
        force_mode=force_mode,
        return_raw_stats=return_raw_stats,
    )


def _time_gate1(
    args,
    query,
    key,
    value,
    *,
    force_mode: int,
    skip_predicate: Optional[str] = None,
    value_norm_bounds: Optional[torch.Tensor] = None,
) -> float:
    return _time_cuda(
        lambda: _run_gate1(
            args,
            query,
            key,
            value,
            force_mode=force_mode,
            skip_predicate=skip_predicate,
            value_norm_bounds=value_norm_bounds,
            return_raw_stats=False,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )


def _measure_bounds(args, value) -> Optional[float]:
    if args.skip_predicate != "value_bound":
        return None
    return _time_cuda(
        lambda: _build_bounds(args, value),
        warmup=max(1, args.warmup // 10),
        iters=max(1, args.iters // 10),
    )


def _run_suite(args):
    query, key, value = _make_tensors(args)
    bounds_build_ms = _measure_bounds(args, value)
    value_norm_bounds = _maybe_build_bounds(args, value)
    if value_norm_bounds is not None:
        torch.cuda.synchronize()

    dense_ms = _time_gate1(
        args,
        query,
        key,
        value,
        force_mode=5,
        skip_predicate="mass",
    )
    qk_only_ms = _time_gate1(
        args,
        query,
        key,
        value,
        force_mode=7,
        skip_predicate="mass",
    )
    qk_log_predicate_no_pv_ms = _time_gate1(
        args,
        query,
        key,
        value,
        force_mode=8,
        skip_predicate="mass",
    )
    qk_exp_predicate_no_pv_ms = _time_gate1(
        args,
        query,
        key,
        value,
        force_mode=9,
        skip_predicate="mass",
    )
    gate1_ms = _time_gate1(
        args,
        query,
        key,
        value,
        force_mode=0,
        value_norm_bounds=value_norm_bounds,
    )
    predicate_no_skip_ms = _time_gate1(
        args,
        query,
        key,
        value,
        force_mode=1,
        value_norm_bounds=value_norm_bounds,
    )

    _, raw_stats = _run_gate1(
        args,
        query,
        key,
        value,
        force_mode=0,
        value_norm_bounds=value_norm_bounds,
        return_raw_stats=True,
    )
    torch.cuda.synchronize()
    stats = _summarize_stats(raw_stats)
    total_cta = stats["cta_tiles_total"]
    active_frac = stats["cta_pv_executed"] / total_cta if total_cta else 0.0
    pv_ms = max(dense_ms - qk_only_ms, 0.0)
    predicted_gate1_ms = qk_only_ms + active_frac * pv_ms
    prediction_error_ms = gate1_ms - predicted_gate1_ms
    prediction_ratio = (
        gate1_ms / predicted_gate1_ms if predicted_gate1_ms > 0.0 else None
    )

    return {
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
        "requested_active_fraction": args.active_fraction,
        "requested_active_blocks": args.active_blocks,
        "block_size": args.block_size,
        "tile_size_q": args.tile_size_q,
        "skip_predicate": args.skip_predicate,
        "precompute_bounds": args.precompute_bounds,
        "bounds_build_ms": bounds_build_ms,
        "dense_no_predicate_ms": dense_ms,
        "qk_only_ms": qk_only_ms,
        "qk_log_predicate_no_pv_ms": qk_log_predicate_no_pv_ms,
        "qk_exp_predicate_no_pv_ms": qk_exp_predicate_no_pv_ms,
        "gate1_ms": gate1_ms,
        "predicate_no_skip_ms": predicate_no_skip_ms,
        "active_pv_fraction": active_frac,
        "estimated_pv_ms": pv_ms,
        "predicted_gate1_ms": predicted_gate1_ms,
        "prediction_error_ms": prediction_error_ms,
        "observed_over_predicted": prediction_ratio,
        "stats": stats,
    }


def _write_cost_model(args, result, path: str) -> None:
    shape = result["shape"]
    request = AttentionRouteRequest(
        batch=shape["batch"],
        seq_q=shape["seq_q"],
        seq_k=shape["seq_k"],
        heads=shape["heads"],
        dim=shape["dim"],
        dtype=shape["dtype"],
        device=result["device"],
        tile_size_q=result["tile_size_q"],
        block_size=result["block_size"],
        causal=args.causal,
    )
    entry = CostEntry.from_active_fraction_curve(
        dense_ms=result["dense_no_predicate_ms"],
        qk_only_ms=result["qk_only_ms"],
        observations=[(result["active_pv_fraction"], result["gate1_ms"])],
        gate1_no_skip_ms=result["predicate_no_skip_ms"],
        gate1_all_skip_ms=result["qk_only_ms"],
    )
    model = Gate1CostModel()
    model.update(CostKey.from_request(request), entry)
    model.to_json(path)


def _write_curve_cost_model(args, results, path: str) -> None:
    first = results[0]
    shape = first["shape"]
    dense_ms = sorted(row["dense_no_predicate_ms"] for row in results)[len(results) // 2]
    qk_only_ms = sorted(row["qk_only_ms"] for row in results)[len(results) // 2]
    request = AttentionRouteRequest(
        batch=shape["batch"],
        seq_q=shape["seq_q"],
        seq_k=shape["seq_k"],
        heads=shape["heads"],
        dim=shape["dim"],
        dtype=shape["dtype"],
        device=first["device"],
        tile_size_q=first["tile_size_q"],
        block_size=first["block_size"],
        causal=args.causal,
    )
    entry = CostEntry.from_active_fraction_curve(
        dense_ms=dense_ms,
        qk_only_ms=qk_only_ms,
        observations=[
            (row["active_pv_fraction"], row["gate1_ms"])
            for row in results
        ],
        gate1_no_skip_ms=max(row["predicate_no_skip_ms"] for row in results),
        gate1_all_skip_ms=min(row["qk_only_ms"] for row in results),
    )
    model = Gate1CostModel()
    model.update(CostKey.from_request(request), entry)
    model.to_json(path)


def _parse_fraction_list(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


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
    parser.add_argument("--active-fraction", type=float, default=None)
    parser.add_argument("--active-blocks", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--skip-predicate", choices=["mass", "value_bound"], default="value_bound")
    parser.add_argument("--bounds-builder", choices=["triton", "torch"], default="triton")
    parser.add_argument("--force-mode", type=int, default=0)
    parser.add_argument("--precompute-bounds", action="store_true")
    parser.add_argument("--return-stats", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--sweep-active-fracs", default=None)
    parser.add_argument("--cost-json-out", default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    if args.sweep_active_fracs:
        rows = []
        for frac in _parse_fraction_list(args.sweep_active_fracs):
            sweep_args = copy.copy(args)
            sweep_args.active_fraction = frac
            sweep_args.active_blocks = None
            rows.append(_run_suite(sweep_args))
        payload = {"sweep": rows}
        if args.cost_json_out:
            _write_curve_cost_model(args, rows, args.cost_json_out)
            payload["cost_json_out"] = args.cost_json_out
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.suite:
        result = _run_suite(args)
        if args.cost_json_out:
            _write_cost_model(args, result, args.cost_json_out)
            result["cost_json_out"] = args.cost_json_out
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    query, key, value = _make_tensors(args)

    bounds_build_ms = _measure_bounds(args, value)
    value_norm_bounds = _maybe_build_bounds(args, value)
    if value_norm_bounds is not None:
        torch.cuda.synchronize()

    def run_once(return_raw_stats=False):
        return _run_gate1(
            args,
            query,
            key,
            value,
            force_mode=args.force_mode,
            value_norm_bounds=value_norm_bounds,
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
                "requested_active_fraction": args.active_fraction,
                "requested_active_blocks": args.active_blocks,
                "block_size": args.block_size,
                "tile_size_q": args.tile_size_q,
                "skip_predicate": args.skip_predicate,
                "bounds_builder": args.bounds_builder,
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
