"""Profile Gate-0 summary bound tightness for long-KV decode.

This is an offline experiment, not a production Gate-0 runtime.  It answers the
first Gate-0 question: can cached K-block summaries safely reject enough blocks
before loading full K/V and computing QK?
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from stream_attention.certified.bounds import block_score_upper_bound
from stream_attention.certified.summaries import build_block_summaries
from stream_attention.kernels.gate0_summary_scan_triton import (
    TRITON_AVAILABLE as GATE0_TRITON_AVAILABLE,
    gate0_summary_scan_triton,
)


def _parse_values(values: Iterable[str], cast):
    parsed = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(cast(item))
    return parsed


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_call(fn, *, device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        _sync(device)
        return start.elapsed_time(end) / iters

    start_s = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start_s) * 1000.0 / iters


def _active_blocks(seq_k: int, block_size: int, active_fraction: float) -> Tuple[int, int]:
    num_blocks = (seq_k + block_size - 1) // block_size
    active = max(0, min(num_blocks, round(active_fraction * num_blocks)))
    if active_fraction > 0.0 and num_blocks > 0 and active == 0:
        active = 1
    return active, num_blocks


def _make_pattern(
    *,
    batch: int,
    query_len: int,
    kv_len: int,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    pattern: str,
    active_fraction: float,
    block_size: int,
    peak: float,
    sink_blocks: int,
    recent_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    q = torch.zeros(batch, query_len, heads, dim, device=device, dtype=dtype)
    k = torch.zeros(batch, kv_len, heads, dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_len, heads, dim, device=device, dtype=dtype)

    if pattern == "random":
        q = torch.randn_like(q)
        k = torch.randn_like(k)
        _, num_blocks = _active_blocks(kv_len, block_size, active_fraction)
        return q, k, v, 1.0 if num_blocks else 0.0

    q[..., 0] = peak
    k[..., 0] = -peak
    active_blocks, num_blocks = _active_blocks(kv_len, block_size, active_fraction)

    if pattern == "peaked":
        active_block_ids: List[int] = list(range(active_blocks))
    elif pattern == "sink_local":
        sink = min(sink_blocks, active_blocks, num_blocks)
        remaining = max(0, active_blocks - sink)
        recent = min(max(recent_blocks, remaining), remaining, max(0, num_blocks - sink))
        active_block_ids = list(range(sink))
        if recent:
            active_block_ids.extend(range(num_blocks - recent, num_blocks))
    elif pattern == "sliding_recent":
        active_block_ids = list(range(max(0, num_blocks - active_blocks), num_blocks))
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    for block_idx in sorted(set(active_block_ids)):
        start = block_idx * block_size
        end = min(start + block_size, kv_len)
        if start < end:
            k[:, start:end, :, 0] = peak

    actual_active = len(set(active_block_ids)) / num_blocks if num_blocks else 0.0
    return q, k, v, actual_active


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), q).item())


def _scan_summary(q_bhsd: torch.Tensor, summaries, *, scale: float) -> torch.Tensor:
    dot_centroid = torch.einsum("bhsd,bhkd->bhsk", q_bhsd, summaries.centroid)
    q_norm = torch.linalg.vector_norm(q_bhsd, dim=-1)
    residual_bound = (dot_centroid + q_norm[..., None] * summaries.radius[:, :, None, :]) * scale
    if summaries.outlier_keys is None or summaries.outlier_mask is None:
        return residual_bound.sum()

    outlier_scores = torch.einsum(
        "bhsd,bhkod->bhsko",
        q_bhsd,
        summaries.outlier_keys,
    ) * scale
    outlier_scores = outlier_scores.masked_fill(
        ~summaries.outlier_mask[:, :, None, :, :],
        -float("inf"),
    )
    upper = torch.maximum(residual_bound, outlier_scores.amax(dim=-1))
    return upper.sum()


def _scan_full_qk(
    q_bhsd: torch.Tensor,
    k_bhnd: torch.Tensor,
    *,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    seq_k = k_bhnd.shape[2]
    scores = torch.einsum("bhsd,bhnd->bhsn", q_bhsd, k_bhnd) * scale
    pad = (-seq_k) % block_size
    if pad:
        scores = F.pad(scores, (0, pad), value=-float("inf"))
    block_scores = scores.reshape(*scores.shape[:-1], -1, block_size)
    return block_scores.amax(dim=-1).sum()


def _time_summary_scan(args, q: torch.Tensor, q_bhsd: torch.Tensor, summaries, *, scale: float, outliers: int, device: torch.device) -> Tuple[float, str]:
    if args.scan_backend == "torch":
        return (
            _time_call(
                lambda: _scan_summary(q_bhsd, summaries, scale=scale),
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            ),
            "torch",
        )
    if args.scan_backend != "triton":
        raise ValueError(f"unknown scan backend: {args.scan_backend}")
    if not GATE0_TRITON_AVAILABLE:
        raise RuntimeError("Triton Gate-0 scan backend is not available")
    if device.type != "cuda":
        raise RuntimeError("Triton Gate-0 scan backend requires CUDA")
    if outliers != 0:
        raise ValueError("Triton Gate-0 scan backend currently supports only --summary-outliers 0")
    output = torch.empty(
        q.shape[0],
        q.shape[2],
        q.shape[1],
        summaries.num_blocks,
        device=q.device,
        dtype=torch.float32,
    )
    return (
        _time_call(
            lambda: gate0_summary_scan_triton(
                q,
                summaries.centroid,
                summaries.radius,
                scale=scale,
                blocks_per_program=args.blocks_per_program,
                output=output,
            ),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        ),
        "triton",
    )


def _analyze_bounds(
    q: torch.Tensor,
    k: torch.Tensor,
    summaries,
    *,
    block_size: int,
    error_budget: float,
    bound_tolerance: float,
) -> dict:
    batch, query_len, heads, dim = q.shape
    seq_k = k.shape[1]
    scale = 1.0 / math.sqrt(dim)
    log_eps = math.log(error_budget) if error_budget > 0.0 else -float("inf")

    q_bhsd = q.permute(0, 2, 1, 3).contiguous().float()
    k_bhnd = k.permute(0, 2, 1, 3).contiguous().float()

    running_max = torch.full(
        (batch, heads, query_len),
        -float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    acc_den = torch.zeros(batch, heads, query_len, device=q.device, dtype=torch.float32)

    total = batch * heads * query_len * summaries.num_blocks
    actual_skip_count = 0
    predicted_skip_count = 0
    false_negative_count = 0
    false_positive_count = 0
    unsafe_bound_count = 0
    gap_values = []

    for block_idx in range(summaries.num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_k)
        block_len = int(summaries.block_lengths[block_idx].item())

        scores = torch.einsum("bhsd,bhnd->bhsn", q_bhsd, k_bhnd[:, :, start:end, :]) * scale
        tile_max = scores.amax(dim=-1)
        upper = block_score_upper_bound(q_bhsd, summaries, block_idx, scale=scale)
        gap = upper - tile_max
        gap_values.append(gap.detach().flatten())
        unsafe_bound_count += int((gap < -bound_tolerance).sum().item())

        has_state = torch.isfinite(running_max) & (acc_den > 0)
        lse = torch.where(
            has_state,
            running_max + torch.log(acc_den.clamp_min(1.0e-30)),
            torch.full_like(running_max, -float("inf")),
        )
        threshold = lse + log_eps
        log_block_len = math.log(max(1, block_len))

        actual_skip = has_state & (tile_max + log_block_len <= threshold)
        summary_skip = has_state & (upper + log_block_len <= threshold)

        actual_skip_count += int(actual_skip.sum().item())
        predicted_skip_count += int(summary_skip.sum().item())
        false_negative_count += int((summary_skip & ~actual_skip).sum().item())
        false_positive_count += int((~summary_skip & actual_skip).sum().item())

        compute_row = ~actual_skip
        if not bool(compute_row.any()):
            continue

        tile_valid = torch.isfinite(tile_max)
        prev_valid = torch.isfinite(running_max)
        new_valid = prev_valid | tile_valid
        new_max = torch.maximum(running_max, tile_max)
        safe_new_max = torch.where(new_valid, new_max, torch.zeros_like(new_max))
        correction = torch.where(
            prev_valid,
            torch.exp(running_max - safe_new_max),
            torch.zeros_like(acc_den),
        )
        exp_scores = torch.exp(scores - safe_new_max[..., None])
        exp_scores = torch.where(torch.isfinite(scores), exp_scores, torch.zeros_like(exp_scores))
        next_den = acc_den * correction + exp_scores.sum(dim=-1)

        running_max = torch.where(compute_row, new_max, running_max)
        acc_den = torch.where(compute_row, next_den, acc_den)

    gaps = torch.cat(gap_values) if gap_values else torch.empty(0, device=q.device)
    predicted_skip_fraction = predicted_skip_count / total if total else 0.0
    actual_skip_fraction = actual_skip_count / total if total else 0.0
    predicted_compute_fraction = 1.0 - predicted_skip_fraction
    return {
        "total_row_blocks": total,
        "actual_skip_count": actual_skip_count,
        "predicted_skip_count": predicted_skip_count,
        "false_negative_count": false_negative_count,
        "false_positive_count": false_positive_count,
        "unsafe_bound_count": unsafe_bound_count,
        "predicted_skip_fraction": predicted_skip_fraction,
        "actual_gate1_skip_fraction": actual_skip_fraction,
        "actual_skip_fraction": actual_skip_fraction,
        "predicted_compute_fraction": predicted_compute_fraction,
        "false_negative_rate": false_negative_count / total if total else 0.0,
        "false_positive_rate": false_positive_count / total if total else 0.0,
        "unsafe_bound_rate": unsafe_bound_count / total if total else 0.0,
        "bound_gap_mean": float(gaps.mean().item()) if gaps.numel() else 0.0,
        "bound_gap_p50": _quantile(gaps, 0.50),
        "bound_gap_p90": _quantile(gaps, 0.90),
        "bound_gap_p99": _quantile(gaps, 0.99),
        "bound_gap_min": float(gaps.min().item()) if gaps.numel() else 0.0,
        "bound_gap_max": float(gaps.max().item()) if gaps.numel() else 0.0,
    }


def _profile_one(args, *, kv_len: int, heads: int, active_fraction: float, block_size: int, outliers: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = _dtype(args.dtype)
    q, k, v, block_active = _make_pattern(
        batch=args.batch,
        query_len=args.query_len,
        kv_len=kv_len,
        heads=heads,
        dim=args.dim,
        dtype=dtype,
        device=device,
        pattern=args.pattern,
        active_fraction=active_fraction,
        block_size=block_size,
        peak=args.peak,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )
    scale = 1.0 / math.sqrt(args.dim)
    q_bhsd = q.permute(0, 2, 1, 3).contiguous().float()
    k_bhnd = k.permute(0, 2, 1, 3).contiguous().float()

    build_fn = lambda: build_block_summaries(
        k,
        v,
        block_size=block_size,
        num_outliers=outliers,
    )
    summary_build_ms = _time_call(
        build_fn,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    summaries = build_fn()
    _sync(device)

    summary_scan_ms, actual_scan_backend = _time_summary_scan(
        args,
        q,
        q_bhsd,
        summaries,
        scale=scale,
        outliers=outliers,
        device=device,
    )
    full_qk_scan_ms = _time_call(
        lambda: _scan_full_qk(q_bhsd, k_bhnd, block_size=block_size, scale=scale),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    metrics = _analyze_bounds(
        q,
        k,
        summaries,
        block_size=block_size,
        error_budget=args.error_budget,
        bound_tolerance=args.bound_tolerance,
    )

    estimated_gate0_qk_ms = summary_scan_ms + metrics["predicted_compute_fraction"] * full_qk_scan_ms
    estimated_gate0_speedup_vs_gate1 = (
        full_qk_scan_ms / estimated_gate0_qk_ms if estimated_gate0_qk_ms > 0.0 else None
    )
    summary_scan_over_qk = summary_scan_ms / full_qk_scan_ms if full_qk_scan_ms > 0.0 else None
    summary_type = "centroid_radius" if outliers == 0 else f"centroid_radius_outlier{outliers}"
    gate0_promising = (
        metrics["false_negative_rate"] <= args.max_false_negative_rate
        and metrics["predicted_skip_fraction"] >= args.min_predicted_skip_fraction
        and (summary_scan_over_qk is not None and summary_scan_over_qk < 1.0)
        and estimated_gate0_qk_ms < full_qk_scan_ms
    )

    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "shape": {
            "batch": args.batch,
            "query_len": args.query_len,
            "kv_len": kv_len,
            "heads": heads,
            "dim": args.dim,
            "dtype": args.dtype,
            "attention_type": "mha",
        },
        "block_size": block_size,
        "num_blocks": summaries.num_blocks,
        "pattern": args.pattern,
        "requested_active_fraction": active_fraction,
        "block_quantized_active_fraction": block_active,
        "summary_type": summary_type,
        "num_summary_outliers": outliers,
        "rope_mode": args.rope_mode,
        "error_budget": args.error_budget,
        "summary_build_ms": summary_build_ms,
        "requested_scan_backend": args.scan_backend,
        "scan_backend": actual_scan_backend,
        "blocks_per_program": args.blocks_per_program,
        "summary_scan_ms": summary_scan_ms,
        "full_qk_scan_ms": full_qk_scan_ms,
        "summary_scan_over_qk": summary_scan_over_qk,
        "estimated_gate0_qk_ms": estimated_gate0_qk_ms,
        "estimated_gate0_speedup_vs_gate1": estimated_gate0_speedup_vs_gate1,
        "estimated_gate0_speedup_vs_flashinfer": None,
        "gate0_promising": gate0_promising,
        **metrics,
        "timing_note": "offline_pytorch_cuda_events" if device.type == "cuda" else "offline_pytorch_wall_time",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--kv-lens", nargs="+", default=["8192", "16384", "32768"])
    parser.add_argument("--heads", nargs="+", default=["16", "32"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--pattern", choices=["random", "peaked", "sink_local", "sliding_recent"], default="peaked")
    parser.add_argument("--active-fraction", nargs="+", default=["0.0625", "0.125", "0.25", "1.0"])
    parser.add_argument("--block-size", nargs="+", default=["64", "128"])
    parser.add_argument("--summary-outliers", nargs="+", default=["0", "1", "2", "4"])
    parser.add_argument("--scan-backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--blocks-per-program", nargs="+", default=["32"])
    parser.add_argument("--rope-mode", choices=["none"], default="none")
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--bound-tolerance", type=float, default=1e-4)
    parser.add_argument("--min-predicted-skip-fraction", type=float, default=0.25)
    parser.add_argument("--max-false-negative-rate", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()

    torch.manual_seed(0)
    kv_lens = _parse_values(args.kv_lens, int)
    heads_values = _parse_values(args.heads, int)
    active_fractions = _parse_values(args.active_fraction, float)
    block_sizes = _parse_values(args.block_size, int)
    outlier_values = _parse_values(args.summary_outliers, int)
    blocks_per_program_values = _parse_values(args.blocks_per_program, int)

    rows = []
    for kv_len, heads, active_fraction, block_size, outliers, blocks_per_program in itertools.product(
        kv_lens,
        heads_values,
        active_fractions,
        block_sizes,
        outlier_values,
        blocks_per_program_values,
    ):
        run_args = argparse.Namespace(**vars(args))
        run_args.blocks_per_program = blocks_per_program
        try:
            rows.append(
                _profile_one(
                    run_args,
                    kv_len=kv_len,
                    heads=heads,
                    active_fraction=active_fraction,
                    block_size=block_size,
                    outliers=outliers,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "shape": {
                        "batch": args.batch,
                        "query_len": args.query_len,
                        "kv_len": kv_len,
                        "heads": heads,
                        "dim": args.dim,
                        "dtype": args.dtype,
                        "attention_type": "mha",
                    },
                    "block_size": block_size,
                    "blocks_per_program": blocks_per_program,
                    "pattern": args.pattern,
                    "requested_active_fraction": active_fraction,
                    "num_summary_outliers": outliers,
                }
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {"rows": rows}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
