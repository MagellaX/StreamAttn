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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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


def _load_pt_tensor(path: str, candidate_keys: Iterable[str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = None
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                tensor = value
                break
        if tensor is None:
            keys = ", ".join(str(key) for key in payload.keys())
            raise ValueError(f"{path} does not contain one of {list(candidate_keys)}; found keys: {keys}")
    else:
        raise ValueError(f"{path} must contain a tensor or a tensor dictionary")
    return tensor.to(device=device, dtype=dtype).contiguous()


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: Optional[torch.Tensor]) -> None:
    if q.dim() != 4:
        raise ValueError("q tensor must have shape [batch, query_len, heads, dim]")
    if k.dim() != 4:
        raise ValueError("k tensor must have shape [batch, kv_len, heads, dim]")
    if q.shape[0] != k.shape[0]:
        raise ValueError("q and k batch dimensions must match")
    if q.shape[2:] != k.shape[2:]:
        raise ValueError("q and k must have matching heads and dim")
    if v is not None and v.shape != k.shape:
        raise ValueError("v tensor must have the same shape as k")


def _load_real_tensors(args, *, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if args.tensor_format != "pt":
        raise ValueError(f"unsupported tensor format: {args.tensor_format}")
    if not args.q_path or not args.k_path:
        raise ValueError("--q-path and --k-path are required for real tensor profiling")
    q = _load_pt_tensor(
        args.q_path,
        ("q", "query", "post_rope_q", "pre_rope_q"),
        device=device,
        dtype=dtype,
    )
    k = _load_pt_tensor(
        args.k_path,
        ("k", "key", "post_rope_k", "pre_rope_k"),
        device=device,
        dtype=dtype,
    )
    v = None
    if args.v_path:
        v = _load_pt_tensor(
            args.v_path,
            ("v", "value"),
            device=device,
            dtype=dtype,
        )
    _validate_qkv(q, k, v)
    return q, k, v


def _selected_head_indices(args, heads: int) -> List[int]:
    if args.head_indices:
        indices = _parse_values(args.head_indices, int)
    elif args.per_head:
        indices = list(range(heads))
    else:
        indices = []
    for head_idx in indices:
        if head_idx < 0 or head_idx >= heads:
            raise ValueError(f"head index {head_idx} is outside [0, {heads})")
    return indices


def _real_tensor_cases(args, *, device: torch.device, dtype: torch.dtype) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]]:
    q, k, v = _load_real_tensors(args, device=device, dtype=dtype)
    head_indices = _selected_head_indices(args, q.shape[2])
    if not head_indices:
        yield q, k, v, {"head_id": -1, "real_case": "all_heads"}
        return
    for head_idx in head_indices:
        q_h = q[:, :, head_idx : head_idx + 1, :].contiguous()
        k_h = k[:, :, head_idx : head_idx + 1, :].contiguous()
        v_h = None if v is None else v[:, :, head_idx : head_idx + 1, :].contiguous()
        yield q_h, k_h, v_h, {"head_id": head_idx, "real_case": "single_head"}


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, List[int]]:
    q = torch.zeros(batch, query_len, heads, dim, device=device, dtype=dtype)
    k = torch.zeros(batch, kv_len, heads, dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_len, heads, dim, device=device, dtype=dtype)

    if pattern == "random":
        q = torch.randn_like(q)
        k = torch.randn_like(k)
        _, num_blocks = _active_blocks(kv_len, block_size, active_fraction)
        return q, k, v, 1.0 if num_blocks else 0.0, list(range(num_blocks))

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
    return q, k, v, actual_active, sorted(set(active_block_ids))


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
        return (
            _time_call(
                lambda: _scan_summary(q_bhsd, summaries, scale=scale),
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            ),
            "torch_outlier_fallback",
        )
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


def _dedup_order(blocks: Iterable[int], *, num_blocks: int) -> List[int]:
    seen = set()
    order = []
    for block_idx in blocks:
        if 0 <= block_idx < num_blocks and block_idx not in seen:
            seen.add(block_idx)
            order.append(block_idx)
    return order


def _resolve_block_order(
    args,
    *,
    q_bhsd: torch.Tensor,
    k_bhnd: torch.Tensor,
    summaries,
    active_block_ids: List[int],
    block_size: int,
    scale: float,
) -> List[int]:
    num_blocks = summaries.num_blocks
    if args.block_order == "sequential":
        return list(range(num_blocks))
    if args.block_order == "recent_first":
        return list(reversed(range(num_blocks)))
    if args.block_order == "sink_recent_first":
        sink = list(range(min(args.sink_blocks, num_blocks)))
        recent_start = max(0, num_blocks - min(args.recent_blocks, num_blocks))
        recent = list(range(recent_start, num_blocks))
        rest = list(range(num_blocks))
        return _dedup_order([*sink, *reversed(recent), *rest], num_blocks=num_blocks)
    if args.block_order == "oracle_active_first":
        if active_block_ids:
            return _dedup_order([*active_block_ids, *range(num_blocks)], num_blocks=num_blocks)
        scores = []
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, k_bhnd.shape[2])
            qk = torch.einsum("bhsd,bhnd->bhsn", q_bhsd, k_bhnd[:, :, start:end, :]) * scale
            scores.append((float(qk.amax(dim=-1).mean().item()), block_idx))
        return [block_idx for _, block_idx in sorted(scores, reverse=True)]
    if args.block_order == "summary_desc":
        scores = []
        for block_idx in range(num_blocks):
            upper = block_score_upper_bound(q_bhsd, summaries, block_idx, scale=scale)
            scores.append((float(upper.mean().item()), block_idx))
        return [block_idx for _, block_idx in sorted(scores, reverse=True)]
    raise ValueError(f"unknown block order: {args.block_order}")


def _analyze_bounds(
    q: torch.Tensor,
    k: torch.Tensor,
    summaries,
    *,
    block_size: int,
    error_budget: float,
    bound_tolerance: float,
    block_order: List[int],
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

    for block_idx in block_order:
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


def _profile_one(
    args,
    *,
    kv_len: Optional[int],
    heads: Optional[int],
    active_fraction: Optional[float],
    block_size: int,
    outliers: int,
    q_override: Optional[torch.Tensor] = None,
    k_override: Optional[torch.Tensor] = None,
    v_override: Optional[torch.Tensor] = None,
    row_extra: Optional[Dict[str, Any]] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = _dtype(args.dtype)
    tensor_source = "synthetic"
    real_case = None
    head_id = None
    if q_override is None or k_override is None:
        if kv_len is None or heads is None or active_fraction is None:
            raise ValueError("synthetic profiling requires kv_len, heads, and active_fraction")
        q, k, v, block_active, active_block_ids = _make_pattern(
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
        pattern = args.pattern
    else:
        q = q_override.to(device=device, dtype=dtype).contiguous()
        k = k_override.to(device=device, dtype=dtype).contiguous()
        v = None if v_override is None else v_override.to(device=device, dtype=dtype).contiguous()
        _validate_qkv(q, k, v)
        tensor_source = "tensor"
        pattern = "real_k"
        block_active = None
        active_block_ids = []
        active_fraction = None
        if row_extra:
            head_id = row_extra.get("head_id")
            real_case = row_extra.get("real_case")

    batch, query_len, actual_heads, dim = q.shape
    actual_kv_len = k.shape[1]
    scale = 1.0 / math.sqrt(dim)
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
    block_order = _resolve_block_order(
        args,
        q_bhsd=q_bhsd,
        k_bhnd=k_bhnd,
        summaries=summaries,
        active_block_ids=active_block_ids,
        block_size=block_size,
        scale=scale,
    )

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
        block_order=block_order,
    )

    estimated_gate0_qk_ms = summary_scan_ms + metrics["predicted_compute_fraction"] * full_qk_scan_ms
    estimated_gate0_speedup_vs_gate1 = (
        full_qk_scan_ms / estimated_gate0_qk_ms if estimated_gate0_qk_ms > 0.0 else None
    )
    summary_scan_over_qk = summary_scan_ms / full_qk_scan_ms if full_qk_scan_ms > 0.0 else None
    summary_type = "centroid_radius" if outliers == 0 else f"centroid_radius_outlier{outliers}"
    tensor_space = args.tensor_space
    if tensor_source == "tensor" and tensor_space == "synthetic":
        tensor_space = "unknown"
    gate0_promising = (
        metrics["false_negative_rate"] <= args.max_false_negative_rate
        and metrics["predicted_skip_fraction"] >= args.min_predicted_skip_fraction
        and (summary_scan_over_qk is not None and summary_scan_over_qk < 1.0)
        and estimated_gate0_qk_ms < full_qk_scan_ms
    )

    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "tensor_source": tensor_source,
        "tensor_space": tensor_space,
        "model_id": args.model_id or None,
        "layer_id": args.layer_id,
        "head_id": head_id,
        "real_case": real_case,
        "shape": {
            "batch": batch,
            "query_len": query_len,
            "kv_len": actual_kv_len,
            "heads": actual_heads,
            "dim": dim,
            "dtype": args.dtype,
            "attention_type": "mha",
        },
        "block_size": block_size,
        "num_blocks": summaries.num_blocks,
        "pattern": pattern,
        "block_order": args.block_order,
        "active_block_ids": active_block_ids,
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
    parser.add_argument(
        "--block-order",
        nargs="+",
        default=["sequential"],
    )
    parser.add_argument("--q-path", default="", help="Optional saved Q tensor path for real-K profiling")
    parser.add_argument("--k-path", default="", help="Optional saved K tensor path for real-K profiling")
    parser.add_argument("--v-path", default="", help="Optional saved V tensor path for metadata/value-norm summaries")
    parser.add_argument("--tensor-format", choices=["pt"], default="pt")
    parser.add_argument(
        "--tensor-space",
        choices=["synthetic", "unknown", "pre_rope", "post_rope"],
        default="synthetic",
    )
    parser.add_argument("--model-id", default="")
    parser.add_argument("--layer-id", type=int, default=None)
    parser.add_argument("--per-head", action="store_true")
    parser.add_argument("--head-indices", nargs="+", default=[])
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
    block_orders = _parse_values(args.block_order, str)
    real_tensor_mode = bool(args.q_path or args.k_path)

    rows = []
    if real_tensor_mode:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        dtype = _dtype(args.dtype)
        try:
            cases = _real_tensor_cases(args, device=device, dtype=dtype)
            for q, k, v, row_extra in cases:
                for block_size, outliers, blocks_per_program, block_order in itertools.product(
                    block_sizes,
                    outlier_values,
                    blocks_per_program_values,
                    block_orders,
                ):
                    run_args = argparse.Namespace(**vars(args))
                    run_args.blocks_per_program = blocks_per_program
                    run_args.block_order = block_order
                    try:
                        rows.append(
                            _profile_one(
                                run_args,
                                kv_len=None,
                                heads=None,
                                active_fraction=None,
                                block_size=block_size,
                                outliers=outliers,
                                q_override=q,
                                k_override=k,
                                v_override=v,
                                row_extra=row_extra,
                            )
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "error": f"{type(exc).__name__}: {exc}",
                                "tensor_source": "tensor",
                                "tensor_space": (
                                    "unknown" if args.tensor_space == "synthetic" else args.tensor_space
                                ),
                                "model_id": args.model_id or None,
                                "layer_id": args.layer_id,
                                "head_id": row_extra.get("head_id"),
                                "real_case": row_extra.get("real_case"),
                                "shape": {
                                    "batch": int(q.shape[0]),
                                    "query_len": int(q.shape[1]),
                                    "kv_len": int(k.shape[1]),
                                    "heads": int(q.shape[2]),
                                    "dim": int(q.shape[3]),
                                    "dtype": args.dtype,
                                    "attention_type": "mha",
                                },
                                "block_size": block_size,
                                "blocks_per_program": blocks_per_program,
                                "block_order": block_order,
                                "pattern": "real_k",
                                "requested_active_fraction": None,
                                "num_summary_outliers": outliers,
                            }
                        )
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        except Exception as exc:
            rows.append(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "tensor_source": "tensor",
                    "tensor_space": "unknown" if args.tensor_space == "synthetic" else args.tensor_space,
                    "model_id": args.model_id or None,
                    "layer_id": args.layer_id,
                    "shape": {
                        "batch": None,
                        "query_len": None,
                        "kv_len": None,
                        "heads": None,
                        "dim": None,
                        "dtype": args.dtype,
                        "attention_type": "mha",
                    },
                    "pattern": "real_k",
                }
            )
    else:
        for kv_len, heads, active_fraction, block_size, outliers, blocks_per_program, block_order in itertools.product(
            kv_lens,
            heads_values,
            active_fractions,
            block_sizes,
            outlier_values,
            blocks_per_program_values,
            block_orders,
        ):
            run_args = argparse.Namespace(**vars(args))
            run_args.blocks_per_program = blocks_per_program
            run_args.block_order = block_order
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
                        "tensor_source": "synthetic",
                        "tensor_space": args.tensor_space,
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
                        "block_order": block_order,
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
