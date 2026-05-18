"""Profile calibrated Gate-0 candidate filters.

This is an offline experiment. It tests whether cheap metadata can propose
pre-QK skips with enough recall to justify a future Gate-0 runtime. Unlike the
certified bound profiler, candidate filters may produce false skips; those are
reported explicitly for calibration and safety analysis.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_summary_bounds import (
    _dtype,
    _load_real_tensors,
    _make_pattern,
    _parse_values,
    _real_tensor_cases,
    _region_for_block,
    _scan_full_qk,
    _sync,
    _time_call,
)
from stream_attention.certified.bounds import block_score_upper_bound
from stream_attention.certified.summaries import build_block_summaries


def _block_lengths(seq_k: int, block_size: int, *, device: torch.device) -> torch.Tensor:
    num_blocks = (seq_k + block_size - 1) // block_size
    lengths = [
        min(block_size, max(0, seq_k - block_idx * block_size))
        for block_idx in range(num_blocks)
    ]
    return torch.tensor(lengths, device=device, dtype=torch.long)


def _parse_str_values(values: Iterable[str]) -> List[str]:
    parsed = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def _block_order(
    order: str,
    *,
    num_blocks: int,
    sink_blocks: int,
    recent_blocks: int,
) -> List[int]:
    if order == "sequential":
        return list(range(num_blocks))
    if order == "recent_first":
        return list(reversed(range(num_blocks)))
    if order == "sink_recent_first":
        sink = list(range(min(sink_blocks, num_blocks)))
        recent_start = max(0, num_blocks - min(recent_blocks, num_blocks))
        recent = list(range(recent_start, num_blocks))
        rest = list(range(num_blocks))
        seen = set()
        result = []
        for block_idx in [*sink, *reversed(recent), *rest]:
            if block_idx not in seen:
                seen.add(block_idx)
                result.append(block_idx)
        return result
    raise ValueError(f"unknown block order: {order}")


def _scan_blocks(
    scan_region: str,
    *,
    num_blocks: int,
    sink_blocks: int,
    recent_blocks: int,
) -> List[int]:
    if scan_region == "all":
        return list(range(num_blocks))
    if scan_region == "middle_only":
        return [
            block_idx
            for block_idx in range(num_blocks)
            if _region_for_block(
                block_idx,
                num_blocks,
                sink_blocks=sink_blocks,
                recent_blocks=recent_blocks,
            )
            == "middle"
        ]
    if scan_region == "middle_plus_old":
        return [
            block_idx
            for block_idx in range(num_blocks)
            if _region_for_block(
                block_idx,
                num_blocks,
                sink_blocks=sink_blocks,
                recent_blocks=recent_blocks,
            )
            != "recent"
        ]
    raise ValueError(f"unknown scan region: {scan_region}")


def _actual_skip_labels(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    error_budget: float,
    block_order: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, query_len, heads, dim = q.shape
    seq_k = k.shape[1]
    num_blocks = (seq_k + block_size - 1) // block_size
    scale = 1.0 / math.sqrt(dim)
    log_eps = math.log(error_budget) if error_budget > 0.0 else -float("inf")
    block_lengths = _block_lengths(seq_k, block_size, device=q.device)

    q_bhsd = q.permute(0, 2, 1, 3).contiguous().float()
    k_bhnd = k.permute(0, 2, 1, 3).contiguous().float()
    running_max = torch.full(
        (batch, heads, query_len),
        -float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    acc_den = torch.zeros(batch, heads, query_len, device=q.device, dtype=torch.float32)
    actual_skip = torch.zeros(batch, heads, query_len, num_blocks, device=q.device, dtype=torch.bool)
    has_state_by_block = torch.zeros_like(actual_skip)
    threshold_by_block = torch.full(
        (batch, heads, query_len, num_blocks),
        -float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    tile_max_by_block = torch.full_like(threshold_by_block, -float("inf"))

    for block_idx in block_order:
        start = block_idx * block_size
        end = min(start + block_size, seq_k)
        block_len = int(block_lengths[block_idx].item())
        scores = torch.einsum("bhsd,bhnd->bhsn", q_bhsd, k_bhnd[:, :, start:end, :]) * scale
        tile_max = scores.amax(dim=-1)

        has_state = torch.isfinite(running_max) & (acc_den > 0)
        lse = torch.where(
            has_state,
            running_max + torch.log(acc_den.clamp_min(1.0e-30)),
            torch.full_like(running_max, -float("inf")),
        )
        threshold = lse + log_eps
        skip = has_state & (tile_max + math.log(max(1, block_len)) <= threshold)
        actual_skip[..., block_idx] = skip
        has_state_by_block[..., block_idx] = has_state
        threshold_by_block[..., block_idx] = threshold
        tile_max_by_block[..., block_idx] = tile_max

        compute_row = ~skip
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

    return actual_skip, has_state_by_block, threshold_by_block, tile_max_by_block


def _hadamard_matrix(dim: int, *, device: torch.device) -> torch.Tensor:
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError("Hadamard projection requires power-of-two head dim")
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < dim:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return h / math.sqrt(dim)


def _projection_matrix(
    kind: str,
    *,
    dim: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if rank <= 0 or rank > dim:
        raise ValueError("projection rank must be in [1, dim]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if kind == "random":
        matrix = torch.randn(rank, dim, generator=generator, dtype=torch.float32)
        matrix = F.normalize(matrix, p=2, dim=-1)
        return matrix.to(device=device)
    if kind == "hadamard":
        h = _hadamard_matrix(dim, device=torch.device("cpu"))
        indices = torch.randperm(dim, generator=generator)[:rank]
        return h[indices].to(device=device)
    raise ValueError(f"unknown projection kind: {kind}")


def _projection_metadata(
    k: torch.Tensor,
    *,
    block_size: int,
    projection: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_bhnd = k.permute(0, 2, 1, 3).contiguous().float()
    k_proj = torch.einsum("bhnd,rd->bhnr", k_bhnd, projection)
    batch, heads, seq_k, rank = k_proj.shape
    num_blocks = (seq_k + block_size - 1) // block_size
    mins = torch.empty(batch, heads, num_blocks, rank, device=k.device, dtype=torch.float32)
    maxs = torch.empty_like(mins)
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_k)
        block = k_proj[:, :, start:end, :]
        mins[:, :, block_idx, :] = block.amin(dim=2)
        maxs[:, :, block_idx, :] = block.amax(dim=2)
    return mins, maxs


def _projection_scores(
    q: torch.Tensor,
    *,
    projection: torch.Tensor,
    proj_min: torch.Tensor,
    proj_max: torch.Tensor,
    selected_blocks: List[int],
) -> torch.Tensor:
    batch, query_len, heads, dim = q.shape
    num_blocks = proj_min.shape[2]
    scale = 1.0 / math.sqrt(dim)
    rank = projection.shape[0]
    q_bhsd = q.permute(0, 2, 1, 3).contiguous().float()
    q_proj = torch.einsum("bhsd,rd->bhsr", q_bhsd, projection)
    scores = torch.full(
        (batch, heads, query_len, num_blocks),
        float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    if not selected_blocks:
        return scores
    idx = torch.tensor(selected_blocks, device=q.device, dtype=torch.long)
    mins = proj_min.index_select(2, idx)
    maxs = proj_max.index_select(2, idx)
    chosen = torch.where(q_proj[:, :, :, None, :] >= 0, maxs[:, :, None, :, :], mins[:, :, None, :, :])
    approx_upper = (q_proj[:, :, :, None, :] * chosen).sum(dim=-1)
    approx_upper = approx_upper * (dim / rank) * scale
    scores.index_copy_(3, idx, approx_upper)
    return scores


def _certified_scores(
    q: torch.Tensor,
    summaries,
    *,
    selected_blocks: List[int],
) -> torch.Tensor:
    batch, query_len, heads, dim = q.shape
    scale = 1.0 / math.sqrt(dim)
    q_bhsd = q.permute(0, 2, 1, 3).contiguous().float()
    scores = torch.full(
        (batch, heads, query_len, summaries.num_blocks),
        float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for block_idx in selected_blocks:
        scores[..., block_idx] = block_score_upper_bound(q_bhsd, summaries, block_idx, scale=scale)
    return scores


def _candidate_metrics(
    *,
    scores: torch.Tensor,
    actual_skip: torch.Tensor,
    has_state: torch.Tensor,
    thresholds: torch.Tensor,
    block_lengths: torch.Tensor,
    filter_margin: float,
    sink_blocks: int,
    recent_blocks: int,
) -> Dict[str, Any]:
    batch, heads, query_len, num_blocks = scores.shape
    log_lengths = block_lengths.float().clamp_min(1).log().view(1, 1, 1, num_blocks)
    predicted = has_state & torch.isfinite(scores) & (scores + log_lengths <= thresholds + filter_margin)
    recovered = predicted & actual_skip
    false_skip = predicted & ~actual_skip
    total = int(predicted.numel())
    actual_count = int(actual_skip.sum().item())
    predicted_count = int(predicted.sum().item())
    recovered_count = int(recovered.sum().item())
    false_skip_count = int(false_skip.sum().item())
    region_metrics: Dict[str, float] = {}
    for region in ("sink", "middle", "recent"):
        mask = torch.zeros(num_blocks, device=scores.device, dtype=torch.bool)
        for block_idx in range(num_blocks):
            if _region_for_block(
                block_idx,
                num_blocks,
                sink_blocks=sink_blocks,
                recent_blocks=recent_blocks,
            ) == region:
                mask[block_idx] = True
        region_total = int(mask.sum().item()) * batch * heads * query_len
        if region_total == 0:
            region_metrics[f"{region}_actual_skip_fraction"] = 0.0
            region_metrics[f"{region}_predicted_skip_fraction"] = 0.0
            region_metrics[f"{region}_actual_skip_recovery"] = 0.0
            region_metrics[f"{region}_false_skip_rate"] = 0.0
            continue
        region_actual = actual_skip[..., mask]
        region_predicted = predicted[..., mask]
        region_recovered = recovered[..., mask]
        region_false_skip = false_skip[..., mask]
        region_actual_count = int(region_actual.sum().item())
        region_predicted_count = int(region_predicted.sum().item())
        region_metrics[f"{region}_actual_skip_fraction"] = region_actual_count / region_total
        region_metrics[f"{region}_predicted_skip_fraction"] = int(region_predicted.sum().item()) / region_total
        region_metrics[f"{region}_actual_skip_recovery"] = (
            int(region_recovered.sum().item()) / region_actual_count if region_actual_count else 0.0
        )
        region_metrics[f"{region}_false_skip_rate"] = (
            int(region_false_skip.sum().item()) / region_predicted_count
            if region_predicted_count
            else 0.0
        )
    return {
        "total_row_blocks": total,
        "actual_skip_count": actual_count,
        "predicted_skip_count": predicted_count,
        "recovered_skip_count": recovered_count,
        "false_skip_count": false_skip_count,
        "actual_skip_fraction": actual_count / total if total else 0.0,
        "predicted_skip_fraction": predicted_count / total if total else 0.0,
        "actual_skip_recovery": recovered_count / actual_count if actual_count else 0.0,
        "false_skip_fraction": false_skip_count / total if total else 0.0,
        "false_skip_rate": false_skip_count / predicted_count if predicted_count else 0.0,
        "precision": recovered_count / predicted_count if predicted_count else 0.0,
        **region_metrics,
    }


def _profile_case(
    args,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    filter_mode: str,
    projection_dim: Optional[int],
    filter_margin: float,
    scan_region: str,
    block_order_name: str,
    row_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    device = q.device
    batch, query_len, heads, dim = q.shape
    seq_k = k.shape[1]
    num_blocks = (seq_k + block_size - 1) // block_size
    block_lengths = _block_lengths(seq_k, block_size, device=device)
    order = _block_order(
        block_order_name,
        num_blocks=num_blocks,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )
    selected_blocks = _scan_blocks(
        scan_region,
        num_blocks=num_blocks,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )

    actual_skip, has_state, thresholds, _tile_max = _actual_skip_labels(
        q,
        k,
        block_size=block_size,
        error_budget=args.error_budget,
        block_order=order,
    )
    q_bhsd = q.permute(0, 2, 1, 3).contiguous().float()
    k_bhnd = k.permute(0, 2, 1, 3).contiguous().float()
    full_qk_scan_ms = _time_call(
        lambda: _scan_full_qk(q_bhsd, k_bhnd, block_size=block_size, scale=1.0 / math.sqrt(dim)),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    projection_kind = None
    metadata_build_ms = 0.0
    if filter_mode in ("projection_random", "projection_hadamard"):
        if projection_dim is None:
            raise ValueError("projection_dim is required for projection filters")
        projection_kind = "random" if filter_mode == "projection_random" else "hadamard"
        projection = _projection_matrix(
            projection_kind,
            dim=dim,
            rank=projection_dim,
            seed=args.seed,
            device=device,
        )
        metadata_build_ms = _time_call(
            lambda: _projection_metadata(k, block_size=block_size, projection=projection),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        proj_min, proj_max = _projection_metadata(k, block_size=block_size, projection=projection)
        scan_fn = lambda: _projection_scores(
            q,
            projection=projection,
            proj_min=proj_min,
            proj_max=proj_max,
            selected_blocks=selected_blocks,
        )
    elif filter_mode in ("certified_centroid", "certified_outlier2"):
        num_outliers = 0 if filter_mode == "certified_centroid" else 2
        metadata_build_ms = _time_call(
            lambda: build_block_summaries(k, block_size=block_size, num_outliers=num_outliers),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        summaries = build_block_summaries(k, block_size=block_size, num_outliers=num_outliers)
        scan_fn = lambda: _certified_scores(q, summaries, selected_blocks=selected_blocks)
    else:
        raise ValueError(f"unknown filter mode: {filter_mode}")

    candidate_scan_ms = _time_call(
        lambda: scan_fn().sum(),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    scores = scan_fn()
    metrics = _candidate_metrics(
        scores=scores,
        actual_skip=actual_skip,
        has_state=has_state,
        thresholds=thresholds,
        block_lengths=block_lengths,
        filter_margin=filter_margin,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )
    predicted_compute_fraction = 1.0 - float(metrics["predicted_skip_fraction"])
    estimated_qk_path_ms = candidate_scan_ms + predicted_compute_fraction * full_qk_scan_ms
    scan_over_qk = candidate_scan_ms / full_qk_scan_ms if full_qk_scan_ms > 0.0 else None
    estimated_speedup = full_qk_scan_ms / estimated_qk_path_ms if estimated_qk_path_ms > 0.0 else None
    promising = (
        scan_over_qk is not None
        and scan_over_qk <= args.max_scan_over_qk
        and float(metrics["actual_skip_recovery"]) >= args.min_recovery
        and float(metrics["false_skip_rate"]) <= args.max_false_skip_rate
    )

    row: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(q.device) if q.device.type == "cuda" else str(q.device),
        "tensor_source": "tensor" if args.q_path else "synthetic",
        "tensor_space": args.tensor_space if args.q_path else "synthetic",
        "model_id": args.model_id or None,
        "layer_id": None if args.layer_id < 0 else args.layer_id,
        "filter_mode": filter_mode,
        "projection_kind": projection_kind,
        "projection_dim": projection_dim,
        "projection_seed": args.seed if projection_kind else None,
        "filter_margin": filter_margin,
        "scan_region": scan_region,
        "block_order": block_order_name,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "scan_block_count": len(selected_blocks),
        "scan_block_fraction": len(selected_blocks) / num_blocks if num_blocks else 0.0,
        "shape": {
            "batch": batch,
            "query_len": query_len,
            "kv_len": seq_k,
            "heads": heads,
            "dim": dim,
            "dtype": args.dtype,
        },
        "error_budget": args.error_budget,
        "metadata_build_ms": metadata_build_ms,
        "candidate_scan_ms": candidate_scan_ms,
        "full_qk_scan_ms": full_qk_scan_ms,
        "scan_over_qk": scan_over_qk,
        "estimated_qk_path_ms": estimated_qk_path_ms,
        "estimated_speedup_vs_qk": estimated_speedup,
        "candidate_promising": promising,
        "torch_version": torch.__version__,
        "timing_note": "offline_pytorch_cuda_events",
        **metrics,
    }
    if row_extra:
        row.update(row_extra)
    return row


def _synthetic_cases(args, *, device: torch.device, dtype: torch.dtype):
    for kv_len, heads, active_fraction, block_size in itertools.product(
        _parse_values(args.kv_lens, int),
        _parse_values(args.heads, int),
        _parse_values(args.active_fraction, float),
        _parse_values(args.block_size, int),
    ):
        q, k, _v, actual_active, active_block_ids = _make_pattern(
            batch=1,
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
        yield q, k, block_size, {
            "pattern": args.pattern,
            "requested_active_fraction": active_fraction,
            "block_quantized_active_fraction": actual_active,
            "active_block_ids": active_block_ids,
        }


def _iter_cases(args, *, device: torch.device, dtype: torch.dtype):
    if args.q_path:
        for q, k, _v, extra in _real_tensor_cases(args, device=device, dtype=dtype):
            for block_size in _parse_values(args.block_size, int):
                yield q, k, block_size, extra
        return
    yield from _synthetic_cases(args, device=device, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", default="")
    parser.add_argument("--k-path", default="")
    parser.add_argument("--v-path", default="")
    parser.add_argument("--tensor-format", choices=["pt"], default="pt")
    parser.add_argument("--tensor-space", default="unknown")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--layer-id", type=int, default=-1)
    parser.add_argument("--per-head", action="store_true")
    parser.add_argument("--head-indices", nargs="*", default=[])
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--kv-lens", nargs="+", default=["4096"])
    parser.add_argument("--heads", nargs="+", default=["16"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--pattern", choices=["random", "peaked", "sink_local", "sliding_recent"], default="peaked")
    parser.add_argument("--active-fraction", nargs="+", default=["0.25"])
    parser.add_argument("--block-size", nargs="+", default=["64"])
    parser.add_argument(
        "--filter-mode",
        nargs="+",
        default=["projection_random"],
    )
    parser.add_argument("--projection-dim", nargs="+", default=["8", "16", "32"])
    parser.add_argument("--filter-margin", nargs="+", default=["0.0"])
    parser.add_argument("--scan-region", nargs="+", default=["all"])
    parser.add_argument("--block-order", nargs="+", default=["recent_first"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--min-recovery", type=float, default=0.50)
    parser.add_argument("--max-false-skip-rate", type=float, default=0.01)
    parser.add_argument("--max-scan-over-qk", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype(args.dtype)
    rows = []
    with torch.no_grad():
        for q, k, block_size, extra in _iter_cases(args, device=device, dtype=dtype):
            for filter_mode, margin, scan_region, block_order_name in itertools.product(
                _parse_str_values(args.filter_mode),
                _parse_values(args.filter_margin, float),
                _parse_str_values(args.scan_region),
                _parse_str_values(args.block_order),
            ):
                projection_dims: List[Optional[int]]
                if filter_mode.startswith("projection_"):
                    projection_dims = _parse_values(args.projection_dim, int)
                else:
                    projection_dims = [None]
                for projection_dim in projection_dims:
                    try:
                        rows.append(
                            _profile_case(
                                args,
                                q,
                                k,
                                block_size=block_size,
                                filter_mode=filter_mode,
                                projection_dim=projection_dim,
                                filter_margin=margin,
                                scan_region=scan_region,
                                block_order_name=block_order_name,
                                row_extra=extra,
                            )
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "error": f"{type(exc).__name__}: {exc}",
                                "filter_mode": filter_mode,
                                "projection_dim": projection_dim,
                                "filter_margin": margin,
                                "scan_region": scan_region,
                                "block_order": block_order_name,
                                "block_size": block_size,
                                **extra,
                            }
                        )
    payload = {"rows": rows}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
