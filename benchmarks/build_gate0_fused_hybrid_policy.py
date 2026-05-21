"""Build calibrated Gate-0 fused-hybrid split-K policy artifacts.

This consumes hybrid-correction benchmark outputs and promotes trusted sparse
head sets that are both fast and within an error budget.  Unlike the older
split-K policy builder, this policy describes a fused head-mode runtime:

* trusted heads use inline projection Gate-0;
* every other head uses exact split-K mode in the same kernel.

The artifact is still a research policy.  It is meant to make calibration
explicit before this path is wired into a production runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.summarize_gate1_inline_projection_splitk_robustness import (
    DEFAULT_ROBUSTNESS_BUDGETS,
    collect_rows_from_payload,
    parse_budgets,
)
from benchmarks.model_head_mode_kv_group_work import model_kv_group_work


SCHEMA = "streamattn.gate0.fused_hybrid_splitk_policy.v1"


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _heads(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(item) for item in raw]


def _head_group(heads: Iterable[int]) -> str:
    return ",".join(str(head) for head in sorted(set(int(item) for item in heads)))


def _kv_len(row: Dict[str, Any]) -> int | None:
    value = row.get("kv_len")
    if value is None:
        value = (row.get("capture_shape") or {}).get("kv_len")
    if value is None:
        value = (row.get("shape") or {}).get("kv_len")
    return int(value) if value is not None else None


def _shape(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("shape") or row.get("capture_shape") or {}


def _kv_heads(row: Dict[str, Any]) -> int | None:
    shape = _shape(row)
    runtime = row.get("runtime") or {}
    for value in (
        shape.get("kv_heads"),
        shape.get("h_kv"),
        runtime.get("true_policy_kv_heads"),
        runtime.get("kv_heads"),
    ):
        if value is not None:
            return int(value)
    return None


def _load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in collect_rows_from_payload(payload, source=str(path)):
            copied = dict(row)
            copied["_source"] = str(path)
            rows.append(copied)
    return rows


def _per_head_map(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    values = payload.get("per_head") or []
    return {int(item.get("head")): dict(item) for item in values if isinstance(item, dict)}


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else default


def _runtime(row: Dict[str, Any], trusted_heads: Sequence[int], exact_heads: Sequence[int]) -> Dict[str, Any]:
    runtime = row.get("runtime") or {}
    policy = row.get("policy") or {}
    aggressive_heads = _heads(policy.get("aggressive_sparse_heads") or row.get("aggressive_heads"))
    head_count = _as_int((row.get("shape") or row.get("capture_shape") or {}).get("heads"), 0)
    head_modes = [1 for _ in range(head_count)]
    for head in trusted_heads:
        if 0 <= int(head) < head_count:
            head_modes[int(head)] = 0
    return {
        "mode": "fused_head_mode_splitk",
        "head_modes": head_modes,
        "aggressive_sparse_heads": aggressive_heads,
        "trusted_sparse_heads": list(trusted_heads),
        "trusted_head_group": _head_group(trusted_heads),
        "exact_heads": list(exact_heads),
        "exact_head_group": _head_group(exact_heads),
        "block_size": runtime.get("block_size"),
        "sink_blocks": runtime.get("sink_blocks"),
        "recent_blocks": runtime.get("recent_blocks"),
        "middle_seed_blocks": runtime.get("middle_seed_blocks"),
        "chunk_anchor_blocks": runtime.get("chunk_anchor_blocks"),
        "block_order": runtime.get("block_order"),
        "num_chunks": runtime.get("num_chunks"),
        "seed_strategy": runtime.get("seed_strategy"),
        "filter_margin": runtime.get("filter_margin"),
        "error_budget": runtime.get("error_budget"),
        "projection_kind": runtime.get("projection_kind"),
        "projection_dim": runtime.get("projection_dim"),
        "projection_seed": runtime.get("projection_seed"),
        "projection_metadata_dtype": runtime.get("projection_metadata_dtype"),
        "splitk_workspace": runtime.get("splitk_workspace"),
    }


def _backend_work(row: Dict[str, Any], trusted_heads: Sequence[int]) -> Dict[str, Any] | None:
    shape = _shape(row)
    runtime = row.get("runtime") or {}
    head_count = _as_int(shape.get("heads") or shape.get("q_heads") or shape.get("h_q"), 0)
    kv_heads = _kv_heads(row)
    kv_len = _kv_len(row)
    block_size = runtime.get("block_size")
    if not head_count or not kv_heads or not kv_len or not block_size:
        return None
    try:
        return model_kv_group_work(
            q_heads=head_count,
            kv_heads=kv_heads,
            kv_len=kv_len,
            tile_size=int(block_size),
            seed_heads=trusted_heads,
            sink_blocks=_as_int(runtime.get("sink_blocks"), 0),
            recent_blocks=_as_int(runtime.get("recent_blocks"), 0),
            middle_seed_blocks=_as_int(runtime.get("middle_seed_blocks"), 0),
            block_order=str(runtime.get("block_order") or "recent_first"),
            padded_group_rows=_as_int(runtime.get("padded_group_rows"), 16),
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _row_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    policy = row.get("policy") or {}
    timing = row.get("timing") or {}
    quality = row.get("quality") or {}
    fused_error = quality.get("fused_hybrid_error_vs_dense_all") or quality.get("corrected_error_vs_dense_all") or {}
    fused_per_head_error = _per_head_map(quality.get("fused_hybrid_error_vs_dense_all_per_head") or {})
    fused_per_head_stats = _per_head_map(quality.get("fused_hybrid_per_head_stats") or {})
    trusted_heads = _heads(policy.get("trusted_sparse_heads") or row.get("trusted_heads"))
    exact_heads = _heads(policy.get("exact_heads") or row.get("exact_heads"))
    backend_work = _backend_work(row, trusted_heads)
    backend_totals = (backend_work or {}).get("totals") or {}
    backend_groups = (backend_work or {}).get("per_kv_group") or []
    seed_only_groups = [
        int(group.get("kv_head"))
        for group in backend_groups
        if group.get("seed_only_whole_group")
    ]
    dense_ms = _as_float(timing.get("dense_all_ms"), 0.0)
    fused_ms = _as_float(timing.get("fused_hybrid_ms"), 0.0)
    speedup = timing.get("fused_hybrid_speedup_vs_dense_all")
    if speedup is None and fused_ms > 0.0:
        speedup = dense_ms / fused_ms
    trusted_stats = [fused_per_head_stats.get(head, {}) for head in trusted_heads]
    trusted_errors = [fused_per_head_error.get(head, {}) for head in trusted_heads]
    trusted_skip = _mean(
        [_as_float(item.get("projection_skip_fraction"), 0.0) for item in trusted_stats],
        default=_as_float((quality.get("fused_hybrid_stats") or {}).get("projection_skip_fraction"), 0.0),
    )
    trusted_pv = _mean(
        [_as_float(item.get("pv_executed_fraction"), 0.0) for item in trusted_stats],
        default=_as_float((quality.get("fused_hybrid_stats") or {}).get("pv_executed_fraction"), 0.0),
    )
    trusted_max_error = max(
        [_as_float(item.get("max_abs_error"), 0.0) for item in trusted_errors],
        default=_as_float(fused_error.get("max_abs_error"), 1.0e9),
    )
    trusted_mean_error = _mean(
        [_as_float(item.get("mean_abs_error"), 0.0) for item in trusted_errors],
        default=_as_float(fused_error.get("mean_abs_error"), 1.0e9),
    )
    return {
        "model_id": row.get("model_id"),
        "prompt_type": row.get("prompt_type"),
        "layer_id": row.get("layer_id"),
        "kv_len": _kv_len(row),
        "shape": row.get("shape") or row.get("capture_shape") or {},
        "trusted_heads": trusted_heads,
        "exact_heads": exact_heads,
        "trusted_head_group": _head_group(trusted_heads),
        "dense_all_ms": dense_ms,
        "fused_hybrid_ms": fused_ms,
        "speedup_vs_dense_all": float(speedup) if speedup is not None else None,
        "max_abs_error": _as_float(fused_error.get("max_abs_error"), 1.0e9),
        "mean_abs_error": _as_float(fused_error.get("mean_abs_error"), 1.0e9),
        "trusted_max_abs_error": trusted_max_error,
        "trusted_mean_abs_error": trusted_mean_error,
        "trusted_projection_skip_fraction": trusted_skip,
        "trusted_pv_executed_fraction": trusted_pv,
        "runtime": _runtime(row, trusted_heads, exact_heads),
        "backend_work": backend_work,
        "kv_tile_load_reduction": _as_float(backend_totals.get("kv_tile_load_reduction"), 0.0),
        "padded_row_work_reduction": _as_float(backend_totals.get("padded_row_work_reduction"), 0.0),
        "row_work_reduction": _as_float(backend_totals.get("row_work_reduction"), 0.0),
        "seed_only_kv_groups": seed_only_groups,
        "source": row.get("_source"),
    }


def _passes(
    row: Dict[str, Any],
    budget: Dict[str, float | str],
    *,
    min_trusted_skip_fraction: float,
    min_kv_tile_load_reduction: float,
) -> bool:
    speedup = row.get("speedup_vs_dense_all")
    return (
        bool(row["trusted_heads"])
        and speedup is not None
        and float(speedup) >= float(budget["min_speedup"])
        and float(row["max_abs_error"]) <= float(budget["max_error"])
        and float(row["mean_abs_error"]) <= float(budget["max_mean_error"])
        and float(row["trusted_projection_skip_fraction"]) >= min_trusted_skip_fraction
        and float(row["kv_tile_load_reduction"]) >= min_kv_tile_load_reduction
    )


def _failed_constraints(
    row: Dict[str, Any],
    budget: Dict[str, float | str],
    *,
    min_trusted_skip_fraction: float,
    min_kv_tile_load_reduction: float,
) -> List[str]:
    failed: List[str] = []
    speedup = row.get("speedup_vs_dense_all")
    if not row["trusted_heads"]:
        failed.append("trusted_heads")
    if speedup is None or float(speedup) < float(budget["min_speedup"]):
        failed.append("speedup")
    if float(row["max_abs_error"]) > float(budget["max_error"]):
        failed.append("max_error")
    if float(row["mean_abs_error"]) > float(budget["max_mean_error"]):
        failed.append("mean_error")
    if float(row["trusted_projection_skip_fraction"]) < min_trusted_skip_fraction:
        failed.append("trusted_skip")
    if float(row["kv_tile_load_reduction"]) < min_kv_tile_load_reduction:
        failed.append("kv_group_load_reduction")
    return failed


def _entry(row: Dict[str, Any], budget: Dict[str, float | str]) -> Dict[str, Any]:
    return {
        "model_id": row["model_id"],
        "prompt_type": row["prompt_type"],
        "layer_id": row["layer_id"],
        "kv_len_bucket": row["kv_len"],
        "safety_budget": {
            "name": budget["name"],
            "max_abs_error": budget["max_error"],
            "max_mean_error": budget["max_mean_error"],
            "min_speedup": budget["min_speedup"],
        },
        "mode": "calibrated_fused_hybrid_splitk_inline_projection",
        "fallback": "exact_head_mode",
        "runtime": row["runtime"],
        "quality": {
            "dense_all_ms": row["dense_all_ms"],
            "fused_hybrid_ms": row["fused_hybrid_ms"],
            "speedup_vs_dense_all": row["speedup_vs_dense_all"],
            "max_abs_error": row["max_abs_error"],
            "mean_abs_error": row["mean_abs_error"],
            "trusted_max_abs_error": row["trusted_max_abs_error"],
            "trusted_mean_abs_error": row["trusted_mean_abs_error"],
            "trusted_projection_skip_fraction": row["trusted_projection_skip_fraction"],
            "trusted_pv_executed_fraction": row["trusted_pv_executed_fraction"],
            "row_work_reduction": row["row_work_reduction"],
            "kv_tile_load_reduction": row["kv_tile_load_reduction"],
            "padded_row_work_reduction": row["padded_row_work_reduction"],
            "seed_only_kv_groups": row["seed_only_kv_groups"],
        },
        "backend_work": row["backend_work"],
        "source": row["source"],
    }


def _entry_key(row: Dict[str, Any], budget_name: str) -> Tuple[Any, ...]:
    return (
        budget_name,
        row.get("model_id"),
        row.get("prompt_type"),
        row.get("layer_id"),
        row.get("kv_len"),
    )


def _stable_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    runtime = entry["runtime"]
    return (
        entry["safety_budget"]["name"],
        entry.get("model_id"),
        entry.get("layer_id"),
        entry.get("kv_len_bucket"),
        runtime.get("trusted_head_group"),
        runtime.get("block_size"),
        runtime.get("num_chunks"),
        runtime.get("filter_margin"),
        runtime.get("projection_dim"),
        runtime.get("projection_seed"),
    )


def _build_stable_entries(entries: Sequence[Dict[str, Any]], *, min_stable_prompts: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(_stable_key(entry), []).append(entry)
    stable: List[Dict[str, Any]] = []
    for group in grouped.values():
        prompt_types = sorted({str(item.get("prompt_type")) for item in group})
        if len(prompt_types) < min_stable_prompts:
            continue
        speeds = [float(item["quality"]["speedup_vs_dense_all"]) for item in group]
        errors = [float(item["quality"]["max_abs_error"]) for item in group]
        means = [float(item["quality"]["mean_abs_error"]) for item in group]
        skips = [float(item["quality"]["trusted_projection_skip_fraction"]) for item in group]
        representative = group[0]
        stable.append(
            {
                "model_id": representative["model_id"],
                "layer_id": representative["layer_id"],
                "kv_len_bucket": representative["kv_len_bucket"],
                "safety_budget": representative["safety_budget"],
                "mode": representative["mode"],
                "fallback": representative["fallback"],
                "runtime": representative["runtime"],
                "robustness": {
                    "prompt_count": len(prompt_types),
                    "prompt_types": prompt_types,
                    "min_speedup_vs_dense_all": min(speeds),
                    "max_speedup_vs_dense_all": max(speeds),
                    "max_abs_error_seen": max(errors),
                    "max_mean_error_seen": max(means),
                    "min_trusted_projection_skip_fraction": min(skips),
                    "min_kv_tile_load_reduction": min(
                        float(item["quality"].get("kv_tile_load_reduction") or 0.0) for item in group
                    ),
                    "min_padded_row_work_reduction": min(
                        float(item["quality"].get("padded_row_work_reduction") or 0.0) for item in group
                    ),
                },
                "sources": sorted({str(item.get("source")) for item in group}),
            }
        )
    stable.sort(
        key=lambda item: (
            item["robustness"]["prompt_count"],
            item["robustness"]["min_speedup_vs_dense_all"],
            -item["robustness"]["max_abs_error_seen"],
        ),
        reverse=True,
    )
    return stable


def _frontier_candidate(
    row: Dict[str, Any],
    budget: Dict[str, float | str],
    failed: Sequence[str],
) -> Dict[str, Any]:
    runtime = row["runtime"]
    experiments: List[Dict[str, Any]] = []
    if "max_error" in failed or "mean_error" in failed:
        margin = runtime.get("filter_margin")
        if margin is not None:
            for delta in (8.0, 16.0, 32.0):
                next_margin = max(0.0, float(margin) - delta)
                if next_margin != float(margin):
                    experiments.append(
                        {
                            "kind": "tighten_margin",
                            "reason": "reduce calibrated projection error",
                            "filter_margin": next_margin,
                            "trusted_head_group": runtime.get("trusted_head_group"),
                        }
                    )
        heads = row.get("trusted_heads") or []
        if len(heads) > 1:
            for head in heads:
                trimmed = [item for item in heads if item != head]
                experiments.append(
                    {
                        "kind": "leave_one_out_trusted_heads",
                        "reason": "isolate unsafe trusted head",
                        "trusted_head_group": _head_group(trimmed),
                        "removed_head": head,
                    }
                )
    if "speedup" in failed and runtime.get("num_chunks") is not None:
        chunks = int(runtime["num_chunks"])
        for next_chunks in sorted({max(2, chunks // 2), chunks * 2}):
            if next_chunks != chunks:
                experiments.append(
                    {
                        "kind": "chunk_sweep",
                        "reason": "test exact/sparse split-K overhead balance",
                        "num_chunks": next_chunks,
                        "trusted_head_group": runtime.get("trusted_head_group"),
                    }
                )
    if "kv_group_load_reduction" in failed:
        heads = row.get("trusted_heads") or []
        shape = row.get("shape") or {}
        kv_heads = _kv_heads(row)
        q_heads = _as_int(shape.get("heads") or shape.get("q_heads") or shape.get("h_q"), 0)
        if kv_heads and q_heads and q_heads % kv_heads == 0:
            group_size = q_heads // kv_heads
            seed_set = set(int(head) for head in heads)
            for kv_head in range(kv_heads):
                group = list(range(kv_head * group_size, (kv_head + 1) * group_size))
                missing = [head for head in group if head not in seed_set]
                if missing and len(missing) <= max(1, group_size // 2):
                    experiments.append(
                        {
                            "kind": "kv_group_coherent_seed_policy",
                            "reason": "whole seed-only KV groups can skip K/V tile scheduling",
                            "kv_head": kv_head,
                            "candidate_trusted_head_group": _head_group(sorted(seed_set | set(group))),
                            "additional_heads_to_verify": missing,
                        }
                    )
    return {
        "model_id": row["model_id"],
        "prompt_type": row["prompt_type"],
        "layer_id": row["layer_id"],
        "kv_len_bucket": row["kv_len"],
        "budget": budget["name"],
        "failed_constraints": list(failed),
        "runtime": runtime,
        "quality": {
            "speedup_vs_dense_all": row["speedup_vs_dense_all"],
            "max_abs_error": row["max_abs_error"],
            "mean_abs_error": row["mean_abs_error"],
            "trusted_projection_skip_fraction": row["trusted_projection_skip_fraction"],
            "kv_tile_load_reduction": row["kv_tile_load_reduction"],
            "padded_row_work_reduction": row["padded_row_work_reduction"],
        },
        "experiments": experiments[:8],
        "source": row["source"],
    }


def build_policy(
    raw_rows: Sequence[Dict[str, Any]],
    budgets: Sequence[Dict[str, float | str]],
    *,
    min_trusted_skip_fraction: float = 0.5,
    min_stable_prompts: int = 2,
    frontier_error_multiplier: float = 1.5,
    frontier_mean_error_multiplier: float = 1.5,
    frontier_speedup_scale: float = 0.95,
    min_kv_tile_load_reduction: float = 0.0,
    frontier_limit: int = 32,
) -> Dict[str, Any]:
    rows = [_row_metrics(row) for row in raw_rows]
    grouped_passes: Dict[Tuple[Any, ...], List[Tuple[Dict[str, Any], Dict[str, float | str]]]] = {}
    frontier: List[Dict[str, Any]] = []
    for budget in budgets:
        for row in rows:
            if _passes(
                row,
                budget,
                min_trusted_skip_fraction=min_trusted_skip_fraction,
                min_kv_tile_load_reduction=min_kv_tile_load_reduction,
            ):
                grouped_passes.setdefault(_entry_key(row, str(budget["name"])), []).append((row, budget))
                continue
            failed = _failed_constraints(
                row,
                budget,
                min_trusted_skip_fraction=min_trusted_skip_fraction,
                min_kv_tile_load_reduction=min_kv_tile_load_reduction,
            )
            speedup = row.get("speedup_vs_dense_all")
            near = (
                speedup is not None
                and float(speedup) >= float(budget["min_speedup"]) * frontier_speedup_scale
                and float(row["max_abs_error"]) <= float(budget["max_error"]) * frontier_error_multiplier
                and float(row["mean_abs_error"])
                <= float(budget["max_mean_error"]) * frontier_mean_error_multiplier
                and float(row["trusted_projection_skip_fraction"]) >= min_trusted_skip_fraction
            )
            if near:
                frontier.append(_frontier_candidate(row, budget, failed))

    entries: List[Dict[str, Any]] = []
    for candidates in grouped_passes.values():
        best_row, budget = max(
            candidates,
            key=lambda item: (
                float(item[0]["speedup_vs_dense_all"] or 0.0),
                -float(item[0]["max_abs_error"]),
                float(item[0]["kv_tile_load_reduction"]),
                float(item[0]["padded_row_work_reduction"]),
                len(item[0]["trusted_heads"]),
                float(item[0]["trusted_projection_skip_fraction"]),
            ),
        )
        entries.append(_entry(best_row, budget))

    entries.sort(
        key=lambda item: (
            str(item["safety_budget"]["name"]),
            str(item.get("model_id")),
            str(item.get("prompt_type")),
            int(item.get("layer_id") or -1),
            -float(item["quality"]["speedup_vs_dense_all"] or 0.0),
        )
    )
    frontier.sort(
        key=lambda item: (
            -float(item["quality"]["speedup_vs_dense_all"] or 0.0),
            float(item["quality"]["max_abs_error"] or 1.0e9),
        )
    )
    stable_entries = _build_stable_entries(entries, min_stable_prompts=min_stable_prompts)
    budget_summary = []
    for budget in budgets:
        name = str(budget["name"])
        matching = [item for item in entries if item["safety_budget"]["name"] == name]
        stable = [item for item in stable_entries if item["safety_budget"]["name"] == name]
        budget_summary.append(
            {
                "budget": name,
                "entries": len(matching),
                "stable_entries": len(stable),
                "prompt_types": sorted({str(item["prompt_type"]) for item in matching}),
                "layers": sorted({int(item["layer_id"]) for item in matching if item["layer_id"] is not None}),
                "best_speedup": max(
                    (float(item["quality"]["speedup_vs_dense_all"]) for item in matching),
                    default=0.0,
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "summary": {
            "rows": len(rows),
            "entries": len(entries),
            "stable_entries": len(stable_entries),
            "frontier_candidates": min(len(frontier), frontier_limit),
            "min_trusted_skip_fraction": min_trusted_skip_fraction,
            "min_stable_prompts": min_stable_prompts,
            "frontier_error_multiplier": frontier_error_multiplier,
            "frontier_mean_error_multiplier": frontier_mean_error_multiplier,
            "frontier_speedup_scale": frontier_speedup_scale,
            "min_kv_tile_load_reduction": min_kv_tile_load_reduction,
        },
        "budgets": list(budgets),
        "budget_summary": budget_summary,
        "entries": entries,
        "stable_entries": stable_entries,
        "frontier": frontier[:frontier_limit],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_entries(entries: Sequence[Dict[str, Any]], *, limit: int) -> None:
    headers = ["budget", "prompt", "layer", "kv", "trusted", "speed", "err", "mean", "skip", "kvred"]
    print(" ".join(f"{header:>12}" for header in headers))
    for row in entries[:limit]:
        quality = row["quality"]
        runtime = row["runtime"]
        print(
            " ".join(
                [
                    f"{_fmt(row['safety_budget']['name']):>12}",
                    f"{_fmt(row.get('prompt_type'))[:12]:>12}",
                    f"{_fmt(row.get('layer_id')):>12}",
                    f"{_fmt(row.get('kv_len_bucket')):>12}",
                    f"{_fmt(runtime.get('trusted_head_group'))[:12]:>12}",
                    f"{_fmt(quality.get('speedup_vs_dense_all')):>12}",
                    f"{_fmt(quality.get('max_abs_error')):>12}",
                    f"{_fmt(quality.get('mean_abs_error')):>12}",
                    f"{_fmt(quality.get('trusted_projection_skip_fraction')):>12}",
                    f"{_fmt(quality.get('kv_tile_load_reduction')):>12}",
                ]
            )
        )


def _print_stable(entries: Sequence[Dict[str, Any]], *, limit: int) -> None:
    if not entries:
        return
    print("stable")
    headers = ["budget", "layer", "kv", "trusted", "prompts", "min_spd", "err", "skip", "kvred"]
    print(" ".join(f"{header:>12}" for header in headers))
    for row in entries[:limit]:
        runtime = row["runtime"]
        robust = row["robustness"]
        print(
            " ".join(
                [
                    f"{_fmt(row['safety_budget']['name']):>12}",
                    f"{_fmt(row.get('layer_id')):>12}",
                    f"{_fmt(row.get('kv_len_bucket')):>12}",
                    f"{_fmt(runtime.get('trusted_head_group'))[:12]:>12}",
                    f"{_fmt(robust.get('prompt_count')):>12}",
                    f"{_fmt(robust.get('min_speedup_vs_dense_all')):>12}",
                    f"{_fmt(robust.get('max_abs_error_seen')):>12}",
                    f"{_fmt(robust.get('min_trusted_projection_skip_fraction')):>12}",
                    f"{_fmt(robust.get('min_kv_tile_load_reduction')):>12}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--budgets", default=DEFAULT_ROBUSTNESS_BUDGETS)
    parser.add_argument("--min-trusted-skip-fraction", type=float, default=0.5)
    parser.add_argument("--min-stable-prompts", type=int, default=2)
    parser.add_argument("--frontier-error-multiplier", type=float, default=1.5)
    parser.add_argument("--frontier-mean-error-multiplier", type=float, default=1.5)
    parser.add_argument("--frontier-speedup-scale", type=float, default=0.95)
    parser.add_argument("--min-kv-tile-load-reduction", type=float, default=0.0)
    parser.add_argument("--frontier-limit", type=int, default=32)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    payload = build_policy(
        _load_rows(args.json_paths),
        parse_budgets(args.budgets),
        min_trusted_skip_fraction=args.min_trusted_skip_fraction,
        min_stable_prompts=args.min_stable_prompts,
        frontier_error_multiplier=args.frontier_error_multiplier,
        frontier_mean_error_multiplier=args.frontier_mean_error_multiplier,
        frontier_speedup_scale=args.frontier_speedup_scale,
        min_kv_tile_load_reduction=args.min_kv_tile_load_reduction,
        frontier_limit=args.frontier_limit,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    _print_entries(payload["entries"], limit=args.limit)
    _print_stable(payload["stable_entries"], limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
