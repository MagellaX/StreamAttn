"""Build calibrated Gate-0 split-K inline projection policy artifacts.

This consumes raw split-K inline projection benchmark JSON files and promotes
passing grouped-head rows into policy entries.  It also reports near-miss
"frontier" rows and concrete follow-up experiment knobs so risky exploration is
explicit instead of hidden in ad hoc spreadsheet notes.
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


SCHEMA = "streamattn.gate0.splitk_inline_projection_policy.v1"


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _nested_float(row: Dict[str, Any], parent: str, key: str, default: float = 0.0) -> float:
    value = (row.get(parent) or {}).get(key)
    return _as_float(value, default)


def _kv_len(row: Dict[str, Any]) -> int | None:
    value = row.get("kv_len")
    if value is None:
        value = (row.get("capture_shape") or {}).get("kv_len")
    if value is None:
        value = (row.get("shape") or {}).get("kv_len")
    return int(value) if value is not None else None


def _head_indices(row: Dict[str, Any]) -> List[int]:
    indices = row.get("head_indices") or row.get("selected_head_indices")
    if indices:
        return [int(item) for item in indices]
    if row.get("head_index") is not None:
        return [int(row["head_index"])]
    raw = row.get("head_group")
    if raw:
        return [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    return []


def _head_group(row: Dict[str, Any]) -> str:
    indices = _head_indices(row)
    return ",".join(str(item) for item in indices)


def _speedup(row: Dict[str, Any]) -> float | None:
    value = row.get("splitk_vs_dense_speedup")
    if value is not None:
        return float(value)
    splitk_ms = _as_float(row.get("splitk_total_ms") or row.get("splitk_ms"), 0.0)
    dense_ms = _as_float(row.get("dense_ms"), 0.0)
    if splitk_ms <= 0.0:
        return None
    return dense_ms / splitk_ms


def _metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    splitk_ms = _as_float(row.get("splitk_total_ms") or row.get("splitk_ms"), 0.0)
    dense_ms = _as_float(row.get("dense_ms"), 0.0)
    speedup = _speedup(row)
    head_indices = _head_indices(row)
    return {
        "model_id": row.get("model_id"),
        "prompt_type": row.get("prompt_type"),
        "layer_id": row.get("layer_id"),
        "kv_len": _kv_len(row),
        "head_indices": head_indices,
        "head_group": _head_group(row),
        "selected_head_count": int(row.get("selected_head_count") or len(head_indices)),
        "block_size": row.get("block_size"),
        "sink_blocks": row.get("sink_blocks"),
        "recent_blocks": row.get("recent_blocks"),
        "middle_seed_blocks": row.get("middle_seed_blocks"),
        "chunk_anchor_blocks": row.get("chunk_anchor_blocks"),
        "block_order": row.get("block_order"),
        "num_chunks": row.get("num_chunks"),
        "seed_strategy": row.get("seed_strategy"),
        "filter_margin": row.get("filter_margin"),
        "error_budget": row.get("error_budget"),
        "projection_kind": row.get("projection_kind"),
        "projection_dim": row.get("projection_dim"),
        "projection_seed": row.get("projection_seed"),
        "projection_metadata_dtype": row.get("projection_metadata_dtype"),
        "qproj_mode": row.get("qproj_mode"),
        "splitk_workspace": row.get("splitk_workspace"),
        "projection_skip_fraction": _nested_float(row, "splitk_stats", "projection_skip_fraction", 0.0),
        "pv_executed_fraction": _nested_float(row, "splitk_stats", "pv_executed_fraction", 0.0),
        "max_abs_error": _nested_float(row, "splitk_error_vs_dense", "max_abs_error", 1.0e9),
        "mean_abs_error": _nested_float(row, "splitk_error_vs_dense", "mean_abs_error", 1.0e9),
        "splitk_ms": splitk_ms,
        "dense_ms": dense_ms,
        "speedup_vs_dense": speedup,
        "splitk_breakdown": row.get("splitk_breakdown"),
        "source": row.get("_source"),
    }


def _load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(collect_rows_from_payload(payload, source=str(path)))
    return rows


def _passes(row: Dict[str, Any], budget: Dict[str, float | str], *, min_skip_fraction: float) -> bool:
    speedup = row.get("speedup_vs_dense")
    return (
        speedup is not None
        and float(speedup) >= float(budget["min_speedup"])
        and float(row["max_abs_error"]) <= float(budget["max_error"])
        and float(row["mean_abs_error"]) <= float(budget["max_mean_error"])
        and float(row["projection_skip_fraction"]) >= min_skip_fraction
    )


def _failed_constraints(
    row: Dict[str, Any],
    budget: Dict[str, float | str],
    *,
    min_skip_fraction: float,
) -> List[str]:
    failed: List[str] = []
    speedup = row.get("speedup_vs_dense")
    if speedup is None or float(speedup) < float(budget["min_speedup"]):
        failed.append("speedup")
    if float(row["max_abs_error"]) > float(budget["max_error"]):
        failed.append("max_error")
    if float(row["mean_abs_error"]) > float(budget["max_mean_error"]):
        failed.append("mean_error")
    if float(row["projection_skip_fraction"]) < min_skip_fraction:
        failed.append("skip_fraction")
    return failed


def _policy_key(row: Dict[str, Any], budget_name: str) -> Tuple[Any, ...]:
    return (
        row.get("model_id"),
        row.get("prompt_type"),
        row.get("layer_id"),
        row.get("kv_len"),
        budget_name,
    )


def _runtime_config(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "head_indices": row["head_indices"],
        "head_group": row["head_group"],
        "selected_head_count": row["selected_head_count"],
        "block_size": row["block_size"],
        "sink_blocks": row["sink_blocks"],
        "recent_blocks": row["recent_blocks"],
        "middle_seed_blocks": row["middle_seed_blocks"],
        "chunk_anchor_blocks": row["chunk_anchor_blocks"],
        "block_order": row["block_order"],
        "num_chunks": row["num_chunks"],
        "seed_strategy": row["seed_strategy"],
        "filter_margin": row["filter_margin"],
        "error_budget": row["error_budget"],
        "projection_kind": row["projection_kind"],
        "projection_dim": row["projection_dim"],
        "projection_seed": row["projection_seed"],
        "projection_metadata_dtype": row["projection_metadata_dtype"],
        "qproj_mode": row["qproj_mode"],
        "splitk_workspace": row["splitk_workspace"],
    }


def _quality(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "projection_skip_fraction": row["projection_skip_fraction"],
        "pv_executed_fraction": row["pv_executed_fraction"],
        "max_abs_error": row["max_abs_error"],
        "mean_abs_error": row["mean_abs_error"],
        "splitk_ms": row["splitk_ms"],
        "dense_ms": row["dense_ms"],
        "speedup_vs_dense": row["speedup_vs_dense"],
        "splitk_breakdown": row["splitk_breakdown"],
    }


def _policy_entry(row: Dict[str, Any], budget: Dict[str, float | str]) -> Dict[str, Any]:
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
        "mode": "calibrated_splitk_inline_projection",
        "fallback": "dense",
        "runtime": _runtime_config(row),
        "quality": _quality(row),
        "source": row["source"],
    }


def _candidate_experiments(row: Dict[str, Any], failed: Sequence[str]) -> List[Dict[str, Any]]:
    experiments: List[Dict[str, Any]] = []
    margin = row.get("filter_margin")
    chunks = row.get("num_chunks")
    heads = row.get("head_indices") or []
    if ("max_error" in failed or "mean_error" in failed) and margin is not None:
        for delta in (8.0, 16.0):
            next_margin = max(0.0, float(margin) - delta)
            if next_margin != float(margin):
                experiments.append(
                    {
                        "kind": "tighten_margin",
                        "reason": "reduce projection false-skip/error risk",
                        "head_group": row["head_group"],
                        "filter_margin": next_margin,
                        "num_chunks": chunks,
                    }
                )
    if "speedup" in failed and chunks is not None:
        for next_chunks in sorted({max(2, int(chunks) // 2), int(chunks) * 2}):
            if next_chunks != int(chunks):
                experiments.append(
                    {
                        "kind": "chunk_sweep",
                        "reason": "test split-K overhead/parallelism balance",
                        "head_group": row["head_group"],
                        "filter_margin": margin,
                        "num_chunks": next_chunks,
                    }
                )
    if ("max_error" in failed or "mean_error" in failed) and len(heads) > 2:
        for head in heads:
            trimmed = [item for item in heads if item != head]
            experiments.append(
                {
                    "kind": "leave_one_out_group",
                    "reason": "find unsafe head inside fast grouped config",
                    "head_group": ",".join(str(item) for item in trimmed),
                    "removed_head": head,
                    "filter_margin": margin,
                    "num_chunks": chunks,
                }
            )
    return experiments[:8]


def _frontier_candidate(
    row: Dict[str, Any],
    budget: Dict[str, float | str],
    failed: Sequence[str],
) -> Dict[str, Any]:
    return {
        "model_id": row["model_id"],
        "prompt_type": row["prompt_type"],
        "layer_id": row["layer_id"],
        "kv_len_bucket": row["kv_len"],
        "budget": budget["name"],
        "failed_constraints": list(failed),
        "runtime": _runtime_config(row),
        "quality": _quality(row),
        "source": row["source"],
        "experiments": _candidate_experiments(row, failed),
    }


def build_policy(
    raw_rows: Sequence[Dict[str, Any]],
    budgets: Sequence[Dict[str, float | str]],
    *,
    min_skip_fraction: float = 0.25,
    frontier_error_multiplier: float = 1.5,
    frontier_mean_error_multiplier: float = 1.5,
    frontier_speedup_scale: float = 0.95,
    frontier_limit: int = 32,
) -> Dict[str, Any]:
    rows = [_metrics(row) for row in raw_rows]
    grouped_passes: Dict[Tuple[Any, ...], List[Tuple[Dict[str, Any], Dict[str, float | str]]]] = {}
    frontier: List[Dict[str, Any]] = []
    for budget in budgets:
        budget_name = str(budget["name"])
        for row in rows:
            if _passes(row, budget, min_skip_fraction=min_skip_fraction):
                grouped_passes.setdefault(_policy_key(row, budget_name), []).append((row, budget))
                continue
            failed = _failed_constraints(row, budget, min_skip_fraction=min_skip_fraction)
            speedup = row.get("speedup_vs_dense")
            near = (
                speedup is not None
                and float(speedup) >= float(budget["min_speedup"]) * frontier_speedup_scale
                and float(row["max_abs_error"]) <= float(budget["max_error"]) * frontier_error_multiplier
                and float(row["mean_abs_error"])
                <= float(budget["max_mean_error"]) * frontier_mean_error_multiplier
                and float(row["projection_skip_fraction"]) >= min_skip_fraction
            )
            if near:
                frontier.append(_frontier_candidate(row, budget, failed))

    entries: List[Dict[str, Any]] = []
    for candidates in grouped_passes.values():
        best_row, budget = max(
            candidates,
            key=lambda item: (
                float(item[0]["speedup_vs_dense"] or 0.0),
                -float(item[0]["max_abs_error"]),
                float(item[0]["projection_skip_fraction"]),
            ),
        )
        entries.append(_policy_entry(best_row, budget))

    entries.sort(
        key=lambda row: (
            str(row.get("model_id")),
            str(row.get("prompt_type")),
            int(row.get("layer_id") or -1),
            int(row.get("kv_len_bucket") or -1),
            str(row.get("safety_budget", {}).get("name")),
            -float(row.get("quality", {}).get("speedup_vs_dense") or 0.0),
        )
    )
    frontier.sort(
        key=lambda row: (
            -float(row.get("quality", {}).get("speedup_vs_dense") or 0.0),
            float(row.get("quality", {}).get("max_abs_error") or 1.0e9),
            -float(row.get("quality", {}).get("projection_skip_fraction") or 0.0),
        )
    )
    budget_summary = []
    for budget in budgets:
        name = str(budget["name"])
        matching = [row for row in entries if row["safety_budget"]["name"] == name]
        budget_summary.append(
            {
                "budget": name,
                "entries": len(matching),
                "prompt_types": sorted({str(row["prompt_type"]) for row in matching}),
                "kv_lens": sorted({int(row["kv_len_bucket"]) for row in matching if row["kv_len_bucket"] is not None}),
                "best_speedup": max(
                    (float(row["quality"]["speedup_vs_dense"]) for row in matching),
                    default=0.0,
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "summary": {
            "rows": len(rows),
            "entries": len(entries),
            "frontier_candidates": min(len(frontier), frontier_limit),
            "min_skip_fraction": min_skip_fraction,
            "frontier_error_multiplier": frontier_error_multiplier,
            "frontier_mean_error_multiplier": frontier_mean_error_multiplier,
            "frontier_speedup_scale": frontier_speedup_scale,
        },
        "budgets": list(budgets),
        "budget_summary": budget_summary,
        "entries": entries,
        "frontier": frontier[:frontier_limit],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_entries(entries: Sequence[Dict[str, Any]], *, limit: int) -> None:
    headers = ["budget", "prompt", "layer", "kv", "heads", "chunks", "margin", "speed", "err", "skip"]
    print(" ".join(f"{header:>12}" for header in headers))
    for row in entries[:limit]:
        runtime = row["runtime"]
        quality = row["quality"]
        print(
            " ".join(
                [
                    f"{_fmt(row['safety_budget']['name']):>12}",
                    f"{_fmt(row.get('prompt_type'))[:12]:>12}",
                    f"{_fmt(row.get('layer_id')):>12}",
                    f"{_fmt(row.get('kv_len_bucket')):>12}",
                    f"{_fmt(runtime.get('head_group'))[:12]:>12}",
                    f"{_fmt(runtime.get('num_chunks')):>12}",
                    f"{_fmt(runtime.get('filter_margin')):>12}",
                    f"{_fmt(quality.get('speedup_vs_dense')):>12}",
                    f"{_fmt(quality.get('max_abs_error')):>12}",
                    f"{_fmt(quality.get('projection_skip_fraction')):>12}",
                ]
            )
        )


def _print_frontier(rows: Sequence[Dict[str, Any]], *, limit: int) -> None:
    if not rows:
        return
    print("frontier")
    headers = ["budget", "fail", "prompt", "heads", "speed", "err", "mean", "skip"]
    print(" ".join(f"{header:>12}" for header in headers))
    for row in rows[:limit]:
        quality = row["quality"]
        runtime = row["runtime"]
        print(
            " ".join(
                [
                    f"{_fmt(row.get('budget')):>12}",
                    f"{','.join(row.get('failed_constraints') or [])[:12]:>12}",
                    f"{_fmt(row.get('prompt_type'))[:12]:>12}",
                    f"{_fmt(runtime.get('head_group'))[:12]:>12}",
                    f"{_fmt(quality.get('speedup_vs_dense')):>12}",
                    f"{_fmt(quality.get('max_abs_error')):>12}",
                    f"{_fmt(quality.get('mean_abs_error')):>12}",
                    f"{_fmt(quality.get('projection_skip_fraction')):>12}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--budgets", default=DEFAULT_ROBUSTNESS_BUDGETS)
    parser.add_argument("--min-skip-fraction", type=float, default=0.25)
    parser.add_argument("--frontier-error-multiplier", type=float, default=1.5)
    parser.add_argument("--frontier-mean-error-multiplier", type=float, default=1.5)
    parser.add_argument("--frontier-speedup-scale", type=float, default=0.95)
    parser.add_argument("--frontier-limit", type=int, default=32)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    payload = build_policy(
        _load_rows(args.json_paths),
        parse_budgets(args.budgets),
        min_skip_fraction=args.min_skip_fraction,
        frontier_error_multiplier=args.frontier_error_multiplier,
        frontier_mean_error_multiplier=args.frontier_mean_error_multiplier,
        frontier_speedup_scale=args.frontier_speedup_scale,
        frontier_limit=args.frontier_limit,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    _print_entries(payload["entries"], limit=args.limit)
    _print_frontier(payload["frontier"], limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
