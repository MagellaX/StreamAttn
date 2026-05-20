"""Build verification-gated Gate-0 split-K policy candidates.

This tool is for the risky path where a large selected-head group is fast but
not globally safe.  It uses per-head error telemetry to split that group into:

* heads allowed to stay on calibrated split-K projection,
* heads that must fall back to dense/Gate-1,
* heads that require online verification before promotion.

The output is not a production policy by itself.  It is the bridge between
"fast but unsafe" evidence and the next runtime experiment.
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


SCHEMA = "streamattn.gate0.verification_gated_splitk_policy.v1"


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _kv_len(row: Dict[str, Any]) -> int | None:
    value = row.get("kv_len")
    if value is None:
        value = (row.get("capture_shape") or {}).get("kv_len")
    if value is None:
        value = (row.get("shape") or {}).get("kv_len")
    return int(value) if value is not None else None


def _head_indices(row: Dict[str, Any]) -> List[int]:
    indices = row.get("head_indices") or row.get("selected_head_indices") or []
    return [int(item) for item in indices]


def _head_group(heads: Iterable[int]) -> str:
    return ",".join(str(head) for head in sorted(set(int(item) for item in heads)))


def _speedup(row: Dict[str, Any]) -> float | None:
    value = row.get("splitk_vs_dense_speedup")
    if value is not None:
        return float(value)
    splitk_ms = _as_float(row.get("splitk_total_ms") or row.get("splitk_ms"), 0.0)
    dense_ms = _as_float(row.get("dense_ms"), 0.0)
    if splitk_ms <= 0.0:
        return None
    return dense_ms / splitk_ms


def _splitk_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("splitk_stats") or {}


def _error(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("splitk_error_vs_dense") or {}


def _per_head_error(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = row.get("splitk_error_vs_dense_per_head") or {}
    values = payload.get("per_head") or []
    return [dict(item) for item in values if isinstance(item, dict)]


def _per_head_stats(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = row.get("splitk_per_head_stats") or {}
    values = payload.get("per_head") or []
    return [dict(item) for item in values if isinstance(item, dict)]


def _runtime(row: Dict[str, Any], heads: Sequence[int] | None = None) -> Dict[str, Any]:
    head_indices = list(_head_indices(row) if heads is None else heads)
    return {
        "head_indices": head_indices,
        "head_group": _head_group(head_indices),
        "selected_head_count": len(head_indices),
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
    }


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


def _head_records(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected = _head_indices(row)
    errors = {int(item.get("head")): item for item in _per_head_error(row)}
    stats = {int(item.get("head")): item for item in _per_head_stats(row)}
    records: List[Dict[str, Any]] = []
    for local_idx, actual_head in enumerate(selected):
        error = errors.get(local_idx, {})
        stat = stats.get(local_idx, {})
        records.append(
            {
                "local_head": local_idx,
                "head": actual_head,
                "max_abs_error": _as_float(error.get("max_abs_error"), 1.0e9),
                "mean_abs_error": _as_float(error.get("mean_abs_error"), 1.0e9),
                "projection_skip_fraction": _as_float(stat.get("projection_skip_fraction"), 0.0),
                "pv_executed_fraction": _as_float(stat.get("pv_executed_fraction"), 0.0),
            }
        )
    return records


def _passes_head(
    head: Dict[str, Any],
    budget: Dict[str, float | str],
    *,
    min_head_skip_fraction: float,
    enforce_mean: bool,
) -> bool:
    if float(head["max_abs_error"]) > float(budget["max_error"]):
        return False
    if enforce_mean and float(head["mean_abs_error"]) > float(budget["max_mean_error"]):
        return False
    return float(head["projection_skip_fraction"]) >= min_head_skip_fraction


def _row_is_fast_enough(row: Dict[str, Any], budget: Dict[str, float | str], *, speedup_scale: float) -> bool:
    speedup = _speedup(row)
    return speedup is not None and speedup >= float(budget["min_speedup"]) * speedup_scale


def _gated_candidate(
    row: Dict[str, Any],
    budget: Dict[str, float | str],
    *,
    min_head_skip_fraction: float,
    speedup_scale: float,
    enforce_mean: bool,
) -> Dict[str, Any] | None:
    heads = _head_records(row)
    if not heads or not _row_is_fast_enough(row, budget, speedup_scale=speedup_scale):
        return None
    safe = [
        head for head in heads
        if _passes_head(
            head,
            budget,
            min_head_skip_fraction=min_head_skip_fraction,
            enforce_mean=enforce_mean,
        )
    ]
    unsafe = [head for head in heads if head not in safe]
    if not safe or not unsafe:
        return None
    safe_heads = [int(head["head"]) for head in safe]
    unsafe_heads = [int(head["head"]) for head in unsafe]
    splitk_ms = _as_float(row.get("splitk_total_ms") or row.get("splitk_ms"), 0.0)
    dense_ms = _as_float(row.get("dense_ms"), 0.0)
    return {
        "model_id": row.get("model_id"),
        "prompt_type": row.get("prompt_type"),
        "layer_id": row.get("layer_id"),
        "kv_len_bucket": _kv_len(row),
        "budget": {
            "name": budget["name"],
            "max_abs_error": budget["max_error"],
            "max_mean_error": budget["max_mean_error"],
            "min_speedup": budget["min_speedup"],
            "min_head_skip_fraction": min_head_skip_fraction,
            "enforce_per_head_mean": enforce_mean,
        },
        "mode": "verification_gated_splitk_inline_projection",
        "runtime": {
            "aggressive_union": _runtime(row),
            "sparse_candidate": _runtime(row, safe_heads),
            "fallback_candidate": _runtime(row, unsafe_heads),
        },
        "verification": {
            "required": True,
            "strategy": "sample_skipped_blocks_then_disable_head",
            "verify_heads": unsafe_heads,
            "promote_heads": safe_heads,
            "fallback": "dense_or_gate1_for_disabled_heads",
        },
        "quality": {
            "union_speedup_vs_dense": _speedup(row),
            "union_splitk_ms": splitk_ms,
            "union_dense_ms": dense_ms,
            "union_max_abs_error": _as_float(_error(row).get("max_abs_error"), 1.0e9),
            "union_mean_abs_error": _as_float(_error(row).get("mean_abs_error"), 1.0e9),
            "projection_skip_fraction": _as_float(_splitk_stats(row).get("projection_skip_fraction"), 0.0),
            "safe_head_count": len(safe_heads),
            "fallback_head_count": len(unsafe_heads),
            "safe_heads": safe,
            "fallback_heads": unsafe,
        },
        "economics": {
            "needs_sparse_safe_group_measurement": True,
            "needs_dense_fallback_group_measurement": True,
            "serial_upper_bound_formula": "sparse_safe_ms + dense_fallback_ms",
            "parallel_lower_bound_formula": "max(sparse_safe_ms, dense_fallback_ms)",
        },
        "source": row.get("_source"),
    }


def build_gated_policy(
    raw_rows: Sequence[Dict[str, Any]],
    budgets: Sequence[Dict[str, float | str]],
    *,
    min_head_skip_fraction: float = 0.25,
    speedup_scale: float = 1.0,
    enforce_mean: bool = False,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for budget in budgets:
        for row in raw_rows:
            candidate = _gated_candidate(
                row,
                budget,
                min_head_skip_fraction=min_head_skip_fraction,
                speedup_scale=speedup_scale,
                enforce_mean=enforce_mean,
            )
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            float(row["quality"]["union_speedup_vs_dense"] or 0.0),
            row["quality"]["safe_head_count"],
            -float(row["quality"]["union_max_abs_error"]),
        ),
        reverse=True,
    )
    return {
        "schema": SCHEMA,
        "summary": {
            "rows": len(raw_rows),
            "candidates": len(candidates),
            "min_head_skip_fraction": min_head_skip_fraction,
            "speedup_scale": speedup_scale,
            "enforce_per_head_mean": enforce_mean,
        },
        "budgets": list(budgets),
        "candidates": candidates,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_candidates(rows: Sequence[Dict[str, Any]], *, limit: int) -> None:
    headers = ["budget", "prompt", "layer", "kv", "sparse", "fallback", "speed", "err", "skip"]
    print(" ".join(f"{header:>12}" for header in headers))
    for row in rows[:limit]:
        quality = row["quality"]
        runtime = row["runtime"]
        print(
            " ".join(
                [
                    f"{_fmt(row['budget']['name']):>12}",
                    f"{_fmt(row.get('prompt_type'))[:12]:>12}",
                    f"{_fmt(row.get('layer_id')):>12}",
                    f"{_fmt(row.get('kv_len_bucket')):>12}",
                    f"{_fmt(runtime['sparse_candidate']['head_group'])[:12]:>12}",
                    f"{_fmt(runtime['fallback_candidate']['head_group'])[:12]:>12}",
                    f"{_fmt(quality.get('union_speedup_vs_dense')):>12}",
                    f"{_fmt(quality.get('union_max_abs_error')):>12}",
                    f"{_fmt(quality.get('projection_skip_fraction')):>12}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--budgets", default=DEFAULT_ROBUSTNESS_BUDGETS)
    parser.add_argument("--min-head-skip-fraction", type=float, default=0.25)
    parser.add_argument("--speedup-scale", type=float, default=1.0)
    parser.add_argument("--enforce-per-head-mean", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    payload = build_gated_policy(
        _load_rows(args.json_paths),
        parse_budgets(args.budgets),
        min_head_skip_fraction=args.min_head_skip_fraction,
        speedup_scale=args.speedup_scale,
        enforce_mean=args.enforce_per_head_mean,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    _print_candidates(payload["candidates"], limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
