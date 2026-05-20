"""Summarize robustness runs for workspace split-K inline projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_ROBUSTNESS_BUDGETS = (
    "strict:1e-3:1e-4:1.00,"
    "moderate:1e-2:1e-3:1.15,"
    "research:5e-2:5e-3:1.25"
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def parse_budgets(raw: str) -> List[Dict[str, float | str]]:
    budgets: List[Dict[str, float | str]] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 4:
            raise ValueError(
                "robustness budgets must use name:max_abs_error:max_mean_error:min_speedup"
            )
        name, max_error, max_mean_error, min_speedup = parts
        budgets.append(
            {
                "name": name,
                "max_error": float(max_error),
                "max_mean_error": float(max_mean_error),
                "min_speedup": float(min_speedup),
            }
        )
    if not budgets:
        raise ValueError("at least one robustness budget is required")
    return budgets


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


def _head_group(row: Dict[str, Any]) -> str:
    value = row.get("head_group")
    if value is not None:
        return str(value)
    indices = row.get("head_indices") or row.get("selected_head_indices")
    if indices:
        return ",".join(str(item) for item in indices)
    index = row.get("head_index")
    return str(index) if index is not None else ""


def _row_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    splitk_ms = _as_float(row.get("splitk_total_ms") or row.get("splitk_ms"), 0.0)
    dense_ms = _as_float(row.get("dense_ms"), 0.0)
    speedup = row.get("splitk_vs_dense_speedup")
    if speedup is None:
        speedup = dense_ms / splitk_ms if splitk_ms > 0 else None
    head_indices = row.get("head_indices") or row.get("selected_head_indices") or []
    selected_count = int(row.get("selected_head_count") or len(head_indices) or 0)
    if selected_count <= 0 and row.get("head_index") is not None:
        selected_count = 1
    return {
        "model_id": row.get("model_id"),
        "prompt_type": row.get("prompt_type"),
        "layer_id": row.get("layer_id"),
        "kv_len": _kv_len(row),
        "head_group": _head_group(row),
        "selected_head_count": selected_count,
        "filter_margin": row.get("filter_margin"),
        "num_chunks": row.get("num_chunks"),
        "projection_dim": row.get("projection_dim"),
        "projection_seed": row.get("projection_seed"),
        "splitk_workspace": row.get("splitk_workspace"),
        "projection_skip_fraction": _nested_float(row, "splitk_stats", "projection_skip_fraction", 0.0),
        "pv_executed_fraction": _nested_float(row, "splitk_stats", "pv_executed_fraction", 0.0),
        "max_abs_error": _nested_float(row, "splitk_error_vs_dense", "max_abs_error", 1.0e9),
        "mean_abs_error": _nested_float(row, "splitk_error_vs_dense", "mean_abs_error", 1.0e9),
        "splitk_ms": splitk_ms,
        "dense_ms": dense_ms,
        "speedup_vs_dense": speedup,
        "splitk_breakdown": row.get("splitk_breakdown"),
        "_source": row.get("_source"),
    }


def collect_rows_from_payload(payload: Any, *, source: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        candidates = []
        for run in payload["runs"]:
            for row in run.get("results", []):
                copied = dict(row)
                copied.setdefault("prompt_type", (run.get("capture") or {}).get("prompt_type"))
                candidates.append(copied)
    elif isinstance(payload, dict):
        candidates = payload.get("results", [payload])
    else:
        candidates = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        copied = dict(row)
        if source:
            copied["_source"] = source
        rows.append(copied)
    return rows


def load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
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


def _config_key(row: Dict[str, Any], budget_name: str) -> Tuple[Any, ...]:
    return (
        budget_name,
        row.get("model_id"),
        row.get("layer_id"),
        row.get("head_group"),
        row.get("selected_head_count"),
        row.get("filter_margin"),
        row.get("num_chunks"),
        row.get("projection_dim"),
        row.get("projection_seed"),
        row.get("splitk_workspace"),
    )


def summarize_rows(
    raw_rows: Sequence[Dict[str, Any]],
    budgets: Sequence[Dict[str, float | str]],
    *,
    min_skip_fraction: float = 0.25,
) -> Dict[str, Any]:
    rows = [_row_metrics(row) for row in raw_rows]
    budget_summaries: List[Dict[str, Any]] = []
    config_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for budget in budgets:
        name = str(budget["name"])
        passed = [row for row in rows if _passes(row, budget, min_skip_fraction=min_skip_fraction)]
        budget_summaries.append(
            {
                "budget": name,
                "rows": len(rows),
                "passed_rows": len(passed),
                "pass_fraction": len(passed) / len(rows) if rows else 0.0,
                "prompt_types_passed": sorted({str(row.get("prompt_type")) for row in passed}),
                "kv_lens_passed": sorted({int(row["kv_len"]) for row in passed if row.get("kv_len") is not None}),
                "layers_passed": sorted({int(row["layer_id"]) for row in passed if row.get("layer_id") is not None}),
                "best_speedup": max((float(row["speedup_vs_dense"]) for row in passed), default=0.0),
            }
        )
        for row in rows:
            config_groups.setdefault(_config_key(row, name), []).append(
                {**row, "budget": name, "passed": _passes(row, budget, min_skip_fraction=min_skip_fraction)}
            )

    robust_configs: List[Dict[str, Any]] = []
    for key, group_rows in config_groups.items():
        passed_rows = [row for row in group_rows if row["passed"]]
        speedups = [float(row["speedup_vs_dense"]) for row in passed_rows if row.get("speedup_vs_dense") is not None]
        robust_configs.append(
            {
                "budget": key[0],
                "model_id": key[1],
                "layer_id": key[2],
                "head_group": key[3],
                "selected_head_count": key[4],
                "filter_margin": key[5],
                "num_chunks": key[6],
                "projection_dim": key[7],
                "projection_seed": key[8],
                "splitk_workspace": key[9],
                "rows": len(group_rows),
                "passed_rows": len(passed_rows),
                "pass_fraction": len(passed_rows) / len(group_rows) if group_rows else 0.0,
                "prompt_types": sorted({str(row.get("prompt_type")) for row in group_rows}),
                "prompt_types_passed": sorted({str(row.get("prompt_type")) for row in passed_rows}),
                "kv_lens": sorted({int(row["kv_len"]) for row in group_rows if row.get("kv_len") is not None}),
                "kv_lens_passed": sorted({int(row["kv_len"]) for row in passed_rows if row.get("kv_len") is not None}),
                "min_pass_speedup": min(speedups) if speedups else 0.0,
                "max_pass_speedup": max(speedups) if speedups else 0.0,
                "max_error_seen": max((float(row["max_abs_error"]) for row in group_rows), default=0.0),
                "max_error_passed": max((float(row["max_abs_error"]) for row in passed_rows), default=0.0),
                "mean_skip_passed": (
                    sum(float(row["projection_skip_fraction"]) for row in passed_rows) / len(passed_rows)
                    if passed_rows
                    else 0.0
                ),
            }
        )
    robust_configs.sort(
        key=lambda row: (
            row["pass_fraction"],
            row["passed_rows"],
            row["min_pass_speedup"],
            -float(row["max_error_passed"]),
        ),
        reverse=True,
    )
    top_rows = sorted(
        rows,
        key=lambda row: (
            float(row["speedup_vs_dense"] or 0.0),
            -float(row["max_abs_error"]),
            float(row["projection_skip_fraction"]),
        ),
        reverse=True,
    )
    return {
        "summary": {
            "rows": len(rows),
            "min_skip_fraction": min_skip_fraction,
            "budgets": list(budgets),
        },
        "budget_summaries": budget_summaries,
        "top_rows": top_rows[:24],
        "robust_configs": robust_configs[:48],
    }


def _print_budget_summaries(rows: Sequence[Dict[str, Any]]) -> None:
    headers = ["budget", "pass", "rows", "frac", "best", "prompts", "kv"]
    print(" ".join(f"{header:>12}" for header in headers))
    for row in rows:
        print(
            " ".join(
                [
                    f"{_fmt(row.get('budget')):>12}",
                    f"{_fmt(row.get('passed_rows')):>12}",
                    f"{_fmt(row.get('rows')):>12}",
                    f"{_fmt(row.get('pass_fraction')):>12}",
                    f"{_fmt(row.get('best_speedup')):>12}",
                    f"{','.join(map(str, row.get('prompt_types_passed') or [])):>12}",
                    f"{','.join(map(str, row.get('kv_lens_passed') or [])):>12}",
                ]
            )
        )


def _print_configs(rows: Sequence[Dict[str, Any]], *, limit: int) -> None:
    headers = ["budget", "layer", "heads", "chunks", "margin", "pass", "rows", "min_spd", "err", "skip"]
    print(" ".join(f"{header:>10}" for header in headers))
    for row in rows[:limit]:
        print(
            " ".join(
                [
                    f"{_fmt(row.get('budget')):>10}",
                    f"{_fmt(row.get('layer_id')):>10}",
                    f"{_fmt(row.get('selected_head_count')):>10}",
                    f"{_fmt(row.get('num_chunks')):>10}",
                    f"{_fmt(row.get('filter_margin')):>10}",
                    f"{_fmt(row.get('passed_rows')):>10}",
                    f"{_fmt(row.get('rows')):>10}",
                    f"{_fmt(row.get('min_pass_speedup')):>10}",
                    f"{_fmt(row.get('max_error_passed')):>10}",
                    f"{_fmt(row.get('mean_skip_passed')):>10}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--budgets", default=DEFAULT_ROBUSTNESS_BUDGETS)
    parser.add_argument("--min-skip-fraction", type=float, default=0.25)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    payload = summarize_rows(
        load_rows(args.json_paths),
        parse_budgets(args.budgets),
        min_skip_fraction=args.min_skip_fraction,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    _print_budget_summaries(payload["budget_summaries"])
    _print_configs(payload["robust_configs"], limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
