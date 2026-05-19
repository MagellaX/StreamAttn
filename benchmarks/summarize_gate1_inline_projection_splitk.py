"""Summarize split-K inline projection sweeps into selective-head plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_SAFETY_BUDGETS = "strict:1e-3,moderate:1e-2,research:5e-2"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else payload.get("results", [payload])
        for row in candidates:
            if not isinstance(row, dict):
                continue
            copied = dict(row)
            copied["_source"] = str(path)
            rows.append(copied)
    return rows


def _as_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def _nested_float(row: Dict[str, Any], parent: str, key: str, default: float = 0.0) -> float:
    value = (row.get(parent) or {}).get(key)
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


def _case_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("model_id"),
        row.get("prompt_type"),
        row.get("layer_id"),
        row.get("head_index"),
        _kv_len(row),
    )


def _parse_safety_budgets(raw: str, *, mean_error_ratio: float) -> List[Dict[str, float | str]]:
    budgets: List[Dict[str, float | str]] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, value = item.split(":", 1)
            max_error = float(value.strip())
        else:
            max_error = float(item)
            name = f"abs_{max_error:g}"
        budgets.append(
            {
                "name": name.strip(),
                "max_error": max_error,
                "max_mean_error": max_error * mean_error_ratio,
            }
        )
    if not budgets:
        raise ValueError("at least one safety budget is required")
    return budgets


def _valid_profile(row: Dict[str, Any]) -> bool:
    return row.get("returncode", 0) == 0 and row.get("splitk_total_ms") is not None


def _projection_skip(row: Dict[str, Any]) -> float:
    return _nested_float(row, "splitk_stats", "projection_skip_fraction", 0.0)


def _pv_fraction(row: Dict[str, Any]) -> float:
    return _nested_float(row, "splitk_stats", "pv_executed_fraction", 0.0)


def _max_error(row: Dict[str, Any]) -> float:
    return _nested_float(row, "splitk_error_vs_dense", "max_abs_error", 1.0e9)


def _mean_error(row: Dict[str, Any]) -> float:
    return _nested_float(row, "splitk_error_vs_dense", "mean_abs_error", 1.0e9)


def _is_safe(
    row: Dict[str, Any],
    *,
    max_error: float,
    max_mean_error: float,
    min_skip_fraction: float,
) -> bool:
    return (
        _valid_profile(row)
        and _max_error(row) <= max_error
        and _mean_error(row) <= max_mean_error
        and _projection_skip(row) >= min_skip_fraction
    )


def _row_score(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        _as_float(row, "splitk_total_ms", 1.0e9),
        -_projection_skip(row),
        _max_error(row),
    )


def _best_ms(rows: List[Dict[str, Any]], key: str) -> float | None:
    values = [_as_float(row, key, 1.0e9) for row in rows if row.get(key) is not None]
    return min(values) if values else None


def _calibrate_heads(
    rows: List[Dict[str, Any]],
    *,
    max_error: float,
    max_mean_error: float,
    min_skip_fraction: float,
    safety_budget: str = "custom",
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("head_index") is None or int(row.get("head_index")) < 0:
            continue
        grouped.setdefault(_case_key(row), []).append(row)

    calibrated: List[Dict[str, Any]] = []
    for key, case_rows in grouped.items():
        safe_rows = [
            row
            for row in case_rows
            if _is_safe(
                row,
                max_error=max_error,
                max_mean_error=max_mean_error,
                min_skip_fraction=min_skip_fraction,
            )
        ]
        best_safe = min(safe_rows, key=_row_score) if safe_rows else None
        best_seen = min((row for row in case_rows if _valid_profile(row)), key=_row_score, default=None)
        source = best_safe or best_seen or case_rows[0]
        dense_ms = _best_ms(case_rows, "dense_ms")
        splitk_total_ms = _as_float(source, "splitk_total_ms", 1.0e9) if best_safe else None
        calibrated.append(
            {
                "model_id": key[0],
                "prompt_type": key[1],
                "layer_id": key[2],
                "head": key[3],
                "kv_len": key[4],
                "safety_budget": safety_budget,
                "max_error_budget": max_error,
                "max_mean_error_budget": max_mean_error,
                "safe": best_safe is not None,
                "projection_dim": source.get("projection_dim"),
                "projection_seed": source.get("projection_seed"),
                "filter_margin": source.get("filter_margin"),
                "num_chunks": source.get("num_chunks"),
                "chunk_anchor_blocks": source.get("chunk_anchor_blocks"),
                "middle_seed_blocks": source.get("middle_seed_blocks"),
                "block_size": source.get("block_size"),
                "block_order": source.get("block_order"),
                "projection_skip_fraction": _projection_skip(source),
                "pv_executed_fraction": _pv_fraction(source),
                "max_abs_error": _max_error(source),
                "mean_abs_error": _mean_error(source),
                "splitk_total_ms": splitk_total_ms,
                "splitk_ms": source.get("splitk_ms"),
                "dense_ms": dense_ms,
                "fallback_ms": dense_ms,
                "selected_ms": splitk_total_ms if best_safe else dense_ms,
                "splitk_speedup_vs_dense": (
                    dense_ms / splitk_total_ms if best_safe and dense_ms is not None and splitk_total_ms else None
                ),
            }
        )
    return sorted(
        calibrated,
        key=lambda row: (
            str(row.get("model_id")),
            str(row.get("prompt_type")),
            row.get("layer_id") if row.get("layer_id") is not None else -1,
            row.get("kv_len") if row.get("kv_len") is not None else -1,
            row.get("head") if row.get("head") is not None else -1,
            str(row.get("safety_budget")),
        ),
    )


def _layer_oracle(heads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in heads:
        grouped.setdefault(
            (
                row.get("model_id"),
                row.get("prompt_type"),
                row.get("layer_id"),
                row.get("kv_len"),
                row.get("safety_budget"),
            ),
            [],
        ).append(row)

    summaries: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        dense_values = [_as_float(row, "dense_ms", 0.0) for row in rows if row.get("dense_ms") is not None]
        splitk_safe_values = [_as_float(row, "splitk_total_ms", 0.0) for row in rows if row.get("safe")]
        dense_unsafe_values = [_as_float(row, "dense_ms", 0.0) for row in rows if not row.get("safe")]
        dense_sum = sum(dense_values)
        splitk_safe_sum = sum(splitk_safe_values)
        dense_unsafe_sum = sum(dense_unsafe_values)
        dense_max = max(dense_values) if dense_values else 0.0
        splitk_safe_max = max(splitk_safe_values) if splitk_safe_values else 0.0
        dense_unsafe_max = max(dense_unsafe_values) if dense_unsafe_values else 0.0
        selected_sum = splitk_safe_sum + dense_unsafe_sum
        group_max_lower_bound = max(splitk_safe_sum, dense_unsafe_sum)
        head_parallel_lower_bound = max(splitk_safe_max, dense_unsafe_max)
        safe_heads = [row["head"] for row in rows if row.get("safe")]
        unsafe_heads = [row["head"] for row in rows if not row.get("safe")]
        summaries.append(
            {
                "model_id": key[0],
                "prompt_type": key[1],
                "layer_id": key[2],
                "kv_len": key[3],
                "safety_budget": key[4],
                "heads": len(rows),
                "safe_sparse_heads": safe_heads,
                "unsafe_heads": unsafe_heads,
                "safe_head_count": len(safe_heads),
                "unsafe_head_count": len(unsafe_heads),
                "safe_head_fraction": len(safe_heads) / len(rows) if rows else 0.0,
                "dense_all_sum_ms": dense_sum,
                "dense_all_max_ms": dense_max,
                "splitk_safe_sum_ms": splitk_safe_sum,
                "splitk_safe_max_ms": splitk_safe_max,
                "dense_unsafe_sum_ms": dense_unsafe_sum,
                "dense_unsafe_max_ms": dense_unsafe_max,
                "selective_serial_sum_ms": selected_sum,
                "selective_group_max_lower_bound_ms": group_max_lower_bound,
                "selective_head_parallel_lower_bound_ms": head_parallel_lower_bound,
                "selective_serial_speedup_vs_dense_sum": dense_sum / selected_sum if selected_sum > 0 else None,
                "selective_group_max_speedup_vs_dense_sum": (
                    dense_sum / group_max_lower_bound if group_max_lower_bound > 0 else None
                ),
                "selective_head_parallel_speedup_vs_dense_max": (
                    dense_max / head_parallel_lower_bound if head_parallel_lower_bound > 0 else None
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            str(row.get("model_id")),
            str(row.get("prompt_type")),
            row.get("layer_id") if row.get("layer_id") is not None else -1,
            row.get("kv_len") if row.get("kv_len") is not None else -1,
            str(row.get("safety_budget")),
        ),
    )


def _print_heads(rows: List[Dict[str, Any]], *, limit: int) -> None:
    headers = ["budget", "head", "safe", "rank", "seed", "margin", "skip", "err", "splitk", "dense"]
    print(" ".join(f"{header:>9}" for header in headers))
    for row in rows[:limit]:
        print(
            " ".join(
                [
                    f"{_fmt(row.get('safety_budget')):>9}",
                    f"{_fmt(row.get('head')):>9}",
                    f"{_fmt(row.get('safe')):>9}",
                    f"{_fmt(row.get('projection_dim')):>9}",
                    f"{_fmt(row.get('projection_seed')):>9}",
                    f"{_fmt(row.get('filter_margin')):>9}",
                    f"{_fmt(row.get('projection_skip_fraction')):>9}",
                    f"{_fmt(row.get('max_abs_error')):>9}",
                    f"{_fmt(row.get('splitk_total_ms')):>9}",
                    f"{_fmt(row.get('dense_ms')):>9}",
                ]
            )
        )


def _print_layers(rows: List[Dict[str, Any]], *, limit: int) -> None:
    headers = ["budget", "layer", "kv", "safe", "heads", "serial", "sum_gmax", "headmax", "dense_sum"]
    print(" ".join(f"{header:>10}" for header in headers))
    for row in rows[:limit]:
        print(
            " ".join(
                [
                    f"{_fmt(row.get('safety_budget')):>10}",
                    f"{_fmt(row.get('layer_id')):>10}",
                    f"{_fmt(row.get('kv_len')):>10}",
                    f"{_fmt(row.get('safe_head_count')):>10}",
                    f"{_fmt(row.get('heads')):>10}",
                    f"{_fmt(row.get('selective_serial_speedup_vs_dense_sum')):>10}",
                    f"{_fmt(row.get('selective_group_max_speedup_vs_dense_sum')):>10}",
                    f"{_fmt(row.get('selective_head_parallel_speedup_vs_dense_max')):>10}",
                    f"{_fmt(row.get('dense_all_sum_ms')):>10}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--safety-budgets", default=DEFAULT_SAFETY_BUDGETS)
    parser.add_argument("--mean-error-ratio", type=float, default=0.1)
    parser.add_argument("--max-error", type=float, default=None)
    parser.add_argument("--max-mean-error", type=float, default=None)
    parser.add_argument("--min-skip-fraction", type=float, default=0.25)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    rows = _load_rows(args.json_paths)
    if args.max_error is not None:
        mean_error = args.max_mean_error if args.max_mean_error is not None else args.max_error * args.mean_error_ratio
        budgets = [{"name": "custom", "max_error": args.max_error, "max_mean_error": mean_error}]
    else:
        budgets = _parse_safety_budgets(args.safety_budgets, mean_error_ratio=args.mean_error_ratio)

    heads: List[Dict[str, Any]] = []
    for budget in budgets:
        heads.extend(
            _calibrate_heads(
                rows,
                max_error=float(budget["max_error"]),
                max_mean_error=float(budget["max_mean_error"]),
                min_skip_fraction=args.min_skip_fraction,
                safety_budget=str(budget["name"]),
            )
        )
    layers = _layer_oracle(heads)
    summary = {
        "rows": len(rows),
        "calibrated_heads": len(heads),
        "min_skip_fraction": args.min_skip_fraction,
        "budgets": budgets,
    }
    for budget in budgets:
        name = str(budget["name"])
        summary[f"safe_{name}_heads"] = len(
            [row for row in heads if row.get("safety_budget") == name and row.get("safe")]
        )
    payload = {
        "summary": summary,
        "heads": heads,
        "layer_selective_oracle": layers,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    _print_heads(heads, limit=args.limit)
    _print_layers(layers, limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
