"""Summarize inline projection Gate-1 sweeps into per-head calibration rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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
        if isinstance(payload, list):
            candidates = payload
        elif "results" in payload:
            candidates = payload.get("results") or []
        elif "profile" in payload:
            profile = dict(payload["profile"])
            capture = payload.get("capture") or {}
            profile.update(
                {
                    "model_id": capture.get("model_id"),
                    "prompt_type": capture.get("prompt_type"),
                    "layer_id": capture.get("layer_id"),
                    "head_index": capture.get("head_index"),
                }
            )
            candidates = [profile]
        else:
            candidates = [payload]
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


def _case_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("model_id"),
        row.get("prompt_type"),
        row.get("layer_id"),
        row.get("head_index"),
    )


def _valid_profile(row: Dict[str, Any]) -> bool:
    return row.get("returncode", 0) == 0 and row.get("inline_total_ms") is not None


def _is_safe(
    row: Dict[str, Any],
    *,
    max_error: float,
    max_mean_error: float,
    min_skip_fraction: float,
) -> bool:
    return (
        _valid_profile(row)
        and _as_float(row, "max_abs_error", 1.0e9) <= max_error
        and _as_float(row, "mean_abs_error", 1.0e9) <= max_mean_error
        and _as_float(row, "projection_skip_fraction", 0.0) >= min_skip_fraction
    )


def _row_score(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        _as_float(row, "inline_total_ms", 1.0e9),
        -_as_float(row, "projection_skip_fraction", 0.0),
        _as_float(row, "max_abs_error", 1.0e9),
    )


def _best_dense_ms(rows: List[Dict[str, Any]]) -> float | None:
    values = [_as_float(row, "dense_ms", 1.0e9) for row in rows if row.get("dense_ms") is not None]
    return min(values) if values else None


def _best_gate1_ms(rows: List[Dict[str, Any]]) -> float | None:
    values = [_as_float(row, "gate1_mass_ms", 1.0e9) for row in rows if row.get("gate1_mass_ms") is not None]
    return min(values) if values else None


def _calibrate_heads(
    rows: List[Dict[str, Any]],
    *,
    max_error: float,
    max_mean_error: float,
    min_skip_fraction: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("head_index") is None:
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
        dense_ms = _best_dense_ms(case_rows)
        gate1_ms = _best_gate1_ms(case_rows)
        inline_total_ms = _as_float(source, "inline_total_ms", 1.0e9) if best_safe else None
        calibrated.append(
            {
                "model_id": key[0],
                "prompt_type": key[1],
                "layer_id": key[2],
                "head": key[3],
                "safe": best_safe is not None,
                "best_margin": source.get("filter_margin"),
                "best_seed": source.get("middle_seed_blocks"),
                "best_block_size": source.get("block_size"),
                "block_order": source.get("block_order"),
                "qproj_mode": source.get("qproj_mode"),
                "max_abs_error": source.get("max_abs_error"),
                "mean_abs_error": source.get("mean_abs_error"),
                "skip_fraction": source.get("projection_skip_fraction"),
                "pv_executed_fraction": source.get("pv_executed_fraction"),
                "inline_total_ms": inline_total_ms,
                "inline_kernel_ms": source.get("inline_projection_ms"),
                "q_projection_ms": source.get("q_projection_ms"),
                "dense_ms": dense_ms,
                "gate1_mass_ms": gate1_ms,
                "fallback_ms": dense_ms,
                "selected_ms": inline_total_ms if best_safe else dense_ms,
            }
        )
    return sorted(
        calibrated,
        key=lambda row: (
            str(row.get("model_id")),
            str(row.get("prompt_type")),
            row.get("layer_id") if row.get("layer_id") is not None else -1,
            row.get("head") if row.get("head") is not None else -1,
        ),
    )


def _layer_oracle(heads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in heads:
        grouped.setdefault((row.get("model_id"), row.get("prompt_type"), row.get("layer_id")), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        dense_sum = sum(_as_float(row, "dense_ms", 0.0) for row in rows if row.get("dense_ms") is not None)
        gate1_sum = sum(_as_float(row, "gate1_mass_ms", 0.0) for row in rows if row.get("gate1_mass_ms") is not None)
        selected_sum = sum(_as_float(row, "selected_ms", 0.0) for row in rows if row.get("selected_ms") is not None)
        safe_heads = [row["head"] for row in rows if row.get("safe")]
        unsafe_heads = [row["head"] for row in rows if not row.get("safe")]
        summaries.append(
            {
                "model_id": key[0],
                "prompt_type": key[1],
                "layer_id": key[2],
                "latency_model": "sum_of_head_latencies",
                "heads": len(rows),
                "safe_sparse_heads": safe_heads,
                "unsafe_heads": unsafe_heads,
                "dense_all_sum_ms": dense_sum,
                "gate1_all_sum_ms": gate1_sum,
                "selective_oracle_sum_ms": selected_sum,
                "selective_speedup_vs_dense_sum": dense_sum / selected_sum if selected_sum > 0 else None,
                "selective_speedup_vs_gate1_sum": gate1_sum / selected_sum if selected_sum > 0 else None,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            str(row.get("model_id")),
            str(row.get("prompt_type")),
            row.get("layer_id") if row.get("layer_id") is not None else -1,
        ),
    )


def _print_heads(rows: List[Dict[str, Any]], *, limit: int) -> None:
    headers = [
        "prompt",
        "layer",
        "head",
        "safe",
        "B",
        "seed",
        "margin",
        "order",
        "qproj",
        "skip",
        "maxerr",
        "inline",
        "dense",
    ]
    print(" ".join(f"{header:>10}" for header in headers))
    for row in rows[:limit]:
        print(
            " ".join(
                [
                    f"{_fmt(row.get('prompt_type')):>10}",
                    f"{_fmt(row.get('layer_id')):>10}",
                    f"{_fmt(row.get('head')):>10}",
                    f"{_fmt(row.get('safe')):>10}",
                    f"{_fmt(row.get('best_block_size')):>10}",
                    f"{_fmt(row.get('best_seed')):>10}",
                    f"{_fmt(row.get('best_margin')):>10}",
                    f"{_fmt(row.get('block_order')):>10}",
                    f"{_fmt(row.get('qproj_mode')):>10}",
                    f"{_fmt(row.get('skip_fraction')):>10}",
                    f"{_fmt(row.get('max_abs_error')):>10}",
                    f"{_fmt(row.get('inline_total_ms')):>10}",
                    f"{_fmt(row.get('dense_ms')):>10}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--max-error", type=float, default=1e-3)
    parser.add_argument("--max-mean-error", type=float, default=1e-4)
    parser.add_argument("--min-skip-fraction", type=float, default=0.25)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    rows = _load_rows(args.json_paths)
    heads = _calibrate_heads(
        rows,
        max_error=args.max_error,
        max_mean_error=args.max_mean_error,
        min_skip_fraction=args.min_skip_fraction,
    )
    layers = _layer_oracle(heads)
    payload = {
        "summary": {
            "rows": len(rows),
            "calibrated_heads": len(heads),
            "safe_sparse_heads": len([row for row in heads if row.get("safe")]),
            "max_error": args.max_error,
            "max_mean_error": args.max_mean_error,
            "min_skip_fraction": args.min_skip_fraction,
        },
        "heads": heads,
        "layer_selective_oracle": layers,
    }
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.table:
        _print_heads(heads, limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
