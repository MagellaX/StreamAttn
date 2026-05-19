"""Summarize Gate-0 candidate-filter profiler JSON."""

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
        for row in payload.get("rows", []):
            row = dict(row)
            row["_source"] = str(path)
            rows.append(row)
    return rows


def _summary_row(row: Dict[str, Any]) -> Dict[str, Any]:
    shape = row.get("shape", {})
    return {
        "source": row.get("_source"),
        "error": row.get("error"),
        "tensor_source": row.get("tensor_source"),
        "tensor_space": row.get("tensor_space"),
        "model_id": row.get("model_id"),
        "prompt_type": row.get("prompt_type"),
        "layer_id": row.get("layer_id"),
        "head_id": row.get("head_id"),
        "filter_mode": row.get("filter_mode"),
        "projection_kind": row.get("projection_kind"),
        "projection_dim": row.get("projection_dim"),
        "projection_metadata_dtype": row.get("projection_metadata_dtype"),
        "filter_margin": row.get("filter_margin"),
        "scan_region": row.get("scan_region"),
        "candidate_scan_backend": row.get("candidate_scan_backend"),
        "block_order": row.get("block_order"),
        "block_size": row.get("block_size"),
        "query_len": shape.get("query_len"),
        "kv_len": shape.get("kv_len"),
        "heads": shape.get("heads"),
        "dim": shape.get("dim"),
        "error_budget": row.get("error_budget"),
        "actual_skip_fraction": row.get("actual_skip_fraction"),
        "predicted_skip_fraction": row.get("predicted_skip_fraction"),
        "actual_skip_recovery": row.get("actual_skip_recovery"),
        "false_skip_rate": row.get("false_skip_rate"),
        "precision": row.get("precision"),
        "middle_actual_skip_fraction": row.get("middle_actual_skip_fraction"),
        "middle_predicted_skip_fraction": row.get("middle_predicted_skip_fraction"),
        "middle_actual_skip_recovery": row.get("middle_actual_skip_recovery"),
        "middle_false_skip_rate": row.get("middle_false_skip_rate"),
        "scan_block_fraction": row.get("scan_block_fraction"),
        "q_projection_ms": row.get("q_projection_ms"),
        "projection_score_scan_ms": row.get("projection_score_scan_ms"),
        "projection_mask_scan_ms": row.get("projection_mask_scan_ms"),
        "projection_bitmask_scan_ms": row.get("projection_bitmask_scan_ms"),
        "candidate_scan_ms": row.get("candidate_scan_ms"),
        "full_qk_scan_ms": row.get("full_qk_scan_ms"),
        "scan_over_qk": row.get("scan_over_qk"),
        "estimated_qk_path_ms": row.get("estimated_qk_path_ms"),
        "estimated_speedup_vs_qk": row.get("estimated_speedup_vs_qk"),
        "candidate_promising": row.get("candidate_promising"),
    }


def _case_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("tensor_source"),
        row.get("tensor_space"),
        row.get("model_id"),
        row.get("prompt_type"),
        row.get("layer_id"),
        row.get("head_id"),
        row.get("query_len"),
        row.get("kv_len"),
        row.get("heads"),
        row.get("dim"),
        row.get("block_size"),
        row.get("error_budget"),
    )


def _score(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    false_skip_rate = float(row.get("false_skip_rate") or 0.0)
    recovery = float(row.get("actual_skip_recovery") or 0.0)
    scan_over_qk = float(row.get("scan_over_qk") or 1.0e9)
    predicted = float(row.get("predicted_skip_fraction") or 0.0)
    return (false_skip_rate, -recovery, scan_over_qk, -predicted)


def _best_by_case(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        if row.get("error"):
            continue
        key = _case_key(row)
        existing = best.get(key)
        if existing is None or _score(row) < _score(existing):
            best[key] = row
    return sorted(best.values(), key=_sort_key)


def _sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("tensor_source")),
        str(row.get("model_id")),
        str(row.get("prompt_type")),
        row.get("layer_id") if row.get("layer_id") is not None else -1,
        row.get("head_id") if row.get("head_id") is not None else -9999,
        row.get("kv_len") or -1,
        row.get("block_size") or -1,
        str(row.get("filter_mode")),
        row.get("projection_dim") or -1,
        str(row.get("projection_metadata_dtype")),
        row.get("filter_margin") or -1,
    )


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    promising = [row for row in valid if row.get("candidate_promising")]
    return {
        "rows": len(valid),
        "errors": len([row for row in rows if row.get("error")]),
        "promising_rows": len(promising),
        "max_actual_skip_fraction": (
            max(float(row.get("actual_skip_fraction") or 0.0) for row in valid)
            if valid
            else None
        ),
        "max_predicted_skip_fraction": (
            max(float(row.get("predicted_skip_fraction") or 0.0) for row in valid)
            if valid
            else None
        ),
        "max_actual_skip_recovery": (
            max(float(row.get("actual_skip_recovery") or 0.0) for row in valid)
            if valid
            else None
        ),
        "min_false_skip_rate": (
            min(float(row.get("false_skip_rate") or 0.0) for row in valid) if valid else None
        ),
        "max_false_skip_rate": (
            max(float(row.get("false_skip_rate") or 0.0) for row in valid) if valid else None
        ),
        "min_scan_over_qk": (
            min(float(row.get("scan_over_qk") or 1.0e9) for row in valid) if valid else None
        ),
        "max_estimated_speedup_vs_qk": (
            max(float(row.get("estimated_speedup_vs_qk") or 0.0) for row in valid)
            if valid
            else None
        ),
        "max_middle_actual_skip_recovery": (
            max(float(row.get("middle_actual_skip_recovery") or 0.0) for row in valid)
            if valid
            else None
        ),
    }


def _print_table(rows: List[Dict[str, Any]], *, limit: int) -> None:
    headers = [
        "src",
        "prompt",
        "layer",
        "head",
        "N",
        "B",
        "mode",
        "R",
        "mdtype",
        "margin",
        "region",
        "backend",
        "order",
        "pred",
        "actual",
        "recov",
        "false",
        "mid_rec",
        "scan/qk",
        "qproj",
        "mask",
        "bitmask",
        "speed",
        "ok",
    ]
    print(" ".join(f"{header:>10}" for header in headers))
    for row in rows[:limit]:
        print(
            " ".join(
                [
                    f"{_fmt(row.get('tensor_source')):>10}",
                    f"{_fmt(row.get('prompt_type')):>10}",
                    f"{_fmt(row.get('layer_id')):>10}",
                    f"{_fmt(row.get('head_id')):>10}",
                    f"{_fmt(row.get('kv_len')):>10}",
                    f"{_fmt(row.get('block_size')):>10}",
                    f"{_fmt(row.get('filter_mode')):>10}",
                    f"{_fmt(row.get('projection_dim')):>10}",
                    f"{_fmt(row.get('projection_metadata_dtype')):>10}",
                    f"{_fmt(row.get('filter_margin')):>10}",
                    f"{_fmt(row.get('scan_region')):>10}",
                    f"{_fmt(row.get('candidate_scan_backend')):>10}",
                    f"{_fmt(row.get('block_order')):>10}",
                    f"{_fmt(row.get('predicted_skip_fraction')):>10}",
                    f"{_fmt(row.get('actual_skip_fraction')):>10}",
                    f"{_fmt(row.get('actual_skip_recovery')):>10}",
                    f"{_fmt(row.get('false_skip_rate')):>10}",
                    f"{_fmt(row.get('middle_actual_skip_recovery')):>10}",
                    f"{_fmt(row.get('scan_over_qk')):>10}",
                    f"{_fmt(row.get('q_projection_ms')):>10}",
                    f"{_fmt(row.get('projection_mask_scan_ms')):>10}",
                    f"{_fmt(row.get('projection_bitmask_scan_ms')):>10}",
                    f"{_fmt(row.get('estimated_speedup_vs_qk')):>10}",
                    f"{_fmt(row.get('candidate_promising')):>10}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--table", choices=["all", "best", "none"], default="best")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    rows = sorted((_summary_row(row) for row in _load_rows(args.json_paths)), key=_sort_key)
    best = _best_by_case(rows)
    payload = {
        "summary": _summary(rows),
        "rows": rows,
        "best_by_case": best,
    }
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.table != "none":
        _print_table(best if args.table == "best" else rows, limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
