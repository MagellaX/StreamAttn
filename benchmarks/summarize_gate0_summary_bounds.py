"""Summarize Gate-0 summary-bound profiler JSON."""

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
    if row.get("error"):
        return {
            "source": row.get("_source"),
            "error": row.get("error"),
            "query_len": shape.get("query_len"),
            "kv_len": shape.get("kv_len"),
            "heads": shape.get("heads"),
            "dim": shape.get("dim"),
            "block_size": row.get("block_size"),
            "block_order": row.get("block_order"),
            "pattern": row.get("pattern"),
            "requested_active_fraction": row.get("requested_active_fraction"),
            "num_summary_outliers": row.get("num_summary_outliers"),
            "blocks_per_program": row.get("blocks_per_program"),
        }
    return {
        "source": row.get("_source"),
        "device": row.get("device"),
        "query_len": shape.get("query_len"),
        "kv_len": shape.get("kv_len"),
        "heads": shape.get("heads"),
        "dim": shape.get("dim"),
        "dtype": shape.get("dtype"),
        "block_size": row.get("block_size"),
        "block_order": row.get("block_order"),
        "num_blocks": row.get("num_blocks"),
        "pattern": row.get("pattern"),
        "requested_active_fraction": row.get("requested_active_fraction"),
        "block_quantized_active_fraction": row.get("block_quantized_active_fraction"),
        "summary_type": row.get("summary_type"),
        "num_summary_outliers": row.get("num_summary_outliers"),
        "scan_backend": row.get("scan_backend"),
        "blocks_per_program": row.get("blocks_per_program"),
        "predicted_skip_fraction": row.get("predicted_skip_fraction"),
        "actual_gate1_skip_fraction": row.get("actual_gate1_skip_fraction"),
        "false_negative_rate": row.get("false_negative_rate"),
        "false_positive_rate": row.get("false_positive_rate"),
        "unsafe_bound_rate": row.get("unsafe_bound_rate"),
        "bound_gap_mean": row.get("bound_gap_mean"),
        "bound_gap_p50": row.get("bound_gap_p50"),
        "bound_gap_p90": row.get("bound_gap_p90"),
        "bound_gap_p99": row.get("bound_gap_p99"),
        "summary_build_ms": row.get("summary_build_ms"),
        "summary_scan_ms": row.get("summary_scan_ms"),
        "full_qk_scan_ms": row.get("full_qk_scan_ms"),
        "summary_scan_over_qk": row.get("summary_scan_over_qk"),
        "estimated_gate0_speedup_vs_gate1": row.get("estimated_gate0_speedup_vs_gate1"),
        "gate0_promising": row.get("gate0_promising"),
    }


def _case_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("query_len"),
        row.get("kv_len"),
        row.get("heads"),
        row.get("dim"),
        row.get("block_size"),
        row.get("pattern"),
        row.get("requested_active_fraction"),
    )


def _sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("query_len") or -1,
        row.get("kv_len") or -1,
        row.get("heads") or -1,
        row.get("block_size") or -1,
        row.get("requested_active_fraction") or -1,
        row.get("blocks_per_program") or -1,
        str(row.get("pattern")),
        str(row.get("block_order")),
        row.get("num_summary_outliers") or 0,
    )


def _best_by_case(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        if row.get("error"):
            continue
        key = _case_key(row)
        existing = best.get(key)
        if existing is None:
            best[key] = row
            continue
        score = _score(row)
        existing_score = _score(existing)
        if score < existing_score:
            best[key] = row
    return sorted(best.values(), key=_sort_key)


def _score(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    false_negative = float(row.get("false_negative_rate") or 0.0)
    unsafe = float(row.get("unsafe_bound_rate") or 0.0)
    predicted_skip = float(row.get("predicted_skip_fraction") or 0.0)
    scan_over_qk = float(row.get("summary_scan_over_qk") or 1.0e9)
    return (false_negative, unsafe, -predicted_skip, scan_over_qk)


def _print_table(rows: List[Dict[str, Any]], *, limit: int) -> None:
    headers = [
        "N",
        "H",
        "B",
        "a",
        "pattern",
        "order",
        "out",
        "scan",
        "bpp",
        "pred",
        "actual",
        "fn",
        "fp",
        "gap90",
        "scan/qk",
        "spd",
        "ok",
    ]
    print(" ".join(f"{header:>9}" for header in headers))
    for row in rows[:limit]:
        if row.get("error"):
            print(
                " ".join(
                    [
                        f"{_fmt(row.get('kv_len')):>9}",
                        f"{_fmt(row.get('heads')):>9}",
                        f"{_fmt(row.get('block_size')):>9}",
                        f"{_fmt(row.get('requested_active_fraction')):>9}",
                        f"{_fmt(row.get('pattern')):>9}",
                        f"{_fmt(row.get('block_order')):>9}",
                        f"{_fmt(row.get('num_summary_outliers')):>9}",
                        f"{'n/a':>9}",
                        f"{_fmt(row.get('blocks_per_program')):>9}",
                        f"{'ERROR':>9}",
                        f"{'n/a':>9}",
                        f"{'n/a':>9}",
                        f"{'n/a':>9}",
                        f"{'n/a':>9}",
                        f"{'n/a':>9}",
                        f"{'n/a':>9}",
                        f"{'no':>9}",
                    ]
                )
            )
            continue
        print(
            " ".join(
                [
                    f"{_fmt(row.get('kv_len')):>9}",
                    f"{_fmt(row.get('heads')):>9}",
                    f"{_fmt(row.get('block_size')):>9}",
                    f"{_fmt(row.get('requested_active_fraction')):>9}",
                    f"{_fmt(row.get('pattern')):>9}",
                    f"{_fmt(row.get('block_order')):>9}",
                    f"{_fmt(row.get('num_summary_outliers')):>9}",
                    f"{_fmt(row.get('scan_backend')):>9}",
                    f"{_fmt(row.get('blocks_per_program')):>9}",
                    f"{_fmt(row.get('predicted_skip_fraction')):>9}",
                    f"{_fmt(row.get('actual_gate1_skip_fraction')):>9}",
                    f"{_fmt(row.get('false_negative_rate')):>9}",
                    f"{_fmt(row.get('false_positive_rate')):>9}",
                    f"{_fmt(row.get('bound_gap_p90')):>9}",
                    f"{_fmt(row.get('summary_scan_over_qk')):>9}",
                    f"{_fmt(row.get('estimated_gate0_speedup_vs_gate1')):>9}",
                    f"{_fmt(row.get('gate0_promising')):>9}",
                ]
            )
        )


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    promising = [row for row in valid if row.get("gate0_promising")]
    zero_fn = [row for row in valid if float(row.get("false_negative_rate") or 0.0) == 0.0]
    return {
        "rows": len(valid),
        "errors": len([row for row in rows if row.get("error")]),
        "promising_rows": len(promising),
        "zero_false_negative_rows": len(zero_fn),
        "max_predicted_skip_fraction": (
            max(float(row.get("predicted_skip_fraction") or 0.0) for row in valid)
            if valid
            else None
        ),
    }


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
