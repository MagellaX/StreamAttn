"""Summarize Gate-1 decode profiler JSON into routing ratios."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den <= 0.0:
        return None
    return float(num) / float(den)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
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


def _winner(row: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    candidates = {"dense": row.get("dense_decode_ms")}
    if row.get("gate1_mass_ms") is not None:
        candidates["mass"] = row.get("gate1_mass_ms")
    if row.get("gate1_value_bound_ms") is not None:
        candidates["value_bound"] = row.get("gate1_value_bound_ms")
    candidates = {name: value for name, value in candidates.items() if value is not None}
    if not candidates:
        return "error", None
    return min(candidates.items(), key=lambda item: item[1])


def _summary_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("error"):
        shape = row.get("shape", {})
        return {
            "source": row.get("_source"),
            "error": row.get("error"),
            "query_len": shape.get("query_len"),
            "kv_len": shape.get("kv_len"),
            "heads": shape.get("heads"),
            "kv_heads": shape.get("kv_heads"),
            "dim": shape.get("dim"),
            "attention_type": shape.get("attention_type"),
            "block_size": row.get("block_size"),
            "tile_size_q": row.get("tile_size_q"),
            "num_warps": row.get("num_warps"),
            "num_stages": row.get("num_stages"),
            "pattern": row.get("pattern"),
            "requested_active_fraction": row.get("requested_active_fraction"),
        }

    shape = row["shape"]
    winner, winner_ms = _winner(row)
    return {
        "source": row.get("_source"),
        "device": row.get("device"),
        "query_len": shape.get("query_len"),
        "kv_len": shape.get("kv_len"),
        "heads": shape.get("heads"),
        "kv_heads": shape.get("kv_heads"),
        "dim": shape.get("dim"),
        "dtype": shape.get("dtype"),
        "attention_type": shape.get("attention_type"),
        "block_size": row.get("block_size"),
        "tile_size_q": row.get("tile_size_q"),
        "num_warps": row.get("num_warps"),
        "num_stages": row.get("num_stages"),
        "pattern": row.get("pattern"),
        "requested_active_fraction": row.get("requested_active_fraction"),
        "active_pv_fraction_mass": row.get("active_pv_fraction_mass"),
        "active_pv_fraction_value_bound": row.get("active_pv_fraction_value_bound"),
        "dense_decode_ms": row.get("dense_decode_ms"),
        "gate1_mass_ms": row.get("gate1_mass_ms"),
        "gate1_value_bound_ms": row.get("gate1_value_bound_ms"),
        "gate1_dense_equiv_ms": row.get("gate1_dense_equiv_ms"),
        "gate1_qk_scan_ms": row.get("gate1_qk_scan_ms"),
        "metadata_full_build_ms": row.get("metadata_full_build_ms"),
        "metadata_update_wall_ms": row.get("metadata_update_wall_ms"),
        "metadata_update_cuda_ms": row.get("metadata_update_cuda_ms"),
        "mass_dense_ratio": _ratio(row.get("gate1_mass_ms"), row.get("dense_decode_ms")),
        "value_dense_ratio": _ratio(
            row.get("gate1_value_bound_ms"),
            row.get("dense_decode_ms"),
        ),
        "dense_equiv_ratio": _ratio(
            row.get("gate1_dense_equiv_ms"),
            row.get("dense_decode_ms"),
        ),
        "qk_dense_ratio": _ratio(row.get("gate1_qk_scan_ms"), row.get("dense_decode_ms")),
        "metadata_update_wall_dense_ratio": _ratio(
            row.get("metadata_update_wall_ms"),
            row.get("dense_decode_ms"),
        ),
        "metadata_full_build_dense_ratio": _ratio(
            row.get("metadata_full_build_ms"),
            row.get("dense_decode_ms"),
        ),
        "router_backend": row.get("router_backend"),
        "router_reason": row.get("router_reason"),
        "router_regret_pct": row.get("router_regret_pct"),
        "winner_backend": winner,
        "winner_ms": winner_ms,
    }


def _case_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("query_len"),
        row.get("kv_len"),
        row.get("heads"),
        row.get("kv_heads"),
        row.get("dim"),
        row.get("dtype"),
        row.get("attention_type"),
        row.get("pattern"),
        row.get("requested_active_fraction"),
        row.get("block_size"),
    )


def _best_by_case(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        if row.get("error"):
            continue
        key = _case_key(row)
        score = min(
            value
            for value in [row.get("mass_dense_ratio"), row.get("value_dense_ratio"), 1.0]
            if value is not None
        )
        existing = best.get(key)
        if existing is None:
            best[key] = row
            continue
        existing_score = min(
            value
            for value in [
                existing.get("mass_dense_ratio"),
                existing.get("value_dense_ratio"),
                1.0,
            ]
            if value is not None
        )
        if score < existing_score:
            best[key] = row
    return sorted(best.values(), key=_sort_key)


def _sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("query_len") or -1,
        row.get("kv_len") or -1,
        row.get("heads") or -1,
        row.get("block_size") or -1,
        row.get("tile_size_q") or -1,
        row.get("requested_active_fraction") or -1,
        str(row.get("pattern")),
    )


def _print_table(rows: List[Dict[str, Any]], *, limit: int) -> None:
    headers = [
        "M",
        "N",
        "H",
        "B",
        "TQ",
        "a",
        "dense",
        "mass/d",
        "val/d",
        "qk/d",
        "upd/d",
        "winner",
        "router",
        "regret",
    ]
    print(" ".join(f"{header:>9}" for header in headers))
    for row in rows[:limit]:
        if row.get("error"):
            print(
                " ".join(
                    [
                        f"{_fmt(row.get('query_len')):>9}",
                        f"{_fmt(row.get('kv_len')):>9}",
                        f"{_fmt(row.get('heads')):>9}",
                        f"{_fmt(row.get('block_size')):>9}",
                        f"{_fmt(row.get('tile_size_q')):>9}",
                        f"{_fmt(row.get('requested_active_fraction')):>9}",
                        f"{'ERROR':>9}",
                        f"{'n/a':>9}",
                        f"{'n/a':>9}",
                        f"{'n/a':>9}",
                        f"{'n/a':>9}",
                        f"{'error':>9}",
                        f"{'n/a':>9}",
                        f"{'n/a':>9}",
                    ]
                )
            )
            continue
        print(
            " ".join(
                [
                    f"{_fmt(row.get('query_len')):>9}",
                    f"{_fmt(row.get('kv_len')):>9}",
                    f"{_fmt(row.get('heads')):>9}",
                    f"{_fmt(row.get('block_size')):>9}",
                    f"{_fmt(row.get('tile_size_q')):>9}",
                    f"{_fmt(row.get('requested_active_fraction')):>9}",
                    f"{_fmt(row.get('dense_decode_ms')):>9}",
                    f"{_fmt(row.get('mass_dense_ratio')):>9}",
                    f"{_fmt(row.get('value_dense_ratio')):>9}",
                    f"{_fmt(row.get('qk_dense_ratio')):>9}",
                    f"{_fmt(row.get('metadata_update_wall_dense_ratio')):>9}",
                    f"{_fmt(row.get('winner_backend')):>9}",
                    f"{_fmt(row.get('router_backend')):>9}",
                    f"{_fmt(row.get('router_regret_pct')):>9}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--table", choices=["all", "best"], default="all")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    rows = sorted((_summary_row(row) for row in _load_rows(args.json_paths)), key=_sort_key)
    best = _best_by_case(rows)
    payload = {
        "rows": rows,
        "best_by_case": best,
        "num_rows": len(rows),
        "num_errors": sum(1 for row in rows if row.get("error")),
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _print_table(best if args.table == "best" else rows, limit=args.limit)


if __name__ == "__main__":
    main()
