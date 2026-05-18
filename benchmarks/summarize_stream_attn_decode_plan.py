"""Summarize planned StreamAttn decode profiler JSON."""

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
            "pattern": row.get("pattern"),
            "requested_active_fraction": row.get("requested_active_fraction"),
            "plan_mode": row.get("plan_mode"),
        }
    shape = row["shape"]
    prev = row.get("prev_token_step_summary") or {}
    regret = row.get("regret_pct")
    if row.get("plan_mode") == "prev_token_plan":
        regret = prev.get("mean_regret_pct")
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
        "actual_active_fraction": row.get("actual_active_fraction"),
        "plan_mode": row.get("plan_mode"),
        "cost_model_hit": row.get("cost_model_hit"),
        "plan_backend": row.get("plan_backend"),
        "plan_reason": row.get("plan_reason"),
        "plan_predicted_ms": row.get("plan_predicted_ms"),
        "plan_run_ms": row.get("plan_run_ms"),
        "plan_wall_ms": row.get("plan_wall_ms"),
        "dense_decode_ms": row.get("dense_decode_ms"),
        "gate1_mass_ms": row.get("gate1_mass_ms"),
        "gate1_value_bound_ms": row.get("gate1_value_bound_ms"),
        "oracle_backend": row.get("oracle_backend"),
        "oracle_ms": row.get("oracle_ms"),
        "regret_pct": regret,
        "prev_token_max_regret_pct": prev.get("max_regret_pct"),
        "mass_dense_ratio": row.get("mass_dense_ratio"),
        "value_dense_ratio": row.get("value_dense_ratio"),
        "plan_oracle_ratio": _ratio(row.get("plan_run_ms"), row.get("oracle_ms")),
    }


def _sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("query_len") or -1,
        row.get("kv_len") or -1,
        row.get("heads") or -1,
        row.get("block_size") or -1,
        row.get("tile_size_q") or -1,
        row.get("requested_active_fraction") or -1,
        str(row.get("pattern")),
        str(row.get("plan_mode")),
    )


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
        row.get("tile_size_q"),
    )


def _best_by_case(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        if row.get("error"):
            continue
        regret = row.get("regret_pct")
        if regret is None:
            continue
        key = _case_key(row)
        if key not in best or regret < (best[key].get("regret_pct") or float("inf")):
            best[key] = row
    return sorted(best.values(), key=_sort_key)


def _print_table(rows: List[Dict[str, Any]], *, limit: int) -> None:
    headers = [
        "M",
        "N",
        "H",
        "B",
        "TQ",
        "a",
        "mode",
        "hit",
        "plan",
        "oracle",
        "dense",
        "mass/d",
        "val/d",
        "plan/o",
        "regret",
        "reason",
    ]
    print(" ".join(f"{header:>10}" for header in headers))
    for row in rows[:limit]:
        if row.get("error"):
            print(
                " ".join(
                    [
                        f"{_fmt(row.get('query_len')):>10}",
                        f"{_fmt(row.get('kv_len')):>10}",
                        f"{_fmt(row.get('heads')):>10}",
                        f"{_fmt(row.get('block_size')):>10}",
                        f"{_fmt(row.get('tile_size_q')):>10}",
                        f"{_fmt(row.get('requested_active_fraction')):>10}",
                        f"{_fmt(row.get('plan_mode')):>10}",
                        f"{'ERR':>10}",
                        f"{'error':>10}",
                        f"{'n/a':>10}",
                        f"{'n/a':>10}",
                        f"{'n/a':>10}",
                        f"{'n/a':>10}",
                        f"{'n/a':>10}",
                        f"{'n/a':>10}",
                        f"{_fmt(row.get('error')):>10}",
                    ]
                )
            )
            continue
        print(
            " ".join(
                [
                    f"{_fmt(row.get('query_len')):>10}",
                    f"{_fmt(row.get('kv_len')):>10}",
                    f"{_fmt(row.get('heads')):>10}",
                    f"{_fmt(row.get('block_size')):>10}",
                    f"{_fmt(row.get('tile_size_q')):>10}",
                    f"{_fmt(row.get('requested_active_fraction')):>10}",
                    f"{_fmt(row.get('plan_mode')):>10}",
                    f"{_fmt(row.get('cost_model_hit')):>10}",
                    f"{_fmt(row.get('plan_backend')):>10}",
                    f"{_fmt(row.get('oracle_backend')):>10}",
                    f"{_fmt(row.get('dense_decode_ms')):>10}",
                    f"{_fmt(row.get('mass_dense_ratio')):>10}",
                    f"{_fmt(row.get('value_dense_ratio')):>10}",
                    f"{_fmt(row.get('plan_oracle_ratio')):>10}",
                    f"{_fmt(row.get('regret_pct')):>10}",
                    f"{_fmt(row.get('plan_reason')):>10}",
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
