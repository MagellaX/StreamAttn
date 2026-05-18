"""Summarize FlashInfer single-decode vs StreamAttn decode JSON results."""

import argparse
import json
from pathlib import Path


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _rows(payload):
    for row in payload.get("rows", []):
        if row.get("error"):
            yield row
            continue
        shape = row.get("shape", {})
        yield {
            **row,
            "kv_len": shape.get("kv_len"),
            "heads": shape.get("heads"),
            "kv_heads": shape.get("kv_heads"),
            "attention_type": shape.get("attention_type"),
            "layout_note": row.get("comparison_note"),
        }


def _sort_key(row):
    return (
        row.get("attention_type") or "",
        int(row.get("kv_len") or 0),
        int(row.get("heads") or 0),
        int(row.get("kv_heads") or 0),
        str(row.get("pattern") or ""),
        float(row.get("requested_active_fraction") or 0.0),
    )


def _print_table(rows, *, only_wins: bool, limit: int) -> None:
    rows = [row for row in rows if not row.get("error")]
    if only_wins:
        rows = [
            row
            for row in rows
            if row.get("streamattn_oracle_wins") or row.get("streamattn_wrapper_wins")
        ]
    rows = sorted(rows, key=_sort_key)
    if limit > 0:
        rows = rows[:limit]
    header = (
        f"{'N':>8} {'H':>4} {'Hkv':>4} {'a':>7} {'pattern':>14} "
        f"{'FI':>8} {'SA-oracle':>10} {'SA-wrap':>9} {'O/FI':>7} {'W/FI':>7} "
        f"{'oracle':>18} {'wrap':>14} {'O-win':>6} {'W-win':>6} {'hit':>5} {'note':>14}"
    )
    print(header)
    for row in rows:
        print(
            f"{int(row.get('kv_len') or 0):8d} "
            f"{int(row.get('heads') or 0):4d} "
            f"{int(row.get('kv_heads') or 0):4d} "
            f"{float(row.get('requested_active_fraction') or 0.0):7.3f} "
            f"{str(row.get('pattern') or ''):>14} "
            f"{_fmt(row.get('best_flashinfer_ms')):>8} "
            f"{_fmt(row.get('streamattn_oracle_ms')):>10} "
            f"{_fmt(row.get('streamattn_wrapper_ms')):>9} "
            f"{_fmt(row.get('streamattn_oracle_vs_flashinfer_speedup')):>7} "
            f"{_fmt(row.get('streamattn_wrapper_vs_flashinfer_speedup')):>7} "
            f"{str(row.get('streamattn_oracle_backend') or ''):>18} "
            f"{str(row.get('streamattn_wrapper_backend') or ''):>14} "
            f"{_fmt(row.get('streamattn_oracle_wins')):>6} "
            f"{_fmt(row.get('streamattn_wrapper_wins')):>6} "
            f"{_fmt(row.get('cost_model_hit')):>5} "
            f"{str(row.get('layout_note') or ''):>14}"
        )


def _summary(rows) -> dict:
    valid = [row for row in rows if not row.get("error")]
    oracle_wins = [row for row in valid if row.get("streamattn_oracle_wins")]
    wrapper_wins = [row for row in valid if row.get("streamattn_wrapper_wins")]
    fair = [row for row in valid if row.get("fair_mha_comparison")]
    return {
        "rows": len(valid),
        "errors": len([row for row in rows if row.get("error")]),
        "oracle_wins": len(oracle_wins),
        "wrapper_wins": len(wrapper_wins),
        "fair_mha_rows": len(fair),
        "fair_mha_oracle_wins": len(
            [row for row in fair if row.get("streamattn_oracle_wins")]
        ),
        "fair_mha_wrapper_wins": len(
            [row for row in fair if row.get("streamattn_wrapper_wins")]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--table", choices=["all", "wins", "none"], default="all")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    payload = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    rows = list(_rows(payload))
    summary = _summary(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.table != "none":
        _print_table(rows, only_wins=args.table == "wins", limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
