"""Summarize selected-exact floor audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def _scope(rows: Iterable[Dict[str, Any]], scope: str, mode: str) -> Dict[str, Any] | None:
    for row in rows:
        if row.get("scope") == scope and row.get("input_mode") == mode:
            return row
    return None


def _extract(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["rows"][0] if "rows" in payload else payload
    prompt = (row.get("capture") or {}).get("prompt_type") or path.stem
    scopes = row.get("flashinfer_scopes", [])
    all_scope = _scope(scopes, "all_heads", "original_true_gqa_contiguous") or {}
    group0 = _scope(scopes, "kv_group_0", "precompact_contiguous") or {}
    group1 = _scope(scopes, "kv_group_1", "precompact_contiguous") or {}
    group0_view = _scope(scopes, "kv_group_0", "view_internal_contiguous") or {}
    group1_view = _scope(scopes, "kv_group_1", "view_internal_contiguous") or {}
    repair = _scope(scopes, "repair_rows", "precompact_contiguous") or {}
    ratios = row.get("ratios", {})
    return {
        "artifact": str(path),
        "prompt": prompt,
        "flash_all_ms": all_scope.get("best_median_ms"),
        "group0_compact_ms": group0.get("best_median_ms"),
        "group1_compact_ms": group1.get("best_median_ms"),
        "group0_view_ms": group0_view.get("best_median_ms"),
        "group1_view_ms": group1_view.get("best_median_ms"),
        "repair_ms": repair.get("best_median_ms"),
        "two_group_oracle_ms": ratios.get("two_kv_group_parallel_oracle_ms"),
        "two_group_oracle_speedup": ratios.get("two_kv_group_oracle_speedup_vs_all"),
        "repair_speedup": ratios.get("repair_speedup_vs_all"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = [_extract(Path(path)) for path in args.artifacts]
    print(
        "prompt\tflash_all\tg0_compact\tg1_compact\t2group_oracle\t"
        "2group_speedup\trepair\trepair_speedup\tg0_view\tg1_view"
    )
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["prompt"]),
                    _fmt(row["flash_all_ms"]),
                    _fmt(row["group0_compact_ms"]),
                    _fmt(row["group1_compact_ms"]),
                    _fmt(row["two_group_oracle_ms"]),
                    _fmt(row["two_group_oracle_speedup"]),
                    _fmt(row["repair_ms"]),
                    _fmt(row["repair_speedup"]),
                    _fmt(row["group0_view_ms"]),
                    _fmt(row["group1_view_ms"]),
                ]
            )
        )
    positive = [
        row
        for row in rows
        if row["two_group_oracle_speedup"] is not None and row["two_group_oracle_speedup"] > 1.0
    ]
    print()
    if len(positive) == len(rows):
        print("decision: selected exact decomposition has a positive floor on all rows")
    else:
        print("decision: selected exact decomposition is not a stable positive floor")
    print(
        "next: compact exact repair may help group0 repair rows, but group1 full exact "
        "must become seed+repair or move inside a dense-quality scheduler."
    )

    if args.json_out:
        output = {"schema": "streamattn.gate0.selected_exact_floor_summary.v1", "rows": rows}
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
