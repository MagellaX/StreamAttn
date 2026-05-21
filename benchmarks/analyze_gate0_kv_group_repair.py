"""Analyze speculative whole-KV-group seed-only policies with exact repair.

The compact TK head-mode spike shows that StreamAttn gets real scheduler
savings when an entire true-GQA KV group can run seed-only.  Real calibration is
messier: a whole group may contain a few unsafe Q heads.  This utility reads
seed-only true-GQA benchmark artifacts and asks a narrower question:

* if we seed the whole KV group speculatively;
* which heads must be repaired exactly for a given error budget;
* what optimistic fused/oracle timing remains before exact-repair cost is known.

It is an evidence tool, not a production policy builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_BUDGETS = "strict:1e-2,moderate:1.5e-2,research:5e-2"


def _heads(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))
    return sorted(set(int(item) for item in raw))


def parse_budgets(raw: str) -> List[Dict[str, Any]]:
    budgets = []
    for item in raw.split(","):
        if not item.strip():
            continue
        name, value = item.split(":", 1)
        budgets.append({"name": name.strip(), "max_abs_error": float(value)})
    return budgets


def _rows_from_paths(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            copied = dict(row)
            copied["_source"] = str(path)
            copied["_capture"] = payload.get("capture") or {}
            rows.append(copied)
    return rows


def _per_head_errors(row: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    quality = row.get("quality") or {}
    payload = quality.get("hybrid_seed_error_vs_true_dense_per_head") or {}
    return {
        int(item["head"]): {
            "max_abs_error": float(item.get("max_abs_error") or 0.0),
            "mean_abs_error": float(item.get("mean_abs_error") or 0.0),
        }
        for item in payload.get("per_head", [])
    }


def _kv_group(head: int, *, q_heads: int, kv_heads: int) -> int:
    return int(head) // (q_heads // kv_heads)


def analyze_row(row: Dict[str, Any], budgets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    policy = row.get("policy") or {}
    shape = row.get("shape") or {}
    timing = row.get("timing") or {}
    capture = row.get("_capture") or row.get("capture") or {}
    q_heads = int(shape.get("q_heads") or (row.get("capture") or {}).get("shape", {}).get("heads") or 0)
    kv_heads = int(shape.get("true_kv_heads") or (row.get("capture") or {}).get("logical_num_kv_heads") or 0)
    seed_heads = _heads(policy.get("seed_heads"))
    per_head = _per_head_errors(row)
    exact_remaining_ms = timing.get("exact_remaining_flashinfer_group_parallel_oracle_ms")
    if exact_remaining_ms is None:
        exact_remaining_ms = timing.get("exact_remaining_group_parallel_oracle_ms")
    seed_oracle_ms = timing.get("seed_only_group_parallel_oracle_ms")
    reference_ms = timing.get("reference_exact_ms") or timing.get("flashinfer_all_true_gqa_ms")
    dense_ms = timing.get("true_gqa_dense_all_ms")
    optimistic_ms = max(
        float(exact_remaining_ms or 0.0),
        float(seed_oracle_ms or 0.0),
    )
    group_size = q_heads // kv_heads if q_heads and kv_heads else 0
    seeded_groups = sorted({_kv_group(head, q_heads=q_heads, kv_heads=kv_heads) for head in seed_heads}) if group_size else []

    rows = []
    for budget in budgets:
        max_error = float(budget["max_abs_error"])
        repair_heads = [
            head
            for head in seed_heads
            if float(per_head.get(head, {}).get("max_abs_error", 0.0)) > max_error
        ]
        trusted_seed_heads = [head for head in seed_heads if head not in set(repair_heads)]
        corrected_max = max(
            (float(per_head.get(head, {}).get("max_abs_error", 0.0)) for head in trusted_seed_heads),
            default=0.0,
        )
        repair_by_group: Dict[int, List[int]] = {}
        for head in repair_heads:
            repair_by_group.setdefault(_kv_group(head, q_heads=q_heads, kv_heads=kv_heads), []).append(head)
        rows.append(
            {
                "model_id": capture.get("model_id") or row.get("model_id"),
                "prompt_type": capture.get("prompt_type") or row.get("prompt_type"),
                "layer_id": (row.get("capture") or {}).get("layer_id") or row.get("layer_id"),
                "kv_len": shape.get("kv_len"),
                "budget": budget["name"],
                "max_abs_error_budget": max_error,
                "speculative_seed_heads": seed_heads,
                "speculative_seed_kv_groups": seeded_groups,
                "repair_heads": repair_heads,
                "repair_heads_by_kv_group": [
                    {"kv_head": kv_head, "heads": heads} for kv_head, heads in sorted(repair_by_group.items())
                ],
                "trusted_seed_heads_after_repair": trusted_seed_heads,
                "corrected_max_abs_error": corrected_max,
                "optimistic_oracle_ms_without_repair_cost": optimistic_ms,
                "reference_exact_ms": float(reference_ms or 0.0),
                "dense_all_ms": float(dense_ms or 0.0),
                "optimistic_speedup_vs_reference_exact": (float(reference_ms) / optimistic_ms)
                if reference_ms and optimistic_ms
                else None,
                "needs_exact_selected_repair_measurement": bool(repair_heads),
                "source": row.get("_source"),
            }
        )
    return rows


def analyze(paths: Sequence[str], budgets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    entries = []
    for row in _rows_from_paths(paths):
        entries.extend(analyze_row(row, budgets))
    return {
        "schema": "streamattn.gate0.kv_group_repair_analysis.v1",
        "summary": {
            "rows": len(entries),
            "entries_requiring_repair": sum(1 for row in entries if row["repair_heads"]),
            "entries_with_optimistic_reference_win": sum(
                1
                for row in entries
                if (row.get("optimistic_speedup_vs_reference_exact") or 0.0) > 1.0
            ),
        },
        "entries": entries,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(entries: Sequence[Dict[str, Any]], *, limit: int) -> None:
    headers = ["budget", "prompt", "seed_groups", "repair", "err", "opt_ms", "ref_ms", "speed"]
    print(" ".join(f"{header:>14}" for header in headers))
    for row in entries[:limit]:
        print(
            " ".join(
                [
                    f"{row['budget'][:14]:>14}",
                    f"{str(row.get('prompt_type'))[:14]:>14}",
                    f"{str(row.get('speculative_seed_kv_groups'))[:14]:>14}",
                    f"{str(row.get('repair_heads'))[:14]:>14}",
                    f"{_fmt(row.get('corrected_max_abs_error')):>14}",
                    f"{_fmt(row.get('optimistic_oracle_ms_without_repair_cost')):>14}",
                    f"{_fmt(row.get('reference_exact_ms')):>14}",
                    f"{_fmt(row.get('optimistic_speedup_vs_reference_exact')):>14}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    payload = analyze(args.json_paths, parse_budgets(args.budgets))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print_table(payload["entries"], limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
