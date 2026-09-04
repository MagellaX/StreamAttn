"""Summarize an SM90 exact micro-prefill portfolio artifact by regime."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPECTED_SCHEMA = "streamattn.sm90_micro_prefill_canary.v2"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _group_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    speedups = [float(row["median_speedup_vs_flash"]) for row in rows]
    return {
        "cells": len(rows),
        "exact_cells": sum(bool(row["strict_correct"]) for row in rows),
        "paired_flash_gate_cells": sum(
            bool(row["promotion_pass"]) for row in rows
        ),
        "transposed_winners": sum(row["winner"] == "transposed" for row in rows),
        "natural_winners": sum(row["winner"] == "natural" for row in rows),
        "geometric_mean_speedup_vs_flash": statistics.geometric_mean(speedups),
        "p10_speedup_vs_flash": _percentile(speedups, 0.10),
        "minimum_speedup_vs_flash": min(speedups),
        "maximum_speedup_vs_flash": max(speedups),
    }


def summarize(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(
            f"expected schema {EXPECTED_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("artifact must contain at least one row")

    query_lengths = sorted({int(row["query_len"]) for row in rows})
    by_query_length = {
        str(query_len): _group_summary(
            [row for row in rows if int(row["query_len"]) == query_len]
        )
        for query_len in query_lengths
    }
    return {
        "schema": "streamattn.sm90_micro_prefill_summary.v1",
        "source": str(path),
        "device": payload.get("device"),
        "provider": payload.get("provider"),
        "baseline_scope": "graph-captured torch Flash SDPA only",
        "overall": _group_summary(rows),
        "by_query_length": by_query_length,
        "unresolved_cells": [
            {
                key: row[key]
                for key in (
                    "batch",
                    "query_len",
                    "kv_len",
                    "group_size",
                    "head_dim",
                    "winner",
                    "median_speedup_vs_flash",
                )
            }
            for row in rows
            if not bool(row["promotion_pass"])
        ],
        "decision": (
            "resolve_fastest_exact_baselines_and_fill_m64_gap"
            if all(bool(row["strict_correct"]) for row in rows)
            else "fix_exactness_before_performance_work"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    summary = summarize(args.artifact)
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
