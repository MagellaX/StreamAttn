"""Estimate split-K feasibility for inline projection Gate-0 artifacts.

The grouped inline profiler proves whether projection filtering finds blocks,
but the current kernel is serial over KV blocks. This script uses the profiler
artifacts to estimate how much a split-K middle-block schedule could help for
chunk counts such as 2, 4, 8, and 16.

Artifacts currently contain aggregate block stats, not per-block traces, so the
estimator reports a range:

* ``uniform`` assumes active middle work is evenly distributed across chunks.
* ``clustered`` assumes active middle work is concentrated in one chunk.

The result is a scheduling feasibility signal, not a replacement for the
split-K kernel benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CHUNKS = "2,4,8,16"


def _parse_ints(raw: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("at least one integer value is required")
    return values


def _load_results(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            candidates = payload
        elif "results" in payload:
            candidates = payload.get("results") or []
        else:
            candidates = [payload]
        for row in candidates:
            if isinstance(row, dict):
                copied = dict(row)
                copied["_source"] = str(path)
                rows.append(copied)
    return rows


def _as_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def _as_int(row: Dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None:
        return default
    return int(value)


def _split_middle_counts(row: Dict[str, Any]) -> Dict[str, int]:
    stats = row.get("stats") or {}
    sparse_heads = row.get("sparse_heads") or []
    head_count = len(sparse_heads) if sparse_heads else _as_int(stats, "mode", 0)
    sink_blocks = _as_int(row, "sink_blocks", 0)
    recent_blocks = _as_int(row, "recent_blocks", 0)
    configured_middle_seed = _as_int(row, "middle_seed_blocks", 0)
    non_middle_seed = head_count * (sink_blocks + recent_blocks)
    middle_seed = head_count * configured_middle_seed
    seed_computed = _as_int(stats, "seed_computed_blocks", non_middle_seed + middle_seed)
    middle_blocks = _as_int(stats, "middle_blocks", 0)
    projection_computed = _as_int(stats, "projection_computed_blocks", 0)
    projection_skipped = _as_int(stats, "projection_skipped_blocks", 0)
    gate1_skipped = _as_int(stats, "gate1_post_qk_skipped_blocks", 0)
    pv_executed = _as_int(stats, "pv_executed_blocks", 0)

    # Projection-computed includes middle seed blocks because those blocks are
    # forced past Gate-0. Split-K keeps those in the seed phase, so remove them
    # from the chunk work estimate.
    chunk_qk_blocks = max(0, projection_computed - middle_seed)
    # Non-middle sink/recent blocks always update state. Remaining PV work is
    # middle work; it may include middle seed and non-seed blocks, but without
    # per-block traces we only need a conservative active-work count.
    middle_pv_blocks = max(0, pv_executed - non_middle_seed)
    chunk_pv_blocks = min(chunk_qk_blocks, middle_pv_blocks)
    chunk_qk_only_blocks = max(0, chunk_qk_blocks - chunk_pv_blocks)
    chunk_projection_only_blocks = max(0, middle_blocks - middle_seed - chunk_qk_blocks)

    return {
        "head_count": head_count,
        "non_middle_seed_blocks": non_middle_seed,
        "middle_seed_blocks": middle_seed,
        "seed_computed_blocks": seed_computed,
        "middle_blocks": middle_blocks,
        "chunk_middle_blocks": max(0, middle_blocks - middle_seed),
        "projection_skipped_blocks": projection_skipped,
        "projection_computed_blocks": projection_computed,
        "chunk_projection_only_blocks": chunk_projection_only_blocks,
        "chunk_qk_blocks": chunk_qk_blocks,
        "chunk_qk_only_blocks": chunk_qk_only_blocks,
        "chunk_pv_blocks": chunk_pv_blocks,
        "gate1_post_qk_skipped_blocks": gate1_skipped,
        "pv_executed_blocks": pv_executed,
    }


def _work_units(
    *,
    projection_only: float,
    qk_only: float,
    pv: float,
    projection_weight: float,
    qk_weight: float,
    pv_weight: float,
) -> float:
    return (
        projection_only * projection_weight
        + qk_only * (projection_weight + qk_weight)
        + pv * (projection_weight + qk_weight + pv_weight)
    )


def estimate_row(
    row: Dict[str, Any],
    *,
    chunks: int,
    projection_weight: float,
    qk_weight: float,
    pv_weight: float,
    merge_base_ms: float,
    merge_per_head_chunk_ms: float,
) -> Dict[str, Any]:
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    counts = _split_middle_counts(row)
    head_count = counts["head_count"]
    inline_ms = _as_float(row, "inline_sparse_group_ms", _as_float(row, "inline_total_ms", 0.0))
    dense_ms = _as_float(row, "dense_all_ms", 0.0)
    seed_blocks = counts["seed_computed_blocks"]
    seed_units = seed_blocks * (projection_weight + qk_weight + pv_weight)
    middle_units = _work_units(
        projection_only=counts["chunk_projection_only_blocks"],
        qk_only=counts["chunk_qk_only_blocks"],
        pv=counts["chunk_pv_blocks"],
        projection_weight=projection_weight,
        qk_weight=qk_weight,
        pv_weight=pv_weight,
    )
    total_units = max(seed_units + middle_units, 1.0)

    uniform_projection_only = math.ceil(counts["chunk_projection_only_blocks"] / chunks)
    uniform_qk_only = math.ceil(counts["chunk_qk_only_blocks"] / chunks)
    uniform_pv = math.ceil(counts["chunk_pv_blocks"] / chunks)
    uniform_middle_units = _work_units(
        projection_only=uniform_projection_only,
        qk_only=uniform_qk_only,
        pv=uniform_pv,
        projection_weight=projection_weight,
        qk_weight=qk_weight,
        pv_weight=pv_weight,
    )

    clustered_projection_only = math.ceil(counts["chunk_projection_only_blocks"] / chunks)
    clustered_qk_only = counts["chunk_qk_only_blocks"]
    clustered_pv = counts["chunk_pv_blocks"]
    clustered_middle_units = _work_units(
        projection_only=clustered_projection_only,
        qk_only=clustered_qk_only,
        pv=clustered_pv,
        projection_weight=projection_weight,
        qk_weight=qk_weight,
        pv_weight=pv_weight,
    )

    merge_ms = merge_base_ms + merge_per_head_chunk_ms * head_count * chunks
    uniform_ms = inline_ms * ((seed_units + uniform_middle_units) / total_units) + merge_ms
    clustered_ms = inline_ms * ((seed_units + clustered_middle_units) / total_units) + merge_ms
    return {
        "source": row.get("_source"),
        "model_id": row.get("model_id"),
        "prompt_type": row.get("prompt_type"),
        "layer_id": row.get("layer_id"),
        "kv_len": row.get("kv_len") or (row.get("shape") or {}).get("kv_len"),
        "sparse_head_count": head_count,
        "sparse_heads": row.get("sparse_heads"),
        "chunks": chunks,
        "dense_all_ms": dense_ms,
        "current_inline_sparse_group_ms": inline_ms,
        "current_inline_vs_dense_speedup": dense_ms / inline_ms if inline_ms > 0 else None,
        "merge_overhead_ms": merge_ms,
        "uniform_estimated_ms": uniform_ms,
        "uniform_speedup_vs_dense": dense_ms / uniform_ms if uniform_ms > 0 else None,
        "uniform_speedup_vs_current_inline": inline_ms / uniform_ms if uniform_ms > 0 else None,
        "clustered_estimated_ms": clustered_ms,
        "clustered_speedup_vs_dense": dense_ms / clustered_ms if clustered_ms > 0 else None,
        "clustered_speedup_vs_current_inline": inline_ms / clustered_ms if clustered_ms > 0 else None,
        "projection_skip_fraction": (row.get("stats") or {}).get("projection_skip_fraction"),
        "max_abs_error": row.get("max_abs_error"),
        "mean_abs_error": row.get("mean_abs_error"),
        "counts": counts,
        "work_units": {
            "seed_units": seed_units,
            "middle_units": middle_units,
            "total_units": total_units,
            "uniform_max_middle_units": uniform_middle_units,
            "clustered_max_middle_units": clustered_middle_units,
            "projection_weight": projection_weight,
            "qk_weight": qk_weight,
            "pv_weight": pv_weight,
        },
        "per_chunk_active_work": {
            "uniform_max_projection_only_blocks": uniform_projection_only,
            "uniform_max_qk_only_blocks": uniform_qk_only,
            "uniform_max_pv_blocks": uniform_pv,
            "clustered_max_projection_only_blocks": clustered_projection_only,
            "clustered_max_qk_only_blocks": clustered_qk_only,
            "clustered_max_pv_blocks": clustered_pv,
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "prompt",
        "kv",
        "heads",
        "chunks",
        "skip",
        "dense",
        "inline",
        "u_ms",
        "u_spd",
        "c_ms",
        "c_spd",
        "u_pv",
        "c_pv",
    ]
    print(" ".join(f"{header:>10}" for header in headers))
    for row in rows:
        active = row["per_chunk_active_work"]
        values = [
            row.get("prompt_type"),
            row.get("kv_len"),
            row.get("sparse_head_count"),
            row.get("chunks"),
            row.get("projection_skip_fraction"),
            row.get("dense_all_ms"),
            row.get("current_inline_sparse_group_ms"),
            row.get("uniform_estimated_ms"),
            row.get("uniform_speedup_vs_dense"),
            row.get("clustered_estimated_ms"),
            row.get("clustered_speedup_vs_dense"),
            active.get("uniform_max_pv_blocks"),
            active.get("clustered_max_pv_blocks"),
        ]
        print(" ".join(f"{_fmt(value):>10}" for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--kv-len", type=int, default=16384)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--projection-weight", type=float, default=1.0)
    parser.add_argument("--qk-weight", type=float, default=8.0)
    parser.add_argument("--pv-weight", type=float, default=4.0)
    parser.add_argument("--merge-base-ms", type=float, default=0.005)
    parser.add_argument("--merge-per-head-chunk-ms", type=float, default=0.00005)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    chunk_counts = _parse_ints(args.chunks)
    input_rows = [
        row
        for row in _load_results(args.artifacts)
        if int(row.get("kv_len") or (row.get("shape") or {}).get("kv_len") or 0) == args.kv_len
    ]
    estimates = [
        estimate_row(
            row,
            chunks=chunks,
            projection_weight=args.projection_weight,
            qk_weight=args.qk_weight,
            pv_weight=args.pv_weight,
            merge_base_ms=args.merge_base_ms,
            merge_per_head_chunk_ms=args.merge_per_head_chunk_ms,
        )
        for row in input_rows
        for chunks in chunk_counts
    ]
    payload = {
        "config": {
            "kv_len": args.kv_len,
            "chunks": chunk_counts,
            "projection_weight": args.projection_weight,
            "qk_weight": args.qk_weight,
            "pv_weight": args.pv_weight,
            "merge_base_ms": args.merge_base_ms,
            "merge_per_head_chunk_ms": args.merge_per_head_chunk_ms,
            "artifacts": args.artifacts,
        },
        "rows": estimates,
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(payload["config"], indent=2, sort_keys=True))
        _print_table(estimates)


if __name__ == "__main__":
    main()
