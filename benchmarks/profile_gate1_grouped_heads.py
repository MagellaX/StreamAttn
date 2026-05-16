"""Simulate aggregate versus grouped-head Gate-1 routing.

This is intentionally a simulation, not a launch-splitting implementation. It
uses per-head active PV fractions and the current cost model to estimate whether
grouped sparse/dense head launches are worth building.
"""

import argparse
import json

from stream_attention.router import CostEntry


def _parse_floats(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def _head_costs(entry: CostEntry, active_fraction: float, heads: int):
    dense_h = entry.dense_ms / heads
    qk_h = entry.qk_only_ms / heads
    pv_h = entry.pv_ms / heads
    overhead_h = entry.predicate_overhead_ms / heads
    gate1_h = qk_h + active_fraction * pv_h + overhead_h
    return dense_h, gate1_h


def simulate_grouped_heads(
    *,
    entry: CostEntry,
    per_head_active: list[float],
    safety_margin: float,
    aggregate_threshold: float,
):
    heads = len(per_head_active)
    aggregate_active = sum(per_head_active) / heads
    dense_all_ms = entry.dense_ms
    gate1_all_ms = entry.predict_gate1_ms(aggregate_active)

    aggregate_auto_backend = (
        "gate1"
        if aggregate_active <= aggregate_threshold
        and entry.profitable(aggregate_active, safety_margin=safety_margin)
        else "dense"
    )
    aggregate_auto_ms = gate1_all_ms if aggregate_auto_backend == "gate1" else dense_all_ms

    sparse_heads = []
    dense_heads = []
    grouped_oracle_ms = 0.0
    for head_idx, active in enumerate(per_head_active):
        dense_h, gate1_h = _head_costs(entry, active, heads)
        if gate1_h * safety_margin < dense_h:
            sparse_heads.append(head_idx)
            grouped_oracle_ms += gate1_h
        else:
            dense_heads.append(head_idx)
            grouped_oracle_ms += dense_h

    return {
        "heads": heads,
        "per_head_active": per_head_active,
        "aggregate_active": aggregate_active,
        "dense_all_ms": dense_all_ms,
        "gate1_all_ms": gate1_all_ms,
        "aggregate_auto_backend": aggregate_auto_backend,
        "aggregate_auto_ms": aggregate_auto_ms,
        "grouped_oracle_ms": grouped_oracle_ms,
        "grouped_oracle_speedup_vs_dense": dense_all_ms / grouped_oracle_ms,
        "grouped_oracle_speedup_vs_aggregate_auto": aggregate_auto_ms
        / grouped_oracle_ms,
        "sparse_heads": sparse_heads,
        "dense_heads": dense_heads,
        "num_sparse_heads": len(sparse_heads),
        "num_dense_heads": len(dense_heads),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-head-active",
        default="0.05,0.08,0.80,0.92",
        help="Comma-separated active PV fraction per head.",
    )
    parser.add_argument("--dense-ms", type=float, default=0.10)
    parser.add_argument("--qk-only-ms", type=float, default=0.05)
    parser.add_argument("--predicate-overhead-ms", type=float, default=0.0)
    parser.add_argument("--safety-margin", type=float, default=1.10)
    parser.add_argument("--aggregate-threshold", type=float, default=0.30)
    args = parser.parse_args()

    entry = CostEntry(
        dense_ms=args.dense_ms,
        qk_only_ms=args.qk_only_ms,
        predicate_overhead_ms=args.predicate_overhead_ms,
    )
    result = simulate_grouped_heads(
        entry=entry,
        per_head_active=_parse_floats(args.per_head_active),
        safety_margin=args.safety_margin,
        aggregate_threshold=args.aggregate_threshold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
