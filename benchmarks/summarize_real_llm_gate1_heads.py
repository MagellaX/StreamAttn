"""Summarize real-LLM Gate-1 per-head telemetry JSON.

The profiler emits raw layer/head rows. This script turns them into decision
evidence:

* which heads are sparse under mass/value-bound predicates;
* whether mass is more aggressive than value-bound;
* whether Q heads sharing a KV head behave differently;
* whether grouped-head routing has enough oracle upside to justify a
  selected-head kernel.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


def _number(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    vals = sorted(float(value) for value in values if value is not None)
    if not vals:
        return None
    if percentile <= 0.0:
        return vals[0]
    if percentile >= 100.0:
        return vals[-1]
    pos = (len(vals) - 1) * percentile / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    weight = pos - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def _stats(values: Iterable[float]) -> dict:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": len(vals),
        "min": min(vals),
        "mean": _mean(vals),
        "p50": _percentile(vals, 50.0),
        "p90": _percentile(vals, 90.0),
        "max": max(vals),
    }


def _all_head_rows(payload: dict) -> list[dict]:
    rows = []
    for result in payload.get("results", []):
        if result.get("skipped"):
            continue
        rows.extend(result.get("per_head", []))
    return rows


def _group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return dict(groups)


def _head_profiles(rows: list[dict]) -> list[dict]:
    profiles = []
    for key, group in sorted(
        _group_rows(rows, ("layer_id", "module_name", "head_id")).items()
    ):
        layer_id, module_name, head_id = key
        mass_values = [_number(row.get("active_mass")) for row in group]
        value_values = [_number(row.get("active_value_bound")) for row in group]
        kv_ids = sorted({row.get("kv_head_id") for row in group})
        q_groups = sorted({row.get("q_group_id") for row in group})
        profiles.append(
            {
                "layer_id": layer_id,
                "module_name": module_name,
                "head_id": head_id,
                "kv_head_id": kv_ids[0] if len(kv_ids) == 1 else kv_ids,
                "q_group_id": q_groups[0] if len(q_groups) == 1 else q_groups,
                "mass_active": _stats(mass_values),
                "value_bound_active": _stats(value_values),
                "mass_p99_relative_token_error_max": max(
                    (
                        _number(row.get("mass_p99_relative_token_error")) or 0.0
                        for row in group
                    ),
                    default=None,
                ),
                "value_bound_p99_relative_token_error_max": max(
                    (
                        _number(row.get("value_bound_p99_relative_token_error")) or 0.0
                        for row in group
                    ),
                    default=None,
                ),
            }
        )
    return profiles


def _estimate_grouped_oracle(
    active_values: list[float],
    *,
    dense_ms: float,
    qk_only_ms: float,
    predicate_overhead_ms: float,
    safety_margin: float,
    aggregate_threshold: float,
    head_index_tax_ms: float,
) -> dict:
    heads = len(active_values)
    if heads == 0:
        return {}
    pv_ms = max(0.0, dense_ms - qk_only_ms)
    aggregate_active = sum(active_values) / heads
    gate1_all_ms = qk_only_ms + aggregate_active * pv_ms + predicate_overhead_ms
    dense_h = dense_ms / heads
    qk_h = qk_only_ms / heads
    pv_h = pv_ms / heads
    overhead_h = predicate_overhead_ms / heads

    sparse_heads = []
    dense_heads = []
    grouped_ms = 0.0
    for head_idx, active in enumerate(active_values):
        gate1_h = qk_h + active * pv_h + overhead_h
        if gate1_h * safety_margin < dense_h:
            sparse_heads.append(head_idx)
            grouped_ms += gate1_h
        else:
            dense_heads.append(head_idx)
            grouped_ms += dense_h

    aggregate_backend = (
        "gate1"
        if aggregate_active <= aggregate_threshold
        and gate1_all_ms * safety_margin < dense_ms
        else "dense"
    )
    aggregate_ms = gate1_all_ms if aggregate_backend == "gate1" else dense_ms
    with_tax = grouped_ms + head_index_tax_ms
    return {
        "aggregate_active": aggregate_active,
        "aggregate_backend": aggregate_backend,
        "aggregate_ms": aggregate_ms,
        "dense_all_ms": dense_ms,
        "gate1_all_ms": gate1_all_ms,
        "grouped_oracle_ms": grouped_ms,
        "grouped_oracle_with_head_index_tax_ms": with_tax,
        "grouped_speedup_vs_aggregate": aggregate_ms / grouped_ms
        if grouped_ms > 0.0
        else None,
        "grouped_with_tax_speedup_vs_aggregate": aggregate_ms / with_tax
        if with_tax > 0.0
        else None,
        "sparse_heads": sparse_heads,
        "dense_heads": dense_heads,
        "num_sparse_heads": len(sparse_heads),
        "num_dense_heads": len(dense_heads),
    }


def _kv_group_spread(rows: list[dict]) -> list[dict]:
    profiles = []
    for key, group in sorted(
        _group_rows(rows, ("layer_id", "module_name", "kv_head_id")).items()
    ):
        layer_id, module_name, kv_head_id = key
        by_q = defaultdict(list)
        for row in group:
            by_q[row.get("q_group_id")].append(_number(row.get("active_mass")))
        q_means = {
            q_group_id: _mean(value for value in values if value is not None)
            for q_group_id, values in by_q.items()
        }
        valid = [value for value in q_means.values() if value is not None]
        profiles.append(
            {
                "layer_id": layer_id,
                "module_name": module_name,
                "kv_head_id": kv_head_id,
                "q_group_mass_active_mean": q_means,
                "within_kv_spread": max(valid) - min(valid) if valid else None,
            }
        )
    return profiles


def _layer_summaries(rows: list[dict], args) -> list[dict]:
    summaries = []
    for key, group in sorted(_group_rows(rows, ("layer_id", "module_name")).items()):
        layer_id, module_name = key
        mass_values = [_number(row.get("active_mass")) for row in group]
        value_values = [_number(row.get("active_value_bound")) for row in group]
        valid_mass = [value for value in mass_values if value is not None]
        valid_value = [value for value in value_values if value is not None]
        diffs = [
            value - mass
            for mass, value in zip(mass_values, value_values)
            if mass is not None and value is not None
        ]
        aggressive_disagreements = [
            diff for diff in diffs if diff > args.disagreement_threshold
        ]
        any_disagreements = [
            diff for diff in diffs if abs(diff) > args.disagreement_threshold
        ]
        head_mean_values = []
        for _, head_rows in _group_rows(group, ("head_id",)).items():
            head_mean = _mean(
                _number(row.get("active_mass"))
                for row in head_rows
                if row.get("active_mass") is not None
            )
            if head_mean is not None:
                head_mean_values.append(head_mean)

        grouped = _estimate_grouped_oracle(
            head_mean_values,
            dense_ms=args.dense_ms,
            qk_only_ms=args.qk_only_ms,
            predicate_overhead_ms=args.predicate_overhead_ms,
            safety_margin=args.safety_margin,
            aggregate_threshold=args.aggregate_threshold,
            head_index_tax_ms=args.head_index_tax_ms,
        )
        sparse_025 = sum(1 for value in head_mean_values if value < 0.25)
        sparse_050 = sum(1 for value in head_mean_values if value < 0.50)
        kv_spreads = [
            profile["within_kv_spread"]
            for profile in _kv_group_spread(group)
            if profile["within_kv_spread"] is not None
        ]
        recommendation = _recommendation(
            grouped=grouped,
            sparse_025=sparse_025,
            heads=len(head_mean_values),
            aggressive_disagreement_rate=(
                len(aggressive_disagreements) / len(diffs) if diffs else 0.0
            ),
            max_kv_spread=max(kv_spreads) if kv_spreads else 0.0,
            args=args,
        )
        summaries.append(
            {
                "layer_id": layer_id,
                "module_name": module_name,
                "heads": len(head_mean_values),
                "mass_active": _stats(valid_mass),
                "value_bound_active": _stats(valid_value),
                "heads_mass_active_lt_0_25": sparse_025,
                "heads_mass_active_lt_0_50": sparse_050,
                "head_fraction_mass_active_lt_0_25": sparse_025 / len(head_mean_values)
                if head_mean_values
                else 0.0,
                "head_fraction_mass_active_lt_0_50": sparse_050 / len(head_mean_values)
                if head_mean_values
                else 0.0,
                "mass_more_aggressive_disagreement_rate": (
                    len(aggressive_disagreements) / len(diffs) if diffs else 0.0
                ),
                "any_disagreement_rate": len(any_disagreements) / len(diffs)
                if diffs
                else 0.0,
                "max_within_kv_active_spread": max(kv_spreads) if kv_spreads else 0.0,
                "grouped_oracle": grouped,
                "recommendation": recommendation,
            }
        )
    return summaries


def _recommendation(
    *,
    grouped: dict,
    sparse_025: int,
    heads: int,
    aggressive_disagreement_rate: float,
    max_kv_spread: float,
    args,
) -> str:
    if heads == 0:
        return "collect_more_data"
    grouped_speedup = grouped.get("grouped_with_tax_speedup_vs_aggregate") or 1.0
    sparse_fraction = sparse_025 / heads
    if aggressive_disagreement_rate >= args.value_bound_disagreement_rate:
        return "calibrate_value_bound_or_lower_mass_budget"
    if grouped_speedup >= args.grouped_speedup_threshold and sparse_fraction >= 0.25:
        if max_kv_spread >= args.kv_spread_threshold:
            return "build_q_head_index_grouped_gate1"
        return "build_grouped_gate1"
    if sparse_fraction >= 0.25:
        return "route_mass_gate1_for_sparse_heads"
    return "prioritize_decode_or_gate0"


def summarize(payload: dict, args) -> dict:
    rows = _all_head_rows(payload)
    skipped = [result for result in payload.get("results", []) if result.get("skipped")]
    layer_summaries = _layer_summaries(rows, args)
    head_profiles = _head_profiles(rows)
    kv_profiles = _kv_group_spread(rows)
    recommendations = defaultdict(int)
    for layer in layer_summaries:
        recommendations[layer["recommendation"]] += 1
    all_mass = [_number(row.get("active_mass")) for row in rows]
    return {
        "model": payload.get("model"),
        "prompts": payload.get("prompts"),
        "block_size": payload.get("block_size"),
        "tile_size_q": payload.get("tile_size_q"),
        "error_budget": payload.get("error_budget"),
        "num_profiled_head_rows": len(rows),
        "num_skipped_layers": len(skipped),
        "overall_mass_active": _stats(all_mass),
        "overall_head_fraction_mass_active_lt_0_25": (
            sum(1 for value in all_mass if value is not None and value < 0.25)
            / len([value for value in all_mass if value is not None])
            if any(value is not None for value in all_mass)
            else 0.0
        ),
        "recommendation_counts": dict(sorted(recommendations.items())),
        "layers": layer_summaries,
        "heads": head_profiles,
        "kv_groups": kv_profiles,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--dense-ms", type=float, default=1.0)
    parser.add_argument("--qk-only-ms", type=float, default=0.60)
    parser.add_argument("--predicate-overhead-ms", type=float, default=0.0)
    parser.add_argument("--safety-margin", type=float, default=1.10)
    parser.add_argument("--aggregate-threshold", type=float, default=0.30)
    parser.add_argument("--head-index-tax-ms", type=float, default=0.0)
    parser.add_argument("--disagreement-threshold", type=float, default=0.05)
    parser.add_argument("--grouped-speedup-threshold", type=float, default=1.15)
    parser.add_argument("--kv-spread-threshold", type=float, default=0.25)
    parser.add_argument("--value-bound-disagreement-rate", type=float, default=0.05)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8-sig"))
    summary = summarize(payload, args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
