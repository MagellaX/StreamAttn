"""Build an interpretable stress-route admissibility dataset.

The dataset joins:

* hybrid route-bundle safety results by prompt bucket
* route-conditioned attention/selector coverage by bucket, layer, and selector

It labels each bucket with the least conservative route class that passes the
strict gate. If no tested hybrid route passes, the label is ``exact_required``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_HYBRID_SUMMARY = (
    "artifacts/gate0/qwen25_3b_32k_b8_hybrid_routes/hybrid_stress_route_summary_32step.json"
)
DEFAULT_SELECTOR_COVERAGE = (
    "artifacts/gate0/qwen25_3b_32k_b8_dynamic_selector/"
    "risk_plan_l24_l26_l27_selectors_h100.json"
)

ROUTE_PRIORITY: Tuple[Tuple[str, str], ...] = (
    ("stress_l27_exact_l26_seed", "seed_ok"),
    ("stress_l27_exact_l26_dynamic_extreme4", "dynamic_ok"),
    ("stress_l27_exact_l26_dynamic_qk", "dynamic_ok"),
    ("stress_l26_l27_exact", "late_exact_required"),
    ("stress_l24_l26_l27_exact", "late_exact_required"),
)

FEATURE_FIELDS: Tuple[str, ...] = (
    "mass_omitted",
    "support_out_seed",
    "delta_collapse",
    "value_residual_ratio",
    "dense_vs_route_attention_js",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return ordered[left]
    weight = pos - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _bucket_gate_passed(
    row: Mapping[str, Any],
    *,
    max_kl: float,
    min_topk_overlap: int,
    max_logprob_delta: float,
) -> bool:
    return (
        _as_float(row.get("kl_max")) <= max_kl
        and _as_int(row.get("top1_changed_count")) == 0
        and _as_int(row.get("sample_token_changed_count")) == 0
        and _as_int(row.get("topk_overlap_min"), min_topk_overlap) >= min_topk_overlap
        and _as_float(row.get("reference_top1_logprob_delta_max_abs")) <= max_logprob_delta
    )


def _bucket_failure_score(
    row: Mapping[str, Any],
    *,
    max_kl: float,
    min_topk_overlap: int,
    max_logprob_delta: float,
) -> float:
    kl_ratio = _as_float(row.get("kl_max")) / max(max_kl, 1.0e-12)
    logprob_ratio = _as_float(row.get("reference_top1_logprob_delta_max_abs")) / max(
        max_logprob_delta, 1.0e-12
    )
    return (
        max(0.0, kl_ratio - 1.0)
        + max(0.0, logprob_ratio - 1.0)
        + 10.0 * _as_int(row.get("top1_changed_count"))
        + 10.0 * _as_int(row.get("sample_token_changed_count"))
        + 5.0
        * max(0, min_topk_overlap - _as_int(row.get("topk_overlap_min"), min_topk_overlap))
    )


def load_route_bucket_rows(
    hybrid_summary_path: Path,
    *,
    max_kl: float = 1.0e-4,
    min_topk_overlap: int = 4,
    max_logprob_delta: float = 2.0e-3,
) -> List[Dict[str, Any]]:
    summary = json.loads(hybrid_summary_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for route in summary.get("routes") or []:
        artifact = Path(str(route.get("artifact") or ""))
        if route.get("status") != "complete" or not artifact.exists():
            continue
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        safety = payload.get("safety") or {}
        for bucket, bucket_row in sorted((safety.get("by_prompt_bucket") or {}).items()):
            passed = _bucket_gate_passed(
                bucket_row,
                max_kl=max_kl,
                min_topk_overlap=min_topk_overlap,
                max_logprob_delta=max_logprob_delta,
            )
            rows.append(
                {
                    "route": route.get("name", ""),
                    "bucket": bucket,
                    "seed_layers": route.get("seed_layers", []),
                    "exact_layers": route.get("exact_layers", []),
                    "dynamic_layers": route.get("dynamic_layers", []),
                    "dynamic_profile": route.get("dynamic_profile", ""),
                    "route_speedup_vs_dense": _as_float(route.get("speedup_vs_dense_decode")),
                    "bucket_passed": passed,
                    "bucket_failure_score": _bucket_failure_score(
                        bucket_row,
                        max_kl=max_kl,
                        min_topk_overlap=min_topk_overlap,
                        max_logprob_delta=max_logprob_delta,
                    ),
                    "case_count": _as_int(bucket_row.get("case_count")),
                    "kl_max": _as_float(bucket_row.get("kl_max")),
                    "kl_p99": _as_float(bucket_row.get("kl_p99")),
                    "top1_changes": _as_int(bucket_row.get("top1_changed_count")),
                    "sample_changes": _as_int(bucket_row.get("sample_token_changed_count")),
                    "topk_overlap_min": _as_int(bucket_row.get("topk_overlap_min")),
                    "target_logprob_delta_max_abs": _as_float(
                        bucket_row.get("reference_top1_logprob_delta_max_abs")
                    ),
                }
            )
    return rows


def build_bucket_decisions(route_bucket_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_route_bucket = {
        (str(row.get("route")), str(row.get("bucket"))): row for row in route_bucket_rows
    }
    buckets = sorted({str(row.get("bucket")) for row in route_bucket_rows})
    decisions: List[Dict[str, Any]] = []
    for bucket in buckets:
        recommendation = "exact_required"
        recommended_route = ""
        for route_name, route_class in ROUTE_PRIORITY:
            row = by_route_bucket.get((route_name, bucket))
            if row and row.get("bucket_passed"):
                recommendation = route_class
                recommended_route = route_name
                break
        candidates = [
            row for row in route_bucket_rows if str(row.get("bucket")) == bucket
        ]
        best = (
            min(candidates, key=lambda row: _as_float(row.get("bucket_failure_score")))
            if candidates
            else None
        )
        decisions.append(
            {
                "bucket": bucket,
                "recommended_mode": recommendation,
                "recommended_route": recommended_route,
                "best_observed_route": str(best.get("route")) if best else "",
                "best_observed_route_passed": bool(best.get("bucket_passed")) if best else False,
                "best_observed_failure_score": _as_float(best.get("bucket_failure_score")) if best else 0.0,
            }
        )
    return decisions


def aggregate_selector_coverage(
    selector_coverage_path: Path,
    *,
    condition: str = "route_conditioned",
) -> List[Dict[str, Any]]:
    payload = json.loads(selector_coverage_path.read_text(encoding="utf-8"))
    groups: DefaultDict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in payload.get("rows") or []:
        if condition and str(row.get("condition")) != condition:
            continue
        key = (
            str(row.get("condition")),
            str(row.get("bucket")),
            _as_int(row.get("layer")),
            str(row.get("selector")),
        )
        groups[key].append(row)

    out: List[Dict[str, Any]] = []
    for (row_condition, bucket, layer, selector), rows in sorted(groups.items()):
        record: Dict[str, Any] = {
            "condition": row_condition,
            "bucket": bucket,
            "layer": layer,
            "selector": selector,
            "sample_count": len(rows),
        }
        for field in FEATURE_FIELDS:
            values = [_as_float(row.get(field)) for row in rows]
            record[f"{field}_mean"] = sum(values) / len(values) if values else 0.0
            record[f"{field}_p95"] = _percentile(values, 0.95)
            record[f"{field}_max"] = max(values) if values else 0.0
        record["selector_estimated_dot_token_ratio_mean"] = (
            sum(_as_float(row.get("selector_estimated_dot_token_ratio")) for row in rows)
            / len(rows)
            if rows
            else 0.0
        )
        out.append(record)
    return out


def build_admissibility_rows(
    bucket_decisions: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    decisions = {str(row.get("bucket")): row for row in bucket_decisions}
    out: List[Dict[str, Any]] = []
    for selector_row in selector_rows:
        bucket = str(selector_row.get("bucket"))
        decision = decisions.get(bucket)
        if decision is None:
            continue
        out.append({**selector_row, **decision})
    return out


def build_dataset(
    hybrid_summary_path: Path,
    selector_coverage_path: Path,
    *,
    max_kl: float = 1.0e-4,
    min_topk_overlap: int = 4,
    max_logprob_delta: float = 2.0e-3,
    condition: str = "route_conditioned",
) -> Dict[str, Any]:
    route_bucket_rows = load_route_bucket_rows(
        hybrid_summary_path,
        max_kl=max_kl,
        min_topk_overlap=min_topk_overlap,
        max_logprob_delta=max_logprob_delta,
    )
    bucket_decisions = build_bucket_decisions(route_bucket_rows)
    selector_rows = aggregate_selector_coverage(selector_coverage_path, condition=condition)
    admissibility_rows = build_admissibility_rows(bucket_decisions, selector_rows)
    return {
        "schema": "streamattn.stress_admissibility_dataset.v1",
        "sources": {
            "hybrid_summary": str(hybrid_summary_path),
            "selector_coverage": str(selector_coverage_path),
        },
        "gates": {
            "max_kl": max_kl,
            "min_topk_overlap": min_topk_overlap,
            "max_logprob_delta": max_logprob_delta,
        },
        "condition": condition,
        "route_bucket_rows": route_bucket_rows,
        "bucket_decisions": bucket_decisions,
        "selector_coverage_rows": selector_rows,
        "admissibility_rows": admissibility_rows,
    }


def print_summary(dataset: Mapping[str, Any]) -> None:
    print("Stress admissibility dataset")
    print(f"  condition: {dataset['condition']}")
    print(f"  route-bucket rows: {len(dataset['route_bucket_rows'])}")
    print(f"  selector coverage rows: {len(dataset['selector_coverage_rows'])}")
    print(f"  admissibility rows: {len(dataset['admissibility_rows'])}")
    print()
    for row in dataset["bucket_decisions"]:
        print(
            f"{row['bucket']}: {row['recommended_mode']} "
            f"(best observed: {row['best_observed_route']}, "
            f"score={row['best_observed_failure_score']:.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-summary", type=Path, default=Path(DEFAULT_HYBRID_SUMMARY))
    parser.add_argument("--selector-coverage", type=Path, default=Path(DEFAULT_SELECTOR_COVERAGE))
    parser.add_argument("--condition", default="route_conditioned")
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    parser.add_argument("--min-topk-overlap", type=int, default=4)
    parser.add_argument("--max-logprob-delta", type=float, default=2.0e-3)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    dataset = build_dataset(
        args.hybrid_summary,
        args.selector_coverage,
        max_kl=args.max_kl,
        min_topk_overlap=args.min_topk_overlap,
        max_logprob_delta=args.max_logprob_delta,
        condition=args.condition,
    )
    print_summary(dataset)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
