"""Summarize prompt-aware attention coverage diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _recommendation(row: Dict[str, Any]) -> str:
    support_out = _as_float(row.get("support_out_seed"))
    omitted = _as_float(row.get("mass_omitted"))
    collapse = _as_float(row.get("delta_collapse"))
    value_residual = _as_float(row.get("value_residual_ratio"))
    js = _as_float(row.get("dense_vs_route_attention_js"))
    if support_out >= 0.05 or (omitted >= 0.25 and collapse >= 0.02):
        return "coverage_repair"
    if value_residual >= 0.35:
        return "value_sensitive_repair"
    if js >= 0.02:
        return "composition_repair"
    return "exact_or_margin_gate"


def _metric_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def values(name: str) -> List[float]:
        return [_as_float(row.get(name)) for row in rows]

    recommendations = defaultdict(int)
    for row in rows:
        recommendations[_recommendation(row)] += 1
    top_recommendation = max(recommendations.items(), key=lambda item: item[1])[0] if recommendations else ""
    return {
        "count": len(rows),
        "mass_omitted_p50": _percentile(values("mass_omitted"), 0.50),
        "mass_omitted_p95": _percentile(values("mass_omitted"), 0.95),
        "support_out_seed_p95": _percentile(values("support_out_seed"), 0.95),
        "delta_collapse_p95": _percentile(values("delta_collapse"), 0.95),
        "value_residual_ratio_p95": _percentile(values("value_residual_ratio"), 0.95),
        "dense_vs_route_attention_js_p95": _percentile(values("dense_vs_route_attention_js"), 0.95),
        "top_recommendation": top_recommendation,
        "recommendation_counts": dict(sorted(recommendations.items())),
    }


def _group(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key, "") for key in keys)].append(row)
    summary = []
    for key_values, subset in sorted(buckets.items()):
        item = {key: value for key, value in zip(keys, key_values)}
        item.update(_metric_summary(subset))
        summary.append(item)
    return summary


def summarize_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    return {
        "schema": "streamattn.seed_policy_attention_coverage_summary.v1",
        "source": str(path),
        "model": payload.get("model"),
        "target_layers": payload.get("target_layers", []),
        "routed_layers": payload.get("routed_layers", []),
        "target_buckets": payload.get("target_buckets", []),
        "selector_profiles": payload.get("selector_profiles", []),
        "capture_steps": payload.get("capture_steps", []),
        "overall": _metric_summary(rows),
        "by_selector": _group(rows, ("selector",)) if rows and "selector" in rows[0] else [],
        "by_selector_bucket": _group(rows, ("selector", "bucket")) if rows and "selector" in rows[0] else [],
        "by_selector_layer_bucket": (
            _group(rows, ("selector", "layer", "bucket")) if rows and "selector" in rows[0] else []
        ),
        "by_layer_bucket_condition": _group(rows, ("layer", "bucket", "condition")),
        "by_layer_bucket": _group(rows, ("layer", "bucket")),
        "by_bucket_condition": _group(rows, ("bucket", "condition")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Attention coverage JSON output.")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    summary = summarize_payload(Path(args.json))
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
