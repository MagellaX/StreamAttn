"""Compile margin-forensics artifacts into route-risk action plans.

This is an offline compiler pass.  It consumes route-bundle decode artifacts,
extracts failing rows from step rows or margin-forensics worst rows, and emits
the row/step plans and bucket recommendations needed for the next verifier or
dynamic-selector experiment.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


SCHEMA = "streamattn.seed_route_risk_plan.v1"


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _gate_from_payload(
    payload: Mapping[str, Any],
    *,
    max_kl: float,
    min_topk_overlap: int,
    max_logprob_delta: float,
) -> Dict[str, Any]:
    gates = ((payload.get("decision") or {}).get("gates") or {}) if isinstance(payload, Mapping) else {}
    return {
        "max_kl": _as_float(gates.get("max_kl"), max_kl),
        "min_topk_overlap": _as_int(gates.get("min_topk_overlap"), min_topk_overlap),
        "max_logprob_delta": _as_float(gates.get("max_logprob_delta"), max_logprob_delta),
    }


def row_failure_reasons(row: Mapping[str, Any], gate: Mapping[str, Any]) -> List[str]:
    """Return strict-gate failure reasons for a logit row."""

    reasons: List[str] = []
    max_kl = _as_float(gate.get("max_kl"), 1.0e-4)
    min_topk_overlap = _as_int(gate.get("min_topk_overlap"), 4)
    max_logprob_delta = _as_float(gate.get("max_logprob_delta"), 2.0e-3)
    if _bool(row.get("top1_changed")):
        reasons.append("top1_changed")
    if _bool(row.get("sample_token_changed")):
        reasons.append("sample_token_changed")
    if "kl_ref_to_candidate" in row and _as_float(row.get("kl_ref_to_candidate")) > max_kl:
        reasons.append("kl_over")
    if "topk_overlap" in row and _as_int(row.get("topk_overlap"), min_topk_overlap) < min_topk_overlap:
        reasons.append("topk_under")
    if (
        "reference_top1_logprob_delta" in row
        and abs(_as_float(row.get("reference_top1_logprob_delta"))) > max_logprob_delta
    ):
        reasons.append("logprob_delta_over")
    return reasons


def _iter_step_rows(payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    for step in payload.get("steps") or []:
        step_id = _as_int(step.get("step"), -1)
        if step_id < 0:
            continue
        for row in step.get("rows") or []:
            yield {**row, "step": step_id}


def _iter_worst_rows(payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    margin = payload.get("margin_forensics") or {}
    for row in margin.get("worst_rows") or []:
        if "step" in row and "row" in row:
            yield dict(row)


def _iter_safety_bucket_rows(payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    safety = payload.get("safety") or {}
    for bucket, summary in (safety.get("by_prompt_bucket") or {}).items():
        for key, reason in (
            ("first_divergence", "top1_changed"),
            ("first_sample_divergence", "sample_token_changed"),
        ):
            event = summary.get(key)
            if not event:
                continue
            step = _as_int(event.get("step"), 0)
            for row_id in event.get("rows") or []:
                yield {
                    "step": step,
                    "row": _as_int(row_id),
                    "prompt_bucket": bucket,
                    reason: True,
                }
        worst = summary.get("worst_case_by_kl") or {}
        if "row" in worst:
            yield {
                **worst,
                "step": _as_int(worst.get("step"), 0),
                "prompt_bucket": str(worst.get("prompt_bucket") or bucket),
            }


def _compact_row(row: Mapping[str, Any], reasons: Sequence[str]) -> Dict[str, Any]:
    return {
        "step": _as_int(row.get("step")),
        "row": _as_int(row.get("row")),
        "prompt_bucket": str(row.get("prompt_bucket") or row.get("prompt_kind") or ""),
        "prompt_id": str(row.get("prompt_id") or ""),
        "reasons": list(reasons),
        "kl_ref_to_candidate": _as_float(row.get("kl_ref_to_candidate")),
        "topk_overlap": _as_int(row.get("topk_overlap"), -1),
        "top1_changed": _bool(row.get("top1_changed")),
        "sample_token_changed": _bool(row.get("sample_token_changed")),
        "reference_top1_logprob_delta": _as_float(row.get("reference_top1_logprob_delta")),
        "reference_top1_margin": _as_float(row.get("reference_top1_margin")),
        "topk_lost_ref_mass": _as_float(row.get("topk_lost_ref_mass")),
        "topk_mass_retained_ref": _as_float(row.get("topk_mass_retained_ref")),
        "topk_boundary_logit_margin_ref": _as_float(row.get("topk_boundary_logit_margin_ref")),
    }


def _serialize_step_row_plan(plan: Mapping[int, Set[int]]) -> Dict[str, List[int]]:
    return {str(step): sorted(rows) for step, rows in sorted(plan.items())}


def format_step_row_plan(plan: Mapping[int, Set[int]]) -> str:
    parts = []
    for step, rows in sorted(plan.items()):
        if not rows:
            continue
        parts.append(f"{int(step)}:{','.join(str(row) for row in sorted(rows))}")
    return ";".join(parts)


def _normalize_existing_plan(plan: Any) -> Dict[str, Any]:
    if not isinstance(plan, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for step, rows in sorted(plan.items(), key=lambda item: _as_int(item[0])):
        if isinstance(rows, str):
            out[str(step)] = rows
        elif rows is None:
            out[str(step)] = "all"
        else:
            out[str(step)] = sorted(_as_int(row) for row in rows)
    return out


def _existing_step_row_plan_text(plan: Mapping[str, Any]) -> str:
    parts = []
    for step, rows in sorted(plan.items(), key=lambda item: _as_int(item[0])):
        if isinstance(rows, str) and rows in {"all", "*"}:
            parts.append(f"{step}:*")
        elif isinstance(rows, Sequence) and not isinstance(rows, str):
            parts.append(f"{step}:{','.join(str(row) for row in rows)}")
    return ";".join(parts)


def _summary_fails(row: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    return (
        _as_float(row.get("kl_max")) > _as_float(gate.get("max_kl"))
        or _as_int(row.get("topk_overlap_min"), _as_int(gate.get("min_topk_overlap"))) < _as_int(gate.get("min_topk_overlap"))
        or _as_int(row.get("top1_changed_count")) > 0
        or _as_int(row.get("sample_token_changed_count")) > 0
        or _as_float(row.get("target_logprob_delta_max_abs")) > _as_float(gate.get("max_logprob_delta"))
        or _as_float(row.get("reference_top1_logprob_delta_max_abs")) > _as_float(gate.get("max_logprob_delta"))
    )


def _bucket_risks(payload: Mapping[str, Any], gate: Mapping[str, Any], *, near_kl_ratio: float) -> List[Dict[str, Any]]:
    margin = payload.get("margin_forensics") or {}
    safety = payload.get("safety") or {}
    by_bucket = margin.get("by_bucket") or safety.get("by_prompt_bucket") or {}
    by_failure = margin.get("by_failure_bucket") or {}
    risks: List[Dict[str, Any]] = []
    for bucket, row in sorted(by_bucket.items()):
        kl_max = _as_float(row.get("kl_max"))
        topk_min = _as_int(row.get("topk_overlap_min"), 999)
        failed = bucket in by_failure or _summary_fails(row, gate)
        near_kl = _as_float(gate.get("max_kl")) > 0 and kl_max >= near_kl_ratio * _as_float(gate.get("max_kl"))
        near_topk = topk_min <= _as_int(gate.get("min_topk_overlap"))
        if not failed and not near_kl and not near_topk:
            continue
        if failed:
            level = "fail"
        elif near_kl or near_topk:
            level = "near_gate"
        else:
            level = "watch"
        failure_row = by_failure.get(bucket) or {}
        risks.append(
            {
                "bucket": bucket,
                "level": level,
                "kl_max": kl_max,
                "kl_p99": _as_float(row.get("kl_p99")),
                "topk_overlap_min": topk_min,
                "top1_changed_count": _as_int(row.get("top1_changed_count")),
                "sample_token_changed_count": _as_int(row.get("sample_token_changed_count")),
                "failure_count": _as_int(failure_row.get("count")),
                "topk_lost_ref_mass_max": _as_float(row.get("topk_lost_ref_mass_max")),
                "topk_boundary_logit_margin_min": _as_float(row.get("topk_boundary_logit_margin_min")),
            }
        )
    return risks


def _infer_recommendation(
    *,
    failure_rows: Sequence[Mapping[str, Any]],
    bucket_risks: Sequence[Mapping[str, Any]],
    decision_passed: Optional[bool],
    speedup: Optional[float],
    exact_refresh_backend: str,
    has_existing_exact_refresh_plan: bool,
) -> Dict[str, Any]:
    actions: List[str] = []
    status: str
    if failure_rows:
        status = "needs_risk_repair"
        actions.append("compile failing rows into verifier/dynamic-selector targets")
        token_failures = any(
            row.get("top1_changed") or row.get("sample_token_changed") for row in failure_rows
        )
        if token_failures:
            actions.append("bucket-gate exact or remove culprit layers before speed work")
        else:
            actions.append("prefer dynamic seed selector or bucket gating before exact-refresh production")
    elif decision_passed is True:
        status = "strict_pass"
        actions.append("safe enough for this gate; use as validation evidence")
    elif bucket_risks:
        status = "near_gate"
        actions.append("monitor near-gate buckets and expand stress coverage")
    else:
        status = "unknown"
        actions.append("artifact lacks enough row-level evidence")

    exact_backend_active = bool(exact_refresh_backend and exact_refresh_backend != "none" and has_existing_exact_refresh_plan)
    if exact_backend_active and speedup is not None and speedup < 1.0:
        actions.append("do not promote exact-refresh backend; safety proof only because speedup is below dense")
        if status == "strict_pass":
            status = "strict_pass_speed_negative"

    if speedup is not None and speedup >= 1.0 and not failure_rows and decision_passed is True:
        actions.append("eligible for product-speed comparison on validated buckets")

    return {
        "status": status,
        "actions": actions,
    }


def compile_artifact(
    payload: Mapping[str, Any],
    *,
    artifact: str = "",
    max_kl: float = 1.0e-4,
    min_topk_overlap: int = 4,
    max_logprob_delta: float = 2.0e-3,
    near_kl_ratio: float = 0.8,
    max_failure_rows: int = 48,
) -> Dict[str, Any]:
    gate = _gate_from_payload(
        payload,
        max_kl=max_kl,
        min_topk_overlap=min_topk_overlap,
        max_logprob_delta=max_logprob_delta,
    )
    rows = list(_iter_step_rows(payload))
    source = "steps" if rows else "margin_forensics.worst_rows"
    if not rows:
        rows = list(_iter_worst_rows(payload))
    if not rows:
        source = "safety.by_prompt_bucket"
        rows = list(_iter_safety_bucket_rows(payload))

    plan: Dict[int, Set[int]] = defaultdict(set)
    failures: List[Dict[str, Any]] = []
    for row in rows:
        reasons = row_failure_reasons(row, gate)
        if not reasons:
            continue
        step = _as_int(row.get("step"), -1)
        row_id = _as_int(row.get("row"), -1)
        if step >= 0 and row_id >= 0:
            plan[step].add(row_id)
        failures.append(_compact_row(row, reasons))

    failures.sort(
        key=lambda row: (
            int("top1_changed" not in row.get("reasons", []) and "sample_token_changed" not in row.get("reasons", [])),
            row.get("topk_overlap", 999),
            -float(row.get("kl_ref_to_candidate", 0.0)),
            float(row.get("topk_boundary_logit_margin_ref", 0.0)),
        )
    )
    failures = failures[: max(0, max_failure_rows)]

    route_bundle = payload.get("route_bundle") or {}
    existing_row_plan = _normalize_existing_plan(route_bundle.get("exact_refresh_row_plan") or {})
    existing_layer_plan = _normalize_existing_plan(route_bundle.get("exact_refresh_plan") or {})
    timing = payload.get("timing") or {}
    decision = payload.get("decision") or {}
    speedup = timing.get("speedup_vs_dense_decode")
    speedup_value = None if speedup is None else _as_float(speedup)
    bucket_risks = _bucket_risks(payload, gate, near_kl_ratio=near_kl_ratio)
    recommendation = _infer_recommendation(
        failure_rows=failures,
        bucket_risks=bucket_risks,
        decision_passed=decision.get("passed") if "passed" in decision else None,
        speedup=speedup_value,
        exact_refresh_backend=str(route_bundle.get("exact_refresh_backend") or ""),
        has_existing_exact_refresh_plan=bool(existing_row_plan or existing_layer_plan),
    )
    margin = payload.get("margin_forensics") or {}
    safety = payload.get("safety") or {}
    return {
        "artifact": artifact,
        "schema": payload.get("schema"),
        "shape": payload.get("shape") or {},
        "gate": gate,
        "timing": {
            "dense_decode_ms_per_token": timing.get("dense_decode_ms_per_token"),
            "streamattn_decode_ms_per_token": timing.get("streamattn_decode_ms_per_token"),
            "speedup_vs_dense_decode": speedup,
        },
        "decision": {
            "passed": decision.get("passed"),
            "kl_passed": decision.get("kl_passed"),
            "topk_passed": decision.get("topk_passed"),
            "top1_passed": decision.get("top1_passed"),
            "sample_passed": decision.get("sample_passed"),
        },
        "safety": {
            "kl_max": safety.get("kl_max"),
            "kl_p99": safety.get("kl_p99"),
            "topk_overlap_min": safety.get("topk_overlap_min"),
            "top1_changed_count": safety.get("top1_changed_count"),
            "sample_token_changed_count": safety.get("sample_token_changed_count"),
        },
        "margin_failure_count": margin.get("failure_count"),
        "failure_row_source": source,
        "compiled_step_row_plan": _serialize_step_row_plan(plan),
        "compiled_step_row_plan_text": format_step_row_plan(plan),
        "failure_rows": failures,
        "bucket_risks": bucket_risks,
        "route_bundle": {
            "layers": route_bundle.get("layers") or [],
            "exact_refresh_backend": route_bundle.get("exact_refresh_backend"),
            "exact_refresh_plan": existing_layer_plan,
            "exact_refresh_row_plan": existing_row_plan,
            "exact_refresh_row_plan_text": _existing_step_row_plan_text(existing_row_plan),
            "dynamic_selector_layers": route_bundle.get("dynamic_selector_layers") or [],
            "dynamic_selector_profile": route_bundle.get("dynamic_selector_profile") or "",
        },
        "recommendation": recommendation,
    }


def compile_artifacts(
    artifacts: Sequence[Path],
    *,
    max_kl: float = 1.0e-4,
    min_topk_overlap: int = 4,
    max_logprob_delta: float = 2.0e-3,
    near_kl_ratio: float = 0.8,
    max_failure_rows: int = 48,
) -> Dict[str, Any]:
    summaries = []
    combined_plan: Dict[int, Set[int]] = defaultdict(set)
    for artifact in artifacts:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        summary = compile_artifact(
            payload,
            artifact=str(artifact),
            max_kl=max_kl,
            min_topk_overlap=min_topk_overlap,
            max_logprob_delta=max_logprob_delta,
            near_kl_ratio=near_kl_ratio,
            max_failure_rows=max_failure_rows,
        )
        summaries.append(summary)
        for step, rows in summary["compiled_step_row_plan"].items():
            combined_plan[_as_int(step)].update(_as_int(row) for row in rows)
    return {
        "schema": SCHEMA,
        "gate_defaults": {
            "max_kl": max_kl,
            "min_topk_overlap": min_topk_overlap,
            "max_logprob_delta": max_logprob_delta,
            "near_kl_ratio": near_kl_ratio,
        },
        "artifacts": summaries,
        "combined_compiled_step_row_plan": _serialize_step_row_plan(combined_plan),
        "combined_compiled_step_row_plan_text": format_step_row_plan(combined_plan),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def print_plan(plan: Mapping[str, Any]) -> None:
    print("Seed route risk plan")
    print(f"  artifacts: {len(plan.get('artifacts') or [])}")
    combined = plan.get("combined_compiled_step_row_plan_text") or ""
    print(f"  combined row plan: {combined or '-'}")
    for item in plan.get("artifacts") or []:
        print()
        print(Path(item.get("artifact") or "-").name)
        print(f"  status:      {item['recommendation']['status']}")
        print(f"  speedup:     {_fmt((item.get('timing') or {}).get('speedup_vs_dense_decode'))}")
        print(f"  strict pass: {(item.get('decision') or {}).get('passed')}")
        print(
            "  KL/top-k:    "
            f"{_fmt((item.get('safety') or {}).get('kl_max'))} / "
            f"{(item.get('safety') or {}).get('topk_overlap_min')}"
        )
        print(f"  row plan:    {item.get('compiled_step_row_plan_text') or '-'}")
        existing = (item.get("route_bundle") or {}).get("exact_refresh_row_plan_text") or ""
        if existing:
            print(f"  exact rows:  {existing}")
        risks = item.get("bucket_risks") or []
        if risks:
            brief = ", ".join(f"{row['bucket']}:{row['level']}" for row in risks[:6])
            print(f"  buckets:     {brief}")
        for action in (item.get("recommendation") or {}).get("actions") or []:
            print(f"  action:      {action}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    parser.add_argument("--min-topk-overlap", type=int, default=4)
    parser.add_argument("--max-logprob-delta", type=float, default=2.0e-3)
    parser.add_argument("--near-kl-ratio", type=float, default=0.8)
    parser.add_argument("--max-failure-rows", type=int, default=48)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    plan = compile_artifacts(
        args.artifacts,
        max_kl=args.max_kl,
        min_topk_overlap=args.min_topk_overlap,
        max_logprob_delta=args.max_logprob_delta,
        near_kl_ratio=args.near_kl_ratio,
        max_failure_rows=args.max_failure_rows,
    )
    print_plan(plan)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
