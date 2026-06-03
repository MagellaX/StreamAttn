"""Summarize late-layer hybrid stress route matrix results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_PLAN_JSON = (
    "artifacts/gate0/qwen25_3b_32k_b8_hybrid_routes/hybrid_stress_route_plan_32step.json"
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


def _command_output_json(command: Sequence[str]) -> Optional[Path]:
    try:
        idx = list(command).index("--output-json")
    except ValueError:
        return None
    if idx + 1 >= len(command):
        return None
    return Path(command[idx + 1])


def _worst_bucket(safety: Dict[str, Any]) -> Dict[str, Any]:
    buckets = safety.get("by_prompt_bucket") or {}
    if not isinstance(buckets, dict) or not buckets:
        return {}
    name, payload = max(
        buckets.items(),
        key=lambda item: _as_float((item[1] or {}).get("kl_max")),
    )
    payload = payload or {}
    return {
        "bucket": name,
        "kl_max": _as_float(payload.get("kl_max")),
        "kl_p99": _as_float(payload.get("kl_p99")),
        "topk_overlap_min": _as_int(payload.get("topk_overlap_min")),
        "target_logprob_delta_max_abs": _as_float(
            payload.get("reference_top1_logprob_delta_max_abs")
        ),
    }


def _failure_reasons(
    *,
    kl_max: float,
    top1_changes: int,
    sample_changes: int,
    topk_overlap_min: int,
    target_logprob_delta_max_abs: float,
    speedup: float,
    max_kl: float,
    min_topk_overlap: int,
    max_logprob_delta: float,
    min_speedup: float,
) -> List[str]:
    reasons: List[str] = []
    if kl_max > max_kl:
        reasons.append("kl_failed")
    if top1_changes != 0:
        reasons.append("top1_failed")
    if sample_changes != 0:
        reasons.append("sample_failed")
    if topk_overlap_min < min_topk_overlap:
        reasons.append("topk_failed")
    if target_logprob_delta_max_abs > max_logprob_delta:
        reasons.append("logprob_failed")
    if speedup < min_speedup:
        reasons.append("runtime_slow")
    return reasons


def summarize_route(
    route: Dict[str, Any],
    *,
    max_kl: float,
    min_topk_overlap: int,
    max_logprob_delta: float,
    min_speedup: float,
) -> Dict[str, Any]:
    output_path = _command_output_json(route.get("command") or [])
    row: Dict[str, Any] = {
        "name": route.get("name", ""),
        "seed_layers": route.get("seed_layers", []),
        "exact_layers": route.get("exact_layers", []),
        "dynamic_layers": route.get("dynamic_layers", []),
        "dynamic_profile": route.get("dynamic_profile", ""),
        "artifact": str(output_path) if output_path else "",
        "status": "missing",
        "strict_passed": False,
        "failure_reasons": ["missing_artifact"],
    }
    if output_path is None:
        return row
    if not output_path.exists():
        return row

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    safety = payload.get("safety") or {}
    timing = payload.get("timing") or {}
    decision = payload.get("decision") or {}
    speedup = _as_float(timing.get("speedup_vs_dense_decode"))
    kl_max = _as_float(safety.get("kl_max"))
    top1_changes = _as_int(safety.get("top1_changed_count"))
    sample_changes = _as_int(safety.get("sample_token_changed_count"))
    topk_overlap_min = _as_int(safety.get("topk_overlap_min"))
    target_delta = _as_float(safety.get("reference_top1_logprob_delta_max_abs"))
    failure_reasons = _failure_reasons(
        kl_max=kl_max,
        top1_changes=top1_changes,
        sample_changes=sample_changes,
        topk_overlap_min=topk_overlap_min,
        target_logprob_delta_max_abs=target_delta,
        speedup=speedup,
        max_kl=max_kl,
        min_topk_overlap=min_topk_overlap,
        max_logprob_delta=max_logprob_delta,
        min_speedup=min_speedup,
    )
    row.update(
        {
            "status": "complete",
            "decision_passed": bool(decision.get("passed")),
            "strict_passed": not failure_reasons,
            "failure_reasons": failure_reasons,
            "speedup_vs_dense_decode": speedup,
            "dense_decode_ms_per_token": _as_float(timing.get("dense_decode_ms_per_token")),
            "streamattn_decode_ms_per_token": _as_float(
                timing.get("streamattn_decode_ms_per_token")
            ),
            "case_count": _as_int(safety.get("case_count")),
            "kl_max": kl_max,
            "kl_p99": _as_float(safety.get("kl_p99")),
            "top1_changes": top1_changes,
            "sample_changes": sample_changes,
            "topk_overlap_min": topk_overlap_min,
            "target_logprob_delta_max_abs": target_delta,
            "worst_bucket": _worst_bucket(safety),
        }
    )
    return row


def _route_by_name(rows: Iterable[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("name") == name:
            return row
    return None


def promotion_decision(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the L27/L26 stress-route decision tree."""

    if not rows:
        return {
            "decision": "no_results",
            "route": "",
            "reason": "No routes were summarized.",
        }

    l27_seed = _route_by_name(rows, "stress_l27_exact_l26_seed")
    if l27_seed and l27_seed.get("strict_passed"):
        return {
            "decision": "promote_l27_exact_l26_seed",
            "route": l27_seed["name"],
            "reason": "L27 exact is sufficient; L26 remains seed-only.",
        }

    dynamic_candidates = [
        row
        for row in (
            _route_by_name(rows, "stress_l27_exact_l26_dynamic_extreme4"),
            _route_by_name(rows, "stress_l27_exact_l26_dynamic_qk"),
        )
        if row and row.get("strict_passed")
    ]
    if dynamic_candidates:
        best = max(dynamic_candidates, key=lambda row: _as_float(row.get("speedup_vs_dense_decode")))
        return {
            "decision": "promote_l27_exact_l26_dynamic",
            "route": best["name"],
            "reason": "L26 needs dynamic seed selection while L27 stays exact.",
        }

    l26_l27_exact = _route_by_name(rows, "stress_l26_l27_exact")
    if l26_l27_exact and l26_l27_exact.get("strict_passed"):
        return {
            "decision": "promote_l26_l27_exact",
            "route": l26_l27_exact["name"],
            "reason": "Stress rows require L26 and L27 exact.",
        }

    late_exact = _route_by_name(rows, "stress_l24_l26_l27_exact")
    if late_exact and late_exact.get("strict_passed"):
        return {
            "decision": "promote_l24_l26_l27_exact",
            "route": late_exact["name"],
            "reason": "Stress rows require conservative late-block exact routing.",
        }

    complete = [row for row in rows if row.get("status") == "complete"]
    missing = [row["name"] for row in rows if row.get("status") == "missing"]
    if missing and not complete:
        return {
            "decision": "await_results",
            "route": "",
            "reason": "No hybrid route artifacts exist yet.",
            "missing_routes": missing,
        }
    if missing:
        return {
            "decision": "await_remaining_results",
            "route": "",
            "reason": "Some hybrid route artifacts are still missing.",
            "missing_routes": missing,
        }
    return {
        "decision": "no_stress_promotion",
        "route": "",
        "reason": "No hybrid stress route passed the strict gate.",
    }


def summarize_plan(
    plan_path: Path,
    *,
    max_kl: float = 1e-4,
    min_topk_overlap: int = 4,
    max_logprob_delta: float = 2e-3,
    min_speedup: float = 1.0,
) -> Dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = [
        summarize_route(
            route,
            max_kl=max_kl,
            min_topk_overlap=min_topk_overlap,
            max_logprob_delta=max_logprob_delta,
            min_speedup=min_speedup,
        )
        for route in plan.get("routes", [])
    ]
    complete = [row for row in rows if row.get("status") == "complete"]
    passed = [row for row in complete if row.get("strict_passed")]
    best_strict = (
        max(passed, key=lambda row: _as_float(row.get("speedup_vs_dense_decode")))
        if passed
        else None
    )
    best_speed = (
        max(complete, key=lambda row: _as_float(row.get("speedup_vs_dense_decode")))
        if complete
        else None
    )
    return {
        "schema": "streamattn.hybrid_stress_route_summary.v1",
        "plan": str(plan_path),
        "gates": {
            "max_kl": max_kl,
            "min_topk_overlap": min_topk_overlap,
            "max_logprob_delta": max_logprob_delta,
            "min_speedup": min_speedup,
        },
        "routes": rows,
        "best_strict_route": best_strict["name"] if best_strict else "",
        "best_speed_route": best_speed["name"] if best_speed else "",
        "promotion": promotion_decision(rows),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("Hybrid stress route summary")
    print(f"  plan: {summary['plan']}")
    print(f"  promotion: {summary['promotion']['decision']}")
    if summary["promotion"].get("route"):
        print(f"  route: {summary['promotion']['route']}")
    print(f"  reason: {summary['promotion']['reason']}")
    print()
    for row in summary["routes"]:
        status = row["status"]
        print(row["name"])
        print(f"  status: {status}")
        print(f"  seed:   {row.get('seed_layers', [])}")
        print(f"  exact:  {row.get('exact_layers', [])}")
        if row.get("dynamic_layers"):
            print(f"  dynamic: {row['dynamic_layers']} profile={row.get('dynamic_profile', '')}")
        print(f"  artifact: {row.get('artifact', '')}")
        if status == "complete":
            print(f"  strict_passed: {row['strict_passed']}")
            print(f"  failure_reasons: {row['failure_reasons']}")
            print(f"  speedup: {row['speedup_vs_dense_decode']:.5f}x")
            print(f"  KL max/p99: {row['kl_max']:.6g} / {row['kl_p99']:.6g}")
            print(f"  top1/sample changes: {row['top1_changes']} / {row['sample_changes']}")
            print(f"  top5 min: {row['topk_overlap_min']}")
            print(f"  target logprob delta max: {row['target_logprob_delta_max_abs']:.6g}")
            if row.get("worst_bucket"):
                print(f"  worst bucket: {row['worst_bucket']}")
        else:
            print(f"  failure_reasons: {row['failure_reasons']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", type=Path, default=Path(DEFAULT_PLAN_JSON))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-kl", type=float, default=1e-4)
    parser.add_argument("--min-topk-overlap", type=int, default=4)
    parser.add_argument("--max-logprob-delta", type=float, default=2e-3)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    args = parser.parse_args()

    summary = summarize_plan(
        args.plan_json,
        max_kl=args.max_kl,
        min_topk_overlap=args.min_topk_overlap,
        max_logprob_delta=args.max_logprob_delta,
        min_speedup=args.min_speedup,
    )
    print_summary(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
