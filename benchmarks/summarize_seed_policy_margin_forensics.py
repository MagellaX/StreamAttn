"""Summarize margin-aware safety forensics from route-bundle decode artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _token_list(values: Iterable[Any], *, limit: int = 8) -> str:
    items = list(values)[:limit]
    return "[" + ",".join(str(item) for item in items) + "]"


def summarize_payload(payload: Dict[str, Any], *, max_rows: int = 12) -> Dict[str, Any]:
    margin = payload.get("margin_forensics") or {}
    if not margin:
        raise ValueError("artifact has no margin_forensics section; rerun with --margin-forensics")
    safety = payload.get("safety") or {}
    decision = payload.get("decision") or {}
    timing = payload.get("timing") or {}
    return {
        "artifact_schema": payload.get("schema"),
        "shape": payload.get("shape") or {},
        "timing": {
            "dense_decode_ms_per_token": timing.get("dense_decode_ms_per_token"),
            "streamattn_decode_ms_per_token": timing.get("streamattn_decode_ms_per_token"),
            "speedup_vs_dense_decode": timing.get("speedup_vs_dense_decode"),
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
        "margin": {
            "failure_count": margin.get("failure_count"),
            "kl_over_count": margin.get("kl_over_count"),
            "topk_under_count": margin.get("topk_under_count"),
            "interpretation": margin.get("interpretation") or {},
            "failure_summary": margin.get("failure_summary") or {},
            "by_failure_bucket": margin.get("by_failure_bucket") or {},
            "worst_rows": (margin.get("worst_rows") or [])[:max_rows],
        },
    }


def print_summary(summary: Dict[str, Any]) -> None:
    shape = summary["shape"]
    timing = summary["timing"]
    decision = summary["decision"]
    safety = summary["safety"]
    margin = summary["margin"]
    interp = margin["interpretation"]
    failure = margin["failure_summary"]

    print("Margin forensics")
    print(f"  steps/batch:        {shape.get('steps')} / {shape.get('batch')}")
    print(f"  speedup:            {_fmt(timing.get('speedup_vs_dense_decode'))}")
    print(f"  strict passed:      {decision.get('passed')}")
    print(f"  KL max / p99:       {_fmt(safety.get('kl_max'))} / {_fmt(safety.get('kl_p99'))}")
    print(f"  top-k min:          {safety.get('topk_overlap_min')}")
    print(f"  top1/sample chg:    {safety.get('top1_changed_count')} / {safety.get('sample_token_changed_count')}")
    print(f"  failure count:      {margin.get('failure_count')}")
    print(f"  KL/top-k failures:  {margin.get('kl_over_count')} / {margin.get('topk_under_count')}")
    print(f"  token stable:       {interp.get('token_stable')}")
    print(f"  p99 KL passed:      {interp.get('p99_kl_passed')}")
    print(
        "  retained mass min: "
        f"{_fmt(failure.get('topk_mass_retained_min'))}; "
        f"lost ref mass max: {_fmt(failure.get('topk_lost_ref_mass_max'))}"
    )
    print(f"  boundary margin min:{_fmt(failure.get('topk_boundary_logit_margin_min'))}")

    by_bucket = margin.get("by_failure_bucket") or {}
    if by_bucket:
        print("\nFailures by bucket")
        for bucket, row in by_bucket.items():
            print(
                f"  {bucket:12} count={row.get('count'):>4} "
                f"kl_max={_fmt(row.get('kl_max')):>10} "
                f"topk_min={row.get('topk_overlap_min')} "
                f"retained_min={_fmt(row.get('topk_mass_retained_min')):>10} "
                f"lost_max={_fmt(row.get('topk_lost_ref_mass_max')):>10}"
            )

    worst_rows = margin.get("worst_rows") or []
    if worst_rows:
        print("\nWorst rows")
        for row in worst_rows:
            print(
                f"  step={row.get('step'):>3} row={row.get('row')} "
                f"bucket={row.get('prompt_bucket'):<10} "
                f"kl={_fmt(row.get('kl_ref_to_candidate')):>10} "
                f"topk={row.get('topk_overlap')} "
                f"retained={_fmt(row.get('topk_mass_retained_ref')):>10} "
                f"lost_mass={_fmt(row.get('topk_lost_ref_mass')):>10} "
                f"boundary={_fmt(row.get('topk_boundary_logit_margin_ref')):>10} "
                f"ref={_token_list(row.get('reference_top_tokens') or [])} "
                f"cand={_token_list(row.get('candidate_top_tokens') or [])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--max-rows", type=int, default=12)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    summary = summarize_payload(payload, max_rows=args.max_rows)
    print_summary(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
