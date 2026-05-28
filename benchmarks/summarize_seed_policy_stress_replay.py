"""Summarize seed-policy stress replay JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


def summarize_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    safety = payload.get("safety") or {}
    timing = payload.get("timing") or {}
    decision = payload.get("decision") or {}
    route = payload.get("route_bundle") or {}
    worst = _worst_bucket(safety)
    return {
        "path": str(path),
        "passed": bool(decision.get("passed")),
        "layers": route.get("layers", []),
        "speedup_vs_dense_decode": _as_float(timing.get("speedup_vs_dense_decode")),
        "dense_decode_ms_per_token": _as_float(timing.get("dense_decode_ms_per_token")),
        "streamattn_decode_ms_per_token": _as_float(timing.get("streamattn_decode_ms_per_token")),
        "case_count": _as_int(safety.get("case_count")),
        "kl_max": _as_float(safety.get("kl_max")),
        "kl_p99": _as_float(safety.get("kl_p99")),
        "top1_changes": _as_int(safety.get("top1_changed_count")),
        "sample_changes": _as_int(safety.get("sample_token_changed_count")),
        "topk_overlap_min": _as_int(safety.get("topk_overlap_min")),
        "target_logprob_delta_max_abs": _as_float(
            safety.get("reference_top1_logprob_delta_max_abs")
        ),
        "worst_bucket": worst,
    }


def summarize_paths(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    return [summarize_payload(path) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+", help="Route-bundle stress JSON output(s).")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    summary = summarize_paths(Path(item) for item in args.json)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
