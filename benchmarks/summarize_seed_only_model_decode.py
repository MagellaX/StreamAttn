"""Summarize actual-model seed-only decode route artifacts.

The selected-attention benchmarks answer whether routed attention calls are
profitable.  This summary answers the stricter serving question:

    Does the routed policy speed up the full `use_cache=True` model decode loop?

It also computes a simple Amdahl-style coverage estimate:

    S_total = 1 / ((1 - f) + f / S_region)

where `S_total` is measured full-model speedup, `S_region` is an assumed routed
attention local speedup, and `f` is the effective fraction of total decode time
that the current route accelerates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def _load(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _effective_fraction(*, total_speedup: float, region_speedup: float) -> float:
    if region_speedup <= 1.0:
        raise ValueError("region_speedup must be > 1")
    return (1.0 - (1.0 / total_speedup)) / (1.0 - (1.0 / region_speedup))


def _required_fraction(*, target_speedup: float, region_speedup: float) -> float:
    if target_speedup <= 1.0:
        raise ValueError("target_speedup must be > 1")
    return _effective_fraction(total_speedup=target_speedup, region_speedup=region_speedup)


def _patch_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    patches = payload.get("patch_counts", {})
    layers = sorted(int(layer) for layer in patches)
    total_forward = sum(int(row.get("forward_calls", 0)) for row in patches.values())
    total_seed = sum(int(row.get("seed_only_decode_calls", 0)) for row in patches.values())
    fallback_reasons: Dict[str, int] = {}
    for row in patches.values():
        for reason, count in row.get("fallback_reasons", {}).items():
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + int(count)
    return {
        "layers": layers,
        "layer_count": len(layers),
        "forward_calls": total_forward,
        "seed_only_decode_calls": total_seed,
        "all_calls_seed_only": total_forward == total_seed and total_forward > 0,
        "fallback_reasons": fallback_reasons,
    }


def _case_summary(path: Path, payload: Dict[str, Any], *, region_speedup: float) -> Dict[str, Any]:
    timing = payload["timing"]
    safety = payload["safety"]
    dense_ms = float(timing["dense_decode_total_ms"])
    stream_ms = float(timing["streamattn_decode_total_ms"])
    total_speedup = dense_ms / stream_ms
    saved_ms = dense_ms - stream_ms
    steps = int(payload["shape"]["steps"])
    effective_fraction = _effective_fraction(total_speedup=total_speedup, region_speedup=region_speedup)
    return {
        "artifact": str(path),
        "batch": int(payload["shape"]["batch"]),
        "steps": steps,
        "layers": payload["route_bundle"]["layers"],
        "layer_count": len(payload["route_bundle"]["layers"]),
        "dense_decode_total_ms": dense_ms,
        "streamattn_decode_total_ms": stream_ms,
        "dense_decode_ms_per_token": float(timing["dense_decode_ms_per_token"]),
        "streamattn_decode_ms_per_token": float(timing["streamattn_decode_ms_per_token"]),
        "speedup_vs_dense_decode": total_speedup,
        "saved_ms_total": saved_ms,
        "saved_ms_per_token": saved_ms / max(1, steps),
        "decision_passed": bool(payload.get("decision", {}).get("passed", False)),
        "kl_max": float(safety.get("kl_max", 0.0)),
        "top1_changed_count": int(safety.get("top1_changed_count", 0)),
        "sample_token_changed_count": int(safety.get("sample_token_changed_count", 0)),
        "topk_overlap_min": int(safety.get("topk_overlap_min", 0)),
        "patch_summary": _patch_summary(payload),
        "assumed_region_speedup": region_speedup,
        "effective_routed_fraction": effective_fraction,
    }


def summarize(
    artifact_paths: Sequence[str | Path],
    *,
    region_speedup: float,
    target_speedups: Iterable[float],
) -> Dict[str, Any]:
    cases = [_case_summary(Path(path), _load(path), region_speedup=region_speedup) for path in artifact_paths]
    return {
        "schema": "streamattn.seed_only_model_decode_summary.v1",
        "assumed_region_speedup": region_speedup,
        "cases": cases,
        "targets": [
            {
                "target_speedup": target,
                "required_routed_fraction": _required_fraction(
                    target_speedup=float(target),
                    region_speedup=region_speedup,
                ),
            }
            for target in target_speedups
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--region-speedup", type=float, default=3.0)
    parser.add_argument("--target-speedups", default="1.05,1.10,1.20")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    target_speedups = [float(row) for row in args.target_speedups.split(",") if row.strip()]
    result = summarize(
        args.artifacts,
        region_speedup=args.region_speedup,
        target_speedups=target_speedups,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
