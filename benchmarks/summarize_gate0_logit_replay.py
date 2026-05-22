"""Summarize Gate-0 logit replay artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _policy(row: Dict[str, Any], name: str) -> Dict[str, Any] | None:
    for policy in row.get("policies", []):
        if policy.get("name") == name:
            return policy
    return None


def _best_nontrivial(row: Dict[str, Any], *, max_kl: float, max_logit: float) -> Dict[str, Any] | None:
    candidates = []
    for policy in row.get("policies", []):
        seed_heads = policy.get("seed_heads") or []
        if not seed_heads:
            continue
        metrics = policy.get("logits_vs_dense_patch") or {}
        if metrics.get("top1_changed") or metrics.get("topk_changed"):
            continue
        kl = abs(float(metrics.get("kl_ref_to_candidate") or 0.0))
        max_err = float(metrics.get("max_abs_error") or 0.0)
        if kl <= max_kl and max_err <= max_logit:
            candidates.append(policy)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-len(item.get("seed_heads") or []), len(item.get("repair_heads") or []), item.get("name", "")))
    return candidates[0]


def _extract(path: Path, *, max_kl: float, max_logit: float) -> Dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    prompt = (row.get("capture") or {}).get("prompt_type") or path.stem
    baseline = (row.get("baseline") or {}).get("dense_patch_logits_vs_model_baseline") or {}
    seed_g0 = _policy(row, "seed_kv_groups_0")
    all_seed = _policy(row, "all_seed_only")
    trusted = _policy(row, "trusted_cross_prompt_policy")
    best = _best_nontrivial(row, max_kl=max_kl, max_logit=max_logit)
    return {
        "artifact": str(path),
        "prompt": prompt,
        "dense_patch_vs_model": baseline,
        "seed_g0": seed_g0,
        "all_seed": all_seed,
        "trusted": trusted,
        "best_nontrivial": best,
    }


def _metrics(policy: Dict[str, Any] | None) -> Dict[str, Any]:
    if not policy:
        return {}
    return policy.get("logits_vs_dense_patch") or {}


def _label(policy: Dict[str, Any] | None) -> str:
    if not policy:
        return "none"
    metrics = _metrics(policy)
    return (
        f"{policy.get('name')} seed={policy.get('seed_heads')} repair={policy.get('repair_heads')} "
        f"kl={_fmt(metrics.get('kl_ref_to_candidate'))} max={_fmt(metrics.get('max_abs_error'))} "
        f"top1={metrics.get('top1_changed')} topk={metrics.get('topk_changed')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--max-kl", type=float, default=1e-5)
    parser.add_argument("--max-logit", type=float, default=0.1)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = [
        _extract(Path(path), max_kl=args.max_kl, max_logit=args.max_logit)
        for path in args.artifacts
    ]
    print("prompt\tdense_patch_top_change\tseed_g0\tall_seed\ttrusted\tbest_nontrivial")
    for row in rows:
        baseline = row["dense_patch_vs_model"]
        print(
            "\t".join(
                [
                    str(row["prompt"]),
                    f"top1={baseline.get('top1_changed')} topk={baseline.get('topk_changed')} max={_fmt(baseline.get('max_abs_error'))}",
                    _label(row["seed_g0"]),
                    _label(row["all_seed"]),
                    _label(row["trusted"]),
                    _label(row["best_nontrivial"]),
                ]
            )
        )
    print()
    if all(row["best_nontrivial"] is not None for row in rows):
        print("decision: logit replay found a nontrivial seed policy under thresholds on all rows")
    else:
        print("decision: logit replay did not find a stable nontrivial seed policy under thresholds")

    if args.json_out:
        output = {
            "schema": "streamattn.gate0.logit_replay_summary.v1",
            "max_kl": args.max_kl,
            "max_logit": args.max_logit,
            "rows": rows,
        }
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
