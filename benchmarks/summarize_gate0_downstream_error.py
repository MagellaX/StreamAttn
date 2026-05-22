"""Summarize Gate-0 downstream-error profiler artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def _policy(row: Dict[str, Any], name: str) -> Dict[str, Any] | None:
    for policy in row.get("fixed_policies", []):
        if policy.get("name") == name:
            return policy
    return None


def _best_sweep(
    rows: Iterable[Dict[str, Any]],
    *,
    budget: float,
    ranking_fragment: str = "wo_l2",
    require_seed_remaining: bool = False,
) -> Dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if ranking_fragment in row.get("name", "")
        and float((row.get("post_o_proj_error") or {}).get("max_abs_error") or 0.0) <= budget
        and (not require_seed_remaining or bool(row.get("seed_heads")))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (len(row.get("repair_heads") or []), row.get("name", "")))
    return candidates[0]


def _extract(path: Path, budget: float) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["rows"][0] if "rows" in payload else payload
    prompt = (row.get("capture") or {}).get("prompt_type") or path.stem
    seed_g0 = _policy(row, "seed_kv_groups_0") or {}
    all_seed = _policy(row, "all_seed_only") or {}
    trusted = _policy(row, "trusted_cross_prompt_policy") or {}
    sweeps = row.get("repair_sweeps") or {}
    best_g0 = _best_sweep(sweeps.get("kv_group_0_seed_repair", []), budget=budget)
    best_g1 = _best_sweep(sweeps.get("kv_group_1_seed_repair", []), budget=budget)
    best_all = _best_sweep(sweeps.get("all_seed_repair", []), budget=budget)
    best_g0_nontrivial = _best_sweep(
        sweeps.get("kv_group_0_seed_repair", []),
        budget=budget,
        require_seed_remaining=True,
    )
    best_g1_nontrivial = _best_sweep(
        sweeps.get("kv_group_1_seed_repair", []),
        budget=budget,
        require_seed_remaining=True,
    )
    best_all_nontrivial = _best_sweep(
        sweeps.get("all_seed_repair", []),
        budget=budget,
        require_seed_remaining=True,
    )
    return {
        "artifact": str(path),
        "prompt": prompt,
        "seed_g0_post_max": (seed_g0.get("post_o_proj_error") or {}).get("max_abs_error"),
        "seed_g0_raw_max": (seed_g0.get("raw_attention_error") or {}).get("max_abs_error"),
        "all_seed_post_max": (all_seed.get("post_o_proj_error") or {}).get("max_abs_error"),
        "all_seed_raw_max": (all_seed.get("raw_attention_error") or {}).get("max_abs_error"),
        "trusted_post_max": (trusted.get("post_o_proj_error") or {}).get("max_abs_error"),
        "trusted_raw_max": (trusted.get("raw_attention_error") or {}).get("max_abs_error"),
        "best_g0": best_g0,
        "best_g1": best_g1,
        "best_all": best_all,
        "best_g0_nontrivial": best_g0_nontrivial,
        "best_g1_nontrivial": best_g1_nontrivial,
        "best_all_nontrivial": best_all_nontrivial,
    }


def _best_label(row: Dict[str, Any] | None) -> str:
    if row is None:
        return "none"
    return f"{len(row.get('repair_heads') or [])}:{row.get('repair_heads')}->{_fmt((row.get('post_o_proj_error') or {}).get('max_abs_error'))}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--budget", type=float, default=0.015)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = [_extract(Path(path), args.budget) for path in args.artifacts]
    print(
        "prompt\tg0_raw\tg0_post\tall_raw\tall_post\ttrusted_raw\ttrusted_post\t"
        "best_g0_repair@budget\tbest_g1_repair@budget\tbest_all_repair@budget"
        "\tbest_g1_nontrivial\tbest_all_nontrivial"
    )
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["prompt"]),
                    _fmt(row["seed_g0_raw_max"]),
                    _fmt(row["seed_g0_post_max"]),
                    _fmt(row["all_seed_raw_max"]),
                    _fmt(row["all_seed_post_max"]),
                    _fmt(row["trusted_raw_max"]),
                    _fmt(row["trusted_post_max"]),
                    _best_label(row["best_g0"]),
                    _best_label(row["best_g1"]),
                    _best_label(row["best_all"]),
                    _best_label(row["best_g1_nontrivial"]),
                    _best_label(row["best_all_nontrivial"]),
                ]
            )
        )
    print()
    if all(row["best_g1_nontrivial"] is not None for row in rows):
        print("decision: group1 has a post-Wo seed+repair candidate under budget on all rows")
    else:
        print("decision: group1 still lacks a stable post-Wo seed+repair candidate under budget")
    if all(row["best_all_nontrivial"] is not None for row in rows):
        print("decision: whole-layer all-seed plus Wo-ranked repair has a candidate under budget")
    else:
        print("decision: whole-layer all-seed plus Wo-ranked repair is not stable under budget")

    if args.json_out:
        output = {
            "schema": "streamattn.gate0.downstream_error_summary.v1",
            "budget": args.budget,
            "rows": rows,
        }
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
