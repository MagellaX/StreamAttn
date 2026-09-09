"""Summarize the adaptive canary without converting diagnostics into promotion."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(result):
    if result.get("schema") != "streamattn.adaptive_two_gate.v1":
        raise ValueError("expected adaptive two-gate canary artifact")
    rows = []
    for case in result["cases"]:
        counts = case["correctness"]["adaptive"]["counters"]
        times = case["graph_ms"]
        for samples in times.values():
            if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("timings must be nonempty, finite, and positive")
        ratios = {}
        for name in ("mask_only_control", "zero_budget_control", "torch_flash_sdpa"):
            if name not in times:
                continue
            if len(times[name]) != len(times["adaptive"]):
                raise ValueError("paired timing lengths differ")
            pairs = [other / own for own, other in zip(times["adaptive"], times[name])]
            ratios[name + "_over_adaptive"] = dict(median=statistics.median(pairs),
                                                   minimum=min(pairs), wins=sum(x > 1 for x in pairs),
                                                   trials=len(pairs))
        rows.append(dict(kind=case["kind"], dtype=case["dtype"],
                         pre_row_blocks=counts[0], post_row_blocks=counts[1],
                         valid_cta_tiles=counts[3], executed_qk_tiles=counts[4],
                         executed_pv_tiles=counts[5],
                         error=case["correctness"]["adaptive"]["max_row_l2"],
                         omission_bound=case["correctness"]["adaptive"]["max_omission_bound"],
                         median_graph_ms=case["median_graph_ms"], ratios=ratios,
                         baseline_error=case.get("baseline_error")))
    return dict(schema="streamattn.adaptive_two_gate_summary.v1", provider=result["provider"],
                device=result["device"], input_complete=result["complete"],
                performance_promotion=False, model_validation=False,
                completed_cases=len(rows), failures=result["failures"], rows=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    result = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["rows"]:
        print(row["kind"], row["dtype"], row["executed_qk_tiles"], row["executed_pv_tiles"], row["ratios"])
    print(f"completed_cases={result['completed_cases']} failures={len(result['failures'])}; no promotion")


if __name__ == "__main__":
    main()
