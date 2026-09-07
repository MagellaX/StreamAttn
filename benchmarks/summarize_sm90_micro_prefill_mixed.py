"""Summarize complete same-worker ratios, preserving input-layout boundaries."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(payload):
    if payload.get("schema") != "streamattn.sm90_micro_prefill_mixed.v1":
        raise ValueError("wrong mixed benchmark schema")
    result = dict(schema="streamattn.sm90_micro_prefill_mixed_summary.v1",
                  complete=payload.get("complete", False), environment=payload["environment"],
                  cases=len(payload["rows"]), correctness_passed=sum(r.get("passed", False) for r in payload["rows"]),
                  promotion=False, interfaces={})
    for interface in ("padded", "packed"):
        comparisons = []
        for row in payload["rows"]:
            graph = row.get("graphs", {}).get(interface, {})
            baseline = graph.get("fastest_tested_baseline")
            native = row.get("loaded_binary_provenance", {})
            if (not row.get("passed") or not baseline or not baseline["correctness_passed"]
                    or not all(native.get(n, {}).get("resolved") for n in ("natural", "transposed"))):
                continue
            bname = baseline["baseline_id"] + "/" + interface
            medians = graph["median_us"]
            times = {n: medians[n + "/" + interface] for n in ("transposed", "natural")}
            best = min(times, key=times.get)
            pairs = [t["us"][bname] / t["us"][best + "/" + interface] for t in graph["paired_trials"]]
            comparisons.append(dict(case=row["case"], baseline=baseline["baseline_id"],
                baseline_us=medians[bname], native_us=times, oracle_family=best,
                paired_speedups=pairs, median_speedup=statistics.median(pairs),
                all_pairs_win=all(p > 1 for p in pairs)))
        result["interfaces"][interface] = dict(
            comparable_cases=len(comparisons),
            oracle_all_pair_winning_cases=sum(r["all_pairs_win"] for r in comparisons),
            oracle_geomean=(math.exp(statistics.mean(math.log(r["median_speedup"]) for r in comparisons))
                            if comparisons else None),
            fixed_family_geomeans={n: math.exp(statistics.mean(
                math.log(r["baseline_us"] / r["native_us"][n]) for r in comparisons))
                for n in ("transposed", "natural")} if comparisons else {},
            rows=comparisons,
        )
    result["selection_contract"] = "per-case best-family oracle, not held-out or public dispatch"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.inputs:
        result = summarize(json.loads(path.read_text(encoding="utf-8")))
        result["input"] = str(path)
        results.append(result)
        print(path)
        print(f"correct cases: {result['correctness_passed']}/{result['cases']}")
        for name, row in result["interfaces"].items():
            ratio = row["oracle_geomean"]
            label = f"{ratio:.3f}x" if ratio is not None else "unresolved"
            print(f"{name}: {row['comparable_cases']} comparable; oracle {label}; "
                  f"{row['oracle_all_pair_winning_cases']} all-pair wins")
    if args.output_json:
        if args.output_json.exists():
            raise FileExistsError("preserve existing summaries")
        args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
