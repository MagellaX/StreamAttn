"""Summarize complete same-worker ratios, preserving input-layout boundaries."""

import argparse
import json
import math
from pathlib import Path
import statistics


def schedule_geometry(case, family, splits):
    """Derive empty CTA work from the retained rectangular launch, not counters."""
    qs, ns = case["query_lengths"], case["kv_lengths"]
    q_per_tile = 64 // case["g"] if family == "natural" else 1
    kv_heads = case["hq"] // case["g"]
    query_tiles = [(q + q_per_tile - 1) // q_per_tile for q in qs]
    kv_tiles = [(n + 63) // 64 for n in ns]
    launched = len(qs) * max(query_tiles) * kv_heads * splits
    live_by_request = [qt * kv_heads * min(splits, kt) for qt, kt in zip(query_tiles, kv_tiles)]
    live = sum(live_by_request)
    return dict(launched_ctas=launched, nonempty_ctas=live, empty_cta_fraction=1-live/launched,
                nonempty_ctas_per_request=live_by_request,
                maximum_kv_tiles_per_cta=[(kt+splits-1)//splits for kt in kv_tiles],
                contract="derived launch geometry, not measured SM occupancy")


def summarize(payload):
    if payload.get("schema") != "streamattn.sm90_micro_prefill_mixed.v1":
        raise ValueError("wrong mixed benchmark schema")
    result = dict(schema="streamattn.sm90_micro_prefill_mixed_summary.v1",
                  complete=payload.get("complete", False), environment=payload["environment"],
                  cases=len(payload["rows"]), correctness_passed=sum(r.get("passed", False) for r in payload["rows"]),
                  promotion=False, interfaces={})
    for interface in ("padded", "packed"):
        comparisons = []
        compact = []
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
            candidate = "natural_compact/" + interface
            if candidate in medians and native.get("natural_compact", {}).get("resolved"):
                ratios = [t["us"][bname] / t["us"][candidate] for t in graph["paired_trials"]]
                control = [t["us"]["natural/" + interface] / t["us"][candidate]
                           for t in graph["paired_trials"]]
                compact.append(dict(case=row["case"], baseline=baseline["baseline_id"],
                    paired_baseline_speedups=ratios, paired_control_speedups=control,
                    median_baseline_speedup=statistics.median(ratios),
                    median_control_speedup=statistics.median(control),
                    baseline_all_pairs_win=all(x > 1 for x in ratios),
                    control_all_pairs_win=all(x > 1 for x in control),
                    schedule=row["families"][candidate]))
            comparisons.append(dict(case=row["case"], baseline=baseline["baseline_id"],
                baseline_us=medians[bname], native_us=times, oracle_family=best,
                paired_speedups=pairs, median_speedup=statistics.median(pairs),
                all_pairs_win=all(p > 1 for p in pairs),
                derived_schedule={n: schedule_geometry(row["case"], n, config["splits"])
                                  for n in ("natural", "transposed")
                                  if (config := row.get("families", {}).get(n + "/" + interface))}))
        result["interfaces"][interface] = dict(
            comparable_cases=len(comparisons),
            oracle_all_pair_winning_cases=sum(r["all_pairs_win"] for r in comparisons),
            oracle_geomean=(math.exp(statistics.mean(math.log(r["median_speedup"]) for r in comparisons))
                            if comparisons else None),
            fixed_family_geomeans={n: math.exp(statistics.mean(
                math.log(r["baseline_us"] / r["native_us"][n]) for r in comparisons))
                for n in ("transposed", "natural")} if comparisons else {},
            rows=comparisons,
            compact_candidate=dict(cases=len(compact),
                baseline_geomean=math.exp(statistics.mean(math.log(r["median_baseline_speedup"])
                    for r in compact)) if compact else None,
                control_geomean=math.exp(statistics.mean(math.log(r["median_control_speedup"])
                    for r in compact)) if compact else None,
                baseline_all_pair_wins=sum(r["baseline_all_pairs_win"] for r in compact),
                control_all_pair_wins=sum(r["control_all_pairs_win"] for r in compact), rows=compact),
        )
    result["selection_contract"] = "per-case best-family oracle, not held-out or public dispatch"
    result["compact_diagnostics"] = compact_diagnostics(payload["rows"])
    return result


def compact_diagnostics(rows):
    """Matched shapes, not a component-timing claim or a counterfactual proof."""
    valid = [r for r in rows if r.get("passed") and
             r.get("loaded_binary_provenance", {}).get("natural_compact", {}).get("resolved")]
    key = lambda r: json.dumps({k: v for k, v in r["case"].items() if k != "causal"}, sort_keys=True)
    noncausal = {key(r): r for r in valid if not r["case"].get("causal")}
    mask, wrapper, proxy = [], [], []
    for r in valid:
        padded = r.get("graphs", {}).get("padded", {}).get("median_us", {})
        packed = r.get("graphs", {}).get("packed", {}).get("median_us", {})
        if "natural_compact/padded" not in padded:
            continue
        if "natural_compact/packed" in packed:
            wrapper.append(packed["natural_compact/packed"] / padded["natural_compact/padded"])
        baseline = r.get("graphs", {}).get("packed", {}).get("fastest_tested_baseline")
        if baseline and baseline["correctness_passed"]:
            proxy.append(packed[baseline["baseline_id"] + "/packed"] / padded["natural_compact/padded"])
        control = noncausal.get(key(r)) if r["case"].get("causal") else None
        if control is not None:
            times = control["graphs"]["padded"]["median_us"]
            if "natural_compact/padded" in times:
                mask.append(padded["natural_compact/padded"] / times["natural_compact/padded"])
    gm = lambda xs: math.exp(statistics.mean(map(math.log, xs))) if xs else None
    return dict(matched_mask_pairs=len(mask), causal_over_noncausal_latency=gm(mask),
                wrapper_cases=len(wrapper), packed_over_padded_native_latency=gm(wrapper),
                copy_free_proxy_cases=len(proxy), copy_free_baseline_speedup_proxy=gm(proxy),
                contract="separate-graph timing proxies, not isolated component timings or formal bounds")


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
            compact = row["compact_candidate"]
            if compact["cases"]:
                print(f"  compact: {compact['control_geomean']:.3f}x vs natural; "
                      f"{compact['baseline_geomean']:.3f}x vs baseline; "
                      f"{compact['baseline_all_pair_wins']}/{compact['cases']} baseline all-pair wins")
    if args.output_json:
        if args.output_json.exists():
            raise FileExistsError("preserve existing summaries")
        args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
