"""Summarize fixed ablations; discovery and holdout never share a fitted selector."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

CONTROL = "natural_compact_interior"
VARIANTS = (CONTROL, "interior_kv_order", "interior_min2", "interior_min2_kv_order")


def summarize_counters(payload):
    if payload.get("schema") != "streamattn.sm90_mixed_attribution_counters.v1":
        raise ValueError("wrong counter schema")
    keep = {"Grid Size", "Block Size", "Registers Per Thread", "Stack Size",
        "Dynamic Shared Memory Per Block", "Static Shared Memory Per Block", "L2 Hit Rate",
        "Eligible Warps Per Scheduler", "Active Warps Per Scheduler", "No Eligible",
        "Compute (SM) Throughput", "DRAM Throughput", "L2 Cache Throughput", "Executed Instructions"}
    rows = []
    for row in payload.get("rows", []):
        raw = row.get("profiler_csv", "")
        start = raw.find('"ID",')
        kernels = {}
        if start >= 0:
            for metric in csv.DictReader(io.StringIO(raw[start:])):
                name = metric.get("Metric Name")
                if name not in keep:
                    continue
                key = (metric["Process ID"], metric["ID"])
                kernel = kernels.setdefault(key, dict(name=metric["Kernel Name"], metrics={}))
                kernel["metrics"][name] = dict(value=metric["Metric Value"], unit=metric["Metric Unit"])
        checked = row.get("checked_result") or {}
        passed = (row.get("returncode") == 0 and checked.get("complete")
                  and len(checked.get("rows", [])) == 1 and checked["rows"][0].get("passed"))
        rows.append(dict(case_index=row["case_index"], target=row["target"],
                         correctness_passed=bool(passed), kernels=list(kernels.values())))
    return dict(collected=bool(payload.get("collected") and payload.get("complete") and rows
            and all(r["correctness_passed"] and r["kernels"] for r in rows)),
        rows=rows, device=payload.get("device"), status=payload.get("status"),
        contract="instrumented resources, scheduler and throughput counters; no latency gate; no byte-traffic attribution")


def geomean(values):
    return math.exp(statistics.mean(map(math.log, values))) if values else None


def comparison(row, interface, variant, mode):
    graph = row.get("graphs", {}).get(interface, {})
    winner = graph.get("fastest_tested_baseline")
    binaries = row.get("loaded_binary_provenance", {})
    if (not row.get("passed") or not winner or not winner.get("correctness_passed")
            or not all(binaries.get(n, {}).get("resolved") for n in (CONTROL, variant))):
        return None
    graph = graph if mode == "warm" else row.get("cache_perturbed", {}).get(interface, {})
    control, candidate, baseline = [n+"/"+interface for n in (CONTROL, variant, winner["baseline_id"])]
    trials = graph.get("paired_trials", [])
    if not trials or any(not all(isinstance(t.get("us", {}).get(n), (int, float))
            and math.isfinite(t["us"][n]) and t["us"][n] > 0
            for n in (control, candidate, baseline)) for t in trials):
        return None
    internal = [t["us"][control]/t["us"][candidate] for t in trials]
    external = [t["us"][baseline]/t["us"][candidate] for t in trials]
    return dict(case=row["case"], baseline=winner["baseline_id"],
        control_speedup=statistics.median(internal), baseline_speedup=statistics.median(external),
        control_all_pairs_win=all(x > 1 for x in internal),
        baseline_all_pairs_win=all(x > 1 for x in external),
        minimum_control_pair=min(internal), minimum_baseline_pair=min(external))


def aggregate(rows):
    return dict(cases=len(rows), control_geomean=geomean([r["control_speedup"] for r in rows]),
        baseline_geomean=geomean([r["baseline_speedup"] for r in rows]),
        control_all_pair_wins=sum(r["control_all_pairs_win"] for r in rows),
        baseline_all_pair_wins=sum(r["baseline_all_pairs_win"] for r in rows),
        worst_control_pair=min((r["minimum_control_pair"] for r in rows), default=None),
        worst_baseline_pair=min((r["minimum_baseline_pair"] for r in rows), default=None))


def summarize(payload):
    if (payload.get("schema") != "streamattn.sm90_micro_prefill_mixed.v1"
            or payload.get("experiment") != "post_affine_attribution"):
        raise ValueError("expected post-affine attribution artifact")
    rows = payload["rows"]
    complete = bool(payload.get("complete") and len(rows) == payload.get("planned_cases"))
    result = dict(schema="streamattn.sm90_post_affine_attribution_summary.v1",
        suite=payload["suite"], environment=payload["environment"], seed=payload["seed"],
        complete=complete, cases=len(rows), correctness_passed=sum(bool(r.get("passed")) for r in rows),
        promotion=False, interfaces={}, attribution=[],
        contract="fixed variants; cache perturbation not guaranteed cold; isolated times not additive; no public dispatch change")
    for interface in ("padded", "packed"):
        result["interfaces"][interface] = {}
        for mode in ("warm", "perturbed"):
            reports = {}
            for variant in VARIANTS:
                measured = [r for row in rows if (r := comparison(row, interface, variant, mode))]
                reports[variant] = dict(**aggregate(measured), rows=measured,
                    by_trace={trace: aggregate([r for r in measured if r["case"]["trace"] == trace])
                              for trace in sorted({r["case"]["trace"] for r in measured})})
            result["interfaces"][interface][mode] = reports
    for row in rows:
        data = row.get("attribution", {})
        if not row.get("passed"):
            continue
        result["attribution"].append(dict(case=row["case"],
            native={name: {k: entry[k] for k in ("geometry", "attributes", "workspace_allocated_bytes", "isolated_median_us")}
                    for name, entry in data.get("native", {}).items()},
            interface_median_us=data.get("interface_components", {}).get("median_us", {}),
            flashinfer=data.get("baselines", {}), launch_traces=data.get("launch_traces", {})))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--counters", type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("preserve existing summary")
    reports = [dict(source=str(path), **summarize(json.loads(path.read_text(encoding="utf-8")))) for path in args.inputs]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = dict(runs=reports)
    if args.counters:
        result["counters"] = summarize_counters(json.loads(args.counters.read_text(encoding="utf-8")))
    args.output_json.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for report in reports:
        print(f"{report['suite']}: {report['correctness_passed']}/{report['cases']} correct; complete={report['complete']}")
        for interface in ("padded", "packed"):
            for variant, data in report["interfaces"][interface]["warm"].items():
                print(interface, variant, "control", data["control_geomean"],
                      "baseline", data["baseline_geomean"], "all-pair external wins", data["baseline_all_pair_wins"])


if __name__ == "__main__":
    main()
