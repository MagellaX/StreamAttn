"""Checked paged-producer source attribution; sampled stalls are not runtime."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.sm90_mixed_attribution import CONTROL, geometry, useful_work
from benchmarks.summarize_sm90_micro_source_counters import source_pcs


def flashinfer_work(case, telemetry, kernel):
    """Pinned causal FA2 loop accounting from measured plan and kernel traits."""
    match = re.search(r"KernelTraits<([0-9, ]+),", kernel)
    if not match or not telemetry.get("decoded") or telemetry.get("version") != "0.6.13":
        return dict(decoded=False, reason="unresolved FA2 traits or plan")
    traits = list(map(int, match[1].split(",")))
    if len(traits) != 9:
        return dict(decoded=False, reason="unrecognized kernel traits")
    mask, rq, _, mma_kv, dq, dv, wq, wkv, rope = traits
    plan = telemetry["plan"]
    if (mask != 1 or rope != 0 or not case["causal"] or rq != plan["cta_tile_q"]
            or not plan["split_kv"] or 16*dq != case["d"] or dq != dv):
        return dict(decoded=False, reason="unsupported measured kernel semantics")
    tile_k = mma_kv*wkv*16
    chunk = telemetry["kv_chunk_tokens"]
    iterations = []
    for request, qtile, ktile in telemetry["tasks"]:
        m, n = case["query_lengths"][request], case["kv_lengths"][request]
        begin, end = min(ktile*chunk, n), min((ktile+1)*chunk, n)
        visible_end = n-m+((qtile+1)*rq+case["g"]-1)//case["g"]
        extent = min(end-begin, max(0, visible_end-begin))
        iterations.append((extent+tile_k-1)//tile_k)
    pairs = sum(iterations)*rq*tile_k*(case["hq"]//case["g"])
    return dict(decoded=True, query_tile=rq, kv_tile=tile_k, iterations_per_live_task=iterations,
                scheduled_head_pairs=pairs, scheduled_qk_pv_flops=4*case["d"]*pairs,
                convention="v0.6.13 causal loop and runtime plan; no sliding window; not measured tensor-op counters")


def regions(source):
    """Derive exclusive ranges from the captured generated source, not current files."""
    begin = source.index("void streamattn_natural_wgmma_micro_prefill_partial_kernel(")
    end = source.index("void streamattn_natural_wgmma_micro_prefill_merge_kernel(", begin)
    tokens = (
        ("setup", begin),
        ("q_staging", source.index("  for (int idx = threadIdx.x", begin)),
        ("setup_fragments", source.index("  PrefillTiledMma tiled_qk;", begin)),
        ("kv_staging", source.index("  auto copy_k_tile =", begin)),
        ("qk", source.index("    Tensor scores = partition_fragment_C(", begin)),
        ("v_issue", source.index("    copy_v_tile(tile);", begin)),
        ("softmax_rescale", source.index("    Tensor score_rows = make_tensor(", begin)),
        ("pv_copy_wait", source.index("    Tensor p_acc = make_tensor(", begin)),
        ("writeback", source.index("  CUTE_UNROLL\n  for (int row = 0; row < size<0>(output_rows)", begin)),
    )
    tokens = sorted(tokens, key=lambda item: item[1])
    result = [(name, source.count("\n", 0, start)+1, source.count("\n", 0, stop)+1)
              for (name, start), (_, stop) in zip(tokens, (*tokens[1:], ("end", end)))]
    loader = source.index("__forceinline__ __device__ void streamattn_micro_load_page16(")
    stop = source.index("\n}\n", loader)+3
    result.append(("paged_address_copy", source.count("\n", 0, loader)+1, source.count("\n", 0, stop)+1))
    pair = source.find("__forceinline__ __device__ void streamattn_micro_load_page16_pair(")
    if pair >= 0:
        stop = source.index("\n}\n", pair)+3
        result.append(("paged_address_copy", source.count("\n", 0, pair)+1, source.count("\n", 0, stop)+1))
    return result


def raw_metrics(text):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    # Nsight raw page has a units row followed by one row per filtered launch.
    rows = [r for r in rows if r.get("ID", "").isdigit()]
    if len(rows) != 1:
        raise ValueError("expected one filtered producer launch")
    return rows[0]


def number(metrics, key):
    value = metrics.get(key)
    if value in (None, "", "n/a"):
        return None
    result = float(value.replace(",", ""))
    if not math.isfinite(result):
        raise ValueError("non-finite counter")
    return result


def machine_instructions(pcs):
    """Counts per warp, preserving opcode widths; never convert them into time."""
    if not pcs or not all("Instructions Executed" in pc["metrics"] for pc in pcs):
        return None
    opcodes, page_loads = {}, 0
    for pc in pcs:
        parts = pc["instruction"].split()
        opcode = parts[1] if parts[0].startswith("@") else parts[0]
        counts = opcodes.setdefault(opcode, dict(static_sites=0, executed_warp_instructions=0))
        counts["static_sites"] += 1
        counts["executed_warp_instructions"] += pc["metrics"]["Instructions Executed"]
        if opcode.startswith("LDG.") and any("const int page = table[" in (c["source"] or "") for c in pc["correlations"]):
            page_loads += pc["metrics"]["Instructions Executed"]
    return dict(opcodes=opcodes, source_correlated_page_load_warp_instructions=page_loads,
                kv_copy_warp_instructions=sum(c["executed_warp_instructions"] for op, c in opcodes.items() if op.startswith("LDGSTS.")))


def summarize(payload):
    if (payload.get("schema") != "streamattn.sm90_mixed_attribution_counters.v1"
            or not payload.get("source_correlated")):
        raise ValueError("expected paged source-counter capture")
    rows, identities = [], set()
    for capture in payload["rows"]:
        identity = (capture["case_index"], capture["target"])
        if identity in identities:
            raise ValueError("duplicate source capture")
        identities.add(identity)
        checked = capture.get("checked_result") or {}
        if (not checked.get("complete") or len(checked.get("rows", [])) != 1
                or not checked["rows"][0].get("passed") or capture.get("returncode")
                or capture.get("source_export_returncode")):
            raise ValueError("unverified source capture")
        row = checked["rows"][0]
        missing_lineinfo = (capture["target"].startswith("flashinfer")
                            and "No lineinfo available" in capture["source_csv"])
        pcs = [] if missing_lineinfo else source_pcs(capture["source_csv"], include_all_metrics=True)
        raw = raw_metrics(capture["profiler_csv"])
        totals = {name: sum(pc["metrics"].get(name, 0) for pc in pcs)
                  for name in sorted({k for pc in pcs for k in pc["metrics"]
                                      if k.startswith("stall_") or k == "# Samples"})}
        for source_name, raw_name in (("# Samples", "smsp__pcsamp_sample_count"),
                ("stall_long_sb", "smsp__pcsamp_warps_issue_stalled_long_scoreboard")):
            if pcs and (source_name not in totals or totals[source_name] != number(raw, raw_name)):
                raise ValueError("PC totals do not match kernel sampling counters")
        instruction_total = (sum(pc["metrics"].get("Instructions Executed", 0) for pc in pcs)
                             if any("Instructions Executed" in pc["metrics"] for pc in pcs) else None)
        if instruction_total is not None and instruction_total != number(raw, "smsp__inst_executed.sum"):
            raise ValueError("PC instructions do not match kernel instruction counter")
        ranges = {}
        for path, generated in capture.get("generated_sources", {}).items():
            text = generated["text"]
            if hashlib.sha256(text.encode()).hexdigest() != generated["sha256"]:
                raise ValueError("generated source hash mismatch")
            ranges[path] = regions(text)
        region_counts = {}
        for pc in pcs:
            labels = {name for correlation in pc["correlations"]
                      for name, start, stop in ranges.get(correlation["file"], [])
                      if correlation["line"] is not None and start <= correlation["line"] < stop}
            label = next(iter(labels)) if len(labels) == 1 else "ambiguous" if labels else "unmapped_or_external"
            pc["region"] = label
            counters = region_counts.setdefault(label, {})
            for metric, value in pc["metrics"].items():
                if metric.startswith("stall_") or metric in ("# Samples", "Instructions Executed", "Predicated-On Thread Instructions Executed"):
                    counters[metric] = counters.get(metric, 0) + value
        work = useful_work(row["case"])
        native_geometry = json.loads(json.dumps(geometry(row["case"], CONTROL)))
        observed_work = row.get("counter_work", {})
        if observed_work and (observed_work["useful"] != work or observed_work["native"] != native_geometry):
            raise ValueError("counter work disagrees with checked case")
        issued = sum(native_geometry["kv_tiles_per_task"])*64*64
        counters = {k: number(raw, k) for k in payload.get("available_metrics", [])}
        normalized = {k: v/work["visible_head_pairs"] for k, v in counters.items() if v is not None}
        rankings = {metric: sorted(pcs, key=lambda p: p["metrics"].get(metric, 0), reverse=True)[:12]
                    for metric in totals if "Not Issued" not in metric and metric != "# Samples"}
        rows.append(dict(case=row["case"], target=capture["target"], kernel=raw["Kernel Name"],
            useful_work=work, native_scheduled_head_pairs=issued,
            native_scheduled_qk_pv_flops=4*row["case"]["d"]*issued,
            flashinfer_issued_tile_work=flashinfer_work(row["case"],
                observed_work.get("baselines", {}).get("flashinfer_fa2", {}), raw["Kernel Name"])
                if capture["target"].startswith("flashinfer") else None,
            observed_plan_telemetry=observed_work.get("baselines", {}),
            traffic_and_instruction_counters=counters, counters_per_visible_head_pair=normalized,
            totals=totals, regions=region_counts, unique_pcs=len(pcs), top_pcs=rankings,
            pc_instruction_total=instruction_total,
            machine_instructions=machine_instructions(pcs),
            counter_attributes=row.get("counter_attributes"),
            source_correlation_available=bool(pcs), aggregate_totals_match=True if pcs else None,
            source_limitation="loaded FA2 binary has no CUDA lineinfo" if missing_lineinfo else None,
            attributed_stalls_available=any("attributed" in k.lower() for pc in pcs for k in pc["metrics"])))
    expected = ({(8, target) for target in ("natural_compact_interior/padded", "interior_q_vector/padded")}
                if payload.get("producer_copy") else {(i, target) for i in (6, 8, 10) for target in
                ("natural_compact_interior/padded", "flashinfer_fa2/packed")})
    if payload.get("page_pair"):
        expected = {(8, target) for target in ("interior_q_vector/padded", "interior_q_vector_page_pair/padded")}
    return dict(schema="streamattn.sm90_paged_source_summary.v1",
        complete=bool(payload.get("complete") and payload.get("collected") and identities == expected), rows=rows,
        missing_metrics=payload.get("unavailable_metrics", []),
        contract="samples not time; integer instructions not exclusively addresses; L2 counters are sectors; no inferred producer-attributed stalls; no latency gate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("preserve existing evidence")
    result = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    args.output_json.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for row in result["rows"]:
        print(row["case"]["trace"], row["target"], row["traffic_and_instruction_counters"])
        print({name: counts.get("# Samples", 0) for name, counts in row["regions"].items()})


if __name__ == "__main__":
    main()
