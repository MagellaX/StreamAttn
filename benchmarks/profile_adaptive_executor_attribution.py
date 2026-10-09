"""Fixed-support state-finalization attribution; no new selector or model capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_adaptive_feasibility import graph_buffer_indices, native_headroom

SCHEMA = "streamattn.adaptive_executor_attribution.v1"
METHODS = ("full", "contribution_triangle")
SOURCE_FILES = (
    "benchmarks/profile_adaptive_executor_attribution.py",
    "benchmarks/profile_adaptive_feasibility.py",
    "benchmarks/run_lightning_adaptive_two_gate.py",
    "stream_attention/paged.py",
    "stream_attention/backends/sm90/transposed_gqa_exact_sources.py",
    "docs/adaptive_executor_attribution_protocol.md",
)


def support_hash(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def selected_reference(q, k, v, rows, block_size=32):
    """FP64 reference over the supplied support, with multiplicity checked."""
    n, hk, dim = k.shape[1:]
    group = q.shape[2] // hk
    if len(rows) != hk:
        raise ValueError("one retained row per complete KV group required")
    mask = torch.zeros(hk, n, dtype=torch.bool)
    for head, blocks in enumerate(rows):
        if not blocks or blocks != sorted(set(blocks)) or min(blocks) < 0:
            raise ValueError("support must be nonempty, unique and increasing")
        if max(blocks) >= math.ceil(n / block_size):
            raise ValueError("support exceeds sequence")
        for block in blocks:
            mask[head, block * block_size:min(n, (block + 1) * block_size)] = True
    kh = k[0].permute(1, 0, 2).double().repeat_interleave(group, 0)
    vh = v[0].permute(1, 0, 2).double().repeat_interleave(group, 0)
    scores = torch.einsum("hd,hnd->hn", q[0, 0].double(), kh) / math.sqrt(dim)
    weights = scores.masked_fill(~mask.repeat_interleave(group, 0), -torch.inf).softmax(-1)
    return torch.einsum("hn,hnd->hd", weights, vh)


def phase_runner(plan, launch, phase):
    if phase not in (0, 1, 2):
        raise ValueError("phase must be complete, producer or merge")

    def run():
        launch(plan.query_group, plan.cache.key, plan.cache.value, plan.routes.row_ptr,
            plan.routes.physical_page_ids, plan.routes.active_head_masks,
            plan.routes.token_valid_masks, plan.workspace["partial_o"],
            plan.workspace["partial_lse"], plan.output_group, plan.max_routes_per_row, phase)
        return plan.output
    return run


def state_geometry(row_counts, *, records_per_cta=1, group_size=8, head_dim=128):
    if not row_counts or min(row_counts) < 1 or records_per_cta < 1:
        raise ValueError("nonempty positive route counts and grouping required")
    partitions = [math.ceil(r / records_per_cta) for r in row_counts]
    padded = len(partitions) * max(partitions)
    state_bytes = 4 * 8 * (head_dim + 1)
    return dict(records_per_row=row_counts, valid_records=sum(row_counts),
        records_per_cta=records_per_cta, valid_producer_tasks=sum(partitions),
        padded_producer_ctas=padded, query_staging_repetitions=sum(partitions),
        valid_state_finalizations=sum(partitions), partial_states_written=padded,
        partial_states_consumed=padded, active_heads=group_size,
        bytes_per_partial_state=state_bytes, allocated_partial_bytes=padded * state_bytes,
        accounting="geometry and emitted-state counts; not measured memory transactions")


def observe_selected(plans, *, iterations=100, trials=7):
    from benchmarks.profile_sm90_micro_prefill import _capture, _elapsed_graph_ms
    from stream_attention.backends.sm90.transposed_gqa_exact import compile_transposed_gqa_exact_extension

    extension = compile_transposed_gqa_exact_extension(head_dim=plans[0].query.shape[-1])
    runs = {name: [phase_runner(p, extension.selected_nhd_phase_out if p.records_per_cta == 1
                              else extension.selected_nhd_grouped_phase_out, phase) for p in plans]
            for name, phase in (("producer", 1), ("merge", 2), ("instrumented_complete", 0))}
    # Merge-only timing must consume genuine completed states, never zeros or random workspace.
    for plan, producer, merge in zip(plans, runs["producer"], runs["merge"]):
        expected = plan.run().clone()
        producer()
        torch.testing.assert_close(merge(), expected, rtol=0, atol=0)
        if not torch.isfinite(plan.workspace["partial_o"]).all():
            raise AssertionError("producer emitted nonfinite partial output")
        lse = plan.workspace["partial_lse"]
        if torch.isnan(lse).any() or torch.isposinf(lse).any():
            raise AssertionError("producer emitted invalid partial LSE")
    resources = extension.selected_nhd_resources()
    keys = ("registers_per_thread", "static_shared_bytes", "local_bytes_per_thread", "max_threads")
    resource_rows = {name: dict(zip(keys, resources[i * 4:(i + 1) * 4]))
        for i, name in enumerate(("selected_producer", "native_producer", "merge", "grouped_selected_producer"))}
    if len(resources) != 16:
        raise AssertionError("incomplete compiled resource report")
    conditions = {}
    for condition in ("warm_fixed_buffer", "rotating_working_set"):
        order = graph_buffer_indices(len(plans), condition)
        graphs = {}
        for name, functions in runs.items():
            def cycle(functions=functions, order=order):
                for i in order:
                    functions[i]()
            graphs[name] = _capture(cycle, warmup=3)
        samples = {name: [] for name in graphs}
        for trial in range(trials):
            names = list(graphs) if trial % 2 == 0 else list(reversed(graphs))
            for name in names:
                samples[name].append(_elapsed_graph_ms(graphs[name], iterations=iterations) / len(plans))
        conditions[condition] = {name: dict(samples_ms=values, median_ms=statistics.median(values))
                                 for name, values in samples.items()}
    plan = plans[0]
    counts = (plan.routes.row_ptr[1:] - plan.routes.row_ptr[:-1]).cpu().tolist()
    geometry = state_geometry(counts, records_per_cta=plan.records_per_cta,
                              group_size=plan.query_group.shape[2], head_dim=plan.query.shape[-1])
    if geometry["allocated_partial_bytes"] != plan.workspace_bytes:
        raise AssertionError("state accounting disagrees with allocation")
    return dict(geometry=geometry, compiled_resources=resource_rows, conditions=conditions,
        partial_states_checked=True, replicas_checked=len(plans),
        component_timings_are_additive=False, hardware_memory_traffic_measured=False,
        spill_load_store_traffic_measured=False,
        caveat="Isolated components have different cache/dependency conditions. Do not sum them as complete-call latency.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--source-report", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--grouped", action="store_true", help="predeclared C1/C4 fixed-support equivalence ladder")
    args = p.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    torch.set_num_threads(4)
    source = json.loads(args.source_report.read_text())
    digest = hashlib.sha256(args.captures.read_bytes()).hexdigest()
    if not source["complete"] or digest != source["capture_archive_sha256"]:
        raise ValueError("complete source report and matching capture archive required")
    captures = torch.load(args.captures, map_location="cpu", weights_only=True)
    from benchmarks.profile_adaptive_long_context import validate_capture_contract
    validate_capture_contract(source["capture_metadata"], captures)
    capture = next(c for c in captures if c["prompt_id"] == "technical" and c["layer"] == 0)
    provenance = next(c for c in source["capture_metadata"]["records"]
                      if c["prompt_id"] == "technical" and c["layer"] == 0)
    for name in ("q", "k", "v"):
        if hashlib.sha256(capture[name].view(torch.uint8).numpy().tobytes()).hexdigest() != provenance["tensor_sha256"][name]:
            raise ValueError("capture tensor hash mismatch")
    record = next(c for c in source["records"] if c["prompt_id"] == "technical" and c["layer"] == 0)
    diagnostic = {name: record[name] for name in ("group_size", "block_size", "budget")}
    diagnostic["schedules"] = [next(s for s in record["schedules"] if s["method"] == method)
                               for method in METHODS]
    q, k, v = [capture[name] for name in ("q", "k", "v")]
    references = {s["method"]: selected_reference(q, k, v, s["kept_blocks"])
                  for s in diagnostic["schedules"]}
    full = references["full"]
    for method, reference in references.items():
        if float((reference - full).norm(dim=-1).max()) > 1e-3 + 1e-10:
            raise AssertionError("saved schedule violates original omission budget")
    with torch.no_grad():
        native = native_headroom(q, k, v, diagnostic, full, references,
                                 buffer_copies=8, selected_observer=observe_selected,
                                 records_per_cta_options=(1, 4) if args.grouped else (1,))
    result = dict(schema="streamattn.adaptive_executor_grouped.v1" if args.grouped else SCHEMA,
        complete=True, diagnostic_only=True, performance_promotion=False,
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        prompt_id="technical", layer=0, archive_sha256=digest,
        source_report_sha256=hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
        contract=source["contract"], schedule_origin="unchanged saved full/contribution_triangle schedules; no search",
        support_hashes={s["method"]: support_hash(s["kept_blocks"]) for s in diagnostic["schedules"]},
        schedules=diagnostic["schedules"], native=native,
        records_per_cta_options=[1, 4] if args.grouped else [1],
        sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES})
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
