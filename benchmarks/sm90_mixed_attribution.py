"""Post-affine diagnostics. Isolated graph times are deliberately nonadditive."""

import json
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace

import torch

from benchmarks.profile_sm90_micro_prefill import _capture
from benchmarks.profile_sm90_micro_prefill_paged import reconstructed_lse
from benchmarks.profile_sm90_micro_prefill_semantics import check_output
from stream_attention.backends.sm90.ragged_schedule import plan_ragged_schedule

CONTROL = "natural_compact_interior"
VARIANTS = {
    CONTROL: dict(min_kv_tiles=1, task_order="query"),
    "interior_kv_order": dict(min_kv_tiles=1, task_order="kv"),
    "interior_min2": dict(min_kv_tiles=2, task_order="query"),
    "interior_min2_kv_order": dict(min_kv_tiles=2, task_order="kv"),
}
COPY_VARIANTS = {
    CONTROL: VARIANTS[CONTROL],
    "interior_q_vector": dict(min_kv_tiles=1, task_order="query", q_vector_copy=True),
}
PAIR_VARIANTS = {
    "interior_q_vector": COPY_VARIANTS["interior_q_vector"],
    "interior_q_vector_page_pair": dict(COPY_VARIANTS["interior_q_vector"], page_pair_reuse=True),
}
ADDRESS_VARIANTS = {
    "interior_q_vector_page_pair": PAIR_VARIANTS["interior_q_vector_page_pair"],
    "interior_page_address_unsigned": dict(PAIR_VARIANTS["interior_q_vector_page_pair"],
                                          unsigned_page_address=True),
}


def variants(args):
    if getattr(args, "page_address", False):
        return ADDRESS_VARIANTS
    if getattr(args, "page_pair", False):
        return PAIR_VARIANTS
    return COPY_VARIANTS if getattr(args, "producer_copy", False) else VARIANTS


def useful_work(c):
    """Visible affine-causal work, not issued tile work or a latency model."""
    qs, ns = c["query_lengths"], c["kv_lengths"]
    if not c["causal"] or len(qs) != len(ns) or any(not 0 <= m <= n for m, n in zip(qs, ns)):
        raise ValueError("useful-work accounting requires affine append lengths N >= M")
    pairs = sum(m*n - m*(m-1)//2 for m, n in zip(qs, ns))
    return dict(visible_pairs_per_head=pairs, visible_head_pairs=pairs*c["hq"],
                useful_qk_pv_flops=4*c["hq"]*c["d"]*pairs,
                convention="equal QK/V dimensions; FMA=2; excludes softmax; not issued/padded work")


def geometry(c, variant):
    qs, ns, g, h, d = (c[k] for k in ("query_lengths", "kv_lengths", "g", "hq", "d"))
    config = ADDRESS_VARIANTS[variant] if variant in ADDRESS_VARIANTS else PAIR_VARIANTS[variant] if variant in PAIR_VARIANTS else COPY_VARIANTS[variant] if variant in COPY_VARIANTS else VARIANTS[variant]
    schedule = plan_ragged_schedule(qs, ns, capacity=max(qs), kv_heads=h//g,
        group_size=g, **{k: config[k] for k in ("min_kv_tiles", "task_order")})
    repeated_rows = sum(q*h*s for q, s in zip(qs, schedule.splits))
    return dict(producer_ctas=len(schedule.tasks), split_counts=schedule.splits,
        kv_tiles_per_task=[end-begin for _, _, begin, end in schedule.tasks],
        merge_ctas=len(qs)*max(qs)*h, valid_output_rows=sum(qs)*h,
        useful_row_slot_fraction=repeated_rows/(len(schedule.tasks)*64) if schedule.tasks else 0,
        partial_state_written_logical_bytes=len(schedule.tasks)*64*(d+1)*4,
        partial_state_read_minimum_logical_bytes=repeated_rows*(d+1)*4,
        traffic_note="logical state sizes, not measured memory transactions or DRAM traffic")


def flashinfer_telemetry(fi, backend):
    import flashinfer

    info = getattr(fi.wrapper, "_plan_info", None)
    result = dict(version=flashinfer.__version__, backend=backend,
                  plan_info=list(info) if isinstance(info, (tuple, list)) else None)
    if backend != "flashinfer_fa2" or flashinfer.__version__ != "0.6.13" or not info or len(info) != 15:
        result["decoded"] = False
        return result
    # Pinned PrefillPlanInfo::ToVector offsets are byte offsets into uint8 workspace.
    keys = ("padded_batch_size", "total_num_rows", "total_num_rows_offset", "cta_tile_q", "request_indices_offset",
            "qo_tile_indices_offset", "kv_tile_indices_offset", "merge_indptr_offset",
            "o_indptr_offset", "kv_chunk_size_ptr_offset", "v_offset", "s_offset",
            "block_valid_mask_offset", "enable_cuda_graph", "split_kv")
    plan = dict(zip(keys, map(int, info)))
    workspace = fi.wrapper._int_workspace_buffer

    def read(offset, count, dtype=torch.int32):
        size = torch.empty((), dtype=dtype).element_size()
        return workspace.narrow(0, offset, count*size).view(dtype).cpu().tolist()

    count = plan["padded_batch_size"]
    if not plan["split_kv"]:
        result.update(decoded=False, plan=plan, reason="non-split plans have no validity mask")
        return result
    valid = read(plan["block_valid_mask_offset"], count, torch.bool)
    result.update(decoded=True, plan=plan,
        kv_chunk_tokens=read(plan["kv_chunk_size_ptr_offset"], 1)[0],
        valid_task_count=sum(valid),
        tasks=[list(task) for task, live in zip(zip(
            read(plan["request_indices_offset"], count),
            read(plan["qo_tile_indices_offset"], count),
            read(plan["kv_tile_indices_offset"], count)), valid) if live])
    return result


def kernel_trace(run):
    """Capture actual CUDA symbols/launch metadata, never use these durations as a gate."""
    try:
        from torch.profiler import profile, ProfilerActivity
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trace.json"
            prof.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        kernels = [dict(name=e["name"], args=e.get("args", {}))
                   for e in events if e.get("cat") == "kernel"]
        return dict(collected=bool(kernels), kernels=kernels,
                    contract="instrumented launch metadata only; not paired timing or hardware counters")
    except Exception as exc:
        return dict(collected=False, error=f"{type(exc).__name__}: {exc}")


def perturbed_timings(graphs, repeats=5, iterations=3):
    """Working-set perturbation outside each timed replay, not a guaranteed cold cache."""
    eviction = torch.empty(128*1024*1024, dtype=torch.uint8, device="cuda")
    names, trials = sorted(graphs), []
    for trial in range(repeats):
        order = names[trial % len(names):] + names[:trial % len(names)]
        if trial % 2:
            order.reverse()
        times = {}
        for name in order:
            events = []
            for _ in range(iterations):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                eviction.fill_(trial+1)
                start.record()
                graphs[name].replay()
                end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            times[name] = statistics.mean(start.elapsed_time(end)*1000 for start, end in events)
        trials.append(dict(order=order, us=times))
    return dict(eviction_bytes=eviction.numel(), paired_trials=trials,
        median_us={name: statistics.median(t["us"][name] for t in trials) for name in names},
        contract="128MiB device fill before each timed replay; fill excluded; not guaranteed cold")


def poison_live_states(plan):
    plan.partial_output.fill_(float("nan"))
    # Unscheduled rectangular slots are identities, not producer outputs.
    plan.partial_lse.fill_(-torch.inf)
    plan.partial_lse[plan.tasks[:, 0].long(), plan.tasks[:, 1].long()] = torch.nan
    plan.output.fill_(float("nan"))


def attribute_case(c, args, plans, baselines, row, qslots, packed_q, reference, reference_lse):
    from benchmarks.profile_sm90_micro_prefill_mixed import paired_timings

    diagnostics = dict(native={}, baselines={}, isolated_timing_additive=False)
    names = ("registers_per_thread", "local_bytes_per_thread", "static_shared_bytes",
             "dynamic_shared_bytes", "resource_limited_ctas_per_sm")
    for variant in variants(args):
        plan = plans[variant+"/padded"]
        poison_live_states(plan)
        plan.run_component("producer")
        if not bool(torch.isnan(plan.output).all()):
            raise AssertionError("producer-only entry point wrote final output")
        plan.run_component("merge")
        check = check_output(SimpleNamespace(query=plan.query, output=plan.output),
            reference, reference_lse, observed_lse=reconstructed_lse(plan))
        if not check["passed"]:
            raise AssertionError(f"separate producer/real-state merge failed: {variant}: {check}")
        graphs = {name: _capture(lambda name=name: plan.run_component(name), warmup=3)
                  for name in ("producer", "merge")}
        trials = paired_timings(graphs, args.iterations, args.repeats)
        attributes = plan.extension.attributes(plan.query, c["layout"] == "NHD")
        diagnostics["native"][variant] = dict(geometry=geometry(c, variant),
            attributes={key: dict(zip(names, attributes[i*5:(i+1)*5]))
                        for i, key in enumerate(("producer", "merge"))},
            workspace_allocated_bytes=plan.workspace_bytes, component_correctness=check,
            isolated_paired_trials=trials,
            isolated_median_us={name: statistics.median(t["us"][name] for t in trials) for name in graphs})
    control = next(iter(variants(args)))
    plan = plans[control+"/packed"]
    _, _, h, d = plan.query.shape
    packed_out = torch.empty_like(packed_q)

    def adapters():
        plan.query.view(-1, h, d).index_copy_(0, qslots, packed_q)
        torch.index_select(plan.output.view(-1, h, d), 0, qslots, out=packed_out)

    graphs = {"packed_adapters": _capture(adapters, warmup=3)}
    for name, fi in baselines.items():
        graphs[name+"/page_id_compaction"] = _capture(fi.refresh_pages, warmup=3)
        graphs[name+"/attention"] = _capture(fi.attention, warmup=3)
        diagnostics["baselines"][name] = flashinfer_telemetry(fi, name)
    trials = paired_timings(graphs, args.iterations, args.repeats)
    diagnostics["interface_components"] = dict(paired_trials=trials,
        median_us={name: statistics.median(t["us"][name] for t in trials) for name in graphs})
    diagnostics["launch_traces"] = {control: kernel_trace(plans[control+"/padded"].run)}
    winner = row["graphs"]["packed"]["fastest_tested_baseline"]
    if winner:
        name = winner["baseline_id"]
        diagnostics["launch_traces"][name] = kernel_trace(baselines[name].run)
    return diagnostics
