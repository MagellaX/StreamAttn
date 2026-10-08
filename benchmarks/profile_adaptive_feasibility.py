"""Offline value-contribution frontier and matched native schedule headroom.

Full-information schedules diagnose feasibility; none is a runtime selector.
The last captured prefill query sees all supplied KV. GQA sharing is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "streamattn.adaptive_feasibility.v1"
SOURCE_FILES = (
    "benchmarks/profile_adaptive_feasibility.py",
    "benchmarks/profile_adaptive_real_activations.py",
    "benchmarks/run_lightning_adaptive_two_gate.py",
    "tests/test_adaptive_feasibility.py",
)
METHODS = ("full", "exact_mass_radius", "contribution_triangle", "hindsight_best_found")


def _search(mass, residual, radius, budget, method):
    """One schedule shared by every consumer of one physical KV head."""
    blocks = mass.shape[-1]
    norms = residual.norm(dim=-1)
    if method == "full" or budget == 0:
        return torch.ones(blocks, dtype=torch.bool)
    mass_order = mass.amax(0).argsort(stable=True).tolist()
    value_order = norms.amax(0).argsort(stable=True).tolist()
    orders = [mass_order] if method == "exact_mass_radius" else [value_order]
    if method == "hindsight_best_found":
        orders += [mass_order, list(range(blocks - 1, -1, -1))]
    best = torch.ones(blocks, dtype=torch.bool)
    for order in orders:
        keep = torch.ones_like(best)
        omitted_mass = torch.zeros(mass.shape[0], dtype=torch.float64)
        omitted_residual = torch.zeros_like(residual[:, 0])
        triangle = torch.zeros_like(omitted_mass)
        for _ in range(3 if method == "hindsight_best_found" else 1):
            changed = False
            for block in order:
                if not keep[block] or int(keep.sum()) == 1:
                    continue
                total_mass = omitted_mass + mass[:, block]
                denominator = 1 - total_mass
                if (denominator <= 1e-12).any():
                    continue
                total_residual = omitted_residual + residual[:, block]
                total_triangle = triangle + norms[:, block]
                if method == "exact_mass_radius":
                    bound = 2 * radius * total_mass
                elif method == "contribution_triangle":
                    bound = total_triangle / denominator
                else:
                    bound = total_residual.norm(dim=-1) / denominator
                if torch.isfinite(bound).all() and bool((bound <= budget).all()):
                    keep[block] = False
                    omitted_mass, omitted_residual = total_mass, total_residual
                    triangle = total_triangle
                    changed = True
            if not changed:
                break
        if int(keep.sum()) < int(best.sum()):
            best = keep
    return best


def frontier(q, k, v, query_positions, *, budget=1e-3, block_size=32):
    """FP64 diagnostic, B1/M1, one shared schedule per complete GQA group.

    With a_b=sum_b p and c_b=sum_b p*v, w_b=c_b-a_b*o_full.
    Error after omitting U is ||sum_U w_b|| / (1-sum_U a_b).
    The triangle bound replaces the numerator by sum_U ||w_b||.
    """
    if not math.isfinite(budget) or budget < 0 or block_size < 1:
        raise ValueError("finite nonnegative budget and positive block size required")
    if q.ndim != 4 or k.ndim != 4 or q.shape[0:2] != (1, 1):
        raise ValueError("feasibility contract requires B1/M1 in BSHD")
    if k.shape != v.shape or k.shape[0] != 1 or q.shape[-1] != k.shape[-1]:
        raise ValueError("matching BSHD Q/K/V dimensions required")
    n, hk, dim = k.shape[1:]
    hq = q.shape[2]
    if not n or not hk or hq % hk or query_positions.tolist() != [n - 1]:
        raise ValueError("complete GQA groups and final all-visible query required")
    if not all(bool(torch.isfinite(t).all()) for t in (q, k, v)):
        raise ValueError("finite activations required")
    group = hq // hk
    qh = q[0, 0].double().cpu()
    kh = k[0].permute(1, 0, 2).double().cpu().repeat_interleave(group, 0)
    vh = v[0].permute(1, 0, 2).double().cpu().repeat_interleave(group, 0)
    scores = torch.einsum("hd,hnd->hn", qh, kh) / math.sqrt(dim)
    p = scores.softmax(-1)
    full = torch.einsum("hn,hnd->hd", p, vh)
    nb = math.ceil(n / block_size)
    mass = torch.stack([p[:, b * block_size:(b + 1) * block_size].sum(-1)
                        for b in range(nb)], 1)
    moment = torch.stack([torch.einsum("hn,hnd->hd",
        p[:, b * block_size:(b + 1) * block_size],
        vh[:, b * block_size:(b + 1) * block_size]) for b in range(nb)], 1)
    residual = moment - mass[..., None] * full[:, None]
    radius = (vh - vh.mean(1, keepdim=True)).norm(dim=-1).amax(-1)
    rows, references = [], {}
    slack = 1e-10 * (1 + full.norm(dim=-1))
    for method in METHODS:
        kept = torch.stack([_search(mass[h * group:(h + 1) * group],
            residual[h * group:(h + 1) * group], radius[h * group:(h + 1) * group],
            budget, method) for h in range(hk)])
        head_keep = kept.repeat_interleave(group, 0)
        omitted_mass = (mass * ~head_keep).sum(-1)
        omitted_residual = (residual * (~head_keep)[..., None]).sum(1)
        triangle = (residual.norm(dim=-1) * ~head_keep).sum(-1) / (1 - omitted_mass)
        token_keep = head_keep.repeat_interleave(block_size, -1)[:, :n]
        weights = p * token_keep
        selected = torch.einsum("hn,hnd->hd", weights, vh) / weights.sum(-1)[:, None]
        error = (selected - full).norm(dim=-1)
        identity_error = (omitted_residual / (1 - omitted_mass)[:, None]
                          - (full - selected)).norm(dim=-1)
        bound = (2 * radius * omitted_mass if method == "exact_mass_radius" else
                 triangle if method == "contribution_triangle" else
                 error if method == "hindsight_best_found" else torch.zeros_like(error))
        if (not torch.isfinite(selected).all() or (identity_error > slack).any()
                or (error > budget + slack).any() or (error > bound + slack).any()):
            raise AssertionError("cumulative schedule check failed")
        lengths = torch.tensor([min(block_size, n - b * block_size) for b in range(nb)])
        retained_tokens = (kept * lengths).sum(-1)
        rows.append(dict(method=method, kept_blocks=[r.nonzero().flatten().tolist() for r in kept],
            retained_tokens_per_kv_head=retained_tokens.tolist(),
            omitted_token_fraction=1 - float(retained_tokens.sum()) / (hk * n),
            max_output_l2_error=float(error.max()), max_bound=float(bound.max()),
            max_identity_residual=float(identity_error.max()),
            full_information=True, deployable_selector=False,
            globally_optimal=False, certified_schedule=method != "hindsight_best_found",
            certificate="offline_contribution_triangle" if method == "contribution_triangle" else
                        "offline_mass_radius" if method == "exact_mass_radius" else
                        "actual_output_check_only" if method == "hindsight_best_found" else "no_omission"))
        references[method] = selected
    return dict(group_size=group, block_size=block_size, budget=budget,
                q_shape=list(q.shape), kv_shape=list(k.shape), schedules=rows), full, references


def graph_buffer_indices(copies, condition):
    """Match graph call count for fixed or rotating prepared buffer sets."""
    if copies < 1:
        raise ValueError("positive buffer copies required")
    if condition == "warm_fixed_buffer":
        return [0] * copies
    if condition == "rotating_working_set":
        if copies < 2:
            raise ValueError("rotation requires independent buffers")
        return list(range(copies))
    raise ValueError("unknown working-set condition")


def native_headroom(q, k, v, diagnostic, full, references, *, iterations=100, trials=7,
                    buffer_copies=1):
    """Offline route preparation; complete allocation-free native call is timed."""
    from benchmarks.micro_prefill_baselines import baseline_versions, prepare_baselines
    from benchmarks.profile_paged_exact_decode import _flashinfer_runner
    from benchmarks.profile_sm90_micro_prefill import _capture, _elapsed_graph_ms
    from stream_attention.paged import PagedExactDecodePlan, PagedKVCache, PagedSelectedDecodePlan
    from stream_attention.planning import AttentionProblem, AttentionTilePlan
    from stream_attention.selected_routes import prepare_paged_routes64

    if torch.cuda.get_device_capability() != (9, 0) or q.dtype != torch.bfloat16:
        raise ValueError("matched executor requires H100 and BF16")
    n, hk, dim = k.shape[1:]
    if n % 16 or diagnostic["group_size"] not in (4, 8) or dim not in (64, 128):
        raise ValueError("native scope requires page-16, G4/G8, D64/D128")
    conditions = ["warm_fixed_buffer"]
    if buffer_copies > 1:
        conditions.append("rotating_working_set")
    indices = {c: graph_buffer_indices(buffer_copies, c) for c in conditions}
    replicas, unavailable, keepalive = [], {}, []
    for replica in range(buffer_copies):
        qc, kc, vc = [t.to(device="cuda", copy=True).contiguous() for t in (q, k, v)]
        cache = PagedKVCache(kc.view(n // 16, 16, hk, dim), vc.view(n // 16, 16, hk, dim),
            torch.arange(n // 16, device="cuda", dtype=torch.int32)[None],
            torch.tensor([n], device="cuda", dtype=torch.int32), "NHD")
        runners, missing = prepare_baselines(qc, kc.transpose(1, 2).contiguous(),
                                            vc.transpose(1, 2).contiguous())
        unavailable[str(replica)] = missing
        for backend in ("fa2", "fa3"):
            name = "paged_flashinfer_" + backend
            try:
                runners[name], _ = _flashinfer_runner(qc, cache, workspace_mb=128, backend=backend)
            except Exception as exc:
                missing[name] = f"{type(exc).__name__}: {exc}"
        exact_plan = PagedExactDecodePlan.build(qc, cache)
        runners["streamattn_full_native"] = exact_plan.run
        replicas.append((qc, kc, vc, cache, runners))
    if len({r[1].data_ptr() for r in replicas}) != buffer_copies:
        raise AssertionError("KV rotation must use independent allocations")

    def check(runs, reference):
        errors, l2s, combined = [], [], []
        for run in runs:
            output = run().detach().float().cpu().reshape_as(reference)
            difference = output.double() - reference
            error = float(difference.abs().max())
            if not torch.isfinite(output).all() or error > 0.02:
                raise AssertionError(f"native reference component error {error}")
            errors.append(error)
            l2s.append(float(difference.norm(dim=-1).max()))
            combined.append(float((output.double() - full).norm(dim=-1).max()))
        return dict(max_abs_error_vs_fp64=max(errors), max_l2_error_vs_fp64=max(l2s),
                    max_combined_l2_error_vs_full_fp64=max(combined), replicas_checked=len(runs))

    def capture_cycle(runs, condition):
        order = indices[condition]
        def cycle():
            for i in order:
                runs[i]()
        return _capture(cycle, warmup=3)

    def elapsed(graph):
        return _elapsed_graph_ms(graph, iterations=iterations) / buffer_copies

    exact, exact_checks = {}, {}
    names = set.intersection(*(set(r[4]) for r in replicas))
    for name in sorted(names):
        runs = [r[4][name] for r in replicas]
        try:
            exact_checks[name] = check(runs, full)
            exact[name] = runs
        except Exception as exc:
            unavailable[name] = f"run:{type(exc).__name__}: {exc}"
    if not any("flashinfer" in name for name in exact):
        raise RuntimeError("no correct FlashInfer comparison; headroom result is incomplete")

    selected, reused = [], {}
    for schedule in diagnostic["schedules"]:
        signature = tuple(map(tuple, schedule["kept_blocks"]))
        if signature in reused:
            selected.append(dict(reused[signature], method=schedule["method"],
                                 reused_identical_schedule=True))
            continue
        start, plans = time.perf_counter(), []
        for qc, _, _, cache, _ in replicas:
            problem = AttentionProblem.from_paged(qc, cache, guarantee="schedule_exact")
            logical = AttentionTilePlan.selected(problem, logical_tile_size=diagnostic["block_size"],
                tile_ids_per_row=signature, policy_id="offline-feasibility",
                reason=schedule["method"], route_granularity="kv_group", schedule_epoch=1)
            routes = prepare_paged_routes64(logical, cache)
            plans.append(PagedSelectedDecodePlan.build(qc, cache, routes, schedule_epoch=1))
        keepalive.extend(plans)
        torch.cuda.synchronize()
        preparation_ms = (time.perf_counter() - start) * 1000
        runs = [p.run for p in plans]
        correctness = check(runs, references[schedule["method"]])
        if schedule["method"] == "full":
            exact["streamattn_selected_full_control"] = runs
            exact_checks["streamattn_selected_full_control"] = correctness
        plan = plans[0]
        item = dict(method=schedule["method"], preparation_ms_excluded=preparation_ms,
            correctness=correctness, route_count=routes.route_count, producer_ctas=plan.producer_ctas,
            max_routes_per_row=plan.max_routes_per_row, metadata_bytes=routes.metadata_bytes,
            workspace_bytes=plan.workspace_bytes, group_route_efficiency=routes.group_route_efficiency,
            active_kv_payload_bytes_per_copy=sum(schedule["retained_tokens_per_kv_head"]) * dim * k.element_size() * 2,
            runs=runs)
        reused[signature] = item
        selected.append(item)

    results = {}
    for condition in conditions:
        graphs, measurements = {}, []
        for name, runs in exact.items():
            try:
                graph = capture_cycle(runs, condition)
                pilot = [elapsed(graph) for _ in range(3)]
                graphs[name] = graph
                measurements.append(dict(name=name, median_ms=statistics.median(pilot),
                                         samples_ms=pilot, **exact_checks[name]))
            except Exception as exc:
                unavailable[f"{condition}/{name}"] = f"graph:{type(exc).__name__}: {exc}"
        if not any("flashinfer" in name for name in graphs):
            raise RuntimeError(f"no graph-correct FlashInfer for {condition}")
        fastest = min(measurements, key=lambda r: r["median_ms"])["name"]
        rows, timed = [], {}
        for item in selected:
            signature = tuple(item["runs"])
            if signature in timed:
                rows.append(dict(timed[signature], method=item["method"], reused_identical_schedule=True))
                continue
            graph = capture_cycle(item["runs"], condition)
            paired = []
            for trial in range(trials):
                order = ("exact", "selected") if trial % 2 == 0 else ("selected", "exact")
                times = {name: elapsed(graphs[fastest] if name == "exact" else graph) for name in order}
                paired.append(dict(trial=trial, order=list(order), **times,
                    headroom_ms=times["exact"] - times["selected"],
                    speedup=times["exact"] / times["selected"]))
            row = dict({key: value for key, value in item.items() if key != "runs"},
                paired=paired, headroom_ms=statistics.median(r["headroom_ms"] for r in paired),
                speedup=statistics.median(r["speedup"] for r in paired),
                positive_trials=sum(r["headroom_ms"] > 0 for r in paired),
                active_kv_payload_bytes_per_cycle=item["active_kv_payload_bytes_per_copy"] * len(set(indices[condition])))
            rows.append(row)
            timed[signature] = row
        results[condition] = dict(fastest_tested_correct_exact=fastest,
            baseline_measurements=measurements, selected=rows, buffer_indices=indices[condition],
            calls_per_graph=buffer_copies, condition=condition)
    common = dict(unavailable=unavailable, versions=baseline_versions(),
        full_native_backend=exact_plan.backend, timing="alternating paired CUDA graph replay; complete call; ms per attention call",
        preparation_and_decisions_included=False, kv_gather_in_timing=False,
        buffer_copies=buffer_copies, full_kv_payload_bytes_per_copy=(k.numel() + v.numel()) * k.element_size(),
        hardware_memory_traffic_measured=False, cache_residency_guaranteed=False,
        rounding_gate=dict(max_abs_per_component=0.02, separate_from_omission_budget=True,
                           formal_native_roundoff_certificate=False))
    if buffer_copies == 1:
        return dict(**common, **results["warm_fixed_buffer"])
    return dict(**common, conditions=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, nargs="+", required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--budget", type=float, default=1e-3)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--native", action="store_true")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--trials", type=int, default=7)
    args = p.parse_args()
    if args.iterations < 1 or args.trials < 1:
        p.error("positive timing iterations/trials required")
    torch.set_num_threads(4)
    records, inputs = [], []
    from benchmarks.profile_adaptive_real_activations import evaluate
    for path in args.captures:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs.append(dict(name=path.name, sha256=digest))
        for capture in torch.load(path, map_location="cpu", weights_only=True):
            q = capture["q"][:, -1:].contiguous()
            positions = capture["query_positions"][-1:]
            k, v = capture["k"], capture["v"]
            diagnostic, full, references = frontier(q, k, v, positions,
                budget=args.budget, block_size=args.block_size)
            cheap = evaluate(q, k, v, positions, budget=args.budget, block_size=args.block_size,
                             head_groups=(diagnostic["group_size"],), query_tiles=(1,))
            record = dict(capture=path.name, archive_sha256=digest, prompt_id=capture["prompt_id"],
                layer=capture["layer"], **diagnostic, current_certificate=cheap["results"])
            print(json.dumps(dict(stage="offline_schedule", capture=path.name, layer=capture["layer"],
                prompt_id=capture["prompt_id"], omitted={r["method"]:r["omitted_token_fraction"]
                                                       for r in diagnostic["schedules"]})), flush=True)
            if args.native:
                # Route metadata validation needs ordinary tensor version counters.
                with torch.no_grad():
                    record["native"] = native_headroom(q, k, v, diagnostic, full, references,
                        iterations=args.iterations, trials=args.trials)
            records.append(record)
            partial = dict(schema=SCHEMA, complete=False, records=records)
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(partial), flush=True)
    result = dict(schema=SCHEMA, complete=True, diagnostic_only=True, performance_promotion=False,
        device=torch.cuda.get_device_name() if args.native else "cpu", torch=torch.__version__,
        conditioning="dense_upstream; final captured prefill query; not autoregressive replay",
        contract=dict(batch=1, query_length=1, head_sharing="full_GQA_group", block_size=args.block_size,
            cumulative_output_l2_budget=args.budget, arithmetic="FP64 omission diagnostic; separate BF16 gate",
            search="deterministic greedy best of three orders; no optimality claim"),
        inputs=inputs, sources={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCE_FILES},
        records=records, interpretation="Nonpositive headroom rejects this schedule/executor combination, "
            "not adaptive attention globally. Offline schedules and excluded decisions are not a deployable speedup.")
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
