"""Whole mixed-ragged batch comparison with explicit query-interface costs.

Synthetic serving boundaries, not production traces or dispatch promotion.
Both padded and packed query contracts are measured so neither input layout
gets a hidden conversion advantage. KV pages are never gathered or repacked.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.micro_prefill_baselines import (
    FLASHINFER_WORKSPACE_BYTES, loaded_binary_provenance, runtime_provenance,
)
from benchmarks.profile_sm90_micro_prefill import _capture, _elapsed_graph_ms
from benchmarks.profile_sm90_micro_prefill_paged import SOURCE_PATHS, reference, reconstructed_lse
from benchmarks.profile_sm90_micro_prefill_semantics import check_output
from stream_attention.backends.sm90.micro_prefill_paged import PagedMicroPrefillPlan
from stream_attention.baseline_resolver import (
    ExactBaselineDescriptor, ExactBaselineMeasurement, fastest_measured_exact_baseline,
)
from stream_attention.inference_workload import AttentionBatchV2
from stream_attention.paged import PagedKVCache

SCHEMA = "streamattn.sm90_micro_prefill_mixed.v1"
BACKENDS = ("flashinfer_fa2", "flashinfer_fa3")
INTERFACES = ("padded", "packed")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def experiment_cases(suite):
    shapes = (
        ("short", [1, 2, 4, 8], [4093, 1021, 511, 2047]),
        ("heterogeneous", [1, 2, 4, 8, 16, 32, 48, 64],
         [63, 257, 1023, 2047, 4093, 8191, 12287, 16381]),
        ("long_tail", [64, 1, 2, 3], [32765, 1021, 4093, 16381]),
    )
    configs = ((64, 4, 16, "bf16"), (128, 8, 16, "bf16"),
               (64, 8, 32, "fp16"), (128, 4, 32, "fp16"))
    cases = [dict(trace=name, query_lengths=q, kv_lengths=k, d=d, g=g, hq=h,
                  dtype=dtype, layout=layout, causal=causal)
             for d, g, h, dtype in configs for name, q, k in shapes
             for layout in ("HND", "NHD") for causal in (False, True)]
    if suite == "smoke":
        return [cases[0], cases[3]]
    if suite == "replay":
        return cases[::4] + cases[3::4]
    if suite == "causal":
        return [c for c in cases if c["causal"]]
    return cases


def metadata(c):
    qs, ns = c["query_lengths"], c["kv_lengths"]
    if not qs or len(qs) != len(ns) or min(qs) < 1 or max(qs) > 64:
        raise ValueError("invalid mixed query lengths")
    if any(n < q for q, n in zip(qs, ns)):
        raise ValueError("append comparison requires KV length >= query length")
    m, pages = max(qs), (max(ns) + 15) // 16
    qi, pi, qslots, pslots, last = [0], [0], [], [], []
    for row, (nq, nk) in enumerate(zip(qs, ns)):
        count = (nk + 15) // 16
        qi.append(qi[-1] + nq)
        pi.append(pi[-1] + count)
        qslots.extend(row * m + i for i in range(nq))
        pslots.extend(row * pages + i for i in range(count))
        last.append((nk - 1) % 16 + 1)
    return dict(m=m, pages=pages, qi=qi, pi=pi, qslots=qslots, pslots=pslots, last=last)


def page_table(c, seed):
    meta = metadata(c)
    b, p = len(c["query_lengths"]), meta["pages"]
    ids = list(range(b * p + 3))
    random.Random(seed).shuffle(ids)
    table = [[-1] * p for _ in range(b)]
    for row, n in enumerate(c["kv_lengths"]):
        count = (n + 15) // 16
        table[row][:count] = ids[row*p:row*p+count]
        table[row][0] = ids[0]  # one read-only shared prefix page
    return table


def workload(c, table):
    requests = []
    for i, (q, n) in enumerate(zip(c["query_lengths"], c["kv_lengths"])):
        requests.append(dict(request_id=str(i), phase="decode" if q == 1 else "micro_prefill",
                             query_len=q, kv_len=n, prefix_group="shared",
                             shared_prefix_len=16, cache_page_ids=table[i][:(n+15)//16],
                             last_page_len=(n-1) % 16+1))
    return AttentionBatchV2.from_dict(dict(
        batch_id="mixed_" + digest(c), architecture="sm90", phase="mixed", requests=requests,
        attention_kind="gqa", q_heads=c["hq"], kv_heads=c["hq"]//c["g"],
        d_qk=c["d"], d_v=c["d"], q_dtype=c["dtype"], kv_dtype=c["dtype"],
        output_dtype=c["dtype"], scale_format="scalar_fp32", cache_kind="paged",
        cache_layout=c["layout"].lower(), page_size=16,
        mask_kind="causal" if c["causal"] else "noncausal", execution_mode="cuda_graph",
        maximum_captured_batch=len(requests), objective="latency",
    ))


def descriptor(name, revision):
    return ExactBaselineDescriptor.from_dict(dict(
        baseline_id=name, implementation="flashinfer.BatchPrefillWithPagedKVCacheWrapper." + name,
        revision=revision, architectures=["sm90"], phases=["mixed"], attention_kinds=["gqa"],
        q_dtypes=["bf16", "fp16"], kv_dtypes=["bf16", "fp16"], output_dtypes=["bf16", "fp16"],
        scale_formats=["scalar_fp32"], d_qk=[64, 128], d_v=[64, 128], cache_kinds=["paged"],
        cache_layouts=["hnd", "nhd"], mask_kinds=["causal", "noncausal"], page_sizes=[16],
        execution_modes=["cuda_graph"], supports_mixed_batches=True,
        supports_ragged_batches=True, supports_shared_prefixes=True,
    ))


def prepare_flashinfer(c, q, cache, meta, slots, packed_q, name):
    import flashinfer

    dev = q.device
    qi, pi, last = [torch.tensor(meta[n], dtype=torch.int32, device=dev) for n in ("qi", "pi", "last")]
    indices = torch.index_select(cache.page_table.flatten(), 0, slots)
    ws = torch.empty(FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout=c["layout"], backend=name.removeprefix("flashinfer_"), use_cuda_graph=True,
        qo_indptr_buf=qi, paged_kv_indptr_buf=pi,
        paged_kv_indices_buf=indices, paged_kv_last_page_len_buf=last,
    )
    wrapper.plan(qi, pi, indices, last, c["hq"], cache.kv_heads, c["d"], 16,
                 causal=c["causal"], q_data_type=q.dtype, kv_data_type=q.dtype)
    out, lse = torch.empty_like(packed_q), torch.empty(packed_q.shape[:2], device=dev, dtype=torch.float32)
    check_out = torch.empty_like(out)

    def refresh_pages():
        torch.index_select(cache.page_table.flatten(), 0, slots, out=indices)

    def run():
        refresh_pages()
        wrapper.run(packed_q, (cache.key, cache.value), out=out)
        return out

    def check_lse():
        # Do not overwrite the graph output: that would conceal stale replay.
        wrapper.run(packed_q, (cache.key, cache.value), out=check_out, lse=lse, return_lse=True)
        # FlashInfer 0.6.13 emits log2 LSE; the independent oracle uses ln.
        return lse * math.log(2)

    return SimpleNamespace(run=run, lse=check_lse, output=out, workspace=ws, wrapper=wrapper)


def initialize(c, q, cache, table, scale=1):
    q.normal_().mul_(scale)
    cache.key.normal_()
    cache.value.normal_()
    cache.page_table.copy_(torch.tensor(table, device=q.device, dtype=torch.int32))
    valid = torch.zeros(cache.num_pages, 16, dtype=torch.bool, device=q.device)
    for row, n in enumerate(c["kv_lengths"]):
        token = torch.arange(n, device=q.device)
        valid[cache.page_table[row, token // 16].long(), token % 16] = True
    mask = valid[:, :, None, None] if c["layout"] == "NHD" else valid[:, None, :, None]
    cache.key.masked_fill_(~mask, float("nan"))
    cache.value.masked_fill_(~mask, float("nan"))
    qlens = torch.tensor(c["query_lengths"], device=q.device)
    q.masked_fill_(torch.arange(q.shape[1], device=q.device)[None, :, None, None]
                   >= qlens[:, None, None, None], float("nan"))


def paired_timings(graphs, iterations, repeats):
    names = sorted(graphs)
    trials = []
    for trial in range(repeats):
        order = names[trial % len(names):] + names[:trial % len(names)]
        if trial % 2:
            order.reverse()
        times = {name: 1000 * _elapsed_graph_ms(graphs[name], iterations=iterations) for name in order}
        trials.append(dict(order=order, us=times))
    return trials


def profile_case(c, args, environment, provenance, binary_cache):
    torch.manual_seed(args.seed)
    meta = metadata(c)
    b, m, p, h, d, hk = len(c["query_lengths"]), meta["m"], meta["pages"], c["hq"], c["d"], c["hq"]//c["g"]
    dtype = torch.float16 if c["dtype"] == "fp16" else torch.bfloat16
    q = torch.empty(b, m, h, d, device="cuda", dtype=dtype)
    shape = (b*p+3, 16, hk, d) if c["layout"] == "NHD" else (b*p+3, hk, 16, d)
    cache = PagedKVCache(torch.empty(shape, device="cuda", dtype=dtype),
                        torch.empty(shape, device="cuda", dtype=dtype),
                        torch.empty(b, p, device="cuda", dtype=torch.int32),
                        torch.tensor(c["kv_lengths"], device="cuda", dtype=torch.int32), c["layout"])
    ql = torch.tensor(c["query_lengths"], device="cuda", dtype=torch.int32)
    qp = (torch.arange(m, device="cuda", dtype=torch.int64)[None, :] + (1 << 40)
          + cache.sequence_lengths[:, None] - ql[:, None]).contiguous()
    kp = (torch.arange(p*16, device="cuda", dtype=torch.int64)[None, :].expand(b, -1) + (1 << 40)).contiguous()
    qslots = torch.tensor(meta["qslots"], device="cuda", dtype=torch.int64)
    pslots = torch.tensor(meta["pslots"], device="cuda", dtype=torch.int64)
    packed_q = torch.empty(len(meta["qslots"]), h, d, device="cuda", dtype=dtype)
    table = page_table(c, args.seed)
    initialize(c, q, cache, table)
    row = dict(case=c, workload=workload(c, table).as_dict(), families={}, baselines={},
               graphs={}, correctness={}, unavailable={}, setup_including_jit_ms={}, loaded_binary_provenance={})
    runs, outputs, lses, plans = {}, {}, {}, {}
    for interface in INTERFACES:
        families = ("transposed", "natural", "natural_compact")
        if c["causal"]:
            families += ("natural_compact_affine", "natural_compact_interior")
        for family in families:
            name = f"{family}/{interface}"
            native_q = q if interface == "padded" else torch.empty_like(q)
            start = time.perf_counter()
            plan = PagedMicroPrefillPlan.build(
                native_q, cache, ql, natural=family != "transposed", cutlass_root=args.cutlass_root,
                compact_schedule=family.startswith("natural_compact"),
                affine_mode={"natural_compact_affine": "index", "natural_compact_interior": "interior"}.get(family, "none"),
                build_dir=args.build_dir, causal=c["causal"], compile_verbose=True,
                query_positions=qp if c["causal"] else None, key_positions=kp if c["causal"] else None,
            )
            row["setup_including_jit_ms"][name] = 1000*(time.perf_counter()-start)
            plans[name] = plan
            if interface == "padded":
                runs[name], outputs[name] = plan.run, plan.output
                lses[name] = lambda plan=plan: reconstructed_lse(plan)
            else:
                out = torch.empty_like(packed_q)

                def packed_run(plan=plan, out=out):
                    plan.query.view(-1, h, d).index_copy_(0, qslots, packed_q)
                    plan.run()
                    torch.index_select(plan.output.view(-1, h, d), 0, qslots, out=out)
                    return out

                runs[name], outputs[name] = packed_run, out
                lses[name] = lambda plan=plan: reconstructed_lse(plan).view(-1, h).index_select(0, qslots)
            row["families"][name] = dict(splits=plan.num_splits, workspace_bytes=plan.workspace_bytes)
            if plan.tasks is not None:
                row["families"][name].update(
                    producer_ctas=plan.tasks.shape[0], split_counts=plan.split_counts.cpu().tolist(),
                    max_kv_tiles_per_cta=int((plan.tasks[:, 3] - plan.tasks[:, 2]).max().item()),
                    lengths_frozen=True, affine_mode=plan.affine_mode)
            if family not in row["loaded_binary_provenance"]:
                row["loaded_binary_provenance"][family] = loaded_binary_provenance(
                    family, extension=plan.extension, cache=binary_cache)
    torch.index_select(q.view(-1, h, d), 0, qslots, out=packed_q)
    for backend in BACKENDS:
        try:
            start = time.perf_counter()
            fi = prepare_flashinfer(c, q, cache, meta, pslots, packed_q, backend)
            row["setup_including_jit_ms"][backend] = 1000*(time.perf_counter()-start)
            fi.run()
            row["baselines"][backend] = dict(workspace_bytes=FLASHINFER_WORKSPACE_BYTES)
            runs[backend + "/packed"], outputs[backend + "/packed"] = fi.run, fi.output
            lses[backend + "/packed"] = fi.lse
            padded_out = torch.empty_like(q)

            def padded_run(fi=fi, out=padded_out):
                torch.index_select(q.view(-1, h, d), 0, qslots, out=packed_q)
                fi.run()
                out.zero_()
                out.view(-1, h, d).index_copy_(0, qslots, fi.output)
                return out

            def padded_lse(fi=fi):
                result = torch.full((b*m, h), -torch.inf, device=q.device)
                result.index_copy_(0, qslots, fi.lse())
                return result.view(b, m, h)

            runs[backend + "/padded"], outputs[backend + "/padded"] = padded_run, padded_out
            lses[backend + "/padded"] = padded_lse
            row["loaded_binary_provenance"][backend] = loaded_binary_provenance(backend, cache=binary_cache)
        except Exception as exc:
            row["unavailable"][backend] = f"{type(exc).__name__}: {exc}"
    graphs = {}
    for stage in ("initial", "mutated_pages_and_values"):
        if stage != "initial":
            initialize(c, q, cache, page_table(c, args.seed+1), scale=3)
        torch.index_select(q.view(-1, h, d), 0, qslots, out=packed_q)
        ref, ref_lse = reference(q, cache, ql, qp, kp, c["causal"])
        for name in runs:
            if stage == "initial":
                graphs[name] = _capture(runs[name], warmup=3)
            else:
                graphs[name].replay()
            torch.cuda.synchronize()
            expected, norm = ref, ref_lse
            if name.endswith("/packed"):
                expected = ref.view(-1, h, d).index_select(0, qslots)
                norm = ref_lse.view(-1, h).index_select(0, qslots)
            check = check_output(SimpleNamespace(query=q, output=outputs[name]), expected, norm,
                                 observed_lse=lses[name]())
            row["correctness"].setdefault(name, {})[stage] = check
            if not check["passed"]:
                row["passed"] = False
                return row
    for interface in INTERFACES:
        selected = {k: v for k, v in graphs.items() if k.endswith("/" + interface)}
        trials = paired_timings(selected, args.iterations, args.repeats)
        medians = {name: statistics.median(t["us"][name] for t in trials) for name in selected}
        versions = {}
        for backend in BACKENDS:
            evidence = row["loaded_binary_provenance"].get(backend, {})
            if evidence.get("resolved") and provenance["versions"][backend] != "unresolved":
                versions[backend] = provenance["versions"][backend] + ";loaded=" + digest(evidence)
        wl = workload(c, page_table(c, args.seed+1))
        row["mutated_workload"] = wl.as_dict()
        measurements = [ExactBaselineMeasurement(
            baseline_id=name, backend_revision=revision, workload_sha256=wl.fingerprint,
            environment_sha256=digest(dict(environment=environment, query_interface=interface)),
            latency_us=medians[name + "/" + interface], correctness_passed=True, graph_replay=True,
        ) for name, revision in versions.items() if name + "/" + interface in medians]
        winner = fastest_measured_exact_baseline(wl, [descriptor(n, v) for n, v in versions.items()], measurements)
        row["graphs"][interface] = dict(paired_trials=trials, median_us=medians,
            workload_sha256=wl.fingerprint, measured_baselines=[asdict(x) for x in measurements],
            fastest_tested_baseline=asdict(winner) if winner else None)
    row["passed"] = True
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "replay", "full", "causal"), default="full")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--seed", type=int, default=17071)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("preserve existing evidence")
    if min(args.iterations, args.repeats) <= 0:
        raise ValueError("positive timing counts required")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = runtime_provenance()
    environment = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                       cuda_version=torch.version.cuda, provider=args.provider)
    paths = SOURCE_PATHS + ("benchmarks/profile_sm90_micro_prefill_mixed.py", "benchmarks/micro_prefill_baselines.py",
        "stream_attention/backends/sm90/ragged_schedule.py", "stream_attention/backends/sm90/micro_prefill_ragged_sources.py")
    result = dict(schema=SCHEMA, complete=False, environment=environment, seed=args.seed,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths},
        provenance=provenance, rows=[], planned_cases=len(experiment_cases(args.suite)),
        contract=dict(source="synthetic boundaries, not serving trace", kv="page16, no gather/repack",
                      timing="warm CUDA graph; complete producer+merge+interface conversion; output only",
                      metadata="lengths/CSR indptr planned once; page-ID compaction included each replay",
                      excludes="JIT, host planning, changed-length replan, request scheduling, KV append",
                      promotion=False, external_baselines=list(BACKENDS)))
    try:
        binary_cache = {}
        for c in experiment_cases(args.suite):
            print(json.dumps(dict(stage="case", case=c)), flush=True)
            row = profile_case(c, args, environment, provenance, binary_cache)
            result["rows"].append(row)
            print(json.dumps(dict(stage="checked", case=c, passed=row["passed"], unavailable=row["unavailable"])), flush=True)
            if not row["passed"]:
                raise AssertionError("mixed output/LSE or graph replay failed")
        result["complete"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
