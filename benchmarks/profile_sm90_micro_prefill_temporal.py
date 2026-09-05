"""Causal R64 scheduling experiment; exactness does not imply public promotion.

The control, temporal, and fully drained candidates use identical split counts,
loads, softmax arithmetic, and normalized partial-output/log2-LSE merge ABI.
Only temporal instruction dependencies change. Component timings are diagnostic
and must not be added to predict complete-call latency.
"""

from __future__ import annotations

import argparse
import faulthandler
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

from benchmarks.profile_sm90_micro_prefill import (  # noqa: E402
    _capture,
    _elapsed_graph_ms,
    _flash_sdpa,
)
from benchmarks.profile_sm90_micro_prefill_128 import (  # noqa: E402
    fp32_reference,
    output_error,
    lse_error,
)
from stream_attention.backends.sm90.micro_prefill import NaturalMicroPrefillPlan  # noqa: E402


SCHEMA = "streamattn.sm90_micro_prefill_temporal.v1"


def experiment_cases(suite: str) -> list[dict]:
    common = dict(m=64, hq=16, g=8, d=128, splits=16)
    anchors = [
        dict(common, batch=b, n=n) for b, n in ((1, 4096), (1, 16384), (2, 4096))
    ]
    boundaries = [
        dict(batch=1, m=17, n=320, hq=16, g=4, d=64, splits=4),
        dict(batch=1, m=17, n=64, hq=16, g=4, d=128, splits=1),
        dict(batch=2, m=9, n=448, hq=16, g=8, d=128, splits=6),
        dict(batch=1, m=64, n=4096, hq=16, g=8, d=128, splits=24),
    ]
    if suite == "anchors":
        return anchors
    if suite == "smoke":
        return boundaries[:2] + anchors[:1]
    if suite == "boundary":
        return boundaries
    if suite == "split_diagnostic":
        return [dict(anchors[0], splits=8)]
    raise ValueError(f"unknown suite: {suite}")


def schedule_geometry(c: dict) -> dict:
    tiles = c["n"] // 64
    query_tiles = math.ceil(c["m"] * c["g"] / 64)
    work_units = c["batch"] * (c["hq"] // c["g"]) * query_tiles
    lengths = [
        ((s + 1) * tiles // c["splits"]) - (s * tiles // c["splits"])
        for s in range(c["splits"])
    ]
    return dict(
        packed_rows=64,
        kv_tile=64,
        query_tiles=query_tiles,
        work_units=work_units,
        producer_ctas=work_units * c["splits"],
        tiles_per_cta_min=min(lengths),
        tiles_per_cta_max=max(lengths),
        tiles_per_cta_mean=tiles / c["splits"],
        cta_tile_iterations=work_units * tiles,
        unique_kv_bytes=2 * c["batch"] * (c["hq"] // c["g"]) * c["n"] * c["d"] * 2,
        logical_kv_bytes=2 * work_units * c["n"] * c["d"] * 2,
        partial_bytes=work_units * c["splits"] * 64 * (c["d"] + 1) * 4,
        qk_pv_flops=4 * c["batch"] * c["m"] * c["hq"] * c["n"] * c["d"],
    )


def partial_lse(plan):
    b, m, hq, _ = plan.query.shape
    hk = plan.key_cache.shape[1]
    g = hq // hk
    return (
        torch.logsumexp(plan.partial_lse * math.log(2), dim=1)
        .view(b, hk, plan.query_tiles, 64 // g, g)
        .reshape(b, hk, plan.query_tiles * (64 // g), g)[:, :, :m]
        .permute(0, 2, 1, 3)
        .reshape(b, m, hq)
    )


def control_component(plan, which):
    return plan.extension.natural_micro_prefill_components_out(
        plan.query,
        plan.key_cache,
        plan.value_cache,
        plan.partial_output,
        plan.partial_lse,
        plan.output,
        plan.num_splits,
        1 if which == "producer" else 2,
    )


def profile_case(c, args):
    from stream_attention.backends.sm90.micro_prefill_temporal import (
        TemporalMicroPrefillPlan,
    )

    print(json.dumps(dict(stage="build", case=c)), flush=True)
    q = torch.empty(
        (c["batch"], c["m"], c["hq"], c["d"]), dtype=torch.bfloat16, device="cuda"
    )
    k = torch.empty(
        (c["batch"], c["hq"] // c["g"], c["n"], c["d"]), dtype=q.dtype, device=q.device
    )
    v = torch.empty_like(k)
    control = NaturalMicroPrefillPlan.build(
        q,
        k,
        v,
        num_splits=c["splits"],
        cutlass_root=args.cutlass_root,
        build_dir=args.build_dir / f"control_d{c['d']}",
        compile_verbose=True,
    )
    plans = {"control64": control}
    for protocol in ("drained", "temporal"):
        plans[protocol] = TemporalMicroPrefillPlan.build(
            q,
            k,
            v,
            num_splits=c["splits"],
            protocol=protocol,
            cutlass_root=args.cutlass_root,
            build_dir=args.build_dir / f"temporal_d{c['d']}",
            compile_verbose=True,
            diagnostic_build=args.binary_diagnostics,
        )
    row = dict(
        case=c,
        geometry=schedule_geometry(c),
        resources={
            name: getattr(plan, "resources", None) for name, plan in plans.items()
        },
    )
    row["workspace_bytes"] = {
        name: plan.workspace_bytes for name, plan in plans.items()
    }
    row["binary_diagnostics"] = {}
    if args.binary_diagnostics:
        from benchmarks.sm90_binary_diagnostics import inspect_extension

        for name, plan in plans.items():
            names = dict(
                producer=(
                    "streamattn_natural_wgmma_micro_prefill_partial_kernel"
                    if name == "control64"
                    else f"streamattn_temporal_micro_prefill_partial_kernel<{str(name == 'drained').lower()}>"
                ),
                merge="streamattn_natural_wgmma_micro_prefill_merge_kernel",
            )
            metadata = dict(
                build_directory=str(Path(plan.extension.__file__).parent),
                intermediates_dir=str(Path(plan.extension.__file__).parent),
                keep_intermediates=name != "control64",
                cutlass_root=str(args.cutlass_root),
            )
            row["binary_diagnostics"][name] = inspect_extension(
                plan.extension,
                args.build_dir / "diagnostics",
                kernel_names=names,
                runtime_resources=getattr(plan, "resources", None),
                build_metadata=metadata,
                include_archive=False,
            )
    if len(set(row["workspace_bytes"].values())) != 1:
        raise RuntimeError("causal experiment requires identical workspace formats")
    row["resource_pass"] = all(
        plan.resource_pass for name, plan in plans.items() if name != "control64"
    )
    if args.mode == "resources" or not row["resource_pass"]:
        row["status"] = (
            "resources_only" if row["resource_pass"] else "rejected_resources"
        )
        return row
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for tensor in (q, k, v):
        tensor.normal_(generator=generator)
    expected, expected_lse = fp32_reference(q, k, v)
    row["accuracy"], row["composition"] = {}, {}
    components = {"control64": lambda which: control_component(control, which)}
    components.update(
        {
            name: plan.run_component
            for name, plan in plans.items()
            if name != "control64"
        }
    )
    for name, plan in plans.items():
        plan.run()
        row["accuracy"][name] = dict(
            output=output_error(plan.output, expected),
            lse=lse_error(partial_lse(plan), expected_lse),
        )
        old = plan.output.clone()
        old_lse = plan.partial_lse.clone()
        plan.partial_output.fill_(float("nan"))
        plan.partial_lse.fill_(float("nan"))
        components[name]("producer")
        plan.output.fill_(float("nan"))
        components[name]("merge")
        row["composition"][name] = bool(
            torch.equal(plan.output, old) and torch.equal(plan.partial_lse, old_lse)
        )
    passed = all(
        x["passed"] for v in row["accuracy"].values() for x in v.values()
    ) and all(row["composition"].values())
    if not passed:
        return dict(row, status="rejected_correctness", exact_pass=False)

    baseline_outputs = {}

    def baseline():
        baseline_outputs["torch_flash"] = _flash_sdpa(q, k, v)
        return baseline_outputs["torch_flash"]

    row["baseline_accuracy"] = output_error(baseline(), expected)
    graphs = {
        name: _capture(plan.run, warmup=args.warmup) for name, plan in plans.items()
    }
    if row["baseline_accuracy"]["passed"]:
        graphs["torch_flash"] = _capture(baseline, warmup=args.warmup)
    for name, run in components.items():
        for which in ("producer", "merge"):
            graphs[f"{name}:{which}"] = _capture(
                lambda run=run, which=which: run(which), warmup=args.warmup
            )
    trials = []
    if args.mode == "benchmark":
        names = list(graphs)
        for trial in range(args.repeats):
            order = names[trial % len(names) :] + names[: trial % len(names)]
            if trial % 2:
                order.reverse()
            trials.append(
                {
                    name: _elapsed_graph_ms(graphs[name], iterations=args.iterations)
                    for name in order
                }
            )
    row["trials_ms"] = trials
    row["median_ms"] = (
        {name: statistics.median(t[name] for t in trials) for name in graphs}
        if trials
        else {}
    )
    row["paired_speedup_vs_control"] = {
        name: [t["control64"] / t[name] for t in trials]
        for name in ("drained", "temporal")
    }
    row["paired_speedup_vs_flash"] = (
        {name: [t["torch_flash"] / t[name] for t in trials] for name in plans}
        if "torch_flash" in graphs
        else {}
    )
    q.mul_(2.5)
    k.neg_()
    v.add_(0.25)
    updated, updated_lse = fp32_reference(q, k, v)
    row["mutation"] = {}
    for name, plan in plans.items():
        before = torch.cuda.memory_allocated()
        for _ in range(3):
            graphs[name].replay()
        torch.cuda.synchronize()
        delta = torch.cuda.memory_allocated() - before
        row["mutation"][name] = dict(
            output=output_error(plan.output, updated),
            lse=lse_error(partial_lse(plan), updated_lse),
            allocated_bytes_delta=delta,
        )
    if "torch_flash" in graphs:
        graphs["torch_flash"].replay()
        row["baseline_mutation"] = output_error(
            baseline_outputs["torch_flash"], updated
        )
        if not row["baseline_mutation"]["passed"]:
            row["paired_speedup_vs_flash"] = {}
    row["exact_pass"] = all(
        v["output"]["passed"] and v["lse"]["passed"] and v["allocated_bytes_delta"] == 0
        for v in row["mutation"].values()
    )
    row["status"] = "passed_canary" if row["exact_pass"] else "rejected_replay"
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("smoke", "anchors", "boundary", "split_diagnostic"),
        default="smoke",
    )
    parser.add_argument(
        "--mode", choices=("resources", "correctness", "benchmark"), default="benchmark"
    )
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binary-diagnostics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=7309)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("timing counts must be positive")
    return args


def main(argv=None):
    faulthandler.enable()
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("SM90 GPU required")
    paths = [
        Path(__file__),
        ROOT / "benchmarks/profile_sm90_micro_prefill_128.py",
        ROOT / "benchmarks/profile_sm90_micro_prefill.py",
    ]
    paths += list(
        (ROOT / "stream_attention/backends/sm90").glob("micro_prefill_temporal*.py")
    )
    paths += [ROOT / "stream_attention/backends/sm90/transposed_gqa_exact_sources.py"]
    props = torch.cuda.get_device_properties(0)
    result = dict(
        schema=SCHEMA,
        device=props.name,
        sm_count=props.multi_processor_count,
        gpu_uuid=str(getattr(props, "uuid", "unavailable")),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        source_files={
            str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in paths
        },
        timing_regime="warm_fixed_buffer_cuda_graph_replay_no_profiler",
        evidence_kind="causal_schedule_experiment_not_public_promotion",
        rows=[],
        complete=False,
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    )

    def checkpoint():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for case in experiment_cases(args.suite):
            row = profile_case(case, args)
            result["rows"].append(row)
            checkpoint()
            print(
                json.dumps(
                    dict(
                        case=case, status=row["status"], median_ms=row.get("median_ms")
                    )
                ),
                flush=True,
            )
            if row["status"].startswith("rejected"):
                break
        result["complete"] = len(result["rows"]) == len(experiment_cases(args.suite))
        result["passed"] = result["complete"] and all(
            not r["status"].startswith("rejected") for r in result["rows"]
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        checkpoint()
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
