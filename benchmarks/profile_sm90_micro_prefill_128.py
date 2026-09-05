"""Bounded M128 exact canary. No launcher, registration, or public promotion.

Examples on an already provisioned H100:
  python benchmarks/profile_sm90_micro_prefill_128.py --mode resources --suite smoke
  python benchmarks/profile_sm90_micro_prefill_128.py --suite smoke --output results.json
  python benchmarks/profile_sm90_micro_prefill_128.py --suite canary --matches-splits

Components are separately captured diagnostic kernels, not an additive latency
model. ``resources`` compiles and queries attributes without launching kernels.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
from pathlib import Path
import platform
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
from stream_attention.backends.sm90.micro_prefill import NaturalMicroPrefillPlan  # noqa: E402
from stream_attention.backends.sm90.micro_prefill_128 import (  # noqa: E402
    Natural128AsyncMicroPrefillPlan,
    balanced_tile_interval,
    resource_gate,
    source_fingerprint,
)


SCHEMA = "streamattn.sm90_micro_prefill_128_canary.v1"


def canary_cases(suite: str, splits: tuple[int, ...] = (1, 8, 16, 32)) -> list[dict]:
    if not splits or any(
        not isinstance(s, int) or isinstance(s, bool) or not 1 <= s <= 512
        for s in splits
    ):
        raise ValueError("splits must contain integers in [1,512]")
    if suite == "dependency":
        return [dict(batch=1, m=64, n=4096, hq=16, g=8, d=128, splits=16)]
    if suite == "smoke":
        return [
            dict(batch=1, m=17, n=320, hq=16, g=4, d=64, splits=4),
            dict(batch=1, m=64, n=4096, hq=16, g=8, d=128, splits=16),
            dict(batch=1, m=17, n=64, hq=16, g=4, d=128, splits=1),
        ]
    if suite == "boundary":
        shapes = [
            (7, 128, 2, 8),
            (9, 448, 6, 8),
            (15, 320, 4, 4),
            (17, 320, 4, 4),
            (31, 704, 7, 4),
            (33, 1216, 13, 4),
            (63, 4160, 16, 8),
        ]
        return [
            dict(batch=2, m=m, n=n, hq=32, g=g, d=d, splits=s)
            for m, n, s, g in shapes
            for d in (64, 128)
        ]
    if suite != "canary":
        raise ValueError("suite must be smoke, canary, or boundary")
    rows = [
        dict(batch=1, m=64, n=n, hq=16, g=g, d=d, splits=s)
        for n in (2048, 4096)
        for g in (4, 8)
        for d in (64, 128)
        for s in dict.fromkeys(splits)
        if s <= n // 64
    ]
    if not rows:
        raise ValueError("split selection leaves no executable canary cases")
    return rows


def fp32_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    out = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    g = q.shape[2] // k.shape[1]
    for b in range(q.shape[0]):
        for h in range(q.shape[2]):
            scores = (q[b, :, h].float() @ k[b, h // g].float().T) * q.shape[-1] ** -0.5
            out[b, :, h] = scores.softmax(-1) @ v[b, h // g].float()
            lse[b, :, h] = torch.logsumexp(scores, dim=-1)
    return out, lse


def output_error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        return dict(passed=False, max_abs=None, relative_l2=None)
    absolute = float((a - b).abs().max())
    relative = float(
        torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)
    )
    return dict(
        passed=absolute <= 0.04 and relative <= 0.02,
        max_abs=absolute,
        relative_l2=relative,
    )


def lse_error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    error = float((actual - expected).abs().max())
    return dict(
        passed=math.isfinite(error) and error <= 0.02,
        max_abs=error if math.isfinite(error) else None,
    )


def reconstructed_lse(plan: Natural128AsyncMicroPrefillPlan) -> torch.Tensor:
    if plan.direct:
        raise ValueError("direct mode has no partial LSE")
    b, m, hq, _ = plan.query.shape
    hk = plan.key_cache.shape[1]
    g = hq // hk
    merged = torch.logsumexp(plan.partial_lse * math.log(2), dim=1)
    return (
        merged.view(b, hk, plan.query_tiles, 128 // g, g)
        .reshape(b, hk, plan.query_tiles * (128 // g), g)[:, :, :m]
        .permute(0, 2, 1, 3)
        .reshape(b, m, hq)
    )


def protocol_fingerprint(baselines: str) -> dict[str, str]:
    paths = [
        "benchmarks/profile_sm90_micro_prefill_128.py",
        "benchmarks/profile_sm90_micro_prefill.py",
        "stream_attention/backends/sm90/micro_prefill_128.py",
        "stream_attention/backends/sm90/micro_prefill.py",
    ]
    if baselines == "all":
        paths += [
            "benchmarks/micro_prefill_baselines.py",
            "benchmarks/micro_prefill_optional_baselines.py",
        ]
    return {
        path: hashlib.sha256(
            (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in paths
        if (ROOT / path).is_file()
    }


def profile_case(c: dict, args: argparse.Namespace) -> dict:
    # No initialization kernels before the resource gate, including resources mode.
    q = torch.empty(
        c["batch"], c["m"], c["hq"], c["d"], device="cuda", dtype=torch.bfloat16
    )
    k = torch.empty(
        c["batch"],
        c["hq"] // c["g"],
        c["n"],
        c["d"],
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.empty(k.shape, device="cuda", dtype=torch.bfloat16)
    protocols = (
        ("overlap", "serial", "overlap_drained")
        if args.protocol == "all"
        else ("overlap", "serial")
        if args.protocol == "both"
        else (args.protocol,)
    )
    plans = {}
    for protocol in protocols:
        for direct in (False, True) if c["splits"] == 1 else (False,):
            name = "m128_" + protocol + ("_direct" if direct else "_partial")
            print(
                json.dumps(dict(stage="build_and_resources", candidate=name, case=c)),
                flush=True,
            )
            plans[name] = Natural128AsyncMicroPrefillPlan.build(
                q,
                k,
                v,
                num_splits=c["splits"],
                protocol=protocol,
                direct=direct,
                cutlass_root=args.cutlass_root,
                build_dir=args.build_dir,
                compile_verbose=args.compile_verbose,
                diagnostic_build=args.binary_diagnostics,
                lineinfo=args.binary_diagnostics,
            )
    intervals = [
        balanced_tile_interval(c["n"] // 64, c["splits"], s) for s in range(c["splits"])
    ]
    result = dict(
        case=c,
        matches_splits=args.matches_splits,
        source_sha256=source_fingerprint(c["d"]),
        state_abi="fp32_normalized_partial_output_and_log2_lse;public_lse_natural_log",
        tile_intervals=intervals,
        resources={name: plan.resources for name, plan in plans.items()},
        resource_gates={
            name: resource_gate(plan.resources, direct=plan.direct)
            for name, plan in plans.items()
        },
        plans={
            name: dict(
                producer_ctas=plan.producer_ctas,
                query_tiles=plan.query_tiles,
                workspace_bytes=plan.workspace_bytes,
                splits=plan.num_splits,
                direct=plan.direct,
                protocol=plan.protocol,
            )
            for name, plan in plans.items()
        },
    )
    result["resource_pass"] = all(plan.resource_pass for plan in plans.values())
    if args.binary_diagnostics:
        from benchmarks.sm90_binary_diagnostics import inspect_plan_binary

        result["binary_diagnostics"] = {
            name: inspect_plan_binary(
                plan, args.build_dir / "diagnostics", include_archive=False
            )
            for name, plan in plans.items()
        }
    if args.mode == "resources" or not result["resource_pass"]:
        result["status"] = (
            "resources_only" if result["resource_pass"] else "rejected_resources"
        )
        return result

    print(json.dumps(dict(stage="initialize_and_reference", case=c)), flush=True)
    generator = torch.Generator(device="cuda").manual_seed(4301)
    for tensor in (q, k, v):
        tensor.normal_(generator=generator)
    reference, reference_lse = fp32_reference(q, k, v)
    accuracy, composition = {}, {}
    for name, plan in plans.items():
        print(
            json.dumps(dict(stage="candidate_correctness", candidate=name)), flush=True
        )
        plan.run()
        accuracy[name] = dict(
            output=output_error(plan.output, reference),
            lse=lse_error(plan.lse, reference_lse),
        )
        if not plan.direct:
            accuracy[name]["reconstructed_lse"] = lse_error(
                reconstructed_lse(plan), plan.lse
            )
            combined, combined_lse = plan.output.clone(), plan.lse.clone()
            plan.partial_output.fill_(float("nan"))
            plan.partial_lse.fill_(float("nan"))
            plan.run_component("producer")
            plan.output.fill_(float("nan"))
            plan.lse.fill_(float("nan"))
            plan.run_component("merge")
            composition[name] = bool(
                torch.equal(plan.output, combined) & torch.equal(plan.lse, combined_lse)
            )
    print(json.dumps(dict(stage="natural64_build", case=c)), flush=True)
    m64 = NaturalMicroPrefillPlan.build(
        q,
        k,
        v,
        num_splits=c["splits"] if args.matches_splits else None,
        cutlass_root=args.cutlass_root,
        build_dir=(args.build_dir / f"natural64_d{c['d']}") if args.build_dir else None,
        compile_verbose=args.compile_verbose,
    )
    accuracy["natural64"] = dict(output=output_error(m64.run(), reference))
    result["natural64"] = dict(
        splits=m64.num_splits, workspace_bytes=m64.workspace_bytes
    )
    result["matches_splits"] = all(
        plan.num_splits == m64.num_splits for plan in plans.values()
    )
    result["accuracy"], result["composition"] = accuracy, composition
    if not all(
        check["passed"] for checks in accuracy.values() for check in checks.values()
    ) or not all(composition.values()):
        result["status"] = "rejected_correctness"
        result["exact_pass"] = False
        return result

    unavailable = {}
    baseline_runners = {}
    if args.baselines == "flash":
        baseline_runners["torch_flash"] = lambda: _flash_sdpa(q, k, v)
    elif args.baselines == "all":
        from benchmarks.micro_prefill_baselines import prepare_baselines

        baseline_runners, unavailable = prepare_baselines(q, k, v)
    baseline_accuracy, baseline_outputs = {}, {}
    primary = {name: plan.run for name, plan in plans.items()}
    primary["natural64"] = m64.run
    graphs = {name: _capture(run, warmup=args.warmup) for name, run in primary.items()}
    for name, run in baseline_runners.items():
        try:
            baseline_accuracy[name] = output_error(run(), reference)
            if not baseline_accuracy[name]["passed"]:
                unavailable[name] = "correctness_failed"
                continue

            def retained_run(name=name, run=run):
                baseline_outputs[name] = run()
                return baseline_outputs[name]

            graphs[name] = _capture(retained_run, warmup=args.warmup)
        except Exception as exc:
            unavailable[name] = f"{type(exc).__name__}: {exc}"
    component_names = []
    for name, plan in plans.items():
        if plan.direct:
            continue
        for which in ("producer", "merge"):
            if args.component not in ("all", which):
                continue
            graph_name = f"{name}:{which}"
            graphs[graph_name] = _capture(
                lambda plan=plan, which=which: plan.run_component(which),
                warmup=args.warmup,
            )
            component_names.append(graph_name)
    timed_names = (
        list(graphs) if args.component in ("all", "combined") else component_names
    )
    trials = []
    if args.mode == "benchmark":
        for repeat in range(args.repeats):
            order = (
                timed_names[repeat % max(1, len(timed_names)) :]
                + timed_names[: repeat % max(1, len(timed_names))]
            )
            if repeat % 2:
                order.reverse()
            trials.append(
                {
                    name: _elapsed_graph_ms(graphs[name], iterations=args.iterations)
                    for name in order
                }
            )
    medians = (
        {name: statistics.median(row[name] for row in trials) for name in timed_names}
        if trials
        else {}
    )

    q.mul_(2.5)
    k.neg_()
    v.add_(0.25)
    updated, updated_lse = fp32_reference(q, k, v)
    mutation = {}
    for name in primary:
        before = torch.cuda.memory_allocated()
        for _ in range(3):
            graphs[name].replay()
        torch.cuda.synchronize()
        delta = torch.cuda.memory_allocated() - before
        plan = m64 if name == "natural64" else plans[name]
        mutation[name] = dict(
            output=output_error(plan.output, updated), allocated_bytes_delta=delta
        )
        if name != "natural64":
            mutation[name]["lse"] = lse_error(plan.lse, updated_lse)
    baseline_mutation = {}
    for name in baseline_outputs:
        try:
            graphs[name].replay()
            baseline_mutation[name] = output_error(baseline_outputs[name], updated)
        except Exception as exc:
            baseline_mutation[name] = dict(
                passed=False, error=f"{type(exc).__name__}: {exc}"
            )
    eligible = [
        name
        for name in baseline_outputs
        if name in medians
        and baseline_accuracy[name]["passed"]
        and baseline_mutation.get(name, {}).get("passed", False)
    ]
    fastest = min(eligible, key=medians.get) if eligible else None
    result.update(
        exact_pass=all(
            check["output"]["passed"]
            and check.get("lse", {"passed": True})["passed"]
            and check["allocated_bytes_delta"] == 0
            for check in mutation.values()
        ),
        mutation=mutation,
        baseline_accuracy=baseline_accuracy,
        baseline_mutation=baseline_mutation,
        unavailable_baselines=unavailable,
        baseline_scope=args.baselines,
        fastest_measured_baseline=fastest,
        median_ms=medians,
        trials_ms=trials,
        speedup_vs_baseline={
            name: medians[fastest] / medians[name]
            for name in plans
            if fastest and name in medians
        },
        serial_over_overlap={
            suffix: medians[f"m128_serial_{suffix}"] / medians[f"m128_overlap_{suffix}"]
            for suffix in ("partial", "direct")
            if f"m128_serial_{suffix}" in medians
            and f"m128_overlap_{suffix}" in medians
        },
    )
    result["status"] = "passed_canary" if result["exact_pass"] else "rejected_replay"
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("resources", "correctness", "benchmark"), default="benchmark"
    )
    parser.add_argument(
        "--suite",
        choices=("smoke", "canary", "boundary", "dependency"),
        default="smoke",
    )
    parser.add_argument(
        "--protocol",
        choices=("overlap", "serial", "overlap_drained", "both", "all"),
        default="both",
    )
    parser.add_argument(
        "--component", choices=("all", "combined", "producer", "merge"), default="all"
    )
    parser.add_argument(
        "--matches-splits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument(
        "--baselines", choices=("flash", "all", "none"), default="flash"
    )
    parser.add_argument("--cutlass-root", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--binary-diagnostics", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile-verbose", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    if min(args.warmup, args.iterations, args.repeats) <= 0:
        parser.error("warmup, iterations, and repeats must be positive")
    if args.binary_diagnostics and args.build_dir is None:
        parser.error("binary diagnostics require --build-dir")
    canary_cases(args.suite, tuple(args.splits))
    return args


def main(argv=None) -> int:
    faulthandler.enable()
    args = parse_args(argv)
    if args.output and args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("this canary requires an already provisioned SM90 GPU")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = dict(
        schema=SCHEMA,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=props.name,
        sm_count=props.multi_processor_count,
        gpu_uuid=str(getattr(props, "uuid", "unavailable")),
        python=platform.python_version(),
        protocol_files=protocol_fingerprint(args.baselines),
        rows=[],
        evidence_kind="calibration_canary_not_public_promotion",
        configuration={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for case in canary_cases(args.suite, tuple(args.splits)):
            row = profile_case(case, args)
            result["rows"].append(row)
            print(json.dumps(dict(case=case, status=row["status"])), flush=True)
            if args.mode != "resources" and row["status"].startswith("rejected"):
                break
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    result["passed"] = all(
        not row["status"].startswith("rejected") for row in result["rows"]
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        print(payload)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
