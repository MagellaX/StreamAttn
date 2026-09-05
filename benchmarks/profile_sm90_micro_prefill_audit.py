"""Cross-provider exactness, incumbent and component audit for micro-prefill.

This is a calibration experiment, not public dispatch promotion. Baselines read
the original HND storage; output correctness is checked again after graph input
mutation before the exact baseline resolver may select a winner.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_sm90_micro_prefill import (  # noqa: E402
    _capture,
    _elapsed_graph_ms,
)
from benchmarks.micro_prefill_baselines import (  # noqa: E402
    BASELINE_IDS,
    loaded_binary_provenance,
    prepare_baselines,
    resolve_measurements,
    runtime_provenance,
)
from stream_attention.backends.sm90.micro_prefill import (  # noqa: E402
    MicroPrefillPlan,
    NaturalMicroPrefillPlan,
)
from stream_attention.backends.sm90.transposed_gqa_exact_sources import (  # noqa: E402
    CPP_SOURCE,
    CUDA_SOURCE,
)

SCHEMA = "streamattn.sm90_micro_prefill_audit.v2"


def cases(cohort: str) -> list[dict]:
    """Common replay anchors plus disjoint expansion; all are calibration."""
    rows = []
    for m in (2, 8, 16, 32, 64):
        for d in (64, 128):
            rows.append(
                dict(
                    batch=1,
                    m=m,
                    n=16384,
                    hq=16,
                    g=8,
                    d=d,
                    splits=None,
                    purpose="common",
                )
            )
    for m, n, s in (
        (3, 320, 4),
        (9, 448, 6),
        (17, 4160, None),
        (33, 704, 7),
        (63, 1216, 13),
    ):
        for d in (64, 128):
            rows.append(
                dict(batch=2, m=m, n=n, hq=32, g=4, d=d, splits=s, purpose="boundary")
            )
    if cohort != "smoke":
        batch, hq = (1, 16) if cohort == "modal" else (2, 32)
        for m in (4, 16, 32, 64):
            for n in (4096, 32768):
                for g in (4, 8):
                    for d in (64, 128):
                        rows.append(
                            dict(
                                batch=batch,
                                m=m,
                                n=n,
                                hq=hq,
                                g=g,
                                d=d,
                                splits=None,
                                purpose="expansion",
                            )
                        )
    return (
        rows
        if cohort != "smoke"
        else [rows[1], rows[11]]
        + [
            dict(
                batch=1,
                m=64,
                n=4096,
                hq=16,
                g=8,
                d=d,
                splits=None,
                purpose="m64_boundary",
            )
            for d in (64, 128)
        ]
    )


def fp32_reference(q, k, v):
    """Independent full QK softmax PV, one head at a time to bound memory."""
    out = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    group = q.shape[2] // k.shape[1]
    for b in range(q.shape[0]):
        for h in range(q.shape[2]):
            score = q[b, :, h].float() @ k[b, h // group].float().T
            score *= q.shape[-1] ** -0.5
            lse[b, :, h] = torch.logsumexp(score, dim=-1)
            out[b, :, h] = score.softmax(-1) @ v[b, h // group].float()
    return out, lse


def errors(actual, expected):
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all())
    if not finite:
        return dict(finite=False, max_abs=None, relative_l2=None, passed=False)
    max_abs = float((a - b).abs().max())
    relative_l2 = float(
        torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)
    )
    return dict(
        finite=True,
        max_abs=max_abs,
        relative_l2=relative_l2,
        passed=max_abs <= 0.04 and relative_l2 <= 0.02,
    )


def natural_lse(plan):
    b, m, h, _ = plan.query.shape
    hk = plan.key_cache.shape[1]
    g = h // hk
    merged = torch.logsumexp(plan.partial_lse * math.log(2), dim=1)
    return (
        merged.view(b, hk, plan.query_tiles, 64 // g, g)
        .reshape(b, hk, plan.query_tiles * (64 // g), g)[:, :, :m]
        .permute(0, 2, 1, 3)
        .reshape(b, m, h)
    )


def component(plan, which):
    plan.extension.natural_micro_prefill_components_out(
        plan.query,
        plan.key_cache,
        plan.value_cache,
        plan.partial_output,
        plan.partial_lse,
        plan.output,
        plan.num_splits,
        which,
    )
    return plan.output


def measure_case(c, args):
    print(json.dumps(dict(case=c, stage="native_plan_build")), flush=True)
    b, m, n, hq, g, d = (c[k] for k in ("batch", "m", "n", "hq", "g", "d"))
    ident = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    generator = torch.Generator(device="cuda").manual_seed(int(ident[:8], 16))
    kw = dict(device="cuda", dtype=torch.bfloat16, generator=generator)
    q = torch.randn(b, m, hq, d, **kw)
    k = torch.randn(b, hq // g, n, d, **kw)
    v = torch.randn(b, hq // g, n, d, **kw)
    build = dict(
        cutlass_root=args.cutlass_root, build_dir=args.build_dir, num_splits=c["splits"]
    )
    natural = NaturalMicroPrefillPlan.build(q, k, v, **build)
    transposed = MicroPrefillPlan.build(q, k, v, **build)
    print(json.dumps(dict(case=c, stage="fp32_reference")), flush=True)
    reference, reference_lse = fp32_reference(q, k, v)
    requested = getattr(args, "requested_baselines", BASELINE_IDS)
    runners, unavailable = prepare_baselines(q, k, v, requested)
    runners.update(
        natural=natural.run,
        transposed=transposed.run,
    )
    accuracy, graphs, captured_outputs = {}, {}, {}
    for name, run in runners.items():
        print(
            json.dumps(dict(case=c, stage="check_and_capture", runner=name)), flush=True
        )
        try:
            accuracy[name] = errors(run(), reference)
            if accuracy[name]["passed"]:

                def retained_run(name=name, run=run):
                    captured_outputs[name] = run()
                    return captured_outputs[name]

                graphs[name] = _capture(retained_run, warmup=args.warmup)
            else:
                unavailable[name] = "correctness_failed"
        except Exception as exc:
            if name in ("natural", "transposed"):
                raise
            unavailable[name] = f"{type(exc).__name__}: {exc}"
    natural.run()
    lse_max_abs = float((natural_lse(natural) - reference_lse).abs().max())
    if not math.isfinite(lse_max_abs):
        lse_max_abs = None
    combined = natural.output.clone()
    natural.partial_output.fill_(float("nan"))
    component(natural, 1)
    component(natural, 2)
    composed_correct = errors(natural.output, combined)
    graphs["natural_producer"] = _capture(
        lambda: component(natural, 1), warmup=args.warmup
    )
    graphs["natural_merge"] = _capture(
        lambda: component(natural, 2), warmup=args.warmup
    )

    trials = []
    names = list(graphs)
    for repeat in range(args.repeats):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        if repeat % 2:
            order.reverse()
        trials.append(
            {
                name: _elapsed_graph_ms(graphs[name], iterations=args.iterations)
                for name in order
            }
        )
    medians = {name: statistics.median(t[name] for t in trials) for name in names}
    native_names = [x for x in ("natural", "transposed") if x in medians]
    winner = min(native_names, key=medians.get) if native_names else None

    # Replay must read updated device contents, not the original captured output.
    mutation = {}
    q.mul_(2.5)
    k.neg_()
    v.add_(0.25)
    updated, _ = fp32_reference(q, k, v)
    for name in runners:
        if name in graphs:
            before = torch.cuda.memory_allocated()
            for _ in range(3):
                graphs[name].replay()
            torch.cuda.synchronize()
            delta = torch.cuda.memory_allocated() - before
            mutation[name] = dict(
                **errors(captured_outputs[name], updated), allocated_bytes_delta=delta
            )
    print(json.dumps(dict(case=c, stage="loaded_binary_provenance")), flush=True)
    cache = getattr(args, "binary_hash_cache", None)
    loaded = {name: loaded_binary_provenance(name, cache=cache) for name in requested}
    loaded.update(
        natural=loaded_binary_provenance("natural", extension=natural.extension, cache=cache),
        transposed=loaded_binary_provenance("transposed", extension=transposed.extension, cache=cache),
    )
    resolution = resolve_measurements(
        c, args.baseline_versions, args.environment_sha256, medians, accuracy, mutation,
        requested=requested, loaded_binary_provenance=loaded,
    )
    baseline = resolution["winner"]["baseline_id"] if resolution["winner"] else None
    exact = (
        all(accuracy.get(x, {}).get("passed", False) for x in ("natural", "transposed"))
        and lse_max_abs is not None
        and lse_max_abs <= 0.02
        and composed_correct["passed"]
        and all(
            mutation.get(x, {}).get("passed", False) for x in ("natural", "transposed")
        )
        and all(
            mutation.get(x, {}).get("allocated_bytes_delta") == 0
            for x in ("natural", "transposed")
        )
    )
    return dict(
        case=c,
        case_sha256=ident,
        accuracy=accuracy,
        mutation=mutation,
        natural_lse_max_abs=lse_max_abs,
        composed_correct=composed_correct,
        exact_pass=exact,
        trials_ms=trials,
        median_ms=medians,
        unavailable_baselines=unavailable,
        fastest_measured_baseline=baseline,
        baseline_resolution=resolution,
        loaded_binary_provenance=loaded,
        baseline_set_complete=resolution["complete"],
        requested_baseline_set_complete=all(
            any(r["baseline_id"] == name and r["correctness_passed"]
                for r in resolution["measurements"]) for name in requested
        ),
        oracle_native_winner=winner,
        oracle_speedup=medians[baseline] / medians[winner]
        if exact and baseline and winner
        else None,
        natural_splits=natural.num_splits,
        transposed_splits=transposed.num_splits,
        natural_workspace_bytes=natural.workspace_bytes,
        transposed_workspace_bytes=transposed.workspace_bytes,
        natural_producer_fraction=medians["natural_producer"]
        / medians.get("natural", float("inf")),
    )


def write_checkpoint(path, result):
    """Replace only this run's reserved output, never expose truncated JSON."""
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    faulthandler.enable()
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True)
    p.add_argument("--cohort", choices=("modal", "lightning", "smoke"), required=True)
    p.add_argument("--cutlass-root", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--baseline", choices=("all",) + BASELINE_IDS, default="all")
    args = p.parse_args(argv)
    args.requested_baselines = list(BASELINE_IDS) if args.baseline == "all" else [args.baseline]
    args.binary_hash_cache = {}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8"):
        pass
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("SM90 H100 required")
    if min(args.warmup, args.iterations, args.repeats) <= 0:
        raise ValueError("timing counts must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = runtime_provenance()
    result = dict(
        schema=SCHEMA,
        provider=args.provider,
        cohort=args.cohort,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        timing=dict(
            warmup=args.warmup, iterations=args.iterations, repeats=args.repeats
        ),
        baseline_provenance=provenance,
        requested_baselines=args.requested_baselines,
        loaded_binary_provenance_required=True,
        complete=False,
        device=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256((CPP_SOURCE + CUDA_SOURCE).encode()).hexdigest(),
        protocol_sha256=hashlib.sha256(
            b"".join(
                (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
                for path in (
                    "benchmarks/profile_sm90_micro_prefill_audit.py",
                    "benchmarks/profile_sm90_micro_prefill.py",
                    "stream_attention/backends/sm90/micro_prefill.py",
                    "benchmarks/micro_prefill_baselines.py",
                    "benchmarks/micro_prefill_optional_baselines.py",
                    "stream_attention/baseline_resolver.py",
                )
            )
        ).hexdigest(),
        evidence_kind="calibration_only_not_public_promotion",
        baseline_layout="direct_HND_no_KV_repack_per_request_FlashInfer_launches_included",
        rows=[],
    )
    result["gpu_inventory"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    args.baseline_versions = provenance["versions"]
    args.environment_sha256 = hashlib.sha256(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "torch",
                    "cuda",
                    "device",
                    "gpu_inventory",
                    "baseline_provenance",
                    "protocol_sha256",
                    "source_sha256",
                    "timing",
                )
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    result["environment_sha256"] = args.environment_sha256
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    write_checkpoint(args.output_json, result)
    for c in cases(args.cohort):
        try:
            row = measure_case(c, args)
        except BaseException as exc:
            result["failure"] = dict(case=c, type=type(exc).__name__, message=str(exc))
            write_checkpoint(args.output_json, result)
            raise
        result["rows"].append(row)
        print(
            json.dumps(
                dict(
                    case=c,
                    exact_pass=row["exact_pass"],
                    baseline=row["fastest_measured_baseline"],
                    speedup=row["oracle_speedup"],
                )
            ),
            flush=True,
        )
        write_checkpoint(args.output_json, result)
    result["complete"] = True
    write_checkpoint(args.output_json, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
