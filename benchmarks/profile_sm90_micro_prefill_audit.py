"""Cross-provider exactness, incumbent and component audit for micro-prefill.

This is a calibration experiment, not public dispatch promotion. Baselines get
prepared NHD KV outside timing; no claim includes their layout conversion cost.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_sm90_micro_prefill import (  # noqa: E402
    _capture,
    _elapsed_graph_ms,
    _flash_sdpa,
)
from stream_attention.backends.sm90.micro_prefill import (  # noqa: E402
    MicroPrefillPlan,
    NaturalMicroPrefillPlan,
)
from stream_attention.backends.sm90.transposed_gqa_exact_sources import (  # noqa: E402
    CPP_SOURCE,
    CUDA_SOURCE,
)

SCHEMA = "streamattn.sm90_micro_prefill_audit.v1"


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
    return rows if cohort != "smoke" else [rows[1], rows[11]]


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


def flashinfer_runner(q, k, v, backend):
    import flashinfer

    b, m, hq, d = q.shape
    hk, n = k.shape[1:3]
    # Give incumbents a prepared native layout; conversion is not timed.
    kn = k.transpose(1, 2).contiguous().view(b * n, hk, d)
    vn = v.transpose(1, 2).contiguous().view(b * n, hk, d)
    qn = q.view(b * m, hq, d)
    out = torch.empty_like(qn)
    qi = torch.arange(b + 1, device=q.device, dtype=torch.int32) * m
    ki = torch.arange(b + 1, device=q.device, dtype=torch.int32) * n
    workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qi,
        kv_indptr_buf=ki,
        backend=backend,
    )
    wrapper.plan(
        qi, ki, hq, hk, d, causal=False, q_data_type=q.dtype, kv_data_type=k.dtype
    )

    def run():
        wrapper.run(qn, kn, vn, out=out)
        return out.view_as(q)

    return run


def measure_case(c, args):
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
    reference, reference_lse = fp32_reference(q, k, v)
    runners = dict(
        natural=natural.run,
        transposed=transposed.run,
        torch_flash=lambda: _flash_sdpa(q, k, v),
    )
    unavailable = {}
    for backend in ("fa2", "fa3"):
        try:
            runners[f"flashinfer_{backend}"] = flashinfer_runner(q, k, v, backend)
        except Exception as exc:
            unavailable[f"flashinfer_{backend}"] = f"{type(exc).__name__}: {exc}"
    accuracy, graphs = {}, {}
    for name, run in runners.items():
        try:
            accuracy[name] = errors(run(), reference)
            if accuracy[name]["passed"]:
                graphs[name] = _capture(run, warmup=args.warmup)
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
    baseline_names = [
        x for x in ("torch_flash", "flashinfer_fa2", "flashinfer_fa3") if x in medians
    ]
    baseline = min(baseline_names, key=medians.get) if baseline_names else None
    native_names = [x for x in ("natural", "transposed") if x in medians]
    winner = min(native_names, key=medians.get) if native_names else None

    # Replay must read updated device contents, not the original captured output.
    mutation = {}
    q.mul_(2.5)
    k.neg_()
    v.add_(0.25)
    updated, _ = fp32_reference(q, k, v)
    for name, plan in (("natural", natural), ("transposed", transposed)):
        if name in graphs:
            before = torch.cuda.memory_allocated()
            for _ in range(3):
                graphs[name].replay()
            torch.cuda.synchronize()
            delta = torch.cuda.memory_allocated() - before
            mutation[name] = dict(
                **errors(plan.output, updated), allocated_bytes_delta=delta
            )
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
        baseline_set_complete=all(
            x in medians for x in ("torch_flash", "flashinfer_fa2", "flashinfer_fa3")
        ),
        oracle_native_winner=winner,
        oracle_speedup=(
            medians[baseline] / medians[winner] if baseline and winner else None
        ),
        natural_splits=natural.num_splits,
        transposed_splits=transposed.num_splits,
        natural_workspace_bytes=natural.workspace_bytes,
        transposed_workspace_bytes=transposed.workspace_bytes,
        natural_producer_fraction=medians["natural_producer"]
        / medians.get("natural", float("inf")),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True)
    p.add_argument("--cohort", choices=("modal", "lightning", "smoke"), required=True)
    p.add_argument("--cutlass-root", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("SM90 H100 required")
    if min(args.warmup, args.iterations, args.repeats) <= 0:
        raise ValueError("timing counts must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(
        schema=SCHEMA,
        provider=args.provider,
        cohort=args.cohort,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        flashinfer=importlib.metadata.version("flashinfer-python"),
        device=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256((CPP_SOURCE + CUDA_SOURCE).encode()).hexdigest(),
        protocol_sha256=hashlib.sha256(
            b"".join(
                (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
                for path in (
                    "benchmarks/profile_sm90_micro_prefill_audit.py",
                    "benchmarks/profile_sm90_micro_prefill.py",
                    "stream_attention/backends/sm90/micro_prefill.py",
                )
            )
        ).hexdigest(),
        evidence_kind="calibration_only_not_public_promotion",
        baseline_layout="prepared_NHD_conversion_excluded",
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
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    for c in cases(args.cohort):
        row = measure_case(c, args)
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
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    result["complete"] = True
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
