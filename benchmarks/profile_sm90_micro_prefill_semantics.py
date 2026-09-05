"""Exact FP16/BF16 and explicit-position causal-append correctness matrix.

No baseline victory is inferred from this semantics experiment. Timings are
complete native graph replays, not comparisons to an unaligned causal baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_sm90_micro_prefill import _capture, _elapsed_graph_ms  # noqa: E402
from stream_attention.backends.sm90.micro_prefill import (  # noqa: E402
    MicroPrefillPlan, NaturalMicroPrefillPlan,
)

SCHEMA = "streamattn.sm90_micro_prefill_semantics.v1"
SOURCE_PATHS = (
    "stream_attention/backends/sm90/micro_prefill.py",
    "stream_attention/backends/sm90/micro_prefill_semantics.py",
    "stream_attention/backends/sm90/micro_prefill_semantics_sources.py",
    "stream_attention/backends/sm90/transposed_gqa_exact_sources.py",
    "benchmarks/profile_sm90_micro_prefill_semantics.py",
)


def experiment_cases(suite):
    geometries = [
        (1, 2, 64, 16, 8, 1),
        (2, 3, 320, 32, 4, 4),
        (1, 17, 448, 16, 8, 6),
        (2, 9, 320, 16, 4, 5),
        (1, 32, 4096, 16, 8, 24),
        (1, 64, 4096, 16, 8, 16),
        (2, 8, 16384, 32, 4, 32),
    ]
    if suite == "smoke":
        geometries = geometries[:2]
    return [dict(batch=b, m=m, n=n, hq=h, g=g, splits=s, d=d,
                 dtype=dtype, mask=mask)
            for d in (64, 128) for dtype in ("fp16", "bf16")
            for b, m, n, h, g, s in geometries
            for mask in ("append", "permuted", "noncausal")]


def logical_positions(c, device):
    b, m, n = c["batch"], c["m"], c["n"]
    # Large origins catch int32 truncation; batches have distinct cache origins.
    origin = (torch.arange(b, device=device, dtype=torch.int64) * 10000 + (1 << 40))[:, None]
    kp = (torch.arange(n, device=device)[None, :] + origin).contiguous()
    qp = (torch.arange(m, device=device)[None, :] + origin + n - m).contiguous()
    if c["mask"] == "permuted":
        kp = kp.flip(1).contiguous()
        qp[:, 0] = origin[:, 0] - 1  # an entirely invisible query
        qp[:, 1] = origin[:, 0]      # one visible key, at physical cache end
        if m > 2:
            qp[:, 2] = origin[:, 0] + 63  # masks across both tile and split boundaries
    return qp, kp


def fp32_reference(q, k, v, qp=None, kp=None):
    b, m, h, d = q.shape
    hk = k.shape[1]
    g = h // hk
    scores = torch.einsum("bmhgd,bhnd->bhgmn", q.float().view(b, m, hk, g, d), k.float())
    scores *= d ** -0.5
    if qp is not None:
        visible = kp[:, None, :] <= qp[:, :, None]
        scores.masked_fill_(~visible[:, None, None], -torch.inf)
    lse = torch.logsumexp(scores, -1)
    p = torch.exp(scores - lse[..., None])
    p = torch.where(torch.isneginf(lse)[..., None], 0.0, p)
    out = torch.einsum("bhgmn,bhnd->bmhgd", p, v.float()).reshape(b, m, h, d)
    return out, lse.permute(0, 3, 1, 2).reshape(b, m, h)


def reconstructed_lse(plan):
    b, m, h, _ = plan.query.shape
    hk = plan.key_cache.shape[1]
    g = h // hk
    merged = torch.logsumexp(plan.partial_lse * math.log(2), dim=1)
    if isinstance(plan, MicroPrefillPlan):
        return merged.view(b, m, hk, 8)[..., :g].reshape(b, m, h)
    return (merged.view(b, hk, plan.query_tiles, 64 // g, g)
            .reshape(b, hk, plan.query_tiles * (64 // g), g)[:, :, :m]
            .permute(0, 2, 1, 3).reshape(b, m, h))


def check_output(plan, expected, expected_lse, *, observed_lse=None):
    out = plan.output.float()
    lse = reconstructed_lse(plan) if observed_lse is None else observed_lse
    finite = torch.isfinite(expected_lse)
    err = float((out - expected).abs().max())
    lse_err = float((lse[finite] - expected_lse[finite]).abs().max()) if finite.any() else 0.0
    empty_match = bool(torch.equal(torch.isneginf(lse), torch.isneginf(expected_lse)))
    zero_empty = bool((out[~finite] == 0).all())
    # Output rounding plus native low-precision P MMA, against full FP32 QK/PV.
    atol, rtol = (0.02, 0.02) if plan.query.dtype == torch.bfloat16 else (0.003, 0.003)
    passed = (bool(torch.isfinite(out).all()) and empty_match and zero_empty
              and bool(torch.allclose(out, expected, atol=atol, rtol=rtol))
              and math.isfinite(lse_err) and lse_err <= 0.005)
    return dict(passed=passed, max_abs=err if math.isfinite(err) else None,
                lse_max_abs=lse_err if math.isfinite(lse_err) else None,
                empty_lse_match=empty_match, empty_output_zero=zero_empty,
                atol=atol, rtol=rtol)


def profile_case(c, args):
    torch.manual_seed(args.seed + c["m"] + c["n"] + c["d"])
    dtype = torch.float16 if c["dtype"] == "fp16" else torch.bfloat16
    q = torch.randn(c["batch"], c["m"], c["hq"], c["d"], dtype=dtype, device="cuda")
    k = torch.randn(c["batch"], c["hq"] // c["g"], c["n"], c["d"], dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    qp, kp = logical_positions(c, "cuda")
    causal = c["mask"] != "noncausal"
    kwargs = dict(causal=True, query_positions=qp, key_positions=kp) if causal else {}
    plans = {}
    for family, cls in (("transposed", MicroPrefillPlan), ("natural", NaturalMicroPrefillPlan)):
        plans[family] = cls.build(
            q, k, v, num_splits=c["splits"], cutlass_root=args.cutlass_root,
            build_dir=args.build_dir / f"d{c['d']}", compile_verbose=True, **kwargs,
        )
    ref, ref_lse = fp32_reference(q, k, v, qp if causal else None, kp if causal else None)
    result = dict(case=c, families={})
    graphs = {}
    for family, plan in plans.items():
        plan.partial_output.fill_(float("nan"))
        plan.partial_lse.fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        first = check_output(plan, ref, ref_lse)
        graph = _capture(plan.run, warmup=3)
        graphs[family] = graph
        result["families"][family] = dict(
            eager=first, workspace_bytes=plan.workspace_bytes,
            complete_graph_us=[1000 * _elapsed_graph_ms(graph, iterations=50) for _ in range(3)],
        )
    pointers = [t.data_ptr() for t in (q, k, v, qp, kp)]
    q.normal_().mul_(3)
    k.normal_()
    v.normal_()
    if causal:
        qp[:, 0] = -(1 << 40)
        kp.copy_(kp.flip(1))
    ref, ref_lse = fp32_reference(q, k, v, qp if causal else None, kp if causal else None)
    for family, plan in plans.items():
        plan.partial_output.fill_(float("nan"))
        plan.partial_lse.fill_(float("nan"))
        graphs[family].replay()
        torch.cuda.synchronize()
        result["families"][family]["mutated_graph"] = check_output(plan, ref, ref_lse)
    assert pointers == [t.data_ptr() for t in (q, k, v, qp, kp)]
    result["passed"] = all(r["eager"]["passed"] and r["mutated_graph"]["passed"]
                           for r in result["families"].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "full"), default="full")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--seed", type=int, default=6107)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("preserve existing evidence")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = experiment_cases(args.suite)
    result = dict(schema=SCHEMA, complete=False, provider=args.provider, seed=args.seed,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(), planned_cases=len(cases), rows=[],
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                 for p in SOURCE_PATHS})
    try:
        for c in cases:
            print(json.dumps(dict(stage="case", case=c)), flush=True)
            row = profile_case(c, args)
            result["rows"].append(row)
            print(json.dumps(dict(stage="checked", case=c, passed=row["passed"])), flush=True)
            if not row["passed"]:
                raise AssertionError(f"semantic correctness failed: {c}")
        result["complete"] = True
        result["passed_cases"] = sum(r["passed"] for r in result["rows"])
        result["median_native_graph_us"] = {
            name: statistics.median(statistics.median(r["families"][name]["complete_graph_us"])
                                    for r in result["rows"])
            for name in ("transposed", "natural")
        }
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
