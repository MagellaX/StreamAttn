"""Synthetic adaptive-accounting/physical-work canary; not model promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from stream_attention.certified import build_block_summaries
from stream_attention.kernels.certified_fwd_triton import certified_attention_triton_forward

SCHEMA = "streamattn.adaptive_two_gate.v2"
SOURCE_FILES = (
    "stream_attention/certified/attention.py",
    "stream_attention/certified/summaries.py",
    "stream_attention/kernels/certified_fwd_triton.py",
    "benchmarks/profile_adaptive_two_gate.py",
    "benchmarks/adaptive_known_support.py",
)


def make_case(kind, dtype, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(19093)
    b, m, n, hq, hkv, d = (1, 16, 8192, 8, 2, 64)
    budget, causal, pre = 1e-3, False, True
    if kind == "random_tails":
        b, m, n, hq, hkv, d = 2, 19, 137, 8, 2, 32
    elif kind == "causal":
        m, n = 65, 65
        causal = True
    elif kind == "cumulative":
        m, n, hq, hkv, d = 1, 4096, 4, 1, 32
        budget = 0.01
    elif kind == "post_only":
        pre = False
    q = torch.randn((b, m, hq, d), generator=generator, device=device, dtype=dtype)
    k = torch.randn((b, n, hkv, d), generator=generator, device=device, dtype=dtype)
    v = torch.randn(k.shape, generator=generator, device=device, dtype=dtype)
    if kind in ("peaked", "post_only", "mixed_rows"):
        q.zero_()
        k.zero_()
        q[..., 0] = 16.0
        k[:, :32, :, 0] = 8.0
        k[:, 32:, :, 0] = -8.0
        if kind == "mixed_rows":
            q[:, m // 2:, :, 0] = -16.0
    elif kind == "cumulative":
        q.zero_()
        k.zero_()
        v.zero_()
        q[..., 0] = math.sqrt(d)
        k[:, 32:, :, 0] = math.log(1e-3)
        v[:, :32, :, 0] = -1.0
        v[:, 32:, :, 0] = 1.0
    return q, k, v, dict(causal=causal, error_budget=budget, enable_summary_gate=pre)


def reference(q, k, v, causal, retained_blocks=None):
    groups = q.shape[2] // k.shape[2]
    qh = q.float().permute(0, 2, 1, 3)
    kh = k.float().repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)
    vh = v.float().repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)
    scores = qh @ kh.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.arange(k.shape[1], device=q.device)[None, :] <= torch.arange(q.shape[1], device=q.device)[:, None]
        scores.masked_fill_(~mask, -float("inf"))
    if retained_blocks is not None:
        keep = retained_blocks.repeat_interleave(32, dim=-1)[..., :k.shape[1]]
        scores.masked_fill_(~keep, -float("inf"))
    return (scores.softmax(-1) @ vh).permute(0, 2, 1, 3)


def numerical_allowance(selected, dtype):
    # Predeclared v2 diagnostic allowance, separate from the omission budget.
    cast_error = torch.linalg.vector_norm(selected.to(dtype).float() - selected, dim=-1)
    arithmetic = 4 * torch.finfo(dtype).eps * torch.linalg.vector_norm(selected, dim=-1)
    return cast_error + arithmetic + 1e-5 * math.sqrt(selected.shape[-1])


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def time_graph(graph, iterations=40):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def diagnose_causal_roundoff():
    """Keep the failed strict gate; isolate no-skip arithmetic without retiming."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        q, k, v, options = make_case("causal", dtype)
        summaries = build_block_summaries(k, v, block_size=32)
        ref = reference(q, k, v, True)
        record = dict(dtype=str(dtype), original_row_l2_limit=0.006 if dtype == torch.bfloat16 else 0.001)
        outputs = {}
        for name, budget in (("adaptive", 1e-3), ("zero_budget", 0.0)):
            output, stats = certified_attention_triton_forward(
                q, k, v, summaries=summaries, causal=True, error_budget=budget,
                block_size=32, tile_size_q=16, return_raw_stats=True)
            outputs[name] = output
            record[name] = dict(max_row_l2=torch.linalg.vector_norm(output.float() - ref, dim=-1).max().item(),
                                max_abs=(output.float() - ref).abs().max().item(),
                                counters=stats.sum(dim=(0, 1, 2)).tolist())
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            flash = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=True,
                enable_gqa=True).permute(0, 2, 1, 3)
        record["flash_max_row_l2"] = torch.linalg.vector_norm(flash.float() - ref, dim=-1).max().item()
        record["cast_only_max_row_l2"] = torch.linalg.vector_norm(ref.to(dtype).float() - ref, dim=-1).max().item()
        record["adaptive_equals_zero_budget"] = torch.equal(outputs["adaptive"], outputs["zero_budget"])
        record["adaptive_passes_original_limit"] = record["adaptive"]["max_row_l2"] <= record["original_row_l2_limit"]
        rows.append(record)
    return dict(schema="streamattn.adaptive_two_gate_precision.v1", complete=True,
                diagnostic_only=True, performance_promotion=False, cases=rows,
                device=torch.cuda.get_device_name(), torch=torch.__version__,
                sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES})


def run_case(kind, dtype):
    q, k, v, options = make_case(kind, dtype)
    torch.cuda.synchronize()
    start = time.perf_counter()
    summaries = build_block_summaries(k, v, block_size=32)
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - start) * 1000
    variants = {}
    for name, budget, materialize, rowwise in (
        ("adaptive", options["error_budget"], False, False),
        ("mask_only_control", options["error_budget"], True, False),
        ("zero_budget_control", 0.0, False, False),
        ("rowwise_diagnostic", options["error_budget"], False, True),
    ):
        output, bound = torch.empty_like(q), torch.empty(q.shape[:-1], device=q.device)
        stats = torch.empty(q.shape[0], q.shape[2], math.ceil(q.shape[1] / 16), 6,
                            device=q.device, dtype=torch.int32)
        support = torch.empty(q.shape[0], q.shape[2], q.shape[1], math.ceil(k.shape[1] / 32),
                              device=q.device, dtype=torch.bool)
        kwargs = dict(options, error_budget=budget, block_size=32, tile_size_q=16,
                      summaries=summaries, materialize_skipped_work=materialize,
                      out=output, raw_stats_out=stats, error_bound_out=bound,
                      return_raw_stats=True, rowwise_omissions=rowwise, retained_blocks_out=support)
        certified_attention_triton_forward(q, k, v, **kwargs)
        variants[name] = dict(output=output, bound=bound, stats=stats, support=support, kwargs=kwargs)
    torch.cuda.synchronize()
    ref = reference(q, k, v, options["causal"])
    roundoff = 0.006 if dtype == torch.bfloat16 else 0.001
    correctness = {}
    for name, data in variants.items():
        error = torch.linalg.vector_norm(data["output"].float() - ref, dim=-1)
        bound = data["bound"]
        selected = reference(q, k, v, options["causal"], data["support"])
        execution_error = torch.linalg.vector_norm(data["output"].float() - selected, dim=-1)
        omission_error = torch.linalg.vector_norm(selected - ref, dim=-1)
        allowance = numerical_allowance(selected, dtype)
        omission_slack = 2e-5 * (1 + torch.linalg.vector_norm(ref, dim=-1))
        assert torch.isfinite(error).all() and torch.isfinite(bound).all()
        assert torch.all(execution_error <= allowance), (name, "execution", execution_error.max().item())
        assert torch.all(omission_error <= bound + omission_slack), (name, "omission", omission_error.max().item())
        assert bound.max().item() <= data["kwargs"]["error_budget"] + 2e-6
        correctness[name] = dict(max_row_l2=error.max().item(),
                                 max_omission_bound=bound.max().item(),
                                 max_omission_error=omission_error.max().item(),
                                 max_execution_error=execution_error.max().item(),
                                 max_numerical_allowance=allowance.max().item(),
                                 original_v1_limit_passed=bool(torch.all(error <= bound + roundoff)),
                                 counters=data["stats"].sum(dim=(0, 1, 2)).tolist())
    torch.testing.assert_close(variants["adaptive"]["output"],
                               variants["mask_only_control"]["output"], rtol=0, atol=0)
    counters = correctness["adaptive"]["counters"]
    if kind == "peaked":
        assert counters[0] > 0 and counters[4] < counters[3] and counters[5] < counters[3]
    if kind == "post_only":
        assert counters[0] == 0 and counters[1] > 0
        assert counters[4] == counters[3] and counters[5] < counters[3]
    if kind == "cumulative":
        assert 0 < counters[0] + counters[1] < (4096 // 32 - 1) * q.shape[2]
    if kind == "mixed_rows":
        assert counters[0] == counters[1] == 0 and counters[4] == counters[3] and counters[5] == counters[3]
        assert correctness["rowwise_diagnostic"]["counters"][0] > 0
        assert correctness["adaptive"]["max_omission_bound"] == 0
        torch.testing.assert_close(variants["adaptive"]["output"], variants["zero_budget_control"]["output"], atol=0, rtol=0)

    # Diagnostic counters and error-bound stores are absent from timed kernels.
    graphs = {}
    for name, data in variants.items():
        kwargs = dict(data["kwargs"], return_raw_stats=False, error_bound_out=None, retained_blocks_out=None)
        graphs[name] = capture(lambda kwargs=kwargs: certified_attention_triton_forward(q, k, v, **kwargs))
    baseline_error = None
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        def flash():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q.permute(0, 2, 1, 3),
                    k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
                    is_causal=options["causal"], enable_gqa=True)
        flash_out = flash().permute(0, 2, 1, 3)
        assert torch.all(torch.linalg.vector_norm(flash_out.float() - ref, dim=-1) <= numerical_allowance(ref, dtype))
        graphs["torch_flash_sdpa"] = capture(flash)
    except Exception as exc:
        baseline_error = f"{type(exc).__name__}: {exc}"
    oracle = None
    oracle_buffers = []
    if kind in ("peaked", "post_only"):
        from benchmarks.adaptive_known_support import execute_known_support
        support = variants["adaptive"]["support"]
        assert support[..., 0].all() and not support[..., 1:].any()
        selected_ref = reference(q, k[:, :32], v[:, :32], False)
        oracle = dict(shared_retained_blocks=[0], diagnostic_only=True,
                      valid_after_arbitrary_query_mutation=False, errors={})
        for name, count in (("known_support_traversal", k.shape[1] // 32), ("known_support_compact", 1)):
            ids = torch.full((count,), -1, device=q.device, dtype=torch.int32)
            ids[0] = 0
            out = torch.empty_like(q)
            call = lambda ids=ids, out=out: execute_known_support(q, k, v, ids, out)
            call()
            error = torch.linalg.vector_norm(out.float() - selected_ref, dim=-1)
            assert torch.all(error <= numerical_allowance(selected_ref, dtype))
            oracle["errors"][name] = error.max().item()
            oracle_buffers.append((ids, out))
            graphs[name] = capture(call)
    timings = {name: [] for name in graphs}
    names = list(graphs)
    for repeat in range(7):
        for name in names if repeat % 2 == 0 else names[::-1]:
            timings[name].append(time_graph(graphs[name]))

    # Mutate live Q, replay the same graph, compare with a fresh reference.
    old = variants["adaptive"]["output"].clone()
    q.neg_()
    graphs["adaptive"].replay()
    ref2 = reference(q, k, v, options["causal"])
    mutation_error = torch.linalg.vector_norm(variants["adaptive"]["output"].float() - ref2, dim=-1).max().item()
    # Recapture actual support to separate numerical and omission error after Q changes.
    replayed = variants["adaptive"]["output"].clone()
    certified_attention_triton_forward(q, k, v, **variants["adaptive"]["kwargs"])
    torch.testing.assert_close(replayed, variants["adaptive"]["output"], atol=0, rtol=0)
    selected2 = reference(q, k, v, options["causal"], variants["adaptive"]["support"])
    assert torch.all(torch.linalg.vector_norm(replayed.float() - selected2, dim=-1) <= numerical_allowance(selected2, dtype))
    assert torch.all(torch.linalg.vector_norm(selected2 - ref2, dim=-1) <= variants["adaptive"]["bound"] + 2e-5 * (1 + torch.linalg.vector_norm(ref2, dim=-1)))
    assert not torch.equal(old, variants["adaptive"]["output"])
    return dict(kind=kind, dtype=str(dtype), q_shape=list(q.shape), kv_shape=list(k.shape),
                options=options, block_size=32, query_tile=16, correctness=correctness,
                roundoff_allowance_l2=roundoff, query_mutation_max_row_l2=mutation_error,
                summary_build_wall_ms=build_ms, graph_ms=timings,
                median_graph_ms={name: statistics.median(vals) for name, vals in timings.items()},
                baseline_error=baseline_error, known_support_oracle=oracle)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--provider", default="local", choices=("local", "lightning", "modal"))
    p.add_argument("--diagnose-causal", action="store_true")
    args = p.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    if args.diagnose_causal:
        result = diagnose_causal_roundoff()
        result["provider"] = args.provider
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
        return 0
    result = dict(schema=SCHEMA, provider=args.provider, complete=False,
                  input_source="synthetic mechanism tests, not LLM activations",
                  omission_commit_scope="query_tile_and_head",
                  numerical_protocol="v2: cast L2 + 4*dtype_epsilon*selected_output_L2 + 1e-5*sqrt(D); omission measured separately",
                  performance_promotion=False, cases=[], failures=[],
                  torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                  device=torch.cuda.get_device_name(),
                  capability=list(torch.cuda.get_device_capability()),
                  sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES})
    for dtype in (torch.float16, torch.bfloat16):
        for kind in ("cumulative", "random_tails", "causal", "peaked", "post_only", "mixed_rows"):
            print(f"CASE {kind} {dtype}", flush=True)
            try:
                result["cases"].append(run_case(kind, dtype))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                result["failures"].append(dict(kind=kind, dtype=str(dtype), error=f"{type(exc).__name__}: {exc}"))
    result["complete"] = not result["failures"] and len(result["cases"]) == 12
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
