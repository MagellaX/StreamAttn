"""Gate the natural-orientation SM90 grouped exact prefill canary."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stream_attention.backends.sm90.grouped_gqa_prefill import (
    GroupedWgmmaPrefillPlan,
)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None


SCHEMA = "streamattn.sm90_grouped_gqa_prefill_gate.v1"


@contextmanager
def _flash_context() -> Iterator[None]:
    if sdpa_kernel is None or SDPBackend is None:
        yield
        return
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        yield


def _flash_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with _flash_context():
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=0.0,
            enable_gqa=True,
        ).transpose(1, 2)


def _capture(fn: Callable[[], torch.Tensor], *, warmup: int) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _elapsed_graph_ms(graph: torch.cuda.CUDAGraph, *, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _reference_lse(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    group_size = q.shape[2] // k.shape[2]
    scores = torch.matmul(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).repeat_interleave(group_size, dim=1).float().transpose(-1, -2),
    ) / math.sqrt(q.shape[-1])
    row = torch.arange(q.shape[1], device=q.device)[:, None]
    column = torch.arange(k.shape[1], device=q.device)[None, :]
    scores.masked_fill_(column > row, -torch.inf)
    return torch.logsumexp(scores, dim=-1).transpose(1, 2)


def _profile_cell(
    *,
    batch: int,
    sequence_length: int,
    q_heads: int,
    group_size: int,
    head_dim: int,
    warmup: int,
    iterations: int,
    repeats: int,
    cutlass_root: Path,
    build_dir: Path,
    verbose_build: bool,
) -> dict[str, object]:
    kv_heads = q_heads // group_size
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1301 + batch * 101 + sequence_length + group_size)
    query = torch.randn(
        batch,
        sequence_length,
        q_heads,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        sequence_length,
        kv_heads,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        batch,
        sequence_length,
        kv_heads,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    plan = GroupedWgmmaPrefillPlan.build(
        query,
        key,
        value,
        cutlass_root=cutlass_root,
        build_dir=build_dir,
        compile_verbose=verbose_build,
    )
    candidate = plan.run().clone()
    baseline = _flash_sdpa(query, key, value)
    torch.cuda.synchronize()
    difference = (candidate.float() - baseline.float()).flatten()
    max_abs_error = float(difference.abs().max().item())
    relative_l2_error = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(baseline.float().flatten()).clamp_min(1e-12)
    )
    lse_max_abs_error = None
    if sequence_length <= 512:
        reference_lse = _reference_lse(query, key)
        lse_max_abs_error = float((plan.lse - reference_lse).abs().max().item())

    candidate_graph, _ = _capture(plan.run, warmup=warmup)
    baseline_graph, _ = _capture(
        lambda: _flash_sdpa(query, key, value), warmup=warmup
    )
    trials: list[dict[str, float]] = []
    for repeat in range(repeats):
        if repeat % 2:
            baseline_ms = _elapsed_graph_ms(baseline_graph, iterations=iterations)
            candidate_ms = _elapsed_graph_ms(candidate_graph, iterations=iterations)
        else:
            candidate_ms = _elapsed_graph_ms(candidate_graph, iterations=iterations)
            baseline_ms = _elapsed_graph_ms(baseline_graph, iterations=iterations)
        trials.append(
            {
                "candidate_ms": candidate_ms,
                "baseline_ms": baseline_ms,
                "speedup": baseline_ms / candidate_ms,
            }
        )
    speedups = [trial["speedup"] for trial in trials]
    strict_correct = max_abs_error <= 0.04 and relative_l2_error <= 0.02
    return {
        "batch": batch,
        "sequence_length": sequence_length,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "group_size": group_size,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "causal": True,
        "candidate": plan.backend,
        "baseline": "torch_flash_sdpa_cuda_graph",
        "max_abs_error": max_abs_error,
        "relative_l2_error": relative_l2_error,
        "lse_max_abs_error": lse_max_abs_error,
        "resources": plan.resource_info(),
        "trials": trials,
        "median_candidate_ms": statistics.median(
            trial["candidate_ms"] for trial in trials
        ),
        "median_baseline_ms": statistics.median(
            trial["baseline_ms"] for trial in trials
        ),
        "median_speedup": statistics.median(speedups),
        "minimum_speedup": min(speedups),
        "paired_wins": sum(speedup > 1.0 for speedup in speedups),
        "strict_correct": strict_correct,
        "decision": (
            "candidate_win"
            if strict_correct and min(speedups) > 1.0
            else "fallback"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--sequence-lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048]
    )
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("an SM90 CUDA device is required")
    if any(args.q_heads % group_size for group_size in args.group_sizes):
        raise ValueError("q_heads must be divisible by every group size")

    rows: list[dict[str, object]] = []
    for batch in args.batches:
        for group_size in args.group_sizes:
            for sequence_length in args.sequence_lengths:
                row = _profile_cell(
                    batch=batch,
                    sequence_length=sequence_length,
                    q_heads=args.q_heads,
                    group_size=group_size,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    cutlass_root=args.cutlass_root,
                    build_dir=args.build_dir,
                    verbose_build=args.verbose_build,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)

    result = {
        "schema": SCHEMA,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
