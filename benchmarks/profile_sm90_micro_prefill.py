"""Profile the exact SM90 M=2..64 transposed-GQA micro-prefill family."""

from __future__ import annotations

import argparse
import json
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

from stream_attention.backends.sm90.micro_prefill import (  # noqa: E402
    MicroPrefillPlan,
    NaturalMicroPrefillPlan,
)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None


SCHEMA = "streamattn.sm90_micro_prefill_canary.v2"


@contextmanager
def _flash_context() -> Iterator[None]:
    if sdpa_kernel is None or SDPBackend is None:
        raise RuntimeError("forced Flash SDPA is unavailable; refusing an implicit fallback")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        yield


def _flash_sdpa(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    with _flash_context():
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key_cache,
            value_cache,
            is_causal=False,
            dropout_p=0.0,
            enable_gqa=True,
        ).transpose(1, 2)


def _capture(
    fn: Callable[[], torch.Tensor], *, warmup: int
) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _elapsed_graph_ms(graph: torch.cuda.CUDAGraph, *, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _profile_cell(
    *,
    provider: str,
    batch: int,
    query_len: int,
    kv_len: int,
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
    generator.manual_seed(
        4301
        + batch * 1_000_003
        + query_len * 10_007
        + kv_len
        + group_size * 101
        + head_dim
    )
    query = torch.randn(
        batch,
        query_len,
        q_heads,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key_cache = torch.randn(
        batch,
        kv_heads,
        kv_len,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn(
        batch,
        kv_heads,
        kv_len,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    transposed_plan = MicroPrefillPlan.build(
        query,
        key_cache,
        value_cache,
        cutlass_root=cutlass_root,
        build_dir=build_dir,
        compile_verbose=verbose_build,
    )
    natural_plan = NaturalMicroPrefillPlan.build(
        query,
        key_cache,
        value_cache,
        cutlass_root=cutlass_root,
        build_dir=build_dir,
        compile_verbose=verbose_build,
    )
    transposed_output = transposed_plan.run().clone()
    natural_output = natural_plan.run().clone()
    baseline = _flash_sdpa(query, key_cache, value_cache)
    torch.cuda.synchronize()

    transposed_difference = (
        transposed_output.float() - baseline.float()
    ).flatten()
    transposed_max_abs_error = float(transposed_difference.abs().max().item())
    transposed_relative_l2_error = float(
        torch.linalg.vector_norm(transposed_difference)
        / torch.linalg.vector_norm(baseline.float().flatten()).clamp_min(1e-12)
    )
    natural_difference = (natural_output.float() - baseline.float()).flatten()
    natural_max_abs_error = float(natural_difference.abs().max().item())
    natural_relative_l2_error = float(
        torch.linalg.vector_norm(natural_difference)
        / torch.linalg.vector_norm(baseline.float().flatten()).clamp_min(1e-12)
    )

    graphs = {
        "transposed": _capture(transposed_plan.run, warmup=warmup),
        "natural": _capture(natural_plan.run, warmup=warmup),
        "flash": _capture(
            lambda: _flash_sdpa(query, key_cache, value_cache),
            warmup=warmup,
        ),
    }
    orders = (
        ("transposed", "natural", "flash"),
        ("flash", "transposed", "natural"),
        ("natural", "flash", "transposed"),
    )
    trials: list[dict[str, float]] = []
    for repeat in range(repeats):
        times: dict[str, float] = {}
        for name in orders[repeat % len(orders)]:
            times[name] = _elapsed_graph_ms(graphs[name], iterations=iterations)
        trials.append(
            {
                "transposed_ms": times["transposed"],
                "natural_ms": times["natural"],
                "flash_ms": times["flash"],
                "flash_over_transposed": times["flash"]
                / times["transposed"],
                "flash_over_natural": times["flash"] / times["natural"],
            }
        )

    transposed_speedups = [trial["flash_over_transposed"] for trial in trials]
    natural_speedups = [trial["flash_over_natural"] for trial in trials]
    transposed_correct = (
        transposed_max_abs_error <= 0.04
        and transposed_relative_l2_error <= 0.02
    )
    natural_correct = (
        natural_max_abs_error <= 0.04 and natural_relative_l2_error <= 0.02
    )
    median_transposed_ms = statistics.median(
        trial["transposed_ms"] for trial in trials
    )
    median_natural_ms = statistics.median(
        trial["natural_ms"] for trial in trials
    )
    median_flash_ms = statistics.median(trial["flash_ms"] for trial in trials)
    winner = (
        "natural" if median_natural_ms < median_transposed_ms else "transposed"
    )
    winner_speedups = (
        natural_speedups if winner == "natural" else transposed_speedups
    )
    median_speedup = statistics.median(winner_speedups)
    minimum_speedup = min(winner_speedups)
    transposed_groups = batch * query_len * kv_heads
    natural_groups = batch * kv_heads * natural_plan.query_tiles
    return {
        "provider": provider,
        "batch": batch,
        "query_len": query_len,
        "kv_len": kv_len,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "group_size": group_size,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "causal": False,
        "kv_layout": "HND_contiguous",
        "candidates": [transposed_plan.backend, natural_plan.backend],
        "winner": winner,
        "baseline": "torch_flash_sdpa_cuda_graph",
        "transposed_num_splits": transposed_plan.num_splits,
        "natural_num_splits": natural_plan.num_splits,
        "natural_query_tiles": natural_plan.query_tiles,
        "transposed_producer_ctas": (
            transposed_groups * transposed_plan.num_splits
        ),
        "natural_producer_ctas": natural_groups * natural_plan.num_splits,
        "transposed_workspace_bytes": transposed_plan.workspace_bytes,
        "natural_workspace_bytes": natural_plan.workspace_bytes,
        "transposed_max_abs_error": transposed_max_abs_error,
        "transposed_relative_l2_error": transposed_relative_l2_error,
        "natural_max_abs_error": natural_max_abs_error,
        "natural_relative_l2_error": natural_relative_l2_error,
        "transposed_correct": transposed_correct,
        "natural_correct": natural_correct,
        "strict_correct": transposed_correct and natural_correct,
        "trials": trials,
        "median_transposed_ms": median_transposed_ms,
        "median_natural_ms": median_natural_ms,
        "median_baseline_ms": median_flash_ms,
        "median_transposed_speedup_vs_flash": statistics.median(
            transposed_speedups
        ),
        "median_natural_speedup_vs_flash": statistics.median(natural_speedups),
        "median_speedup_vs_flash": median_speedup,
        "minimum_speedup_vs_flash": minimum_speedup,
        "paired_wins_vs_flash": sum(value > 1.0 for value in winner_speedups),
        "canary_pass": transposed_correct
        and natural_correct
        and median_speedup >= 0.75,
        "promotion_pass": transposed_correct
        and natural_correct
        and median_speedup > 1.0
        and minimum_speedup >= 0.98,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="unknown")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--query-lengths", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64]
    )
    parser.add_argument(
        "--kv-lengths", nargs="+", type=int, default=[4096, 16384, 32768]
    )
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--head-dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("an SM90 CUDA device is required")
    if any(args.q_heads % group_size for group_size in args.group_sizes):
        raise ValueError("q_heads must be divisible by every group size")
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if any(not 2 <= length <= 64 for length in args.query_lengths):
        raise ValueError("query lengths must be in [2,64]")
    if any(length <= 0 or length % 64 for length in args.kv_lengths):
        raise ValueError("KV lengths must be positive multiples of 64")
    if any(head_dim not in (64, 128) for head_dim in args.head_dims):
        raise ValueError("head dimensions must be 64 or 128")

    rows: list[dict[str, object]] = []
    for batch in args.batch_sizes:
        for query_len in args.query_lengths:
            for kv_len in args.kv_lengths:
                for group_size in args.group_sizes:
                    for head_dim in args.head_dims:
                        row = _profile_cell(
                            provider=args.provider,
                            batch=batch,
                            query_len=query_len,
                            kv_len=kv_len,
                            q_heads=args.q_heads,
                            group_size=group_size,
                            head_dim=head_dim,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            repeats=args.repeats,
                            cutlass_root=args.cutlass_root,
                            build_dir=args.build_dir,
                            verbose_build=args.verbose_build,
                        )
                        rows.append(row)
                        print(json.dumps(row), flush=True)

    correct_rows = sum(bool(row["strict_correct"]) for row in rows)
    promoted_rows = sum(bool(row["promotion_pass"]) for row in rows)
    result = {
        "schema": SCHEMA,
        "provider": args.provider,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "gate": {
            "cell_count": len(rows),
            "correct_rows": correct_rows,
            "promoted_rows": promoted_rows,
            "all_correct": correct_rows == len(rows),
            "minimum_cell_median_speedup": min(
                float(row["median_speedup_vs_flash"]) for row in rows
            ),
            "geometric_mean_speedup": statistics.geometric_mean(
                float(row["median_speedup_vs_flash"]) for row in rows
            ),
            "decision": (
                "profile_natural_small_m_competitor"
                if correct_rows == len(rows)
                else "fix_exactness_before_schedule_search"
            ),
        },
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
