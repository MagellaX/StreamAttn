"""Gate the lean consumer-owned SM90 grouped RS-PV prefill canary."""

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

from stream_attention.backends.sm90.grouped_gqa_prefill import (  # noqa: E402
    GroupedRSPrefillPlan,
    GroupedWgmmaPrefillPlan,
)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None


SCHEMAS = {
    "canary": "streamattn.sm90_grouped_rs_prefill_canary.v1",
    "promotion": "streamattn.sm90_grouped_rs_prefill_promotion.v1",
}


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


def _sampled_lse_error(
    query: torch.Tensor,
    key: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[float, list[int]]:
    sequence_length = int(query.shape[1])
    q_heads = int(query.shape[2])
    kv_heads = int(key.shape[2])
    group_size = q_heads // kv_heads
    positions = sorted(
        {
            0,
            min(31, sequence_length - 1),
            min(63, sequence_length - 1),
            sequence_length // 2,
            sequence_length - 1,
        }
    )
    kv_indices = torch.arange(q_heads, device=query.device) // group_size
    max_error = 0.0
    scale = 1.0 / math.sqrt(query.shape[-1])
    for position in positions:
        selected_key = key[:, : position + 1].index_select(2, kv_indices).float()
        scores = torch.einsum(
            "bhd,blhd->bhl",
            query[:, position].float(),
            selected_key,
        ) * scale
        reference = torch.logsumexp(scores, dim=-1)
        error = float((lse[:, position] - reference).abs().max().item())
        max_error = max(max_error, error)
    return max_error, positions


def _profile_cell(
    *,
    provider: str,
    batch: int,
    sequence_length: int,
    q_heads: int,
    group_size: int,
    warmup: int,
    iterations: int,
    repeats: int,
    cutlass_root: Path,
    build_dir: Path,
    verbose_build: bool,
) -> dict[str, object]:
    kv_heads = q_heads // group_size
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2901 + batch * 100_003 + sequence_length + group_size)
    query = torch.randn(
        batch,
        sequence_length,
        q_heads,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        sequence_length,
        kv_heads,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        batch,
        sequence_length,
        kv_heads,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    rs_plan = GroupedRSPrefillPlan.build(
        query,
        key,
        value,
        cutlass_root=cutlass_root,
        build_dir=build_dir,
        compile_verbose=verbose_build,
    )
    ss_plan = GroupedWgmmaPrefillPlan.build(
        query,
        key,
        value,
        cutlass_root=cutlass_root,
        build_dir=build_dir,
        compile_verbose=verbose_build,
    )
    rs_output = rs_plan.run().clone()
    ss_output = ss_plan.run().clone()
    baseline = _flash_sdpa(query, key, value)
    torch.cuda.synchronize()

    difference = (rs_output.float() - baseline.float()).flatten()
    max_abs_error = float(difference.abs().max().item())
    relative_l2_error = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(baseline.float().flatten()).clamp_min(1e-12)
    )
    rs_vs_ss_max_abs = float(
        (rs_output.float() - ss_output.float()).abs().max().item()
    )
    lse_max_abs_error, lse_positions = _sampled_lse_error(
        query, key, rs_plan.lse
    )
    resources = rs_plan.resource_info()

    graphs = {
        "rs": _capture(rs_plan.run, warmup=warmup),
        "ss": _capture(ss_plan.run, warmup=warmup),
        "flash": _capture(lambda: _flash_sdpa(query, key, value), warmup=warmup),
    }
    orders = (
        ("rs", "ss", "flash"),
        ("flash", "rs", "ss"),
        ("ss", "flash", "rs"),
    )
    trials: list[dict[str, float]] = []
    for repeat in range(repeats):
        times: dict[str, float] = {}
        for name in orders[repeat % len(orders)]:
            times[name] = _elapsed_graph_ms(graphs[name], iterations=iterations)
        trials.append(
            {
                "rs_ms": times["rs"],
                "ss_ms": times["ss"],
                "flash_ms": times["flash"],
                "flash_over_rs": times["flash"] / times["rs"],
                "ss_over_rs": times["ss"] / times["rs"],
            }
        )

    flash_speedups = [trial["flash_over_rs"] for trial in trials]
    ss_speedups = [trial["ss_over_rs"] for trial in trials]
    strict_correct = (
        max_abs_error <= 0.04
        and relative_l2_error <= 0.02
        and lse_max_abs_error <= 0.01
    )
    zero_local_bytes = int(resources["local_bytes_per_thread"]) == 0
    median_flash_speedup = statistics.median(flash_speedups)
    minimum_flash_speedup = min(flash_speedups)
    return {
        "provider": provider,
        "batch": batch,
        "sequence_length": sequence_length,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "group_size": group_size,
        "head_dim": 128,
        "dtype": "bfloat16",
        "causal": True,
        "candidate": rs_plan.backend,
        "control": ss_plan.backend,
        "baseline": "torch_flash_sdpa_cuda_graph",
        "max_abs_error": max_abs_error,
        "relative_l2_error": relative_l2_error,
        "sampled_lse_max_abs_error": lse_max_abs_error,
        "sampled_lse_positions": lse_positions,
        "rs_vs_ss_max_abs": rs_vs_ss_max_abs,
        "resources": resources,
        "trials": trials,
        "median_candidate_ms": statistics.median(
            trial["rs_ms"] for trial in trials
        ),
        "median_control_ms": statistics.median(
            trial["ss_ms"] for trial in trials
        ),
        "median_baseline_ms": statistics.median(
            trial["flash_ms"] for trial in trials
        ),
        "median_speedup_vs_flash": median_flash_speedup,
        "minimum_speedup_vs_flash": minimum_flash_speedup,
        "median_speedup_vs_ss": statistics.median(ss_speedups),
        "paired_wins_vs_flash": sum(value > 1.0 for value in flash_speedups),
        "strict_correct": strict_correct,
        "zero_local_bytes": zero_local_bytes,
        "canary_pass": strict_correct
        and zero_local_bytes
        and median_flash_speedup >= 0.90,
        "promotion_pass": strict_correct
        and zero_local_bytes
        and median_flash_speedup > 1.0
        and minimum_flash_speedup >= 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="unknown")
    parser.add_argument("--matrix-kind", choices=sorted(SCHEMAS), default="canary")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[2048, 4096])
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[4, 8])
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
    if any(length <= 0 or length % 64 for length in args.sequence_lengths):
        raise ValueError("sequence lengths must be positive multiples of 64")

    rows: list[dict[str, object]] = []
    for batch in args.batch_sizes:
        for group_size in args.group_sizes:
            for sequence_length in args.sequence_lengths:
                row = _profile_cell(
                    provider=args.provider,
                    batch=batch,
                    sequence_length=sequence_length,
                    q_heads=args.q_heads,
                    group_size=group_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    cutlass_root=args.cutlass_root,
                    build_dir=args.build_dir,
                    verbose_build=args.verbose_build,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)

    all_correct = all(bool(row["strict_correct"]) for row in rows)
    zero_local_bytes = all(bool(row["zero_local_bytes"]) for row in rows)
    all_canary = all(bool(row["canary_pass"]) for row in rows)
    all_promoted = all(bool(row["promotion_pass"]) for row in rows)
    promoted_cells = sum(bool(row["promotion_pass"]) for row in rows)
    if args.matrix_kind == "canary":
        decision = (
            "build_broad_promotion_matrix"
            if all_canary
            else "close_h100_grouped_prefill_family"
        )
    else:
        decision = (
            "promote_scoped_phase_cells"
            if all_correct and zero_local_bytes and promoted_cells > 0
            else "reject_complete_kernel"
        )

    result = {
        "schema": SCHEMAS[args.matrix_kind],
        "matrix_kind": args.matrix_kind,
        "provider": args.provider,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "gate": {
            "all_correct": all_correct,
            "zero_local_bytes": zero_local_bytes,
            "canary_pass": all_canary,
            "promotion_pass": all_promoted,
            "promoted_cells": promoted_cells,
            "cell_count": len(rows),
            "minimum_cell_median_speedup": min(
                float(row["median_speedup_vs_flash"]) for row in rows
            ),
            "decision": decision,
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
