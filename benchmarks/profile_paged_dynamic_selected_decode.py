"""Profile no-sync GPU Q-head route lowering plus selected H100 decode."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_paged_exact_decode import _flashinfer_runner, _time_cuda  # noqa: E402
from benchmarks.profile_paged_selected_decode import (  # noqa: E402
    _evenly_spaced_tiles,
)
from stream_attention.paged import (  # noqa: E402
    PagedDynamicSelectedDecodePlan,
    PagedKVCache,
    PagedSelectedDecodePlan,
    paged_selected_reference,
)
from stream_attention.planning import (  # noqa: E402
    ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    AttentionProblem,
    AttentionTilePlan,
)
from stream_attention.selected_routes import prepare_paged_routes64  # noqa: E402


def _time_cuda_amortized(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> float:
    """Measure sub-launch-scale paths with one event interval over many calls."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / repeats


def _qhead_rows(
    *,
    batch: int,
    q_heads: int,
    tile_count: int,
    selected_count: int,
    mode: str,
) -> tuple[tuple[int, ...], ...]:
    base = _evenly_spaced_tiles(tile_count, selected_count)
    if mode == "shared":
        return (base,) * (batch * q_heads)
    if mode == "alternating":
        shifted = tuple(sorted((tile + 1) % tile_count for tile in base))
        return tuple(
            base if head % 2 == 0 else shifted
            for _batch in range(batch)
            for head in range(q_heads)
        )
    if mode == "disjoint":
        if selected_count * q_heads > tile_count:
            raise ValueError("disjoint mode requires selected_tiles * Hq <= total tiles")
        return tuple(
            tuple(range(head * selected_count, (head + 1) * selected_count))
            for _batch in range(batch)
            for head in range(q_heads)
        )
    raise ValueError(f"unknown route mode: {mode}")


def _wall_time_torch_lowering(
    tile_plan: AttentionTilePlan,
    cache: PagedKVCache,
    *,
    repeats: int,
) -> tuple[float, list[float]]:
    prepare_paged_routes64(tile_plan, cache)
    samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        prepare_paged_routes64(tile_plan, cache)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(samples)), samples


def profile(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("dynamic selected profiling requires H100/SM90")
    if args.kv_len <= 0 or args.kv_len % 64:
        raise ValueError("kv_len must be positive and divisible by 64")
    if args.selected_tokens <= 0 or args.selected_tokens % 64:
        raise ValueError("selected_tokens must be positive and divisible by 64")
    if args.selected_tokens > args.kv_len:
        raise ValueError("selected_tokens cannot exceed kv_len")
    if args.page_size != 16:
        raise ValueError("dynamic PackedRoute64 requires page size 16")
    if args.q_heads % args.kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pages_per_request = args.kv_len // args.page_size
    total_pages = args.batch * pages_per_request
    query = torch.randn(
        args.batch,
        1,
        args.q_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    cache_shape = (
        (total_pages, args.page_size, args.kv_heads, args.head_dim)
        if args.layout == "NHD"
        else (total_pages, args.kv_heads, args.page_size, args.head_dim)
    )
    key = torch.randn(*cache_shape, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    page_table = (
        torch.randperm(total_pages, device=device, dtype=torch.int64)
        .to(torch.int32)
        .view(args.batch, pages_per_request)
        .contiguous()
    )
    cache = PagedKVCache(
        key=key,
        value=value,
        page_table=page_table,
        sequence_lengths=torch.full(
            (args.batch,), args.kv_len, device=device, dtype=torch.int32
        ),
        layout=args.layout,
    )
    selected_count = args.selected_tokens // 64
    rows = _qhead_rows(
        batch=args.batch,
        q_heads=args.q_heads,
        tile_count=args.kv_len // 64,
        selected_count=selected_count,
        mode=args.route_mode,
    )
    problem = AttentionProblem.from_paged(
        query,
        cache,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    tile_plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=64,
        tile_ids_per_row=rows,
        policy_id="paged-dynamic-selected-profile",
        reason=args.route_mode,
        route_granularity=ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
        schedule_epoch=1,
    ).with_device_routes(device=device)
    source_routes = tile_plan.schedule.device_routes
    assert source_routes is not None

    torch_lowering_ms, torch_lowering_samples = _wall_time_torch_lowering(
        tile_plan,
        cache,
        repeats=args.lowering_repeats,
    )
    static_routes = prepare_paged_routes64(tile_plan, cache)
    static_plan = PagedSelectedDecodePlan.build(
        query,
        cache,
        static_routes,
        schedule_epoch=1,
    )
    dynamic_plan = PagedDynamicSelectedDecodePlan.build(query, cache, source_routes)

    expected = paged_selected_reference(
        query,
        cache,
        static_routes,
        schedule_epoch=1,
    ).clone()
    static_output = static_plan.run().clone()
    dynamic_output = dynamic_plan.run().clone()
    torch.cuda.synchronize()
    dynamic_plan.check_route_errors()
    static_error = float((static_output.float() - expected.float()).abs().max().item())
    dynamic_error = float((dynamic_output.float() - expected.float()).abs().max().item())
    if static_error > args.atol or dynamic_error > args.atol:
        raise RuntimeError(
            "selected decode mismatch: "
            f"static={static_error:.6g}, dynamic={dynamic_error:.6g}, atol={args.atol:.6g}"
        )

    original_atoms = source_routes.atom_ids.clone()
    mutation_shift = max(1, (args.kv_len // 64) // 3)
    mutated_rows = tuple(
        tuple(sorted((atom + mutation_shift) % (args.kv_len // 64) for atom in row))
        for row in rows
    )
    mutated_atoms = torch.tensor(
        [atom for row in mutated_rows for atom in row],
        device=device,
        dtype=torch.int32,
    )
    source_routes.atom_ids.copy_(mutated_atoms)
    mutated_static_routes = prepare_paged_routes64(tile_plan, cache)
    mutated_expected = paged_selected_reference(
        query,
        cache,
        mutated_static_routes,
        schedule_epoch=1,
    ).clone()
    mutated_output = dynamic_plan.run().clone()
    torch.cuda.synchronize()
    dynamic_plan.check_route_errors()
    mutation_error = float(
        (mutated_output.float() - mutated_expected.float()).abs().max().item()
    )
    mutation_delta = float(
        (mutated_output.float() - dynamic_output.float()).abs().max().item()
    )
    if mutation_error > args.atol or mutation_delta == 0.0:
        raise RuntimeError(
            "dynamic route mutation failed: "
            f"error={mutation_error:.6g}, output_delta={mutation_delta:.6g}"
        )
    source_routes.atom_ids.copy_(original_atoms)
    dynamic_plan.run()
    torch.cuda.synchronize()
    dynamic_plan.check_route_errors()

    prepare_samples = _time_cuda(
        dynamic_plan.prepare,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    static_samples = _time_cuda(
        static_plan.run,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    dynamic_samples = _time_cuda(
        dynamic_plan.run,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    prepare_ms = float(statistics.median(prepare_samples))
    static_ms = float(statistics.median(static_samples))
    dynamic_ms = float(statistics.median(dynamic_samples))
    amortized_prepare_ms = _time_cuda_amortized(
        dynamic_plan.prepare,
        warmup=args.warmup,
        repeats=args.amortized_repeats,
    )
    amortized_static_ms = _time_cuda_amortized(
        static_plan.run,
        warmup=args.warmup,
        repeats=args.amortized_repeats,
    )
    amortized_dynamic_ms = _time_cuda_amortized(
        dynamic_plan.run,
        warmup=args.warmup,
        repeats=args.amortized_repeats,
    )

    flashinfer_candidates: list[dict[str, Any]] = []
    valid_flashinfer: list[tuple[Any, dict[str, Any], list[float]]] = []
    for backend in (
        item.strip() for item in args.flashinfer_backends.split(",") if item.strip()
    ):
        try:
            runner, info = _flashinfer_runner(
                query,
                cache,
                workspace_mb=args.workspace_mb,
                backend=backend,
            )
            samples = _time_cuda(runner, warmup=args.warmup, repeats=args.repeats)
            median_ms = float(statistics.median(samples))
            flashinfer_candidates.append({**info, "status": "ok", "median_ms": median_ms})
            valid_flashinfer.append((runner, info, samples))
        except Exception as exc:
            flashinfer_candidates.append(
                {
                    "requested_backend": backend,
                    "status": "unsupported",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not valid_flashinfer:
        raise RuntimeError("no FlashInfer exact backend was available")
    flashinfer_run, flashinfer_info, flashinfer_samples = min(
        valid_flashinfer,
        key=lambda item: statistics.median(item[2]),
    )
    flashinfer_ms = float(statistics.median(flashinfer_samples))
    amortized_flashinfer_ms = _time_cuda_amortized(
        flashinfer_run,
        warmup=args.warmup,
        repeats=args.amortized_repeats,
    )

    paired_trials: list[dict[str, Any]] = []
    for trial in range(args.paired_trials):
        if trial % 2 == 0:
            dynamic_trial = _time_cuda(
                dynamic_plan.run, warmup=0, repeats=args.paired_repeats
            )
            flashinfer_trial = _time_cuda(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            order = "streamattn_first"
        else:
            flashinfer_trial = _time_cuda(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            dynamic_trial = _time_cuda(
                dynamic_plan.run, warmup=0, repeats=args.paired_repeats
            )
            order = "flashinfer_first"
        dynamic_trial_ms = float(statistics.median(dynamic_trial))
        flashinfer_trial_ms = float(statistics.median(flashinfer_trial))
        paired_trials.append(
            {
                "trial": trial,
                "order": order,
                "streamattn_dynamic_ms": dynamic_trial_ms,
                "flashinfer_ms": flashinfer_trial_ms,
                "speedup_vs_flashinfer": flashinfer_trial_ms / dynamic_trial_ms,
            }
        )
    paired_speedups = [float(row["speedup_vs_flashinfer"]) for row in paired_trials]
    route_counts = dynamic_plan.metadata["route_counts"].detach().cpu()
    active_routes = int(route_counts.sum().item())
    selected_head_routes = int(source_routes.nnz)
    group_size = args.q_heads // args.kv_heads
    union_efficiency = selected_head_routes / max(1, group_size * active_routes)

    return {
        "schema": "streamattn.paged_dynamic_selected_decode_profile.v1",
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "batch": args.batch,
        "kv_len": args.kv_len,
        "selected_tokens_per_head": args.selected_tokens,
        "route_mode": args.route_mode,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "group_size": group_size,
        "head_dim": args.head_dim,
        "layout": args.layout,
        "page_size": args.page_size,
        "source_route_nnz": source_routes.nnz,
        "active_group_routes": active_routes,
        "max_routes_per_group": dynamic_plan.max_routes_per_group,
        "group_route_efficiency": union_efficiency,
        "producer_ctas_launched": dynamic_plan.producer_ctas,
        "producer_ctas_active": active_routes,
        "metadata_bytes": dynamic_plan.metadata_bytes,
        "workspace_bytes": dynamic_plan.workspace_bytes,
        "torch_lowering_ms": torch_lowering_ms,
        "gpu_prepare_ms": prepare_ms,
        "gpu_prepare_amortized_ms": amortized_prepare_ms,
        "gpu_prepare_speedup_vs_torch": torch_lowering_ms / prepare_ms,
        "static_selected_ms": static_ms,
        "static_selected_amortized_ms": amortized_static_ms,
        "dynamic_prepare_and_selected_ms": dynamic_ms,
        "dynamic_prepare_and_selected_amortized_ms": amortized_dynamic_ms,
        "dynamic_overhead_vs_static_ms": dynamic_ms - static_ms,
        "flashinfer_ms": flashinfer_ms,
        "flashinfer_amortized_ms": amortized_flashinfer_ms,
        "dynamic_speedup_vs_flashinfer": flashinfer_ms / dynamic_ms,
        "dynamic_amortized_speedup_vs_flashinfer": (
            amortized_flashinfer_ms / amortized_dynamic_ms
        ),
        "static_speedup_vs_flashinfer": flashinfer_ms / static_ms,
        "max_abs_error_static": static_error,
        "max_abs_error_dynamic": dynamic_error,
        "max_abs_error_after_route_mutation": mutation_error,
        "max_output_delta_after_route_mutation": mutation_delta,
        "torch_lowering_samples_ms": torch_lowering_samples,
        "gpu_prepare_samples_ms": prepare_samples,
        "static_samples_ms": static_samples,
        "dynamic_samples_ms": dynamic_samples,
        "flashinfer_samples_ms": flashinfer_samples,
        "flashinfer": flashinfer_info,
        "flashinfer_candidates": flashinfer_candidates,
        "paired": {
            "trials": paired_trials,
            "speedup_median": float(statistics.median(paired_speedups)),
            "speedup_min": float(min(paired_speedups)),
            "wins": sum(value > 1.0 for value in paired_speedups),
            "trial_count": len(paired_speedups),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--selected-tokens", type=int, default=384)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument(
        "--route-mode",
        choices=("shared", "alternating", "disjoint"),
        default="shared",
    )
    parser.add_argument("--flashinfer-backends", default="auto,fa2,fa3")
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--lowering-repeats", type=int, default=5)
    parser.add_argument("--paired-trials", type=int, default=9)
    parser.add_argument("--paired-repeats", type=int, default=10)
    parser.add_argument("--amortized-repeats", type=int, default=200)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = profile(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
