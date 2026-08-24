"""Benchmark H100 selected paged WGMMA decode against exact FlashInfer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_paged_exact_decode import (  # noqa: E402
    _flashinfer_runner,
    _time_cuda,
)
from stream_attention.paged import (  # noqa: E402
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


def _evenly_spaced_tiles(tile_count: int, selected_count: int) -> tuple[int, ...]:
    if selected_count <= 0 or selected_count > tile_count:
        raise ValueError("selected tile count must be in [1, tile_count]")
    if selected_count == tile_count:
        return tuple(range(tile_count))
    if selected_count == 1:
        return (tile_count - 1,)
    return tuple(
        (index * (tile_count - 1)) // (selected_count - 1)
        for index in range(selected_count)
    )


def _selected_reference(
    query: torch.Tensor,
    cache: PagedKVCache,
    selected_tiles: tuple[int, ...],
) -> torch.Tensor:
    """Vectorized FP32 reference for all-head 64-token selected routes."""

    batch, _query_len, q_heads, dim = map(int, query.shape)
    group_size = q_heads // cache.kv_heads
    kv_head_indices = torch.arange(q_heads, device=query.device) // group_size
    page_offsets = torch.arange(4, device=query.device, dtype=torch.int64)
    logical_pages = (
        torch.tensor(selected_tiles, device=query.device, dtype=torch.int64)[:, None]
        * 4
        + page_offsets[None, :]
    ).reshape(-1)
    output = torch.empty_like(query, dtype=torch.float32)
    scale = 1.0 / math.sqrt(float(dim))
    for row in range(batch):
        physical_pages = cache.page_table[row].to(torch.int64).index_select(
            0, logical_pages
        )
        key_pages = cache.key.index_select(0, physical_pages)
        value_pages = cache.value.index_select(0, physical_pages)
        if cache.normalized_layout == "HND":
            key_pages = key_pages.permute(0, 2, 1, 3)
            value_pages = value_pages.permute(0, 2, 1, 3)
        keys = key_pages.reshape(-1, cache.kv_heads, dim).float()
        values = value_pages.reshape(-1, cache.kv_heads, dim).float()
        keys = keys.index_select(1, kv_head_indices)
        values = values.index_select(1, kv_head_indices)
        scores = torch.einsum("nhd,hd->hn", keys, query[row, 0].float()) * scale
        output[row, 0] = torch.einsum(
            "hn,nhd->hd", torch.softmax(scores, dim=-1), values
        )
    return output


def profile(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("selected paged WGMMA profiling requires H100/SM90")
    if args.kv_len <= 0 or args.kv_len % 64:
        raise ValueError("kv_len must be positive and divisible by 64")
    if args.selected_tokens <= 0 or args.selected_tokens % 64:
        raise ValueError("selected_tokens must be positive and divisible by 64")
    if args.selected_tokens > args.kv_len:
        raise ValueError("selected_tokens cannot exceed kv_len")
    if args.page_size != 16:
        raise ValueError("selected PackedRoute64 benchmark requires page size 16")
    if args.q_heads % args.kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")

    device = torch.device("cuda")
    dtype = torch.bfloat16
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
        dtype=dtype,
    )
    if args.layout == "NHD":
        cache_shape = (
            total_pages,
            args.page_size,
            args.kv_heads,
            args.head_dim,
        )
    else:
        cache_shape = (
            total_pages,
            args.kv_heads,
            args.page_size,
            args.head_dim,
        )
    key = torch.randn(*cache_shape, device=device, dtype=dtype)
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

    tile_count = args.kv_len // 64
    selected_tiles = _evenly_spaced_tiles(
        tile_count,
        args.selected_tokens // 64,
    )
    problem = AttentionProblem.from_paged(
        query,
        cache,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    if args.route_mode == "all_heads":
        route_rows = (selected_tiles,) * args.batch
        route_granularity = "batch"
    elif args.route_mode == "head_private_alternating":
        shifted_tiles = tuple(sorted((tile + 1) % tile_count for tile in selected_tiles))
        route_rows = tuple(
            selected_tiles if head % 2 == 0 else shifted_tiles
            for _batch in range(args.batch)
            for head in range(args.q_heads)
        )
        route_granularity = ATTENTION_ROUTE_GRANULARITY_Q_HEAD
    else:
        raise ValueError("unknown route_mode")
    logical_plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=64,
        tile_ids_per_row=route_rows,
        policy_id="paged-selected-profile",
        reason=args.route_mode,
        route_granularity=route_granularity,
        schedule_epoch=1,
    )
    torch.cuda.synchronize()
    prepare_start = time.perf_counter()
    routes = prepare_paged_routes64(logical_plan, cache)
    torch.cuda.synchronize()
    route_prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
    build_start = time.perf_counter()
    selected_plan = PagedSelectedDecodePlan.build(
        query,
        cache,
        routes,
        schedule_epoch=1,
    )
    torch.cuda.synchronize()
    plan_build_ms = (time.perf_counter() - build_start) * 1000.0

    selected_output = selected_plan.run().clone()
    reference = (
        _selected_reference(query, cache, selected_tiles)
        if args.route_mode == "all_heads"
        else paged_selected_reference(
            query,
            cache,
            routes,
            schedule_epoch=1,
        ).float()
    )
    torch.cuda.synchronize()
    error = (selected_output.float() - reference).abs()
    max_abs_error = float(error.max().item())
    mean_abs_error = float(error.mean().item())
    if max_abs_error > args.atol:
        raise RuntimeError(
            f"selected WGMMA mismatch: max={max_abs_error:.6g} > {args.atol:.6g}"
        )

    selected_samples = _time_cuda(
        selected_plan.run,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    selected_ms = float(statistics.median(selected_samples))

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

    full_route_error = None
    if args.selected_tokens == args.kv_len:
        flashinfer_output = flashinfer_run().view_as(query).clone()
        torch.cuda.synchronize()
        full_route_error = float(
            (selected_output.float() - flashinfer_output.float()).abs().max().item()
        )
        if full_route_error > args.atol:
            raise RuntimeError(
                f"full selected route differs from FlashInfer: {full_route_error:.6g}"
            )

    paired_trials: list[dict[str, Any]] = []
    for trial in range(args.paired_trials):
        if trial % 2 == 0:
            selected_trial = _time_cuda(
                selected_plan.run, warmup=0, repeats=args.paired_repeats
            )
            flashinfer_trial = _time_cuda(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            order = "streamattn_first"
        else:
            flashinfer_trial = _time_cuda(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            selected_trial = _time_cuda(
                selected_plan.run, warmup=0, repeats=args.paired_repeats
            )
            order = "flashinfer_first"
        selected_trial_ms = float(statistics.median(selected_trial))
        flashinfer_trial_ms = float(statistics.median(flashinfer_trial))
        paired_trials.append(
            {
                "trial": trial,
                "order": order,
                "streamattn_ms": selected_trial_ms,
                "flashinfer_ms": flashinfer_trial_ms,
                "speedup_vs_flashinfer": flashinfer_trial_ms / selected_trial_ms,
            }
        )
    paired_speedups = [float(row["speedup_vs_flashinfer"]) for row in paired_trials]

    return {
        "schema": "streamattn.paged_selected_decode_profile.v1",
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "batch": args.batch,
        "kv_len": args.kv_len,
        "selected_tokens": args.selected_tokens,
        "selected_ratio": args.selected_tokens / args.kv_len,
        "route_mode": args.route_mode,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "group_size": args.q_heads // args.kv_heads,
        "head_dim": args.head_dim,
        "dtype": "bf16",
        "layout": args.layout,
        "page_size": args.page_size,
        "route_count": routes.route_count,
        "max_routes_per_row": selected_plan.max_routes_per_row,
        "producer_ctas": selected_plan.producer_ctas,
        "group_route_efficiency": routes.group_route_efficiency,
        "scheduler_hint": routes.scheduler_hint,
        "metadata_bytes": routes.metadata_bytes,
        "workspace_bytes": selected_plan.workspace_bytes,
        "route_prepare_ms": route_prepare_ms,
        "plan_build_ms": plan_build_ms,
        "streamattn_ms": selected_ms,
        "flashinfer_ms": flashinfer_ms,
        "speedup_vs_flashinfer": flashinfer_ms / selected_ms,
        "max_abs_error_vs_selected_reference": max_abs_error,
        "mean_abs_error_vs_selected_reference": mean_abs_error,
        "max_abs_error_vs_flashinfer_full_route": full_route_error,
        "flashinfer": flashinfer_info,
        "flashinfer_candidates": flashinfer_candidates,
        "streamattn_samples_ms": selected_samples,
        "flashinfer_samples_ms": flashinfer_samples,
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
        choices=("all_heads", "head_private_alternating"),
        default="all_heads",
    )
    parser.add_argument("--flashinfer-backends", default="auto,fa2,fa3")
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--paired-trials", type=int, default=9)
    parser.add_argument("--paired-repeats", type=int, default=10)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=17)
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
