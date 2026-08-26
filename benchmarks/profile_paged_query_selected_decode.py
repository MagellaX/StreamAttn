"""Profile query selection plus dynamic selected H100 paged decode."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_paged_dynamic_selected_decode import (  # noqa: E402
    _time_cuda_amortized,
)
from benchmarks.profile_paged_exact_decode import _flashinfer_runner, _time_cuda  # noqa: E402
from stream_attention.paged import (  # noqa: E402
    PagedKVCache,
    PagedQuerySelectedDecodePlan,
    build_paged_support_keys,
    paged_selected_reference,
)
from stream_attention.planning import (  # noqa: E402
    ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    AttentionProblem,
    AttentionTilePlan,
)
from stream_attention.selected_routes import prepare_paged_routes64  # noqa: E402


def _logical_keys(cache: PagedKVCache) -> torch.Tensor:
    pages = cache.key[cache.page_table.to(torch.long)]
    if cache.normalized_layout == "NHD":
        pages = pages.permute(0, 3, 1, 2, 4)
    else:
        pages = pages.permute(0, 2, 1, 3, 4)
    return pages.reshape(
        cache.batch_size,
        cache.kv_heads,
        cache.max_sequence_length // 64,
        64,
        cache.head_dim,
    )


def _oracle_middle_recall(
    query: torch.Tensor,
    cache: PagedKVCache,
    selected: torch.Tensor,
    *,
    sink_atoms: int,
    recent_atoms: int,
) -> float:
    keys = _logical_keys(cache)
    rows = selected.view(cache.batch_size, int(query.shape[2]), -1)
    middle_count = int(rows.shape[-1]) - sink_atoms - recent_atoms
    recalls: list[float] = []
    for batch in range(cache.batch_size):
        valid_atoms = int(cache.sequence_lengths[batch].item()) // 64
        for head in range(int(query.shape[2])):
            kv_head = head // (int(query.shape[2]) // cache.kv_heads)
            scores = torch.mv(
                keys[batch, kv_head].float().reshape(-1, cache.head_dim),
                query[batch, 0, head].float(),
            ).view(valid_atoms, 64).max(dim=1).values
            scores[:sink_atoms] = -float("inf")
            scores[valid_atoms - recent_atoms :] = -float("inf")
            oracle = set(scores.topk(middle_count).indices.cpu().tolist())
            chosen = set(rows[batch, head, sink_atoms + recent_atoms :].cpu().tolist())
            recalls.append(len(oracle & chosen) / middle_count)
    return float(statistics.mean(recalls))


def profile(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("query-selected profiling requires H100/SM90")
    if args.kv_len <= 0 or args.kv_len % 64:
        raise ValueError("kv_len must be positive and divisible by 64")
    if args.selected_tokens <= 0 or args.selected_tokens % 64:
        raise ValueError("selected_tokens must be positive and divisible by 64")
    if args.page_size != 16 or args.q_heads % args.kv_heads:
        raise ValueError("query-selected profile requires page16 and valid GQA")

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

    summary_samples = _time_cuda(
        lambda: build_paged_support_keys(
            cache,
            support_width=args.support_width,
            method=args.support_method,
        ),
        warmup=1,
        repeats=args.summary_repeats,
    )
    support_keys = build_paged_support_keys(
        cache,
        support_width=args.support_width,
        method=args.support_method,
    )
    selected_atoms = args.selected_tokens // 64
    plan = PagedQuerySelectedDecodePlan.build(
        query,
        cache,
        selected_atoms=selected_atoms,
        sink_atoms=args.sink_atoms,
        recent_atoms=args.recent_atoms,
        support_width=args.support_width,
        support_method=args.support_method,
        support_keys=support_keys,
    )

    plan.select()
    torch.cuda.synchronize()
    initial_atoms = plan.routes.atom_ids.clone()
    route_rows = tuple(
        tuple(int(atom) for atom in row)
        for row in initial_atoms.view(args.batch * args.q_heads, -1).cpu().tolist()
    )
    problem = AttentionProblem.from_paged(
        query,
        cache,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    tile_plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=64,
        tile_ids_per_row=route_rows,
        policy_id="paged-query-selected-profile-reference",
        reason=args.support_method,
        route_granularity=ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
        schedule_epoch=1,
    ).with_device_routes(device=device)
    static_routes = prepare_paged_routes64(tile_plan, cache)
    expected = paged_selected_reference(
        query,
        cache,
        static_routes,
        schedule_epoch=1,
    ).clone()
    output = plan.run().clone()
    torch.cuda.synchronize()
    plan.check_route_errors()
    max_error = float((output.float() - expected.float()).abs().max().item())
    if max_error > args.atol:
        raise RuntimeError(
            f"query-selected output mismatch: {max_error:.6g} > {args.atol:.6g}"
        )

    query.copy_(torch.randn_like(query))
    plan.select()
    torch.cuda.synchronize()
    mutated_atoms = plan.routes.atom_ids.clone()
    changed_route_entries = int((mutated_atoms != initial_atoms).sum().item())
    if changed_route_entries == 0:
        raise RuntimeError("query mutation did not change any selected atom")
    query.copy_(torch.randn_like(query))
    plan.run()
    torch.cuda.synchronize()
    plan.check_route_errors()

    selector_samples = _time_cuda(
        plan.select, warmup=args.warmup, repeats=args.repeats
    )
    selected_samples = _time_cuda(
        plan.selected_plan.run, warmup=args.warmup, repeats=args.repeats
    )
    complete_samples = _time_cuda(
        plan.run, warmup=args.warmup, repeats=args.repeats
    )
    selector_ms = float(statistics.median(selector_samples))
    selected_ms = float(statistics.median(selected_samples))
    complete_ms = float(statistics.median(complete_samples))
    complete_amortized_ms = _time_cuda_amortized(
        plan.run,
        warmup=args.warmup,
        repeats=args.amortized_repeats,
    )

    flashinfer_candidates: list[dict[str, Any]] = []
    valid_flashinfer: list[tuple[Any, dict[str, Any], list[float]]] = []
    for backend in (
        value.strip() for value in args.flashinfer_backends.split(",") if value.strip()
    ):
        try:
            runner, info = _flashinfer_runner(
                query, cache, workspace_mb=args.workspace_mb, backend=backend
            )
            samples = _time_cuda(runner, warmup=args.warmup, repeats=args.repeats)
            flashinfer_candidates.append(
                {**info, "status": "ok", "median_ms": statistics.median(samples)}
            )
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
        valid_flashinfer, key=lambda item: statistics.median(item[2])
    )
    flashinfer_ms = float(statistics.median(flashinfer_samples))
    flashinfer_amortized_ms = _time_cuda_amortized(
        flashinfer_run,
        warmup=args.warmup,
        repeats=args.amortized_repeats,
    )

    paired: list[dict[str, Any]] = []
    for trial in range(args.paired_trials):
        if trial % 2 == 0:
            stream_ms = _time_cuda_amortized(
                plan.run, warmup=0, repeats=args.paired_repeats
            )
            flash_ms = _time_cuda_amortized(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            order = "streamattn_first"
        else:
            flash_ms = _time_cuda_amortized(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            stream_ms = _time_cuda_amortized(
                plan.run, warmup=0, repeats=args.paired_repeats
            )
            order = "flashinfer_first"
        paired.append(
            {
                "trial": trial,
                "order": order,
                "streamattn_ms": stream_ms,
                "flashinfer_ms": flash_ms,
                "speedup_vs_flashinfer": flash_ms / stream_ms,
            }
        )
    paired_speedups = [float(row["speedup_vs_flashinfer"]) for row in paired]
    oracle_recall = _oracle_middle_recall(
        query,
        cache,
        plan.routes.atom_ids,
        sink_atoms=args.sink_atoms,
        recent_atoms=args.recent_atoms,
    )

    return {
        "schema": "streamattn.paged_query_selected_decode_profile.v1",
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "batch": args.batch,
        "kv_len": args.kv_len,
        "selected_tokens_per_head": args.selected_tokens,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "layout": args.layout,
        "page_size": args.page_size,
        "support_width": args.support_width,
        "support_method": args.support_method,
        "support_scan_ratio_vs_token_qk": args.support_width / 64.0,
        "oracle_middle_block_recall": oracle_recall,
        "changed_route_entries_after_query_mutation": changed_route_entries,
        "max_abs_error_vs_selected_reference": max_error,
        "support_metadata_bytes": plan.support_metadata_bytes,
        "selector_workspace_bytes": plan.selector_workspace_bytes,
        "summary_build_ms": float(statistics.median(summary_samples)),
        "selector_ms": selector_ms,
        "route_compile_and_attention_ms": selected_ms,
        "complete_selector_and_attention_ms": complete_ms,
        "complete_amortized_ms": complete_amortized_ms,
        "flashinfer_ms": flashinfer_ms,
        "flashinfer_amortized_ms": flashinfer_amortized_ms,
        "speedup_vs_flashinfer": flashinfer_ms / complete_ms,
        "amortized_speedup_vs_flashinfer": (
            flashinfer_amortized_ms / complete_amortized_ms
        ),
        "flashinfer": flashinfer_info,
        "flashinfer_candidates": flashinfer_candidates,
        "paired": {
            "trials": paired,
            "speedup_median": float(statistics.median(paired_speedups)),
            "speedup_min": float(min(paired_speedups)),
            "wins": sum(value > 1.0 for value in paired_speedups),
            "trial_count": len(paired_speedups),
        },
        "note": (
            "Systems evidence only: attention is exact over selected atoms; "
            "model-distribution safety requires a separate policy gate."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--selected-tokens", type=int, default=384)
    parser.add_argument("--sink-atoms", type=int, default=1)
    parser.add_argument("--recent-atoms", type=int, default=1)
    parser.add_argument("--support-width", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument(
        "--support-method",
        choices=("centroid_extremes", "centroid_top_norm"),
        default="centroid_extremes",
    )
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--flashinfer-backends", default="auto,fa2,fa3")
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--summary-repeats", type=int, default=3)
    parser.add_argument("--paired-trials", type=int, default=9)
    parser.add_argument("--paired-repeats", type=int, default=10)
    parser.add_argument("--amortized-repeats", type=int, default=200)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=31)
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
