"""Profile split-K inline projection Gate-0 prototype."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_candidate_filters import (
    _project_query,
    _projection_matrix,
    _projection_metadata,
    _projection_metadata_dtype,
)
from benchmarks.profile_gate0_summary_bounds import _dtype, _load_real_tensors, _sync, _time_call
from benchmarks.profile_gate1_inline_projection import (
    _load_or_make_tensors,
    _max_mean_error,
    _summarize_inline_stats,
    _summarize_inline_stats_per_head,
)
from stream_attention.gate1 import dense_attention_forward
from stream_attention.kernels.gate1_inline_projection_fwd_triton import (
    gate1_inline_projection_attention_triton_forward,
)
from stream_attention.kernels.gate1_inline_projection_splitk_triton import (
    SPLITK_PROJECTION_STATS,
    gate1_inline_projection_splitk_attention_triton_forward,
    make_splitk_workspace,
)
import stream_attention.kernels.gate1_inline_projection_splitk_triton as splitk_triton


def _parse_head_indices(raw: str, *, heads: int) -> list[int]:
    if not raw:
        return []
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    expanded: list[int] = []
    for value in values:
        if value < 0:
            expanded.extend(range(heads))
        else:
            if value >= heads:
                raise ValueError(f"head index {value} is outside [0, {heads})")
            expanded.append(value)
    return sorted(set(expanded))


def _summarize_splitk_stats(raw_stats: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if raw_stats is None:
        return None
    detached = raw_stats.detach().cpu()
    totals = detached.sum(dim=(0, 1, 2))
    result = {
        name: int(totals[index].item())
        for name, index in SPLITK_PROJECTION_STATS.items()
    }
    result["seed_blocks"] = int(detached[..., SPLITK_PROJECTION_STATS["seed_blocks"]].max().item())
    result["chunks"] = int(raw_stats.shape[2])
    result["mode"] = int(detached[..., SPLITK_PROJECTION_STATS["mode"]].max().item())
    middle = result["middle_blocks"]
    result["projection_skip_fraction"] = (
        result["projection_skipped_blocks"] / middle if middle else 0.0
    )
    result["pv_executed_fraction"] = (
        result["pv_executed_blocks"] / middle if middle else 0.0
    )
    return result


def _summarize_splitk_stats_per_head(raw_stats: Optional[torch.Tensor]):
    if raw_stats is None:
        return None
    if raw_stats.dim() != 4 or raw_stats.shape[-1] != 8:
        raise ValueError("split-K raw stats must have shape [batch, heads, chunks, 8]")
    rows = raw_stats.detach().sum(dim=(0, 2)).cpu()
    per_head = []
    skip_fracs = []
    pv_fracs = []
    for head_idx, row in enumerate(rows):
        middle = int(row[SPLITK_PROJECTION_STATS["middle_blocks"]].item())
        skipped = int(row[SPLITK_PROJECTION_STATS["projection_skipped_blocks"]].item())
        pv = int(row[SPLITK_PROJECTION_STATS["pv_executed_blocks"]].item())
        skip_frac = skipped / middle if middle else 0.0
        pv_frac = pv / middle if middle else 0.0
        skip_fracs.append(skip_frac)
        pv_fracs.append(pv_frac)
        per_head.append(
            {
                "head": head_idx,
                "middle_blocks": middle,
                "projection_skipped_blocks": skipped,
                "projection_skip_fraction": skip_frac,
                "pv_executed_blocks": pv,
                "pv_executed_fraction": pv_frac,
            }
        )
    return {
        "per_head": per_head,
        "projection_skip_fraction_mean": sum(skip_fracs) / len(skip_fracs) if skip_fracs else 0.0,
        "pv_executed_fraction_mean": sum(pv_fracs) / len(pv_fracs) if pv_fracs else 0.0,
    }


def _per_head_error(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, Any]:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
    if actual.dim() != 4:
        raise ValueError("attention outputs must have shape [batch, query_len, heads, dim]")
    diff = (actual - expected).detach().abs().float()
    rows = []
    max_values = []
    mean_values = []
    for head_idx in range(diff.shape[2]):
        head_diff = diff[:, :, head_idx, :]
        max_error = float(head_diff.max().item())
        mean_error = float(head_diff.mean().item())
        max_values.append(max_error)
        mean_values.append(mean_error)
        rows.append(
            {
                "head": head_idx,
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
            }
        )
    worst_idx = max(range(len(max_values)), key=lambda idx: max_values[idx]) if max_values else None
    return {
        "per_head": rows,
        "worst_head": worst_idx,
        "max_abs_error": max(max_values) if max_values else 0.0,
        "mean_abs_error": sum(mean_values) / len(mean_values) if mean_values else 0.0,
    }


def _time_recompute_seed_breakdown(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_proj: Optional[torch.Tensor],
    projection: torch.Tensor,
    proj_min: torch.Tensor,
    proj_max: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    if args.seed_strategy != "recompute_seed":
        return {}
    batch, _seq_q, heads, dim = q.shape
    seq_k = k.shape[1]
    rank = projection.shape[0] if args.qproj_mode == "fused" else q_proj.shape[3]  # type: ignore[union-attr]
    num_blocks = math.ceil(seq_k / args.block_size)
    if args.sink_blocks + args.recent_blocks > num_blocks:
        return {}
    middle_blocks = num_blocks - args.sink_blocks - args.recent_blocks
    nonseed_middle = middle_blocks - args.middle_seed_blocks
    if nonseed_middle < 0:
        return {}

    num_chunks = int(args.num_chunks)
    total_states = num_chunks + 1
    chunk_blocks = max(1, math.ceil(nonseed_middle / num_chunks))
    chunk_anchor_blocks = min(int(args.chunk_anchor_blocks), int(chunk_blocks))
    chunk_max = torch.empty(batch, heads, total_states, device=q.device, dtype=torch.float32)
    chunk_den = torch.empty(batch, heads, total_states, device=q.device, dtype=torch.float32)
    chunk_num = torch.empty(batch, heads, total_states, dim, device=q.device, dtype=torch.float32)
    output = torch.empty_like(q)
    raw_stats = torch.empty(1, device=q.device, dtype=torch.int32)
    score_scale = 1.0 / math.sqrt(dim)
    projection_score_scale = (float(dim) / float(rank)) * score_scale
    recent_start = num_blocks - args.recent_blocks
    q_proj_arg = q_proj if q_proj is not None else projection
    projection_arg = projection
    block_order_id = splitk_triton._block_order_id(args.block_order)

    def run_chunk(debug_mode: int):
        return splitk_triton._chunk_recompute_seed_kernel[(batch, heads, num_chunks)](
            q,
            k,
            v,
            q_proj_arg,
            projection_arg,
            proj_min,
            proj_max,
            chunk_max,
            chunk_den,
            chunk_num,
            raw_stats,
            N=seq_k,
            H=heads,
            D=dim,
            RANK=rank,
            NUM_BLOCKS=num_blocks,
            NUM_CHUNKS=num_chunks,
            CHUNK_BLOCKS=chunk_blocks,
            TILE_N=args.block_size,
            SCALE=score_scale,
            PROJ_SCORE_SCALE=projection_score_scale,
            ERROR_BUDGET=float(args.error_budget),
            LOG_ERROR_BUDGET=math.log(max(float(args.error_budget), 1.0e-20)),
            FILTER_MARGIN=float(args.filter_margin),
            SINK_BLOCKS=int(args.sink_blocks),
            RECENT_BLOCKS=int(args.recent_blocks),
            RECENT_START=int(recent_start),
            MIDDLE_SEED_BLOCKS=int(args.middle_seed_blocks),
            CHUNK_ANCHOR_BLOCKS=int(chunk_anchor_blocks),
            BLOCK_ORDER=block_order_id,
            COMPUTE_QPROJ=args.qproj_mode == "fused",
            PV_USE_BF16=v.dtype is torch.bfloat16,
            HAS_STATS=False,
            DEBUG_MODE=debug_mode,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    def run_merge():
        return splitk_triton._merge_states_kernel[(batch, heads)](
            chunk_max,
            chunk_den,
            chunk_num,
            output,
            H=heads,
            D=dim,
            TOTAL_STATES=total_states,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    # Populate valid chunk states before timing merge-only.
    run_chunk(0)
    _sync(q.device)
    full_chunk_ms = _time_call(lambda: run_chunk(0), device=q.device, warmup=args.warmup, iters=args.iters)
    seed_only_ms = _time_call(lambda: run_chunk(1), device=q.device, warmup=args.warmup, iters=args.iters)
    projection_only_ms = _time_call(lambda: run_chunk(2), device=q.device, warmup=args.warmup, iters=args.iters)
    no_pv_ms = _time_call(lambda: run_chunk(3), device=q.device, warmup=args.warmup, iters=args.iters)
    run_chunk(0)
    _sync(q.device)
    merge_ms = _time_call(run_merge, device=q.device, warmup=args.warmup, iters=args.iters)
    return {
        "chunk_full_ms": full_chunk_ms,
        "chunk_seed_only_ms": seed_only_ms,
        "chunk_projection_only_ms": projection_only_ms,
        "chunk_no_pv_ms": no_pv_ms,
        "merge_ms": merge_ms,
        "chunk_plus_merge_ms": full_chunk_ms + merge_ms,
        "active_qk_pv_estimate_ms": max(0.0, full_chunk_ms - projection_only_ms),
        "pv_estimate_ms": max(0.0, full_chunk_ms - no_pv_ms),
        "projection_middle_estimate_ms": max(0.0, projection_only_ms - seed_only_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", default="")
    parser.add_argument("--k-path", default="")
    parser.add_argument("--v-path", default="")
    parser.add_argument("--tensor-format", choices=["pt"], default="pt")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--head-index", type=int, default=-1)
    parser.add_argument("--head-indices", default="")
    parser.add_argument("--pattern", choices=["random", "peaked"], default="peaked")
    parser.add_argument("--active-fraction", type=float, default=0.0625)
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--tile-size-q", type=int, default=16)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--seed-strategy", choices=["separate", "recompute_seed"], default="recompute_seed")
    parser.add_argument("--chunk-anchor-blocks", type=int, default=0)
    parser.add_argument("--error-budget", type=float, default=1e-2)
    parser.add_argument("--filter-margin", type=float, default=32.0)
    parser.add_argument("--post-qk-threshold", type=float, default=0.0)
    parser.add_argument("--projection-kind", choices=["random", "hadamard"], default="random")
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--projection-metadata-dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--qproj-mode", choices=["precomputed", "fused"], default="fused")
    parser.add_argument("--splitk-workspace", choices=["none", "reuse"], default="none")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--splitk-breakdown", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    with torch.no_grad():
        if args.head_indices:
            args.head_index = -1
        q, k, v = _load_or_make_tensors(args)
        selected_head_indices = _parse_head_indices(args.head_indices, heads=q.shape[2])
        if selected_head_indices:
            head_tensor = torch.tensor(selected_head_indices, device=q.device, dtype=torch.long)
            q = q.index_select(2, head_tensor).contiguous()
            k = k.index_select(2, head_tensor).contiguous()
            v = v.index_select(2, head_tensor).contiguous()
        if q.shape[1] != 1:
            raise ValueError("split-K inline projection benchmark requires query_len == 1")
        dim = q.shape[-1]
        projection = _projection_matrix(
            args.projection_kind,
            dim=dim,
            rank=args.projection_dim,
            seed=args.seed,
            device=q.device,
        )
        metadata_dtype = _projection_metadata_dtype(args.projection_metadata_dtype)
        proj_min, proj_max = _projection_metadata(
            k,
            block_size=args.block_size,
            projection=projection,
            metadata_dtype=metadata_dtype,
        )
        q_projection_reference_ms = _time_call(
            lambda: _project_query(q, projection),
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        q_projection_ms = 0.0 if args.qproj_mode == "fused" else q_projection_reference_ms
        q_proj = _project_query(q, projection) if args.qproj_mode == "precomputed" else None
        splitk_workspace = (
            make_splitk_workspace(
                q,
                rank=args.projection_dim,
                num_chunks=args.num_chunks,
                seed_strategy=args.seed_strategy,
            )
            if args.splitk_workspace == "reuse"
            else None
        )

        dense_ms = _time_call(
            lambda: dense_attention_forward(q, k, v, causal=False),
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        serial_inline_ms = _time_call(
            lambda: gate1_inline_projection_attention_triton_forward(
                q,
                k,
                v,
                q_proj if args.qproj_mode == "precomputed" else None,
                proj_min,
                proj_max,
                projection=projection if args.qproj_mode == "fused" else None,
                compute_qproj=args.qproj_mode == "fused",
                error_budget=args.error_budget,
                filter_margin=args.filter_margin,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                post_qk_threshold=args.post_qk_threshold,
                return_raw_stats=False,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )[0],
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        splitk_ms = _time_call(
            lambda: gate1_inline_projection_splitk_attention_triton_forward(
                q,
                k,
                v,
                q_proj if args.qproj_mode == "precomputed" else None,
                proj_min,
                proj_max,
                projection=projection if args.qproj_mode == "fused" else None,
                compute_qproj=args.qproj_mode == "fused",
                num_chunks=args.num_chunks,
                error_budget=args.error_budget,
                filter_margin=args.filter_margin,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                chunk_anchor_blocks=args.chunk_anchor_blocks,
                block_order=args.block_order,
                seed_strategy=args.seed_strategy,
                return_raw_stats=False,
                workspace=splitk_workspace,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )[0],
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        splitk_breakdown = (
            _time_recompute_seed_breakdown(q, k, v, q_proj, projection, proj_min, proj_max, args)
            if args.splitk_breakdown
            else None
        )

        dense_out = dense_attention_forward(q, k, v, causal=False)
        serial_out, serial_raw = gate1_inline_projection_attention_triton_forward(
            q,
            k,
            v,
            q_proj if args.qproj_mode == "precomputed" else None,
            proj_min,
            proj_max,
            projection=projection if args.qproj_mode == "fused" else None,
            compute_qproj=args.qproj_mode == "fused",
            error_budget=args.error_budget,
            filter_margin=args.filter_margin,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            post_qk_threshold=args.post_qk_threshold,
            return_raw_stats=True,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        splitk_out, splitk_raw = gate1_inline_projection_splitk_attention_triton_forward(
            q,
            k,
            v,
            q_proj if args.qproj_mode == "precomputed" else None,
            proj_min,
            proj_max,
            projection=projection if args.qproj_mode == "fused" else None,
            compute_qproj=args.qproj_mode == "fused",
            num_chunks=args.num_chunks,
            error_budget=args.error_budget,
            filter_margin=args.filter_margin,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            chunk_anchor_blocks=args.chunk_anchor_blocks,
            block_order=args.block_order,
            seed_strategy=args.seed_strategy,
            return_raw_stats=True,
            workspace=splitk_workspace,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        _sync(q.device)

    splitk_total_ms = q_projection_ms + splitk_ms
    serial_total_ms = q_projection_ms + serial_inline_ms
    payload = {
        "device": torch.cuda.get_device_name(q.device),
        "torch": torch.__version__,
        "shape": {
            "batch": q.shape[0],
            "query_len": q.shape[1],
            "kv_len": k.shape[1],
            "heads": q.shape[2],
            "dim": q.shape[3],
            "dtype": args.dtype,
        },
        "selected_head_indices": selected_head_indices,
        "selected_head_count": len(selected_head_indices) if selected_head_indices else q.shape[2],
        "block_size": args.block_size,
        "tile_size_q": args.tile_size_q,
        "sink_blocks": args.sink_blocks,
        "recent_blocks": args.recent_blocks,
        "middle_seed_blocks": args.middle_seed_blocks,
        "chunk_anchor_blocks": args.chunk_anchor_blocks,
        "block_order": args.block_order,
        "num_chunks": args.num_chunks,
        "seed_strategy": args.seed_strategy,
        "error_budget": args.error_budget,
        "filter_margin": args.filter_margin,
        "projection_kind": args.projection_kind,
        "projection_dim": args.projection_dim,
        "projection_seed": args.seed,
        "projection_metadata_dtype": args.projection_metadata_dtype,
        "qproj_mode": args.qproj_mode,
        "splitk_workspace": args.splitk_workspace,
        "q_projection_ms": q_projection_ms,
        "q_projection_reference_ms": q_projection_reference_ms,
        "dense_ms": dense_ms,
        "serial_inline_ms": serial_inline_ms,
        "serial_total_ms": serial_total_ms,
        "splitk_ms": splitk_ms,
        "splitk_total_ms": splitk_total_ms,
        "splitk_breakdown": splitk_breakdown,
        "splitk_vs_dense_speedup": dense_ms / splitk_total_ms if splitk_total_ms > 0 else None,
        "splitk_vs_serial_speedup": serial_total_ms / splitk_total_ms if splitk_total_ms > 0 else None,
        "serial_vs_dense_speedup": dense_ms / serial_total_ms if serial_total_ms > 0 else None,
        "serial_stats": _summarize_inline_stats(serial_raw),
        "serial_per_head_stats": _summarize_inline_stats_per_head(serial_raw),
        "splitk_stats": _summarize_splitk_stats(splitk_raw),
        "splitk_per_head_stats": _summarize_splitk_stats_per_head(splitk_raw),
        "splitk_error_vs_dense": _max_mean_error(splitk_out, dense_out),
        "splitk_error_vs_dense_per_head": _per_head_error(splitk_out, dense_out),
        "serial_error_vs_dense": _max_mean_error(serial_out, dense_out),
        "serial_error_vs_dense_per_head": _per_head_error(serial_out, dense_out),
        "splitk_error_vs_serial": _max_mean_error(splitk_out, serial_out),
        "splitk_error_vs_serial_per_head": _per_head_error(splitk_out, serial_out),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
