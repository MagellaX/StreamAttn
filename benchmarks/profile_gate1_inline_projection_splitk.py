"""Profile split-K inline projection Gate-0 prototype."""

from __future__ import annotations

import argparse
import json
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
)


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
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    with torch.no_grad():
        q, k, v = _load_or_make_tensors(args)
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
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )[0],
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
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
        "q_projection_ms": q_projection_ms,
        "q_projection_reference_ms": q_projection_reference_ms,
        "dense_ms": dense_ms,
        "serial_inline_ms": serial_inline_ms,
        "serial_total_ms": serial_total_ms,
        "splitk_ms": splitk_ms,
        "splitk_total_ms": splitk_total_ms,
        "splitk_vs_dense_speedup": dense_ms / splitk_total_ms if splitk_total_ms > 0 else None,
        "splitk_vs_serial_speedup": serial_total_ms / splitk_total_ms if splitk_total_ms > 0 else None,
        "serial_vs_dense_speedup": dense_ms / serial_total_ms if serial_total_ms > 0 else None,
        "serial_stats": _summarize_inline_stats(serial_raw),
        "serial_per_head_stats": _summarize_inline_stats_per_head(serial_raw),
        "splitk_stats": _summarize_splitk_stats(splitk_raw),
        "splitk_per_head_stats": _summarize_splitk_stats_per_head(splitk_raw),
        "splitk_error_vs_dense": _max_mean_error(splitk_out, dense_out),
        "serial_error_vs_dense": _max_mean_error(serial_out, dense_out),
        "splitk_error_vs_serial": _max_mean_error(splitk_out, serial_out),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
