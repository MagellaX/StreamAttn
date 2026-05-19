"""Profile the inline projection Gate-0/Gate-1 prototype.

This benchmark is intentionally narrow. It compares dense SDPA, mass Gate-1,
and the inline calibrated projection prototype for single-token contiguous-KV
decode. It is a science benchmark for the next Gate-0 design decision, not a
production API benchmark.
"""

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
from stream_attention.gate1 import dense_attention_forward
from stream_attention.kernels.gate1_inline_projection_fwd_triton import (
    INLINE_PROJECTION_STATS,
    gate1_inline_projection_attention_triton_forward,
)
from stream_attention.kernels.gate1_mass_fwd_triton import gate1_mass_attention_triton_forward


def _make_synthetic(args, *, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(args.seed)
    shape_q = (args.batch, 1, args.heads, args.dim)
    shape_kv = (args.batch, args.kv_len, args.heads, args.dim)
    if args.pattern == "random":
        return (
            torch.randn(shape_q, device=device, dtype=dtype),
            torch.randn(shape_kv, device=device, dtype=dtype),
            torch.randn(shape_kv, device=device, dtype=dtype),
        )
    if args.pattern != "peaked":
        raise ValueError(f"unknown pattern: {args.pattern}")

    num_blocks = (args.kv_len + args.block_size - 1) // args.block_size
    active_blocks = max(1, min(num_blocks, round(args.active_fraction * num_blocks)))
    active_tokens = min(args.kv_len, active_blocks * args.block_size)
    q = torch.zeros(shape_q, device=device, dtype=dtype)
    k = torch.zeros(shape_kv, device=device, dtype=dtype)
    v = torch.randn(shape_kv, device=device, dtype=dtype)
    q[..., 0] = args.peak
    k[:, :active_tokens, :, 0] = args.peak
    k[:, active_tokens:, :, 0] = -args.peak
    return q, k, v


def _load_or_make_tensors(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype(args.dtype)
    if args.q_path or args.k_path or args.v_path:
        if not (args.q_path and args.k_path and args.v_path):
            raise ValueError("--q-path, --k-path, and --v-path must be provided together")
        q, k, v = _load_real_tensors(args, device=device, dtype=dtype)
    else:
        q, k, v = _make_synthetic(args, device=device, dtype=dtype)
    if args.head_index >= 0:
        if args.head_index >= q.shape[2]:
            raise ValueError(f"head index {args.head_index} is outside [0, {q.shape[2]})")
        q = q[:, :, args.head_index : args.head_index + 1, :].contiguous()
        k = k[:, :, args.head_index : args.head_index + 1, :].contiguous()
        v = v[:, :, args.head_index : args.head_index + 1, :].contiguous()
    return q, k, v


def _summarize_inline_stats(raw_stats: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if raw_stats is None:
        return None
    totals = raw_stats.detach().sum(dim=(0, 1)).cpu()
    result = {
        name: int(totals[index].item())
        for name, index in INLINE_PROJECTION_STATS.items()
    }
    middle = result["middle_blocks"]
    total = result["total_blocks"]
    result["projection_skip_fraction"] = (
        result["projection_skipped_blocks"] / middle if middle else 0.0
    )
    result["pv_executed_fraction"] = (
        result["pv_executed_blocks"] / total if total else 0.0
    )
    return result


def _quantile(values, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _summarize_inline_stats_per_head(raw_stats: Optional[torch.Tensor]):
    if raw_stats is None:
        return None
    if raw_stats.dim() != 3 or raw_stats.shape[-1] != 8:
        raise ValueError("inline raw stats must have shape [batch, heads, 8]")
    rows = raw_stats.detach().sum(dim=0).cpu()
    per_head = []
    skip_fracs = []
    pv_fracs = []
    for head_idx, row in enumerate(rows):
        middle = int(row[INLINE_PROJECTION_STATS["middle_blocks"]].item())
        total = int(row[INLINE_PROJECTION_STATS["total_blocks"]].item())
        skipped = int(row[INLINE_PROJECTION_STATS["projection_skipped_blocks"]].item())
        pv = int(row[INLINE_PROJECTION_STATS["pv_executed_blocks"]].item())
        skip_frac = skipped / middle if middle else 0.0
        pv_frac = pv / total if total else 0.0
        skip_fracs.append(skip_frac)
        pv_fracs.append(pv_frac)
        per_head.append(
            {
                "head": head_idx,
                "projection_skipped_blocks": skipped,
                "projection_skip_fraction": skip_frac,
                "pv_executed_blocks": pv,
                "pv_executed_fraction": pv_frac,
                "middle_blocks": middle,
                "total_blocks": total,
            }
        )
    return {
        "per_head": per_head,
        "projection_skip_fraction_mean": sum(skip_fracs) / len(skip_fracs) if skip_fracs else 0.0,
        "projection_skip_fraction_p50": _quantile(skip_fracs, 0.50),
        "projection_skip_fraction_p90": _quantile(skip_fracs, 0.90),
        "pv_executed_fraction_mean": sum(pv_fracs) / len(pv_fracs) if pv_fracs else 0.0,
        "pv_executed_fraction_p50": _quantile(pv_fracs, 0.50),
        "pv_executed_fraction_p90": _quantile(pv_fracs, 0.90),
    }


def _max_mean_error(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
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
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tile-size-q", type=int, default=16)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=0)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="sink_recent_first")
    parser.add_argument("--error-budget", type=float, default=1e-2)
    parser.add_argument("--filter-margin", type=float, default=32.0)
    parser.add_argument("--post-qk-threshold", type=float, default=0.0)
    parser.add_argument("--projection-kind", choices=["random", "hadamard"], default="random")
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--projection-metadata-dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--qproj-mode", choices=["precomputed", "fused"], default="precomputed")
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
            raise ValueError("inline projection benchmark requires query_len == 1")
        dim = q.shape[-1]
        projection = _projection_matrix(
            args.projection_kind,
            dim=dim,
            rank=args.projection_dim,
            seed=args.seed,
            device=q.device,
        )
        metadata_dtype = _projection_metadata_dtype(args.projection_metadata_dtype)
        metadata_build_ms = _time_call(
            lambda: _projection_metadata(
                k,
                block_size=args.block_size,
                projection=projection,
                metadata_dtype=metadata_dtype,
            ),
            device=q.device,
            warmup=max(1, args.warmup // 5),
            iters=max(1, args.iters // 5),
        )
        proj_min, proj_max = _projection_metadata(
            k,
            block_size=args.block_size,
            projection=projection,
            metadata_dtype=metadata_dtype,
        )
        q_projection_ms = _time_call(
            lambda: _project_query(q, projection),
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        q_proj = _project_query(q, projection)

        dense_ms = _time_call(
            lambda: dense_attention_forward(q, k, v, causal=False),
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        gate1_mass_ms = _time_call(
            lambda: gate1_mass_attention_triton_forward(
                q,
                k,
                v,
                causal=False,
                error_budget=args.error_budget,
                block_size=args.block_size,
                tile_size_q=args.tile_size_q,
                post_qk_threshold=args.post_qk_threshold,
                return_raw_stats=False,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            ),
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        inline_ms = _time_call(
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
            ),
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )

        dense_out = dense_attention_forward(q, k, v, causal=False)
        inline_out, raw_stats = gate1_inline_projection_attention_triton_forward(
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
        _sync(q.device)

    stats = _summarize_inline_stats(raw_stats)
    per_head_stats = _summarize_inline_stats_per_head(raw_stats)
    inline_total_ms = inline_ms if args.qproj_mode == "fused" else q_projection_ms + inline_ms
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
        "block_order": args.block_order,
        "error_budget": args.error_budget,
        "filter_margin": args.filter_margin,
        "projection_kind": args.projection_kind,
        "projection_dim": args.projection_dim,
        "projection_metadata_dtype": args.projection_metadata_dtype,
        "qproj_mode": args.qproj_mode,
        "metadata_build_ms": metadata_build_ms,
        "q_projection_ms": q_projection_ms,
        "dense_ms": dense_ms,
        "gate1_mass_ms": gate1_mass_ms,
        "inline_projection_ms": inline_ms,
        "inline_total_ms": inline_total_ms,
        "inline_vs_gate1_speedup": gate1_mass_ms / inline_ms if inline_ms > 0 else None,
        "inline_vs_dense_speedup": dense_ms / inline_ms if inline_ms > 0 else None,
        "inline_total_vs_gate1_speedup": gate1_mass_ms / inline_total_ms if inline_total_ms > 0 else None,
        "inline_total_vs_dense_speedup": dense_ms / inline_total_ms if inline_total_ms > 0 else None,
        "stats": stats,
        "per_head_stats": per_head_stats,
        **_max_mean_error(inline_out, dense_out),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
