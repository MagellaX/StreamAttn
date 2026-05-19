"""Profile selective-head inline projection routing.

This is a benchmark-level prototype for the next Gate-0 runtime shape. It
groups calibrated sparse heads into one inline projection launch and runs dense
SDPA on the remaining heads. It reports both pre-grouped timing and an indexed
prototype timing that includes per-token ``index_select``/``index_copy`` tax.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

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
    _max_mean_error,
    _summarize_inline_stats,
    _summarize_inline_stats_per_head,
)
from stream_attention.gate1 import dense_attention_forward
from stream_attention.kernels.gate1_inline_projection_fwd_triton import (
    gate1_inline_projection_attention_triton_forward,
)


def _parse_ints(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _complement(indices: Iterable[int], heads: int) -> List[int]:
    selected = set(indices)
    return [head for head in range(heads) if head not in selected]


def _validate_heads(indices: List[int], heads: int, *, name: str) -> None:
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} contains duplicate heads")
    bad = [head for head in indices if head < 0 or head >= heads]
    if bad:
        raise ValueError(f"{name} contains heads outside [0, {heads}): {bad}")


def _index(device: torch.device, heads: List[int]) -> torch.Tensor:
    return torch.tensor(heads, device=device, dtype=torch.long)


def _select_heads(tensor: torch.Tensor, heads: List[int]) -> torch.Tensor:
    if not heads:
        shape = list(tensor.shape)
        shape[2] = 0
        return torch.empty(shape, device=tensor.device, dtype=tensor.dtype)
    return tensor.index_select(2, _index(tensor.device, heads)).contiguous()


def _select_metadata(metadata: torch.Tensor, heads: List[int]) -> torch.Tensor:
    if not heads:
        shape = list(metadata.shape)
        shape[1] = 0
        return torch.empty(shape, device=metadata.device, dtype=metadata.dtype)
    return metadata.index_select(1, _index(metadata.device, heads)).contiguous()


def _make_synthetic(args, *, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, 1, args.heads, args.dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.kv_len, args.heads, args.dim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.kv_len, args.heads, args.dim, device=device, dtype=dtype)
    return q, k, v


def _load_or_make_tensors(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype(args.dtype)
    if args.q_path or args.k_path or args.v_path:
        if not (args.q_path and args.k_path and args.v_path):
            raise ValueError("--q-path, --k-path, and --v-path must be provided together")
        return _load_real_tensors(args, device=device, dtype=dtype)
    return _make_synthetic(args, device=device, dtype=dtype)


def _empty_output_like(q: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(q)


def _scatter_group(out: torch.Tensor, heads: List[int], group_out: torch.Tensor) -> None:
    if heads:
        out.index_copy_(2, _index(out.device, heads), group_out)


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
    parser.add_argument("--sparse-heads", required=True)
    parser.add_argument("--dense-heads", default="")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--tile-size-q", type=int, default=16)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
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
            raise ValueError("grouped inline projection benchmark requires query_len == 1")
        batch, _query_len, heads, dim = q.shape
        sparse_heads = _parse_ints(args.sparse_heads)
        dense_heads = _parse_ints(args.dense_heads) if args.dense_heads else _complement(sparse_heads, heads)
        _validate_heads(sparse_heads, heads, name="sparse_heads")
        _validate_heads(dense_heads, heads, name="dense_heads")
        overlap = sorted(set(sparse_heads) & set(dense_heads))
        if overlap:
            raise ValueError(f"sparse and dense head groups overlap: {overlap}")
        missing = _complement([*sparse_heads, *dense_heads], heads)
        if missing:
            raise ValueError(f"head groups do not cover all heads: {missing}")

        projection = _projection_matrix(
            args.projection_kind,
            dim=dim,
            rank=args.projection_dim,
            seed=args.seed,
            device=q.device,
        )
        metadata_dtype = _projection_metadata_dtype(args.projection_metadata_dtype)
        proj_min_full, proj_max_full = _projection_metadata(
            k,
            block_size=args.block_size,
            projection=projection,
            metadata_dtype=metadata_dtype,
        )

        q_sparse = _select_heads(q, sparse_heads)
        k_sparse = _select_heads(k, sparse_heads)
        v_sparse = _select_heads(v, sparse_heads)
        proj_min_sparse = _select_metadata(proj_min_full, sparse_heads)
        proj_max_sparse = _select_metadata(proj_max_full, sparse_heads)
        q_dense = _select_heads(q, dense_heads)
        k_dense = _select_heads(k, dense_heads)
        v_dense = _select_heads(v, dense_heads)
        q_proj_sparse = (
            _project_query(q_sparse, projection)
            if args.qproj_mode == "precomputed" and sparse_heads
            else None
        )

        def inline_sparse():
            if not sparse_heads:
                return torch.empty_like(q_sparse), None
            return gate1_inline_projection_attention_triton_forward(
                q_sparse,
                k_sparse,
                v_sparse,
                q_proj_sparse,
                proj_min_sparse,
                proj_max_sparse,
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
            )

        def dense_unsafe():
            if not dense_heads:
                return torch.empty_like(q_dense)
            return dense_attention_forward(q_dense, k_dense, v_dense, causal=False)

        dense_all_ms = _time_call(
            lambda: dense_attention_forward(q, k, v, causal=False),
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        inline_sparse_ms = _time_call(
            lambda: inline_sparse()[0],
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        ) if sparse_heads else 0.0
        dense_unsafe_ms = _time_call(
            dense_unsafe,
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        ) if dense_heads else 0.0

        def pregrouped_serial():
            out = _empty_output_like(q)
            sparse_out, _ = inline_sparse()
            dense_out = dense_unsafe()
            _scatter_group(out, sparse_heads, sparse_out)
            _scatter_group(out, dense_heads, dense_out)
            return out

        def indexed_serial():
            out = _empty_output_like(q)
            qs = _select_heads(q, sparse_heads)
            ks = _select_heads(k, sparse_heads)
            vs = _select_heads(v, sparse_heads)
            pmins = _select_metadata(proj_min_full, sparse_heads)
            pmaxs = _select_metadata(proj_max_full, sparse_heads)
            qds = _select_heads(q, dense_heads)
            kds = _select_heads(k, dense_heads)
            vds = _select_heads(v, dense_heads)
            if sparse_heads:
                qps = _project_query(qs, projection) if args.qproj_mode == "precomputed" else None
                sparse_out, _ = gate1_inline_projection_attention_triton_forward(
                    qs,
                    ks,
                    vs,
                    qps,
                    pmins,
                    pmaxs,
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
                )
                _scatter_group(out, sparse_heads, sparse_out)
            if dense_heads:
                dense_out = dense_attention_forward(qds, kds, vds, causal=False)
                _scatter_group(out, dense_heads, dense_out)
            return out

        pregrouped_serial_ms = _time_call(
            pregrouped_serial,
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        indexed_serial_ms = _time_call(
            indexed_serial,
            device=q.device,
            warmup=args.warmup,
            iters=args.iters,
        )

        dense_out = dense_attention_forward(q, k, v, causal=False)
        selective_out = pregrouped_serial()
        sparse_out_for_stats, raw_stats = (
            gate1_inline_projection_attention_triton_forward(
                q_sparse,
                k_sparse,
                v_sparse,
                q_proj_sparse,
                proj_min_sparse,
                proj_max_sparse,
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
            if sparse_heads
            else (torch.empty_like(q_sparse), None)
        )
        _ = sparse_out_for_stats
        _sync(q.device)

    group_max_lower_bound_ms = max(inline_sparse_ms, dense_unsafe_ms)
    payload = {
        "device": torch.cuda.get_device_name(q.device),
        "torch": torch.__version__,
        "shape": {
            "batch": batch,
            "query_len": q.shape[1],
            "kv_len": k.shape[1],
            "heads": heads,
            "dim": dim,
            "dtype": args.dtype,
        },
        "sparse_heads": sparse_heads,
        "dense_heads": dense_heads,
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
        "dense_all_ms": dense_all_ms,
        "inline_sparse_group_ms": inline_sparse_ms,
        "dense_unsafe_group_ms": dense_unsafe_ms,
        "group_max_lower_bound_ms": group_max_lower_bound_ms,
        "pregrouped_serial_ms": pregrouped_serial_ms,
        "indexed_serial_ms": indexed_serial_ms,
        "pregrouped_serial_vs_dense_speedup": dense_all_ms / pregrouped_serial_ms if pregrouped_serial_ms else None,
        "indexed_serial_vs_dense_speedup": dense_all_ms / indexed_serial_ms if indexed_serial_ms else None,
        "group_max_vs_dense_speedup": dense_all_ms / group_max_lower_bound_ms if group_max_lower_bound_ms else None,
        "stats": _summarize_inline_stats(raw_stats),
        "per_head_stats": _summarize_inline_stats_per_head(raw_stats) if raw_stats is not None else None,
        **_max_mean_error(selective_out, dense_out),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
