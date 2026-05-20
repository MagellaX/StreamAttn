"""Profile aggressive sparse union plus exact head correction.

This benchmark tests the next Gate-0 frontier:

1. run a large calibrated split-K projection group to amortize sparse overhead;
2. trust only a subset of that sparse output;
3. compute exact attention for every other head and overwrite/assemble output.

It is a science benchmark, not a production runtime.  The key metric is whether
the oracle overlap bound, ``max(sparse_union_ms, dense_exact_ms)``, can beat
``dense_all_ms`` after correction restores output quality.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_candidate_filters import (
    _projection_matrix,
    _projection_metadata,
    _projection_metadata_dtype,
)
from benchmarks.profile_gate0_summary_bounds import _dtype, _load_real_tensors, _sync, _time_call
from benchmarks.profile_gate1_inline_projection import _load_or_make_tensors, _max_mean_error
from benchmarks.profile_gate1_inline_projection_splitk import (
    _per_head_error,
    _summarize_splitk_stats,
    _summarize_splitk_stats_per_head,
)
from stream_attention.gate1 import dense_attention_forward
from stream_attention.kernels.gate1_inline_projection_splitk_triton import (
    gate1_inline_projection_splitk_attention_triton_forward,
    make_splitk_workspace,
)


def _parse_heads(raw: str, *, total_heads: int) -> List[int]:
    if not raw:
        return []
    heads = sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))
    for head in heads:
        if head < 0 or head >= total_heads:
            raise ValueError(f"head index {head} is outside [0, {total_heads})")
    return heads


def _head_group(heads: Iterable[int]) -> str:
    return ",".join(str(head) for head in sorted(set(int(item) for item in heads)))


def _complement(heads: Sequence[int], *, total_heads: int) -> List[int]:
    selected = set(int(head) for head in heads)
    return [head for head in range(total_heads) if head not in selected]


def _select_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: Sequence[int]):
    if not heads:
        raise ValueError("at least one head is required")
    index = torch.tensor(list(heads), device=q.device, dtype=torch.long)
    return (
        q.index_select(2, index).contiguous(),
        k.index_select(2, index).contiguous(),
        v.index_select(2, index).contiguous(),
    )


def _assemble_corrected_output(
    *,
    total_heads: int,
    dense_exact_out: torch.Tensor,
    exact_heads: Sequence[int],
    sparse_union_out: torch.Tensor,
    aggressive_heads: Sequence[int],
    trusted_heads: Sequence[int],
) -> torch.Tensor:
    if sparse_union_out.dim() != 4 or dense_exact_out.dim() != 4:
        raise ValueError("attention outputs must have shape [batch, query_len, heads, dim]")
    if len(set(trusted_heads) - set(aggressive_heads)) > 0:
        raise ValueError("trusted sparse heads must be a subset of aggressive heads")
    batch, query_len, _exact_count, dim = dense_exact_out.shape
    out = torch.empty(
        batch,
        query_len,
        total_heads,
        dim,
        device=dense_exact_out.device,
        dtype=dense_exact_out.dtype,
    )
    written: set[int] = set()
    exact_local = {int(head): idx for idx, head in enumerate(exact_heads)}
    sparse_local = {int(head): idx for idx, head in enumerate(aggressive_heads)}
    for head in exact_heads:
        out[:, :, int(head), :] = dense_exact_out[:, :, exact_local[int(head)], :]
        written.add(int(head))
    for head in trusted_heads:
        out[:, :, int(head), :] = sparse_union_out[:, :, sparse_local[int(head)], :]
        written.add(int(head))
    missing = sorted(set(range(total_heads)) - written)
    if missing:
        raise ValueError(f"hybrid output did not write heads: {missing}")
    return out


def _time_parallel(
    sparse_fn: Callable[[], torch.Tensor],
    exact_fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    if device.type != "cuda":
        return 0.0
    stream_sparse = torch.cuda.Stream(device=device)
    stream_exact = torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)
    for _ in range(max(0, warmup)):
        gate = torch.cuda.Event()
        stream_sparse.wait_event(gate)
        stream_exact.wait_event(gate)
        gate.record(current)
        with torch.cuda.stream(stream_sparse):
            sparse_fn()
        with torch.cuda.stream(stream_exact):
            exact_fn()
        current.wait_stream(stream_sparse)
        current.wait_stream(stream_exact)
    _sync(device)

    timings = []
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream_sparse.wait_event(start)
        stream_exact.wait_event(start)
        start.record(current)
        with torch.cuda.stream(stream_sparse):
            sparse_fn()
        with torch.cuda.stream(stream_exact):
            exact_fn()
        current.wait_stream(stream_sparse)
        current.wait_stream(stream_exact)
        end.record(current)
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return float(sum(timings) / len(timings))


def _splitk_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    projection: torch.Tensor,
    proj_min: torch.Tensor,
    proj_max: torch.Tensor,
    args: argparse.Namespace,
    workspace: Dict[str, torch.Tensor] | None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    return gate1_inline_projection_splitk_attention_triton_forward(
        q,
        k,
        v,
        None,
        proj_min,
        proj_max,
        projection=projection,
        compute_qproj=True,
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
        workspace=workspace,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def profile_hybrid(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    dtype = _dtype(args.dtype)
    with torch.no_grad():
        if args.q_path or args.k_path or args.v_path:
            if not (args.q_path and args.k_path and args.v_path):
                raise ValueError("--q-path, --k-path, and --v-path must be provided together")
            q, k, v = _load_real_tensors(args, device=device, dtype=dtype)
            if v is None:
                raise ValueError("--v-path is required")
        else:
            q, k, v = _load_or_make_tensors(args)

        if q.shape[1] != 1:
            raise ValueError("hybrid correction benchmark requires query_len == 1")
        total_heads = int(q.shape[2])
        aggressive_heads = _parse_heads(args.aggressive_heads, total_heads=total_heads)
        trusted_heads = _parse_heads(args.trusted_heads, total_heads=total_heads)
        if not aggressive_heads:
            raise ValueError("--aggressive-heads is required")
        if not trusted_heads:
            raise ValueError("--trusted-heads is required")
        if not set(trusted_heads).issubset(set(aggressive_heads)):
            raise ValueError("--trusted-heads must be a subset of --aggressive-heads")
        repair_heads = sorted(set(aggressive_heads) - set(trusted_heads))
        outside_union_heads = _complement(aggressive_heads, total_heads=total_heads)
        exact_heads = sorted(set(range(total_heads)) - set(trusted_heads))

        q_union, k_union, v_union = _select_heads(q, k, v, aggressive_heads)
        q_exact, k_exact, v_exact = _select_heads(q, k, v, exact_heads)
        projection = _projection_matrix(
            args.projection_kind,
            dim=q_union.shape[-1],
            rank=args.projection_dim,
            seed=args.seed,
            device=q_union.device,
        )
        proj_min, proj_max = _projection_metadata(
            k_union,
            block_size=args.block_size,
            projection=projection,
            metadata_dtype=_projection_metadata_dtype(args.projection_metadata_dtype),
        )
        workspace = (
            make_splitk_workspace(
                q_union,
                rank=args.projection_dim,
                num_chunks=args.num_chunks,
                seed_strategy=args.seed_strategy,
            )
            if args.splitk_workspace == "reuse"
            else None
        )

        def sparse_fn():
            return _splitk_projection(
                q_union,
                k_union,
                v_union,
                projection=projection,
                proj_min=proj_min,
                proj_max=proj_max,
                args=args,
                workspace=workspace,
            )[0]

        def exact_fn():
            return dense_attention_forward(q_exact, k_exact, v_exact, causal=False)

        dense_all_ms = _time_call(
            lambda: dense_attention_forward(q, k, v, causal=False),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        dense_exact_ms = _time_call(exact_fn, device=device, warmup=args.warmup, iters=args.iters)
        sparse_union_ms = _time_call(sparse_fn, device=device, warmup=args.warmup, iters=args.iters)
        parallel_stream_ms = (
            _time_parallel(sparse_fn, exact_fn, device=device, warmup=args.warmup, iters=args.iters)
            if args.measure_parallel_streams
            else None
        )

        dense_all_out = dense_attention_forward(q, k, v, causal=False)
        dense_exact_out = exact_fn()
        sparse_union_out, sparse_raw = _splitk_projection(
            q_union,
            k_union,
            v_union,
            projection=projection,
            proj_min=proj_min,
            proj_max=proj_max,
            args=args,
            workspace=workspace,
        )
        corrected_out = _assemble_corrected_output(
            total_heads=total_heads,
            dense_exact_out=dense_exact_out,
            exact_heads=exact_heads,
            sparse_union_out=sparse_union_out,
            aggressive_heads=aggressive_heads,
            trusted_heads=trusted_heads,
        )
        _sync(device)

    serial_corrected_ms = sparse_union_ms + dense_exact_ms
    oracle_max_ms = max(sparse_union_ms, dense_exact_ms)
    payload = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "kv_len": int(k.shape[1]),
            "heads": total_heads,
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "policy": {
            "aggressive_sparse_heads": aggressive_heads,
            "trusted_sparse_heads": trusted_heads,
            "repair_heads": repair_heads,
            "outside_union_exact_heads": outside_union_heads,
            "exact_heads": exact_heads,
            "aggressive_head_group": _head_group(aggressive_heads),
            "trusted_head_group": _head_group(trusted_heads),
            "exact_head_group": _head_group(exact_heads),
        },
        "runtime": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "chunk_anchor_blocks": args.chunk_anchor_blocks,
            "block_order": args.block_order,
            "num_chunks": args.num_chunks,
            "seed_strategy": args.seed_strategy,
            "filter_margin": args.filter_margin,
            "error_budget": args.error_budget,
            "projection_kind": args.projection_kind,
            "projection_dim": args.projection_dim,
            "projection_seed": args.seed,
            "projection_metadata_dtype": args.projection_metadata_dtype,
            "splitk_workspace": args.splitk_workspace,
        },
        "timing": {
            "dense_all_ms": dense_all_ms,
            "sparse_union_ms": sparse_union_ms,
            "dense_exact_ms": dense_exact_ms,
            "serial_corrected_ms": serial_corrected_ms,
            "oracle_max_ms": oracle_max_ms,
            "parallel_stream_ms": parallel_stream_ms,
            "sparse_union_speedup_vs_dense_all": dense_all_ms / sparse_union_ms if sparse_union_ms > 0 else None,
            "serial_corrected_speedup_vs_dense_all": dense_all_ms / serial_corrected_ms if serial_corrected_ms > 0 else None,
            "oracle_max_speedup_vs_dense_all": dense_all_ms / oracle_max_ms if oracle_max_ms > 0 else None,
            "parallel_stream_speedup_vs_dense_all": (
                dense_all_ms / parallel_stream_ms
                if parallel_stream_ms is not None and parallel_stream_ms > 0
                else None
            ),
        },
        "quality": {
            "sparse_union_error_vs_dense_all_selected": _max_mean_error(
                sparse_union_out,
                dense_all_out.index_select(
                    2, torch.tensor(aggressive_heads, device=dense_all_out.device, dtype=torch.long)
                ).contiguous(),
            ),
            "corrected_error_vs_dense_all": _max_mean_error(corrected_out, dense_all_out),
            "corrected_error_vs_dense_all_per_head": _per_head_error(corrected_out, dense_all_out),
            "sparse_union_stats": _summarize_splitk_stats(sparse_raw),
            "sparse_union_per_head_stats": _summarize_splitk_stats_per_head(sparse_raw),
        },
    }
    return payload


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
    parser.add_argument("--aggressive-heads", required=True)
    parser.add_argument("--trusted-heads", required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--tile-size-q", type=int, default=16)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--seed-strategy", choices=["separate", "recompute_seed"], default="recompute_seed")
    parser.add_argument("--chunk-anchor-blocks", type=int, default=0)
    parser.add_argument("--error-budget", type=float, default=1e-2)
    parser.add_argument("--filter-margin", type=float, default=64.0)
    parser.add_argument("--projection-kind", choices=["random", "hadamard"], default="random")
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--projection-metadata-dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--splitk-workspace", choices=["none", "reuse"], default="reuse")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--measure-parallel-streams", action="store_true")
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()

    payload = profile_hybrid(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
