"""Benchmark head-private split-seed decode against direct seed and FlashInfer.

The split-seed path tests the remaining low-batch kernel question from
``seed_autotune.py``: can extra CTAs from
``CTA_count = B * Hq * Csplit`` beat the chunk/merge overhead while preserving
the same seed-only schedule?  Correctness is checked against the direct
seed-only output, not exact dense, because both kernels compute the same
approximation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_seed_only_batch_scaling import (  # noqa: E402
    _make_flashinfer_wrapper,
    _make_paged_kv_cache,
)
from benchmarks.profile_stream_attn_gate0_wrapper import _error, _sync, _time_cuda  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward_out,
    gate0_seed_only_split_seed_triton_forward_out,
    gate0_seed_only_split_seed_triton_merge,
    gate0_seed_only_split_seed_triton_partial,
    make_gate0_seed_only_split_seed_workspace,
)
from stream_attention.seed_autotune import SeedKernelShape, autotune_seed_kernel_mode  # noqa: E402

try:
    import flashinfer  # noqa: F401

    HAS_FLASHINFER = True
except Exception:  # pragma: no cover - optional dependency
    HAS_FLASHINFER = False


def _parse_ints(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"empty integer list: {raw!r}")
    return values


def _dtype(raw: str) -> torch.dtype:
    if raw == "fp16":
        return torch.float16
    if raw == "bf16":
        return torch.bfloat16
    if raw == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {raw}")


def _time_or_error(fn, *, device: torch.device, warmup: int, iters: int) -> tuple[float | None, str | None]:
    try:
        return _time_cuda(fn, device=device, warmup=warmup, iters=iters), None
    except Exception as exc:  # pragma: no cover - backend dependent
        _sync(device)
        return None, f"{type(exc).__name__}: {exc}"


def _seed_chunks(seed_tokens: int, seed_tile_tokens: int) -> int:
    return math.ceil(float(seed_tokens) / float(seed_tile_tokens))


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not HAS_FLASHINFER:
        raise RuntimeError("FlashInfer is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows: list[dict[str, Any]] = []
    seed_tokens = (args.sink_blocks + args.recent_blocks + args.middle_seed_blocks) * args.block_size
    for batch in _parse_ints(args.batch_sizes):
        q = torch.randn(batch, 1, args.q_heads, args.dim, device=device, dtype=dtype)
        k = torch.randn(batch, args.kv_len, args.kv_heads, args.dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        direct_out = torch.empty_like(q)
        split_out = torch.empty_like(q)

        kv_cache, indptr, indices, last_page_len = _make_paged_kv_cache(
            k,
            v,
            page_size=args.page_size,
        )
        flash_workspace = torch.empty(
            args.workspace_mb * 1024 * 1024,
            device=device,
            dtype=torch.uint8,
        )
        flash_wrapper = _make_flashinfer_wrapper(
            workspace=flash_workspace,
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            dim=args.dim,
            page_size=args.page_size,
            dtype=dtype,
            backend=args.flashinfer_backend,
            use_tensor_cores=args.flashinfer_tensor_cores,
            disable_split_kv=args.disable_split_kv,
        )

        def flashinfer_batch_run() -> torch.Tensor:
            return flash_wrapper.run(q[:, 0].contiguous(), kv_cache).view_as(q)

        def direct_seed_run() -> torch.Tensor:
            return gate0_seed_only_attention_triton_forward_out(
                q,
                k,
                v,
                direct_out,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_warps=args.direct_num_warps,
                num_stages=args.direct_num_stages,
            )

        direct_ref = direct_seed_run().clone()
        flash_ref = flashinfer_batch_run().clone()
        _sync(device)
        flash_ms = _time_cuda(flashinfer_batch_run, device=device, warmup=args.warmup, iters=args.iters)
        direct_ms = _time_cuda(direct_seed_run, device=device, warmup=args.warmup, iters=args.iters)

        autotune = autotune_seed_kernel_mode(
            SeedKernelShape(
                batch=batch,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                kv_len=args.kv_len,
                dim=args.dim,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                dtype_bytes=2 if dtype in {torch.float16, torch.bfloat16} else 4,
            ),
            sm_count=args.sm_count,
            target_waves=args.target_waves,
            seed_tile_tokens=_parse_ints(args.seed_tiles),
            duplication_byte_budget=args.duplication_byte_budget,
        )

        for seed_tile in _parse_ints(args.seed_tiles):
            csplit = _seed_chunks(seed_tokens, seed_tile)
            workspace = make_gate0_seed_only_split_seed_workspace(q, seed_chunks=csplit)
            partial_m = workspace["partial_m"]
            partial_l = workspace["partial_l"]
            partial_num = workspace["partial_num"]

            def chunk_run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return gate0_seed_only_split_seed_triton_partial(
                    q,
                    k,
                    v,
                    partial_m,
                    partial_l,
                    partial_num,
                    seed_tile_tokens=seed_tile,
                    block_size=args.block_size,
                    sink_blocks=args.sink_blocks,
                    recent_blocks=args.recent_blocks,
                    middle_seed_blocks=args.middle_seed_blocks,
                    block_order=args.block_order,
                    num_warps=args.partial_num_warps,
                    num_stages=args.partial_num_stages,
                )

            def merge_run() -> torch.Tensor:
                return gate0_seed_only_split_seed_triton_merge(
                    partial_m,
                    partial_l,
                    partial_num,
                    split_out,
                    num_warps=args.merge_num_warps,
                    num_stages=args.merge_num_stages,
                )

            def split_run() -> torch.Tensor:
                return gate0_seed_only_split_seed_triton_forward_out(
                    q,
                    k,
                    v,
                    split_out,
                    seed_tile_tokens=seed_tile,
                    block_size=args.block_size,
                    sink_blocks=args.sink_blocks,
                    recent_blocks=args.recent_blocks,
                    middle_seed_blocks=args.middle_seed_blocks,
                    block_order=args.block_order,
                    workspace=workspace,
                    partial_num_warps=args.partial_num_warps,
                    partial_num_stages=args.partial_num_stages,
                    merge_num_warps=args.merge_num_warps,
                    merge_num_stages=args.merge_num_stages,
                )

            chunk_ms, chunk_failure = _time_or_error(
                chunk_run,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            if chunk_failure is None:
                chunk_run()
                _sync(device)
            merge_ms, merge_failure = _time_or_error(
                merge_run,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            split_ms, split_failure = _time_or_error(
                split_run,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            row: dict[str, Any] = {
                "batch": batch,
                "seed_tile_tokens": seed_tile,
                "csplit": csplit,
                "cta_count": batch * args.q_heads * csplit,
                "autotune_recommended_mode": autotune.recommended_mode,
                "autotune_recommended_seed_tile_tokens": autotune.recommended_seed_tile_tokens,
                "flashinfer_exact_ms": flash_ms,
                "direct_seed_ms": direct_ms,
                "chunk_ms": chunk_ms,
                "merge_ms": merge_ms,
                "total_split_seed_ms": split_ms,
                "chunk_failure": chunk_failure,
                "merge_failure": merge_failure,
                "split_failure": split_failure,
                "direct_speedup_vs_flashinfer": flash_ms / direct_ms,
                "direct_seed_vs_flashinfer_exact": _error(direct_ref, flash_ref),
            }
            if split_ms is not None:
                split_ref = split_run().clone()
                _sync(device)
                row.update(
                    {
                        "speedup_vs_flashinfer": flash_ms / split_ms,
                        "speedup_vs_direct_seed": direct_ms / split_ms,
                        "max_err_vs_direct_seed": _error(split_ref, direct_ref)["max_abs_error"],
                        "split_vs_direct_seed": _error(split_ref, direct_ref),
                    }
                )
            rows.append(row)

    valid_rows = [row for row in rows if row["total_split_seed_ms"] is not None]
    best_by_batch = {}
    for batch in _parse_ints(args.batch_sizes):
        candidates = [row for row in valid_rows if row["batch"] == batch]
        if candidates:
            best_by_batch[str(batch)] = min(candidates, key=lambda row: row["total_split_seed_ms"])
    split_wins_batches = [
        int(batch)
        for batch, row in best_by_batch.items()
        if row["total_split_seed_ms"] < row["flashinfer_exact_ms"]
    ]
    split_beats_direct_batches = [
        int(batch)
        for batch, row in best_by_batch.items()
        if row["total_split_seed_ms"] < row["direct_seed_ms"]
    ]
    direct_wins_batches = sorted(
        {
            int(row["batch"])
            for row in rows
            if row["direct_seed_ms"] < row["flashinfer_exact_ms"]
        }
    )
    return {
        "schema": "streamattn.gate0.seed_only_split_seed_profile.v1",
        "shape": {
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": args.q_heads // args.kv_heads,
            "kv_len": args.kv_len,
            "dim": args.dim,
            "dtype": args.dtype,
        },
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "seed_tokens": seed_tokens,
            "block_order": args.block_order,
        },
        "search": {
            "batch_sizes": _parse_ints(args.batch_sizes),
            "seed_tiles": _parse_ints(args.seed_tiles),
            "target_waves": args.target_waves,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "best_by_batch": best_by_batch,
        "decision": {
            "split_seed_beats_flashinfer_batches": split_wins_batches,
            "split_seed_beats_direct_batches": split_beats_direct_batches,
            "direct_seed_beats_flashinfer_batches": direct_wins_batches,
            "split_seed_profitable": bool(split_wins_batches),
            "split_seed_lowers_batch4_threshold": 4 in split_wins_batches,
            "batch4_threshold_lowered": 4 in split_wins_batches,
        },
        "rows": sorted(
            rows,
            key=lambda row: (
                row["batch"],
                row["total_split_seed_ms"] is None,
                row["total_split_seed_ms"] or math.inf,
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--seed-tiles", default="32,64,96,128,192")
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first", "sink_recent_first"])
    parser.add_argument("--direct-num-warps", type=int, default=4)
    parser.add_argument("--direct-num-stages", type=int, default=2)
    parser.add_argument("--partial-num-warps", type=int, default=4)
    parser.add_argument("--partial-num-stages", type=int, default=3)
    parser.add_argument("--merge-num-warps", type=int, default=1)
    parser.add_argument("--merge-num-stages", type=int, default=3)
    parser.add_argument("--sm-count", type=int, default=132)
    parser.add_argument("--target-waves", type=float, default=0.40)
    parser.add_argument("--duplication-byte-budget", type=float, default=0.15)
    parser.add_argument("--flashinfer-backend", default="auto")
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--disable-split-kv", action="store_true")
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
