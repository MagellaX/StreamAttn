"""Probe physically packed head-private seed K/V cache for seed-only decode.

This benchmark tests the StreamAttn-specific dataflow hypothesis:

    if G * S / N is tiny, duplicating seed K/V per Q head may be cheaper than
    reading seed tokens through a shared true-GQA cache because the packed
    layout gives simpler, coalesced per-head seed reads.

Correctness is checked against the current full-cache seed-only path because
both paths compute the same seed-only schedule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_stream_attn_gate0_wrapper import _error, _sync, _time_cuda  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_pack_seed_cache_bhsd,
    gate0_refresh_packed_seed_cache_recent_bhsd,
    gate0_seed_only_attention_packed_bhsd_triton_forward_out,
    gate0_seed_only_attention_triton_forward_out,
    make_gate0_seed_only_packed_workspace,
)


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


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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
        packed_out = torch.empty_like(q)
        workspace = make_gate0_seed_only_packed_workspace(q, seed_tokens=seed_tokens)
        k_seed = workspace["k_seed"]
        v_seed = workspace["v_seed"]

        def direct_run() -> torch.Tensor:
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

        def pack_run() -> tuple[torch.Tensor, torch.Tensor]:
            return gate0_pack_seed_cache_bhsd(
                k,
                v,
                k_seed,
                v_seed,
                q_heads=args.q_heads,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_warps=args.pack_num_warps,
                num_stages=args.pack_num_stages,
            )

        def recent_refresh_run() -> tuple[torch.Tensor, torch.Tensor]:
            return gate0_refresh_packed_seed_cache_recent_bhsd(
                k,
                v,
                k_seed,
                v_seed,
                q_heads=args.q_heads,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                num_warps=args.refresh_num_warps,
                num_stages=args.refresh_num_stages,
            )

        def packed_run() -> torch.Tensor:
            return gate0_seed_only_attention_packed_bhsd_triton_forward_out(
                q,
                k_seed,
                v_seed,
                packed_out,
                block_size=args.block_size,
                num_warps=args.packed_num_warps,
                num_stages=args.packed_num_stages,
            )

        def pack_plus_packed_run() -> torch.Tensor:
            pack_run()
            return packed_run()

        def refresh_plus_packed_run() -> torch.Tensor:
            recent_refresh_run()
            return packed_run()

        direct_ref = direct_run().clone()
        pack_run()
        packed_ref = packed_run().clone()
        recent_refresh_run()
        refreshed_ref = packed_run().clone()
        _sync(device)

        direct_ms = _time_cuda(direct_run, device=device, warmup=args.warmup, iters=args.iters)
        pack_ms = _time_cuda(pack_run, device=device, warmup=args.warmup, iters=args.iters)
        refresh_ms = _time_cuda(recent_refresh_run, device=device, warmup=args.warmup, iters=args.iters)
        packed_ms = _time_cuda(packed_run, device=device, warmup=args.warmup, iters=args.iters)
        total_ms = _time_cuda(pack_plus_packed_run, device=device, warmup=args.warmup, iters=args.iters)
        refresh_total_ms = _time_cuda(
            refresh_plus_packed_run,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )

        group_size = args.q_heads // args.kv_heads
        seed_ratio = seed_tokens / float(args.kv_len)
        duplicate_ratio = group_size * seed_ratio
        row = {
            "batch": batch,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": group_size,
            "kv_len": args.kv_len,
            "dim": args.dim,
            "seed_tokens": seed_tokens,
            "seed_token_ratio": seed_ratio,
            "head_private_kv_byte_ratio": duplicate_ratio,
            "direct_full_cache_seed_ms": direct_ms,
            "pack_seed_cache_ms": pack_ms,
            "refresh_recent_seed_cache_ms": refresh_ms,
            "packed_seed_ms": packed_ms,
            "pack_plus_packed_seed_ms": total_ms,
            "refresh_plus_packed_seed_ms": refresh_total_ms,
            "packed_speedup_vs_direct_kernel_only": direct_ms / packed_ms,
            "packed_total_speedup_vs_direct": direct_ms / total_ms,
            "refresh_total_speedup_vs_direct": direct_ms / refresh_total_ms,
            "refresh_speedup_vs_full_pack": pack_ms / refresh_ms,
            "packed_vs_direct_seed": _error(packed_ref, direct_ref),
            "refreshed_packed_vs_direct_seed": _error(refreshed_ref, direct_ref),
        }
        rows.append(row)

    best_kernel = max(rows, key=lambda row: row["packed_speedup_vs_direct_kernel_only"])
    best_total = max(rows, key=lambda row: row["packed_total_speedup_vs_direct"])
    best_refresh_total = max(rows, key=lambda row: row["refresh_total_speedup_vs_direct"])
    return {
        "schema": "streamattn.seed_only_packed_seed_cache.v1",
        "shape": {
            "dtype": args.dtype,
            "batch_sizes": _parse_ints(args.batch_sizes),
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "dim": args.dim,
            "kv_len": args.kv_len,
            "seed_tokens": seed_tokens,
        },
        "kernel_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
            "direct_num_warps": args.direct_num_warps,
            "direct_num_stages": args.direct_num_stages,
            "pack_num_warps": args.pack_num_warps,
            "pack_num_stages": args.pack_num_stages,
            "refresh_num_warps": args.refresh_num_warps,
            "refresh_num_stages": args.refresh_num_stages,
            "packed_num_warps": args.packed_num_warps,
            "packed_num_stages": args.packed_num_stages,
        },
        "best_kernel_only": best_kernel,
        "best_total": best_total,
        "best_refresh_total": best_refresh_total,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--batch-sizes", default="4,8,16")
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--direct-num-warps", type=int, default=4)
    parser.add_argument("--direct-num-stages", type=int, default=2)
    parser.add_argument("--pack-num-warps", type=int, default=4)
    parser.add_argument("--pack-num-stages", type=int, default=3)
    parser.add_argument("--refresh-num-warps", type=int, default=4)
    parser.add_argument("--refresh-num-stages", type=int, default=3)
    parser.add_argument("--packed-num-warps", type=int, default=4)
    parser.add_argument("--packed-num-stages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
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
