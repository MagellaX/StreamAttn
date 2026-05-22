"""Synthetic KV-length scaling for direct-output seed-only vs FlashInfer exact.

This benchmark intentionally avoids model capture.  It isolates the backend
economics question: at what KV length does the seed-only direct-output path
beat FlashInfer TC exact when seed work is fixed?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_true_gqa import _flashinfer_single_decode  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _error, _sync, _time_cuda  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward_out,
)


def _parse_ints(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty integer list: {raw!r}")
    return vals


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


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
    for kv_len in _parse_ints(args.kv_lens):
        q = torch.randn(args.batch, 1, args.q_heads, args.dim, device=device, dtype=dtype)
        k = torch.randn(args.batch, kv_len, args.kv_heads, args.dim, device=device, dtype=dtype)
        v = torch.randn(args.batch, kv_len, args.kv_heads, args.dim, device=device, dtype=dtype)
        out = torch.empty_like(q)

        def flashinfer_exact() -> torch.Tensor:
            return _flashinfer_single_decode(q, k, v, use_tensor_cores=args.flashinfer_tensor_cores)

        def seed_direct() -> torch.Tensor:
            return gate0_seed_only_attention_triton_forward_out(
                q,
                k,
                v,
                out,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )

        flash_ref = flashinfer_exact().clone()
        seed_ref = seed_direct().clone()
        _sync(device)
        flash_ms = _time_cuda(flashinfer_exact, device=device, warmup=args.warmup, iters=args.iters)
        seed_ms = _time_cuda(seed_direct, device=device, warmup=args.warmup, iters=args.iters)
        rows.append(
            {
                "kv_len": kv_len,
                "flashinfer_tc_exact_ms": flash_ms,
                "seed_direct_full_prealloc_ms": seed_ms,
                "speedup_vs_flashinfer": flash_ms / seed_ms,
                "seed_vs_flashinfer_exact": _error(seed_ref, flash_ref),
            }
        )

    winning = [r for r in rows if r["seed_direct_full_prealloc_ms"] < r["flashinfer_tc_exact_ms"]]
    return {
        "schema": "streamattn.gate0.seed_only_synthetic_scaling.v1",
        "shape": {
            "batch": args.batch,
            "query_len": 1,
            "q_heads": args.q_heads,
            "true_kv_heads": args.kv_heads,
            "group_size": args.q_heads // args.kv_heads,
            "dim": args.dim,
            "dtype": args.dtype,
        },
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "seed_blocks": args.sink_blocks + args.recent_blocks + args.middle_seed_blocks,
            "seed_tokens": (args.sink_blocks + args.recent_blocks + args.middle_seed_blocks)
            * args.block_size,
            "block_order": args.block_order,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        },
        "search": {
            "kv_lens": _parse_ints(args.kv_lens),
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
        },
        "rows": rows,
        "break_even_kv_len": min((r["kv_len"] for r in winning), default=None),
        "decision": "seed_only_wins_at_some_kv" if winning else "seed_only_never_wins_in_sweep",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--kv-lens", default="4096,8192,16384,32768,65536,131072")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first", "sink_recent_first"])
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
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
