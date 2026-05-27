"""Check fused RoPE+append+seed-only decode parity against the safe path.

This isolates the fused kernel from full-model effects.  The reference path
applies Qwen-style RoPE in PyTorch, appends K/V to a BHND cache, and calls the
existing cache-position seed-only kernel.  The candidate path does the same
work inside the fused RoPE+append+seed kernel.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch

from stream_attention.kernels.gate0_seed_only_triton import (
    gate0_seed_only_attention_triton_forward_out_cachepos_bhnd,
    gate0_seed_only_rope_append_triton_forward_out_cachepos_bhnd,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_qwen_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos.unsqueeze(1)) + (_rotate_half(x) * sin.unsqueeze(1))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    if args.position >= args.max_seq:
        raise ValueError("position must be smaller than max_seq")

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    scale = 1.0 / math.sqrt(args.head_dim)
    q_raw = torch.randn(
        args.batch,
        args.q_heads,
        1,
        args.head_dim,
        device=device,
        dtype=dtype,
    ) * scale
    k_raw = torch.randn(
        args.batch,
        args.kv_heads,
        1,
        args.head_dim,
        device=device,
        dtype=dtype,
    ) * scale
    v_raw = torch.randn(
        args.batch,
        args.kv_heads,
        1,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    k_cache = torch.randn(
        args.batch,
        args.kv_heads,
        args.max_seq,
        args.head_dim,
        device=device,
        dtype=dtype,
    ) * scale
    v_cache = torch.randn_like(k_cache)
    angles = torch.randn(
        1 if args.shared_cos else args.batch,
        1,
        args.head_dim // 2,
        device=device,
        dtype=torch.float32,
    )
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    cos_ref = cos.expand(args.batch, -1, -1) if args.shared_cos else cos
    sin_ref = sin.expand(args.batch, -1, -1) if args.shared_cos else sin

    cache_position = torch.tensor([args.position], device=device, dtype=torch.int64)

    q_ref = _apply_qwen_rope(q_raw, cos_ref, sin_ref)
    k_ref_current = _apply_qwen_rope(k_raw, cos_ref, sin_ref)
    k_ref = k_cache.clone()
    v_ref = v_cache.clone()
    k_ref[:, :, args.position : args.position + 1, :].copy_(k_ref_current)
    v_ref[:, :, args.position : args.position + 1, :].copy_(v_raw)
    out_ref = torch.empty((args.batch, 1, args.q_heads, args.head_dim), device=device, dtype=dtype)
    gate0_seed_only_attention_triton_forward_out_cachepos_bhnd(
        q_ref.transpose(1, 2).contiguous(),
        k_ref,
        v_ref,
        out_ref,
        cache_position,
        block_size=args.block_size,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )

    k_fused = k_cache.clone()
    v_fused = v_cache.clone()
    out_fused = torch.empty_like(out_ref)
    gate0_seed_only_rope_append_triton_forward_out_cachepos_bhnd(
        q_raw,
        k_raw,
        v_raw,
        cos,
        sin,
        k_fused,
        v_fused,
        out_fused,
        cache_position,
        block_size=args.block_size,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    out_diff = (out_fused.float() - out_ref.float()).abs()
    k_diff = (k_fused[:, :, args.position, :].float() - k_ref[:, :, args.position, :].float()).abs()
    v_diff = (v_fused[:, :, args.position, :].float() - v_ref[:, :, args.position, :].float()).abs()
    return {
        "schema": "streamattn.seed_only_fused_rope_parity.v1",
        "shape": {
            "batch": args.batch,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "max_seq": args.max_seq,
            "position": args.position,
            "dtype": args.dtype,
            "shared_cos": bool(args.shared_cos),
        },
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "errors": {
            "out_max_abs": float(out_diff.max().item()),
            "out_mean_abs": float(out_diff.mean().item()),
            "k_store_max_abs": float(k_diff.max().item()),
            "v_store_max_abs": float(v_diff.max().item()),
        },
        "passed": bool(
            out_diff.max().item() <= args.atol
            and k_diff.max().item() <= args.atol
            and v_diff.max().item() <= args.atol
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq", type=int, default=32777)
    parser.add_argument("--position", type=int, default=32768)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=2.0e-3)
    parser.add_argument("--shared-cos", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
