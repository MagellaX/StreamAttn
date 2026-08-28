"""Profile KV-group-owned exact GQA prefill candidates."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from stream_attention.core.fused_online_attention import FusedOnlineAttention
from stream_attention.kernels.grouped_gqa_prefill_triton import (
    effective_kv_reuse,
    grouped_gqa_prefill,
)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch
    SDPBackend = None
    sdpa_kernel = None


def _elapsed_ms(fn: Callable[[], None], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _flash_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    context = nullcontext()
    if sdpa_kernel is not None and SDPBackend is not None:
        context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    with context:
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=0.0,
            enable_gqa=True,
        ).transpose(1, 2)


def _reference_tile_m(capability: tuple[int, int], head_dim: int) -> int:
    sm = capability[0] * 10 + capability[1]
    if head_dim >= 128:
        return 32 if sm >= 100 else 64
    return 64 if sm >= 100 else 128


def _parse_config(config: str) -> tuple[int, int]:
    heads, tile_m = config.lower().split("x", maxsplit=1)
    return int(heads), int(tile_m)


def _profile_candidate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads_per_program: int,
    tile_m: int,
    tile_n: int,
    num_warps: int,
    num_stages: int,
    reference_tile_m: int,
    current_ms: float,
    flash_ms: float,
    flash_output: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    base = {
        "heads_per_program": heads_per_program,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "grouped_rows": heads_per_program * tile_m,
        "effective_kv_reuse": effective_kv_reuse(
            heads_per_program=heads_per_program,
            tile_m=tile_m,
            reference_tile_m=reference_tile_m,
        ),
    }
    try:
        output = grouped_gqa_prefill(
            q,
            k,
            v,
            heads_per_program=heads_per_program,
            tile_m=tile_m,
            tile_n=tile_n,
            num_warps=None if num_warps == 0 else num_warps,
            num_stages=num_stages,
        )
        torch.cuda.synchronize()
        candidate_ms = _elapsed_ms(
            lambda: grouped_gqa_prefill(
                q,
                k,
                v,
                heads_per_program=heads_per_program,
                tile_m=tile_m,
                tile_n=tile_n,
                num_warps=None if num_warps == 0 else num_warps,
                num_stages=num_stages,
            ),
            warmup=warmup,
            iterations=iterations,
        )
        return {
            **base,
            "status": "ok",
            "max_abs_error": float((output - flash_output).abs().max().item()),
            "candidate_ms": candidate_ms,
            "speedup_vs_current": current_ms / candidate_ms,
            "speedup_vs_flash": flash_ms / candidate_ms,
        }
    except Exception as exc:  # benchmark records resource/compiler boundaries
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--configs", nargs="+", default=["2x32", "4x16", "4x32"])
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--num-warps", nargs="+", type=int, default=[0])
    parser.add_argument("--num-stages", nargs="+", type=int, default=[2])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.q_heads % args.kv_heads:
        raise ValueError("q_heads must be a multiple of kv_heads")

    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    dtype = torch.bfloat16
    configs = [_parse_config(config) for config in args.configs]
    rows: list[dict[str, object]] = []
    for batch in args.batches:
        for seq_len in args.seq_lens:
            q = torch.randn(
                batch,
                seq_len,
                args.q_heads,
                args.head_dim,
                device="cuda",
                dtype=dtype,
            )
            k = torch.randn(
                batch,
                seq_len,
                args.kv_heads,
                args.head_dim,
                device="cuda",
                dtype=dtype,
            )
            v = torch.randn_like(k)
            current = FusedOnlineAttention(
                num_heads=args.q_heads,
                num_kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                device=q.device,
                dtype=dtype,
            ).eval()
            with torch.no_grad():
                flash_output = _flash_sdpa(q, k, v)
                current_output = current(q, k, v, causal=True)
                current_ms = _elapsed_ms(
                    lambda: current(q, k, v, causal=True),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                flash_ms = _elapsed_ms(
                    lambda: _flash_sdpa(q, k, v),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                reference_tile_m = _reference_tile_m(capability, args.head_dim)
                for heads_per_program, tile_m in configs:
                    for num_warps in args.num_warps:
                        for num_stages in args.num_stages:
                            result = _profile_candidate(
                                q,
                                k,
                                v,
                                heads_per_program=heads_per_program,
                                tile_m=tile_m,
                                tile_n=args.tile_n,
                                num_warps=num_warps,
                                num_stages=num_stages,
                                reference_tile_m=reference_tile_m,
                                current_ms=current_ms,
                                flash_ms=flash_ms,
                                flash_output=flash_output,
                                warmup=args.warmup,
                                iterations=args.iterations,
                            )
                            row = {
                                "batch": batch,
                                "seq_len": seq_len,
                                "q_heads": args.q_heads,
                                "kv_heads": args.kv_heads,
                                "group_size": args.q_heads // args.kv_heads,
                                "head_dim": args.head_dim,
                                "current_ms": current_ms,
                                "flash_ms": flash_ms,
                                "current_max_abs_error": float(
                                    (current_output - flash_output).abs().max().item()
                                ),
                                **result,
                            }
                            rows.append(row)
                            print(json.dumps(row), flush=True)

    result = {
        "device": device_name,
        "capability": list(capability),
        "torch": torch.__version__,
        "rows": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
