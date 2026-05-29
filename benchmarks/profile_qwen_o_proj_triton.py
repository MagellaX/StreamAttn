"""Microbenchmark decode-shaped Qwen output projection kernels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_stream_attn_gate0_wrapper import _dtype  # noqa: E402
from stream_attention.kernels.qwen_o_proj_triton import qwen_o_proj_triton_forward  # noqa: E402


def _parse_variants(text: str) -> list[tuple[int, int, int]]:
    variants = []
    for spec in text.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        parts = [int(part) for part in spec.replace("x", ",").split(",")]
        if len(parts) != 3:
            raise ValueError(f"invalid variant {spec!r}; expected BLOCK_M,BLOCK_N,BLOCK_K")
        if any(part <= 0 for part in parts):
            raise ValueError("variant block sizes must be positive")
        variants.append((parts[0], parts[1], parts[2]))
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def _bench(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(1, iters))


def profile(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    x = torch.randn(args.batch, 1, args.hidden, device=device, dtype=dtype)
    weight = torch.randn(args.hidden, args.hidden, device=device, dtype=dtype) / (args.hidden**0.5)
    bias = torch.randn(args.hidden, device=device, dtype=dtype) / (args.hidden**0.5) if args.bias else None

    expected = F.linear(x, weight, bias)
    baseline_ms = _bench(lambda: F.linear(x, weight, bias), warmup=args.warmup, iters=args.iters)
    rows = []
    for block_m, block_n, block_k in _parse_variants(args.variants):
        actual = qwen_o_proj_triton_forward(
            x,
            weight,
            bias,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        torch.cuda.synchronize()
        max_abs = float((actual - expected).abs().max().detach().cpu())
        ms = _bench(
            lambda bm=block_m, bn=block_n, bk=block_k: qwen_o_proj_triton_forward(
                x,
                weight,
                bias,
                block_m=bm,
                block_n=bn,
                block_k=bk,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        rows.append(
            {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "ms": ms,
                "speedup_vs_f_linear": baseline_ms / max(ms, 1.0e-12),
                "max_abs_error": max_abs,
            }
        )
    rows.sort(key=lambda row: row["ms"])
    return {
        "schema": "streamattn.qwen_o_proj_triton_profile.v1",
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "dtype": args.dtype,
            "bias": bool(args.bias),
        },
        "baseline": {"backend": "torch_f_linear", "ms": baseline_ms},
        "variants": rows,
        "best": rows[0] if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument(
        "--variants",
        default="16,32,64;16,64,64;16,128,64;16,64,128;32,64,64;32,128,64",
        help="Semicolon-separated BLOCK_M,BLOCK_N,BLOCK_K variants.",
    )
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
