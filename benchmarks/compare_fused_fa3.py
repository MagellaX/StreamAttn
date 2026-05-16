#!/usr/bin/env python3
"""Benchmark FusedOnlineAttention against FlashAttention-3.

This script sweeps sequence lengths and head dimensions to compare the
custom fused online attention kernel with the FlashAttention-3 reference
implementation.  Tokens/s and TFLOP/s are reported along with speedups
for easy sharing in CSV or Markdown formats.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List

import torch

from stream_attention.core.config import StreamAttentionConfig
from stream_attention.core.fused_online_attention import FusedOnlineAttention
from stream_attention.core.flashattention_v3 import FlashAttentionV3


def _benchmark(
    module: torch.nn.Module,
    seq_len: int,
    batch: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    """Benchmark helper returning tokens/s and TFLOP/s."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for _ in range(warmup):
        module(q, k, v, causal=causal)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        module(q, k, v, causal=causal)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters

    tokens_per_s = batch * seq_len / elapsed
    flops = 4.0 * batch * num_heads * seq_len * seq_len * head_dim
    tflops = flops / elapsed / 1e12
    return {"time_s": elapsed, "tokens_s": tokens_per_s, "tflops": tflops}


def run_benchmarks(
    seq_lens: List[int],
    head_dims: List[int],
    num_heads: int,
    batch: int,
    causal: bool,
    dropout: float,
    warmup: int,
    iters: int,
) -> List[Dict[str, float]]:
    """Run benchmarks for all seq lengths and head dimensions."""

    results: List[Dict[str, float]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for hd in head_dims:
        cfg = StreamAttentionConfig(
            num_heads=num_heads,
            head_dim=hd,
            dropout=dropout,
            use_fp16=device.type == "cuda",
        )
        fused = FusedOnlineAttention(
            num_heads=num_heads,
            head_dim=hd,
            dropout=dropout,
            dtype=dtype,
        )
        fa3 = FlashAttentionV3(cfg)
        if dropout > 0.0:
            fused.train()
            fa3.train()
        else:
            fused.eval()
            fa3.eval()

        for L in seq_lens:
            fr = _benchmark(fused, L, batch, num_heads, hd, causal, warmup, iters)
            ar = _benchmark(fa3, L, batch, num_heads, hd, causal, warmup, iters)
            speedup = fr["tokens_s"] / ar["tokens_s"] if ar["tokens_s"] > 0 else float("inf")
            results.append(
                {
                    "seq_len": L,
                    "head_dim": hd,
                    "fused_tokens_s": fr["tokens_s"],
                    "fused_tflops": fr["tflops"],
                    "fa3_tokens_s": ar["tokens_s"],
                    "fa3_tflops": ar["tflops"],
                    "speedup": speedup,
                }
            )
    return results


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    fieldnames = [
        "seq_len",
        "head_dim",
        "fused_tokens_s",
        "fused_tflops",
        "fa3_tokens_s",
        "fa3_tflops",
        "speedup",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_markdown(path: str, rows: List[Dict[str, float]]) -> None:
    with open(path, "w") as f:
        f.write(
            "| Seq Len | Head Dim | Fused Tokens/s | Fused TFLOP/s | FA3 Tokens/s | FA3 TFLOP/s | Speedup |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['seq_len']} | {r['head_dim']} | {r['fused_tokens_s']:.2f} | {r['fused_tflops']:.2f} | "
                f"{r['fa3_tokens_s']:.2f} | {r['fa3_tflops']:.2f} | {r['speedup']:.2f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FusedOnlineAttention with FlashAttention-3"
    )
    parser.add_argument(
        "--seq-lens",
        nargs="*",
        type=int,
        default=[1024, 2048, 4096, 8192, 16384, 32768],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--head-dims",
        nargs="*",
        type=int,
        default=[64, 128, 256],
        help="Head dimensions to benchmark",
    )
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Measured iterations")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument(
        "--causal",
        action="store_true",
        default=False,
        help="Use causal attention (default non-causal)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="",
        help="CUDA architecture (e.g. 80, 90) for kernel compilation",
    )
    parser.add_argument("--csv", type=str, default="", help="Path to CSV output")
    parser.add_argument("--md", type=str, default="", help="Path to Markdown output")
    args = parser.parse_args()

    if args.arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch

    rows = run_benchmarks(
        args.seq_lens,
        args.head_dims,
        args.heads,
        args.batch,
        args.causal,
        args.dropout,
        args.warmup,
        args.iters,
    )

    print(
        "SeqLen\tHeadDim\tFusedTok/s\tFusedTFLOP/s\tFA3Tok/s\tFA3TFLOP/s\tSpeedup"
    )
    for r in rows:
        print(
            f"{r['seq_len']}\t{r['head_dim']}\t{r['fused_tokens_s']:.2f}\t{r['fused_tflops']:.2f}"
            f"\t{r['fa3_tokens_s']:.2f}\t{r['fa3_tflops']:.2f}\t{r['speedup']:.2f}"
        )

    if args.csv:
        write_csv(args.csv, rows)
    if args.md:
        write_markdown(args.md, rows)


if __name__ == "__main__":  # pragma: no cover
    main()
