"""Benchmark suite for StreamAttention.

This module provides utilities to benchmark the fused online attention
implementation against FlashAttention-3.  In addition to the standard
full-sequence benchmark, a *streaming* benchmark is provided which feeds
tokens incrementally in fixed sized chunks.  The streaming benchmark is
useful for measuring latency in autoregressive decoding scenarios where
tokens arrive over time.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import torch

from stream_attention.core.config import StreamAttentionConfig
from stream_attention.core.flashattention_v3 import FlashAttentionV3
from stream_attention.core.fused_online_attention import FusedOnlineAttention


def _benchmark_module(
    module: torch.nn.Module,
    seq_len: int,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> Dict[str, float]:
    """Benchmark a module on a full sequence.

    Args:
        module: Attention module with ``forward`` accepting ``(q, k, v)``.
        seq_len: Sequence length to benchmark.
        batch_size: Batch size.
        warmup: Number of warmup runs.
        iterations: Number of measured iterations.

    Returns:
        Dictionary with ``time_ms``, ``tflops`` and ``bandwidth_gb_s``.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    nh = getattr(module, "num_heads")
    hd = getattr(module, "head_dim")

    q = torch.randn(batch_size, seq_len, nh, hd, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for _ in range(warmup):
        module(q, k, v, causal=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iterations):
        module(q, k, v, causal=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / iterations

    flops = 4.0 * batch_size * nh * seq_len * seq_len * hd
    tflops = flops / elapsed / 1e12
    bytes_per_el = torch.tensor([], dtype=dtype).element_size()
    memory_bytes = 3 * batch_size * seq_len * nh * hd * bytes_per_el
    bandwidth = memory_bytes / elapsed / 1e9
    return {
        "time_ms": elapsed * 1000.0,
        "tflops": tflops,
        "bandwidth_gb_s": bandwidth,
    }


def _streaming_benchmark_module(
    module: torch.nn.Module,
    seq_len: int,
    chunk_size: int,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> Dict[str, float]:
    """Benchmark a module in a streaming setting.

    Tokens are processed incrementally in chunks of ``chunk_size``.  Each
    chunk's queries attend to all keys/values seen so far.  The measured time
    corresponds to the cumulative latency to process the full ``seq_len``.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    nh = getattr(module, "num_heads")
    hd = getattr(module, "head_dim")

    q = torch.randn(batch_size, seq_len, nh, hd, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def run_once() -> None:
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            qc = q[:, start:end]
            kc = k[:, :end]
            vc = v[:, :end]
            module(qc, kc, vc, causal=True)

    for _ in range(warmup):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_t = time.time()
    for _ in range(iterations):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.time() - start_t) / iterations

    # Total FLOPS across all chunks: 4 * B * H * D * sum(q_chunk_len * kv_len)
    total_qk = 0
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        ql = end - start
        total_qk += ql * end
    flops = 4.0 * batch_size * nh * hd * total_qk
    tflops = flops / elapsed / 1e12

    bytes_per_el = torch.tensor([], dtype=dtype).element_size()
    memory_bytes = 3 * batch_size * seq_len * nh * hd * bytes_per_el
    bandwidth = memory_bytes / elapsed / 1e9

    return {
        "time_ms": elapsed * 1000.0,
        "tflops": tflops,
        "bandwidth_gb_s": bandwidth,
    }


def run_bench(
    seq_lens: List[int],
    batch_size: int,
    num_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
) -> Dict[int, Dict[str, float]]:
    """Run the standard full sequence benchmark for multiple lengths."""

    cfg = StreamAttentionConfig(
        num_heads=num_heads, head_dim=head_dim, use_fp16=torch.cuda.is_available()
    )
    fused = FusedOnlineAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    fa3 = FlashAttentionV3(cfg)

    results: Dict[int, Dict[str, float]] = {}
    for L in seq_lens:
        fr = _benchmark_module(fused, L, batch_size, warmup, iters)
        ar = _benchmark_module(fa3, L, batch_size, warmup, iters)
        results[L] = {
            "fused_time_ms": fr["time_ms"],
            "fused_tflops": fr["tflops"],
            "fa3_time_ms": ar["time_ms"],
            "fa3_tflops": ar["tflops"],
            "speedup_vs_fa3": (
                ar["time_ms"] / fr["time_ms"] if fr["time_ms"] > 0 else float("inf")
            ),
        }
    return results


def streaming_benchmark(
    seq_lens: List[int],
    chunk_size: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
) -> Dict[int, Dict[str, float]]:
    """Run the streaming benchmark for multiple sequence lengths."""

    cfg = StreamAttentionConfig(
        num_heads=num_heads, head_dim=head_dim, use_fp16=torch.cuda.is_available()
    )
    fused = FusedOnlineAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    fa3 = FlashAttentionV3(cfg)

    results: Dict[int, Dict[str, float]] = {}
    for L in seq_lens:
        fr = _streaming_benchmark_module(fused, L, chunk_size, batch_size, warmup, iters)
        ar = _streaming_benchmark_module(fa3, L, chunk_size, batch_size, warmup, iters)
        results[L] = {
            "fused_time_ms": fr["time_ms"],
            "fused_tflops": fr["tflops"],
            "fa3_time_ms": ar["time_ms"],
            "fa3_tflops": ar["tflops"],
            "speedup_vs_fa3": (
                ar["time_ms"] / fr["time_ms"] if fr["time_ms"] > 0 else float("inf")
            ),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StreamAttention vs FlashAttention-3 Benchmark"
    )
    parser.add_argument("--seq", nargs="*", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Chunk size for streaming benchmark. If 0, run full benchmark.",
    )
    parser.add_argument("--json_out", type=str, default="")
    args = parser.parse_args()

    if args.chunk > 0:
        res = streaming_benchmark(
            args.seq,
            args.chunk,
            args.batch,
            args.heads,
            args.dim,
            args.warmup,
            args.iters,
        )
        print(
            "SeqLen\tChunk\tFused(ms)\tFused(TF)\tFA3(ms)\tFA3(TF)\tFA3/Fused(ms)"
        )
        for L in args.seq:
            r = res[L]
            print(
                f"{L}\t{args.chunk}\t{r['fused_time_ms']:.3f}\t{r['fused_tflops']:.2f}"
                f"\t{r['fa3_time_ms']:.3f}\t{r['fa3_tflops']:.2f}"
                f"\t{r['speedup_vs_fa3']:.2f}"
            )
    else:
        res = run_bench(
            args.seq,
            args.batch,
            args.heads,
            args.dim,
            args.warmup,
            args.iters,
        )
        print("SeqLen\tFused(ms)\tFused(TF)\tFA3(ms)\tFA3(TF)\tFA3/Fused(ms)")
        for L in args.seq:
            r = res[L]
            print(
                f"{L}\t{r['fused_time_ms']:.3f}\t{r['fused_tflops']:.2f}"
                f"\t{r['fa3_time_ms']:.3f}\t{r['fa3_tflops']:.2f}"
                f"\t{r['speedup_vs_fa3']:.2f}"
            )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

