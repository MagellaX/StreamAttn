"""Profile value-norm metadata build cost against dense and Gate-1 attention."""

import argparse
import itertools
import json
from typing import Iterable

import torch

from stream_attention import StreamAttnMetadataCache
from stream_attention.gate1 import stream_attn_gate1


def _parse_values(values: Iterable[str], cast):
    parsed = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(cast(item))
    return parsed


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) / iters


def _make_peaked(
    *,
    batch: int,
    seq: int,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    active_fraction: float,
    block_size: int,
    peak: float,
):
    device = torch.device("cuda")
    q = torch.zeros(batch, seq, heads, dim, device=device, dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.randn_like(q)
    num_blocks = (seq + block_size - 1) // block_size
    active_blocks = max(0, min(num_blocks, round(active_fraction * num_blocks)))
    active_tokens = min(seq, active_blocks * block_size)
    q[..., 0] = peak
    k[:, :active_tokens, :, 0] = peak
    k[:, active_tokens:, :, 0] = -peak
    return q, k, v


def _profile_one(args, *, seq: int, heads: int, dim: int, block_size: int, dtype_name: str):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    q, k, v = _make_peaked(
        batch=args.batch,
        seq=seq,
        heads=heads,
        dim=dim,
        dtype=dtype,
        active_fraction=args.active_fraction,
        block_size=block_size,
        peak=args.peak,
    )

    metadata_build_ms = _time_cuda(
        lambda: StreamAttnMetadataCache.from_value(
            v,
            block_size=block_size,
            use_triton=args.builder == "triton",
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    metadata = StreamAttnMetadataCache.from_value(
        v,
        block_size=block_size,
        use_triton=args.builder == "triton",
    )
    torch.cuda.synchronize()

    dense_ms = _time_cuda(
        lambda: stream_attn_gate1(
            q,
            k,
            v,
            causal=args.causal,
            mode="dense",
            telemetry=False,
        ),
        warmup=args.attn_warmup,
        iters=args.attn_iters,
    )
    gate1_ms = _time_cuda(
        lambda: stream_attn_gate1(
            q,
            k,
            v,
            causal=args.causal,
            mode="gate1",
            metadata=metadata,
            block_size=block_size,
            tile_size_q=args.tile_size_q,
            error_budget=args.error_budget,
            telemetry=False,
        ),
        warmup=args.attn_warmup,
        iters=args.attn_iters,
    )

    return {
        "device": torch.cuda.get_device_name(0),
        "builder": args.builder,
        "shape": {
            "batch": args.batch,
            "seq": seq,
            "heads": heads,
            "dim": dim,
            "dtype": dtype_name,
        },
        "block_size": block_size,
        "tile_size_q": args.tile_size_q,
        "active_fraction": args.active_fraction,
        "metadata_build_ms": metadata_build_ms,
        "dense_ms": dense_ms,
        "gate1_ms": gate1_ms,
        "metadata_plus_gate1_ms": metadata_build_ms + gate1_ms,
        "metadata_over_dense": metadata_build_ms / dense_ms if dense_ms > 0.0 else None,
        "metadata_over_gate1": metadata_build_ms / gate1_ms if gate1_ms > 0.0 else None,
        "metadata_plus_gate1_over_dense": (
            (metadata_build_ms + gate1_ms) / dense_ms if dense_ms > 0.0 else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", nargs="+", default=["4096", "8192"])
    parser.add_argument("--heads", nargs="+", default=["16"])
    parser.add_argument("--dim", nargs="+", default=["128"])
    parser.add_argument("--block-size", nargs="+", default=["64"])
    parser.add_argument("--dtype", nargs="+", choices=["fp16", "bf16"], default=["fp16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--active-fraction", type=float, default=0.0625)
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--builder", choices=["triton", "torch"], default="triton")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--attn-warmup", type=int, default=5)
    parser.add_argument("--attn-iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    rows = []
    seqs = _parse_values(args.seq, int)
    heads_values = _parse_values(args.heads, int)
    dims = _parse_values(args.dim, int)
    block_sizes = _parse_values(args.block_size, int)
    dtypes = _parse_values(args.dtype, str)

    for seq, heads, dim, block_size, dtype_name in itertools.product(
        seqs,
        heads_values,
        dims,
        block_sizes,
        dtypes,
    ):
        try:
            rows.append(
                _profile_one(
                    args,
                    seq=seq,
                    heads=heads,
                    dim=dim,
                    block_size=block_size,
                    dtype_name=dtype_name,
                )
            )
        except RuntimeError as exc:
            rows.append(
                {
                    "shape": {
                        "batch": args.batch,
                        "seq": seq,
                        "heads": heads,
                        "dim": dim,
                        "dtype": dtype_name,
                    },
                    "block_size": block_size,
                    "error": str(exc),
                }
            )
            torch.cuda.empty_cache()

    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
