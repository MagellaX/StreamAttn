"""Profile Gate-1 competitiveness against SDPA dense at real-ish LLM shapes."""

import argparse
import itertools
import json
from typing import Iterable

import torch

from stream_attention import StreamAttnMetadataCache
from stream_attention.gate1 import dense_attention_forward
from stream_attention.kernels.gate1_fwd_triton import gate1_attention_triton_forward


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
    return q, k, v, active_blocks / num_blocks if num_blocks else 0.0


def _summarize_stats(raw_stats: torch.Tensor):
    totals = raw_stats.detach().sum(dim=(0, 1, 2)).cpu()
    cta_tiles_total = int(totals[2].item())
    cta_pv_executed = int(totals[4].item())
    return {
        "row_skips": int(totals[0].item()),
        "row_computes": int(totals[1].item()),
        "cta_tiles_total": cta_tiles_total,
        "cta_pv_skipped": int(totals[3].item()),
        "cta_pv_executed": cta_pv_executed,
        "active_pv_fraction": (
            cta_pv_executed / cta_tiles_total if cta_tiles_total else 0.0
        ),
    }


def _run_gate1(
    q,
    k,
    v,
    *,
    block_size: int,
    tile_size_q: int,
    causal: bool,
    value_norm_bounds=None,
    force_mode: int,
    skip_predicate: str = "mass",
    error_budget: float = 1e-3,
    return_raw_stats: bool = False,
):
    return gate1_attention_triton_forward(
        q,
        k,
        v,
        causal=causal,
        error_budget=error_budget,
        block_size=block_size,
        tile_size_q=tile_size_q,
        value_norm_bounds=value_norm_bounds,
        skip_predicate=skip_predicate,
        force_mode=force_mode,
        return_raw_stats=return_raw_stats,
    )


def _profile_one(
    args,
    *,
    seq: int,
    heads: int,
    dim: int,
    block_size: int,
    dtype_name: str,
    active_fraction: float,
):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    q, k, v, block_quantized_active = _make_peaked(
        batch=args.batch,
        seq=seq,
        heads=heads,
        dim=dim,
        dtype=dtype,
        active_fraction=active_fraction,
        block_size=block_size,
        peak=args.peak,
    )

    metadata_build_ms = _time_cuda(
        lambda: StreamAttnMetadataCache.from_value(
            v,
            block_size=block_size,
            use_triton=True,
        ),
        warmup=args.metadata_warmup,
        iters=args.metadata_iters,
    )
    metadata = StreamAttnMetadataCache.from_value(
        v,
        block_size=block_size,
        use_triton=True,
    )
    value_norm_bounds = metadata.require_value_norm_bounds()
    torch.cuda.synchronize()

    sdpa_dense_ms = _time_cuda(
        lambda: dense_attention_forward(q, k, v, causal=args.causal),
        warmup=args.warmup,
        iters=args.iters,
    )
    gate1_dense_equiv_ms = _time_cuda(
        lambda: _run_gate1(
            q,
            k,
            v,
            block_size=block_size,
            tile_size_q=args.tile_size_q,
            causal=args.causal,
            force_mode=5,
            skip_predicate="mass",
            error_budget=args.error_budget,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    gate1_true_qk_scan_ms = _time_cuda(
        lambda: _run_gate1(
            q,
            k,
            v,
            block_size=block_size,
            tile_size_q=args.tile_size_q,
            causal=args.causal,
            force_mode=7,
            skip_predicate="mass",
            error_budget=args.error_budget,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    gate1_qk_log_predicate_no_pv_ms = _time_cuda(
        lambda: _run_gate1(
            q,
            k,
            v,
            block_size=block_size,
            tile_size_q=args.tile_size_q,
            causal=args.causal,
            force_mode=8,
            skip_predicate="mass",
            error_budget=args.error_budget,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    gate1_qk_exp_predicate_no_pv_ms = _time_cuda(
        lambda: _run_gate1(
            q,
            k,
            v,
            block_size=block_size,
            tile_size_q=args.tile_size_q,
            causal=args.causal,
            force_mode=9,
            skip_predicate="mass",
            error_budget=args.error_budget,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    gate1_low_active_ms = _time_cuda(
        lambda: _run_gate1(
            q,
            k,
            v,
            block_size=block_size,
            tile_size_q=args.tile_size_q,
            causal=args.causal,
            value_norm_bounds=value_norm_bounds,
            force_mode=0,
            skip_predicate="value_bound",
            error_budget=args.error_budget,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    gate1_mass_ms = _time_cuda(
        lambda: _run_gate1(
            q,
            k,
            v,
            block_size=block_size,
            tile_size_q=args.tile_size_q,
            causal=args.causal,
            force_mode=0,
            skip_predicate="mass",
            error_budget=args.error_budget,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )

    _, raw_stats = _run_gate1(
        q,
        k,
        v,
        block_size=block_size,
        tile_size_q=args.tile_size_q,
        causal=args.causal,
        value_norm_bounds=value_norm_bounds,
        force_mode=0,
        skip_predicate="value_bound",
        error_budget=args.error_budget,
        return_raw_stats=True,
    )
    _, raw_stats_mass = _run_gate1(
        q,
        k,
        v,
        block_size=block_size,
        tile_size_q=args.tile_size_q,
        causal=args.causal,
        force_mode=0,
        skip_predicate="mass",
        error_budget=args.error_budget,
        return_raw_stats=True,
    )
    torch.cuda.synchronize()
    stats = _summarize_stats(raw_stats)
    mass_stats = _summarize_stats(raw_stats_mass)

    return {
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "batch": args.batch,
            "seq": seq,
            "heads": heads,
            "dim": dim,
            "dtype": dtype_name,
        },
        "block_size": block_size,
        "tile_size_q": args.tile_size_q,
        "requested_active_fraction": active_fraction,
        "block_quantized_active_fraction": block_quantized_active,
        "actual_active_fraction": stats["active_pv_fraction"],
        "metadata_build_ms": metadata_build_ms,
        "sdpa_dense_ms": sdpa_dense_ms,
        "gate1_dense_equiv_ms": gate1_dense_equiv_ms,
        "gate1_true_qk_scan_ms": gate1_true_qk_scan_ms,
        "gate1_qk_log_predicate_no_pv_ms": gate1_qk_log_predicate_no_pv_ms,
        "gate1_qk_exp_predicate_no_pv_ms": gate1_qk_exp_predicate_no_pv_ms,
        "gate1_value_bound_ms": gate1_low_active_ms,
        "gate1_mass_ms": gate1_mass_ms,
        "metadata_plus_gate1_value_bound_ms": metadata_build_ms + gate1_low_active_ms,
        "dense_equiv_ratio": gate1_dense_equiv_ms / sdpa_dense_ms,
        "true_qk_scan_ratio": gate1_true_qk_scan_ms / sdpa_dense_ms,
        "qk_log_predicate_no_pv_ratio": gate1_qk_log_predicate_no_pv_ms
        / sdpa_dense_ms,
        "qk_exp_predicate_no_pv_ratio": gate1_qk_exp_predicate_no_pv_ms
        / sdpa_dense_ms,
        "gate1_value_bound_ratio": gate1_low_active_ms / sdpa_dense_ms,
        "gate1_mass_ratio": gate1_mass_ms / sdpa_dense_ms,
        "metadata_over_dense": metadata_build_ms / sdpa_dense_ms,
        "metadata_plus_gate1_value_bound_ratio": (
            metadata_build_ms + gate1_low_active_ms
        )
        / sdpa_dense_ms,
        "value_bound_stats": stats,
        "mass_stats": mass_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", nargs="+", default=["4096"])
    parser.add_argument("--heads", nargs="+", default=["16"])
    parser.add_argument("--dim", nargs="+", default=["128"])
    parser.add_argument("--block-size", nargs="+", default=["64"])
    parser.add_argument("--active-fraction", nargs="+", default=["0.0625", "1.0"])
    parser.add_argument("--dtype", nargs="+", choices=["fp16", "bf16"], default=["fp16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--metadata-warmup", type=int, default=5)
    parser.add_argument("--metadata-iters", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    rows = []
    seqs = _parse_values(args.seq, int)
    heads_values = _parse_values(args.heads, int)
    dims = _parse_values(args.dim, int)
    block_sizes = _parse_values(args.block_size, int)
    active_fractions = _parse_values(args.active_fraction, float)
    dtypes = _parse_values(args.dtype, str)

    for seq, heads, dim, block_size, dtype_name, active_fraction in itertools.product(
        seqs,
        heads_values,
        dims,
        block_sizes,
        dtypes,
        active_fractions,
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
                    active_fraction=active_fraction,
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
                    "active_fraction": active_fraction,
                    "error": str(exc),
                }
            )
            torch.cuda.empty_cache()

    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
