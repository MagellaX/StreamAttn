"""Profile Gate-1 for long-KV decode with cached metadata.

This benchmark is intentionally scoped to contiguous KV cache tensors. Decode
mechanics use ``causal=False`` by default because a query at the end of a KV
cache should attend to all valid cached keys; regular top-left causal masking is
wrong when ``query_len << kv_len``.
"""

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch

from stream_attention import StreamAttnMetadataCache
from stream_attention.gate1 import (
    dense_attention_forward,
    make_route_request,
    stream_attn_gate1,
)
from stream_attention.kernels.gate1_fwd_triton import gate1_attention_triton_forward
from stream_attention.router import (
    CostEntry,
    CostKey,
    Gate1CostModel,
    StreamAttnPolicy,
    StreamAttnRouter,
)
from stream_attention.telemetry import ActiveFractionTelemetry


def _parse_values(values: Iterable[str], cast):
    parsed = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(cast(item))
    return parsed


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


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


def _time_wall_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _active_blocks(seq_k: int, block_size: int, active_fraction: float) -> Tuple[int, int]:
    num_blocks = (seq_k + block_size - 1) // block_size
    active = max(0, min(num_blocks, round(active_fraction * num_blocks)))
    if active_fraction > 0.0 and num_blocks > 0 and active == 0:
        active = 1
    return active, num_blocks


def _make_pattern(
    *,
    batch: int,
    query_len: int,
    kv_len: int,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    pattern: str,
    active_fraction: float,
    block_size: int,
    peak: float,
    sink_blocks: int,
    recent_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    device = torch.device("cuda")
    q = torch.zeros(batch, query_len, heads, dim, device=device, dtype=dtype)
    k = torch.zeros(batch, kv_len, heads, dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_len, heads, dim, device=device, dtype=dtype)

    if pattern == "random":
        q = torch.randn_like(q)
        k = torch.randn_like(k)
        _, num_blocks = _active_blocks(kv_len, block_size, active_fraction)
        return q, k, v, 1.0 if num_blocks else 0.0

    q[..., 0] = peak
    k[..., 0] = -peak
    active_blocks, num_blocks = _active_blocks(kv_len, block_size, active_fraction)

    active_block_ids: List[int] = []
    if pattern == "peaked":
        active_block_ids = list(range(active_blocks))
    elif pattern == "sink_local":
        sink = min(sink_blocks, active_blocks, num_blocks)
        remaining = max(0, active_blocks - sink)
        recent = min(max(recent_blocks, remaining), remaining, max(0, num_blocks - sink))
        active_block_ids = list(range(sink))
        if recent:
            active_block_ids.extend(range(num_blocks - recent, num_blocks))
    elif pattern == "sliding_recent":
        active_block_ids = list(range(max(0, num_blocks - active_blocks), num_blocks))
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    for block_idx in sorted(set(active_block_ids)):
        start = block_idx * block_size
        end = min(start + block_size, kv_len)
        if start < end:
            k[:, start:end, :, 0] = peak

    actual_active = len(set(active_block_ids)) / num_blocks if num_blocks else 0.0
    return q, k, v, actual_active


def _make_decode_tensors(args, *, query_len: int, kv_len: int, heads: int, kv_heads: int, active_fraction: float):
    dtype = _dtype(args.dtype)
    if args.attention_type == "mha":
        logical_kv_heads = heads
    elif args.attention_type == "mqa":
        logical_kv_heads = 1
    else:
        logical_kv_heads = kv_heads
    if heads % logical_kv_heads != 0:
        raise ValueError("heads must be divisible by kv_heads for expanded GQA/MQA")

    q, k, v, block_active = _make_pattern(
        batch=args.batch,
        query_len=query_len,
        kv_len=kv_len,
        heads=logical_kv_heads if args.attention_type != "mha" else heads,
        dim=args.dim,
        dtype=dtype,
        pattern=args.pattern,
        active_fraction=active_fraction,
        block_size=args.block_size,
        peak=args.peak,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )
    if args.attention_type == "mha":
        return q, k, v, logical_kv_heads, block_active

    if args.pattern == "random":
        q = torch.randn(args.batch, query_len, heads, args.dim, device="cuda", dtype=dtype)
    else:
        q = torch.zeros(args.batch, query_len, heads, args.dim, device="cuda", dtype=dtype)
        q[..., 0] = args.peak
    group = heads // logical_kv_heads
    return q, k.repeat_interleave(group, dim=2), v.repeat_interleave(group, dim=2), logical_kv_heads, block_active


def _stats_dict(info) -> Tuple[Optional[float], List[float]]:
    if info.stats is None:
        return None, []
    per_head = (
        [stat.active_pv_fraction for stat in info.per_head_stats]
        if info.per_head_stats is not None
        else []
    )
    return info.stats.active_pv_fraction, per_head


def _error_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = (out.float() - ref.float()).abs()
    ref_norm = ref.float().norm()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "relative_l2_error": float((out.float() - ref.float()).norm().div(ref_norm.clamp_min(1.0e-20)).item()),
    }


def _run_gate1_force(
    q,
    k,
    v,
    *,
    args,
    force_mode: int,
    skip_predicate: str = "mass",
):
    return gate1_attention_triton_forward(
        q,
        k,
        v,
        causal=False,
        error_budget=args.error_budget,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        skip_predicate=skip_predicate,
        force_mode=force_mode,
        return_raw_stats=False,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def _metadata_update_uses_triton(backend: str) -> Optional[bool]:
    if backend == "auto":
        return None
    return backend == "triton"


def _profile_one(args, *, query_len: int, kv_len: int, heads: int, kv_heads: int, active_fraction: float) -> dict:
    if args.causal_mode != "none":
        raise NotImplementedError("decode profiler v1 supports only causal-mode=none")

    q, k, v, logical_kv_heads, block_active = _make_decode_tensors(
        args,
        query_len=query_len,
        kv_len=kv_len,
        heads=heads,
        kv_heads=kv_heads,
        active_fraction=active_fraction,
    )
    expansion_factor = heads / logical_kv_heads

    metadata_full_build_ms = _time_cuda(
        lambda: StreamAttnMetadataCache.from_value(
            v,
            block_size=args.block_size,
            use_triton=True,
        ),
        warmup=args.metadata_warmup,
        iters=args.metadata_iters,
    )
    metadata = StreamAttnMetadataCache.from_value(v, block_size=args.block_size, use_triton=True)
    torch.cuda.synchronize()

    update_tokens = min(query_len, kv_len)
    new_v = v[:, kv_len - update_tokens : kv_len, :, :].contiguous()
    update_start = kv_len - update_tokens
    metadata_update_cuda_ms = _time_cuda(
        lambda: metadata.update_value_bounds_(
            new_v,
            start_pos=update_start,
            use_triton=_metadata_update_uses_triton(args.metadata_update_backend),
        ),
        warmup=args.metadata_warmup,
        iters=args.metadata_iters,
    )
    metadata_update_wall_ms = _time_wall_cuda(
        lambda: metadata.update_value_bounds_(
            new_v,
            start_pos=update_start,
            use_triton=_metadata_update_uses_triton(args.metadata_update_backend),
        ),
        warmup=args.metadata_warmup,
        iters=args.metadata_iters,
    )

    dense_decode_ms = _time_cuda(
        lambda: stream_attn_gate1(q, k, v, causal=False, mode="dense", telemetry=False),
        warmup=args.warmup,
        iters=args.iters,
    )
    run_gate1_suite = args.mode in {"all", "gate1", "auto"}
    run_mass = run_gate1_suite and (
        args.skip_predicate in {"mass", "both"} or args.mode == "auto"
    )
    run_value_bound = run_gate1_suite and (
        args.skip_predicate in {"value_bound", "both"} or args.mode == "auto"
    )

    gate1_dense_equiv_ms = None
    gate1_qk_scan_ms = None
    if run_gate1_suite:
        gate1_dense_equiv_ms = _time_cuda(
            lambda: _run_gate1_force(q, k, v, args=args, force_mode=5),
            warmup=args.warmup,
            iters=args.iters,
        )
        gate1_qk_scan_ms = _time_cuda(
            lambda: _run_gate1_force(q, k, v, args=args, force_mode=7),
            warmup=args.warmup,
            iters=args.iters,
        )

    dense_out = dense_attention_forward(q, k, v, causal=False)

    gate1_mass_ms = None
    gate1_value_bound_ms = None
    mass_info = None
    value_info = None
    mass_out = None
    value_out = None
    if run_mass:
        gate1_mass_ms = _time_cuda(
            lambda: stream_attn_gate1(
                q,
                k,
                v,
                causal=False,
                mode="gate1",
                skip_predicate="mass",
                error_budget=args.error_budget,
                block_size=args.block_size,
                tile_size_q=args.tile_size_q,
                telemetry=False,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        mass_out, mass_info = stream_attn_gate1(
            q,
            k,
            v,
            causal=False,
            mode="gate1",
            skip_predicate="mass",
            error_budget=args.error_budget,
            block_size=args.block_size,
            tile_size_q=args.tile_size_q,
            telemetry=False,
            return_info=True,
        )
    if run_value_bound:
        gate1_value_bound_ms = _time_cuda(
            lambda: stream_attn_gate1(
                q,
                k,
                v,
                causal=False,
                mode="gate1",
                metadata=metadata,
                skip_predicate="value_bound",
                error_budget=args.error_budget,
                block_size=args.block_size,
                tile_size_q=args.tile_size_q,
                telemetry=False,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        value_out, value_info = stream_attn_gate1(
            q,
            k,
            v,
            causal=False,
            mode="gate1",
            metadata=metadata,
            skip_predicate="value_bound",
            error_budget=args.error_budget,
            block_size=args.block_size,
            tile_size_q=args.tile_size_q,
            telemetry=False,
            return_info=True,
        )

    request = make_route_request(
        q,
        k,
        causal=False,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        model_id="decode-profile",
        layer_id=0,
        head_id=-1,
        phase="decode",
        metadata_available=True,
    )
    cost_model = Gate1CostModel()
    qk_for_cost = min(gate1_qk_scan_ms or dense_decode_ms, dense_decode_ms)
    cost_model.update(
        CostKey.from_request(request),
        CostEntry(dense_ms=dense_decode_ms, qk_only_ms=qk_for_cost),
    )
    router = StreamAttnRouter(
        policy=StreamAttnPolicy(min_confidence=0.7, history_min_observations=4),
        telemetry=ActiveFractionTelemetry(min_observations=4),
        cost_model=cost_model,
    )
    if mass_info is not None and mass_info.stats is not None:
        for _ in range(4):
            router.observe(
                request,
                cta_pv_executed=mass_info.stats.cta_pv_executed,
                cta_tiles_total=mass_info.stats.cta_tiles_total,
            )

    auto_skip_predicate = args.auto_skip_predicate
    auto_metadata = metadata if auto_skip_predicate == "value_bound" else None
    router_auto_ms = None
    auto_info = None
    if args.mode in {"all", "auto"}:
        router_auto_ms = _time_cuda(
            lambda: stream_attn_gate1(
                q,
                k,
                v,
                causal=False,
                mode="auto",
                router=router,
                metadata=auto_metadata,
                request=request,
                skip_predicate=auto_skip_predicate,
                error_budget=args.error_budget,
                block_size=args.block_size,
                tile_size_q=args.tile_size_q,
                telemetry=False,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        _, auto_info = stream_attn_gate1(
            q,
            k,
            v,
            causal=False,
            mode="auto",
            router=router,
            metadata=auto_metadata,
            request=request,
            skip_predicate=auto_skip_predicate,
            error_budget=args.error_budget,
            block_size=args.block_size,
            tile_size_q=args.tile_size_q,
            telemetry=False,
            return_info=True,
        )

    candidates = {"dense": dense_decode_ms}
    if gate1_mass_ms is not None:
        candidates["mass"] = gate1_mass_ms
    if gate1_value_bound_ms is not None:
        candidates["value_bound"] = gate1_value_bound_ms
    oracle_backend, oracle_ms = min(candidates.items(), key=lambda item: item[1])
    router_regret_raw_ms = (
        router_auto_ms - oracle_ms if router_auto_ms is not None else None
    )
    router_regret_ms = (
        max(0.0, router_regret_raw_ms) if router_regret_raw_ms is not None else None
    )
    router_regret_pct = (
        router_regret_ms / oracle_ms
        if router_regret_ms is not None and oracle_ms > 0
        else None
    )

    active_mass, per_head_mass = _stats_dict(mass_info) if mass_info is not None else (None, [])
    active_value, per_head_value = _stats_dict(value_info) if value_info is not None else (None, [])
    row = {
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "batch": args.batch,
            "query_len": query_len,
            "kv_len": kv_len,
            "heads": heads,
            "kv_heads": logical_kv_heads,
            "dim": args.dim,
            "dtype": args.dtype,
            "attention_type": args.attention_type,
            "physical_heads_used_by_gate1": heads,
            "expansion_factor": expansion_factor,
        },
        "block_size": args.block_size,
        "tile_size_q": args.tile_size_q,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "pattern": args.pattern,
        "requested_active_fraction": active_fraction,
        "block_quantized_active_fraction": block_active,
        "causal_mode": args.causal_mode,
        "metadata_full_build_ms": metadata_full_build_ms,
        "metadata_update_backend": args.metadata_update_backend,
        "metadata_update_cuda_ms": metadata_update_cuda_ms,
        "metadata_update_wall_ms": metadata_update_wall_ms,
        "metadata_full_build_over_dense": metadata_full_build_ms / dense_decode_ms,
        "metadata_update_wall_over_dense": metadata_update_wall_ms / dense_decode_ms,
        "dense_decode_ms": dense_decode_ms,
        "gate1_dense_equiv_ms": gate1_dense_equiv_ms,
        "gate1_qk_scan_ms": gate1_qk_scan_ms,
        "gate1_mass_ms": gate1_mass_ms,
        "gate1_value_bound_ms": gate1_value_bound_ms,
        "metadata_plus_value_bound_ms": (
            metadata_full_build_ms + gate1_value_bound_ms
            if gate1_value_bound_ms is not None
            else None
        ),
        "router_auto_ms": router_auto_ms,
        "router_backend": auto_info.decision.backend if auto_info is not None else None,
        "router_reason": auto_info.decision.reason if auto_info is not None else None,
        "router_prediction_active_frac": (
            auto_info.decision.prediction.active_frac_hat
            if auto_info is not None
            else None
        ),
        "router_prediction_confidence": (
            auto_info.decision.prediction.confidence
            if auto_info is not None
            else None
        ),
        "router_active_threshold": (
            auto_info.decision.active_threshold if auto_info is not None else None
        ),
        "oracle_backend": oracle_backend,
        "oracle_ms": oracle_ms,
        "router_regret_raw_ms": router_regret_raw_ms,
        "router_regret_ms": router_regret_ms,
        "router_regret_pct": router_regret_pct,
        "active_pv_fraction_mass": active_mass,
        "active_pv_fraction_value_bound": active_value,
        "per_head_active_pv_fraction_mass": per_head_mass,
        "per_head_active_pv_fraction_value_bound": per_head_value,
        "mass_error": _error_metrics(mass_out, dense_out) if mass_out is not None else None,
        "value_bound_error": (
            _error_metrics(value_out, dense_out) if value_out is not None else None
        ),
    }
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-lens", nargs="+", default=["1", "4", "8", "16"])
    parser.add_argument("--kv-lens", nargs="+", default=["4096", "8192", "16384"])
    parser.add_argument("--heads", nargs="+", default=["16"])
    parser.add_argument("--kv-heads", nargs="+", default=["16"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--attention-type", choices=["mha", "gqa", "mqa"], default="mha")
    parser.add_argument("--pattern", choices=["random", "peaked", "sink_local", "sliding_recent"], default="peaked")
    parser.add_argument("--active-fraction", nargs="+", default=["0.0625", "0.25", "1.0"])
    parser.add_argument("--skip-predicate", choices=["mass", "value_bound", "both"], default="both")
    parser.add_argument("--auto-skip-predicate", choices=["mass", "value_bound"], default="mass")
    parser.add_argument("--mode", choices=["dense", "gate1", "auto", "all"], default="all")
    parser.add_argument("--causal-mode", choices=["none", "bottom_right_reference"], default="none")
    parser.add_argument("--block-size", nargs="+", default=["64"])
    parser.add_argument("--tile-size-q", nargs="+", default=["64"])
    parser.add_argument("--num-warps", nargs="+", default=["4"])
    parser.add_argument("--num-stages", nargs="+", default=["3"])
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--metadata-warmup", type=int, default=5)
    parser.add_argument("--metadata-iters", type=int, default=20)
    parser.add_argument(
        "--metadata-update-backend",
        choices=["auto", "triton", "torch"],
        default="auto",
    )
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    query_lens = _parse_values(args.query_lens, int)
    kv_lens = _parse_values(args.kv_lens, int)
    heads_values = _parse_values(args.heads, int)
    kv_head_values = _parse_values(args.kv_heads, int)
    active_fractions = _parse_values(args.active_fraction, float)
    block_sizes = _parse_values(args.block_size, int)
    tile_sizes_q = _parse_values(args.tile_size_q, int)
    num_warps_values = _parse_values(args.num_warps, int)
    num_stages_values = _parse_values(args.num_stages, int)

    rows = []
    for (
        query_len,
        kv_len,
        heads,
        kv_heads,
        active_fraction,
        block_size,
        tile_size_q,
        num_warps,
        num_stages,
    ) in itertools.product(
        query_lens,
        kv_lens,
        heads_values,
        kv_head_values,
        active_fractions,
        block_sizes,
        tile_sizes_q,
        num_warps_values,
        num_stages_values,
    ):
        if query_len > kv_len:
            continue
        run_args = argparse.Namespace(**vars(args))
        run_args.block_size = block_size
        run_args.tile_size_q = tile_size_q
        run_args.num_warps = num_warps
        run_args.num_stages = num_stages
        try:
            rows.append(
                _profile_one(
                    run_args,
                    query_len=query_len,
                    kv_len=kv_len,
                    heads=heads,
                    kv_heads=kv_heads,
                    active_fraction=active_fraction,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "shape": {
                        "batch": args.batch,
                        "query_len": query_len,
                        "kv_len": kv_len,
                        "heads": heads,
                        "kv_heads": kv_heads,
                        "dim": args.dim,
                        "dtype": args.dtype,
                        "attention_type": args.attention_type,
                    },
                    "block_size": block_size,
                    "tile_size_q": tile_size_q,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                    "pattern": args.pattern,
                    "requested_active_fraction": active_fraction,
                    "causal_mode": args.causal_mode,
                }
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {"rows": rows}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
