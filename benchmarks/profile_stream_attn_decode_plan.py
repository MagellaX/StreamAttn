"""Profile planned StreamAttn decode against dense/Gate-1/oracle.

This benchmark consumes a decode cost table exported from
``profile_gate1_decode.py`` and measures whether ``stream_attn_decode_plan`` plus
``stream_attn_decode_run`` picks the same backend as the measured oracle.
It is scoped to contiguous KV cache tensors and ``causal=False`` decode
mechanics.
"""

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import torch

from stream_attention import StreamAttnMetadataCache
from stream_attention.decode import (
    DecodeCostKey,
    DecodeCostModel,
    StreamAttnDecodePolicy,
    stream_attn_decode_plan,
    stream_attn_decode_run,
)
from stream_attention.gate1 import dense_attention_forward, stream_attn_gate1

from benchmarks.profile_gate1_decode import (
    _make_decode_tensors,
    _metadata_update_uses_triton,
    _parse_values,
    _time_cuda,
    _time_wall_cuda,
)


def _time_wall(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iters


def _winner(candidates: dict) -> tuple[str, float]:
    candidates = {name: value for name, value in candidates.items() if value is not None}
    return min(candidates.items(), key=lambda item: item[1])


def _active_fraction(info, fallback: float) -> float:
    if info is not None and info.stats is not None:
        return float(info.stats.active_pv_fraction)
    return float(fallback)


def _error_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = (out.float() - ref.float()).abs()
    ref_norm = ref.float().norm()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "relative_l2_error": float(
            (out.float() - ref.float()).norm().div(ref_norm.clamp_min(1.0e-20)).item()
        ),
    }


def _policy(args) -> StreamAttnDecodePolicy:
    return StreamAttnDecodePolicy(
        max_router_regret_pct=args.max_router_regret_pct,
        safety_margin=args.safety_margin,
        allow_mass=not args.disable_mass,
        allow_value_bound=not args.disable_value_bound,
        prefer_value_bound_if_within=args.prefer_value_bound_if_within,
        collect_telemetry_every=args.collect_telemetry_every,
        min_kv_len_for_gate1=args.min_kv_len_for_gate1,
        max_active_fraction_mass=args.max_active_fraction_mass,
        max_active_fraction_value_bound=args.max_active_fraction_value_bound,
        min_confidence=args.min_confidence,
        require_metadata_for_value_bound=not args.allow_value_bound_without_metadata,
    )


def _cost_key(args, q, k, *, logical_kv_heads: int) -> DecodeCostKey:
    return DecodeCostKey.from_tensors(
        q,
        k,
        kv_heads=logical_kv_heads,
        attention_type=args.attention_type,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def _profile_plan_mode(
    args,
    *,
    q,
    k,
    v,
    metadata,
    logical_kv_heads: int,
    block_active: float,
    actual_active: float,
    plan_mode: str,
    dense_ms: float,
    mass_ms: float,
    value_ms: Optional[float],
    oracle_backend: str,
    oracle_ms: float,
    dense_out,
    cost_model: DecodeCostModel,
) -> dict:
    active_hint = actual_active if plan_mode == "hint_plan" else None
    plan_args = dict(
        metadata=metadata,
        decode_cost_model=cost_model,
        policy=_policy(args),
        active_fraction_hint=active_hint,
        attention_type=args.attention_type,
        kv_heads=logical_kv_heads,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        error_budget=args.error_budget,
    )
    plan_wall_ms = _time_wall(
        lambda: stream_attn_decode_plan(q, k, **plan_args),
        warmup=args.plan_warmup,
        iters=args.plan_iters,
    )
    plan = stream_attn_decode_plan(q, k, **plan_args)
    plan_run_ms = _time_cuda(
        lambda: stream_attn_decode_run(
            q,
            k,
            v,
            plan=plan,
            metadata=metadata,
            error_budget=args.error_budget,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    plan_out = stream_attn_decode_run(
        q,
        k,
        v,
        plan=plan,
        metadata=metadata,
        error_budget=args.error_budget,
    )
    regret_raw_ms = plan_run_ms - oracle_ms
    regret_ms = max(0.0, regret_raw_ms)
    key = _cost_key(args, q, k, logical_kv_heads=logical_kv_heads)
    cost_entry = cost_model.lookup(key)
    return {
        "plan_mode": plan_mode,
        "cost_model_hit": cost_entry is not None,
        "plan_backend": plan.backend,
        "plan_reason": plan.reason,
        "plan_predicted_ms": plan.predicted_ms,
        "plan_wall_ms": plan_wall_ms,
        "plan_run_ms": plan_run_ms,
        "plan_total_wall_plus_run_ms": plan_wall_ms + plan_run_ms,
        "plan_active_threshold": plan.active_threshold,
        "plan_predicted_active_fraction": plan.predicted_active_fraction,
        "plan_expected_regret_ms": plan.expected_regret_ms,
        "plan_collect_telemetry": plan.collect_telemetry,
        "oracle_backend": oracle_backend,
        "oracle_ms": oracle_ms,
        "regret_raw_ms": regret_raw_ms,
        "regret_ms": regret_ms,
        "regret_pct": regret_ms / oracle_ms if oracle_ms > 0.0 else None,
        "actual_active_fraction": actual_active,
        "block_quantized_active_fraction": block_active,
        "dense_decode_ms": dense_ms,
        "gate1_mass_ms": mass_ms,
        "gate1_value_bound_ms": value_ms,
        "mass_dense_ratio": mass_ms / dense_ms if dense_ms > 0.0 else None,
        "value_dense_ratio": (
            value_ms / dense_ms if value_ms is not None and dense_ms > 0.0 else None
        ),
        "plan_error": _error_metrics(plan_out, dense_out),
    }


def _make_step_queries(args, *, steps: int, heads: int, dtype: torch.dtype) -> torch.Tensor:
    if args.pattern == "random":
        return torch.randn(args.batch, steps, heads, args.dim, device="cuda", dtype=dtype)
    q = torch.zeros(args.batch, steps, heads, args.dim, device="cuda", dtype=dtype)
    q[..., 0] = args.peak
    return q


def _profile_prev_token_plan(
    args,
    *,
    k,
    v,
    metadata,
    logical_kv_heads: int,
    block_active: float,
    cost_model: DecodeCostModel,
) -> Optional[dict]:
    if args.decode_steps <= 0:
        return None
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    q_steps = _make_step_queries(args, steps=args.decode_steps, heads=k.shape[2], dtype=dtype)
    previous_active = None
    rows = []
    policy = _policy(args)
    for step in range(args.decode_steps):
        q_step = q_steps[:, step : step + 1, :, :]
        dense_ms = _time_cuda(
            lambda: dense_attention_forward(q_step, k, v, causal=False),
            warmup=args.step_warmup,
            iters=args.step_iters,
        )
        mass_ms = _time_cuda(
            lambda: stream_attn_gate1(
                q_step,
                k,
                v,
                causal=False,
                mode="gate1",
                skip_predicate="mass",
                error_budget=args.error_budget,
                block_size=args.block_size,
                tile_size_q=args.tile_size_q,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                telemetry=False,
            ),
            warmup=args.step_warmup,
            iters=args.step_iters,
        )
        _, mass_info = stream_attn_gate1(
            q_step,
            k,
            v,
            causal=False,
            mode="gate1",
            skip_predicate="mass",
            error_budget=args.error_budget,
            block_size=args.block_size,
            tile_size_q=args.tile_size_q,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
            telemetry=False,
            return_info=True,
        )
        value_ms = None
        if not args.disable_value_bound:
            value_ms = _time_cuda(
                lambda: stream_attn_gate1(
                    q_step,
                    k,
                    v,
                    causal=False,
                    mode="gate1",
                    metadata=metadata,
                    skip_predicate="value_bound",
                    error_budget=args.error_budget,
                    block_size=args.block_size,
                    tile_size_q=args.tile_size_q,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                    telemetry=False,
                ),
                warmup=args.step_warmup,
                iters=args.step_iters,
            )
        oracle_backend, oracle_ms = _winner(
            {"dense": dense_ms, "gate1_mass": mass_ms, "gate1_value_bound": value_ms}
        )
        plan = stream_attn_decode_plan(
            q_step,
            k,
            metadata=metadata,
            decode_cost_model=cost_model,
            policy=policy,
            active_fraction_hint=previous_active,
            attention_type=args.attention_type,
            kv_heads=logical_kv_heads,
            block_size=args.block_size,
            tile_size_q=args.tile_size_q,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
            error_budget=args.error_budget,
            step_index=step,
        )
        plan_run_ms = _time_cuda(
            lambda: stream_attn_decode_run(
                q_step,
                k,
                v,
                plan=plan,
                metadata=metadata,
                error_budget=args.error_budget,
            ),
            warmup=args.step_warmup,
            iters=args.step_iters,
        )
        actual_active = _active_fraction(mass_info, block_active)
        regret_ms = max(0.0, plan_run_ms - oracle_ms)
        rows.append(
            {
                "step": step,
                "active_fraction_hint": previous_active,
                "actual_active_fraction": actual_active,
                "plan_backend": plan.backend,
                "plan_reason": plan.reason,
                "plan_predicted_ms": plan.predicted_ms,
                "plan_run_ms": plan_run_ms,
                "dense_ms": dense_ms,
                "gate1_mass_ms": mass_ms,
                "gate1_value_bound_ms": value_ms,
                "oracle_backend": oracle_backend,
                "oracle_ms": oracle_ms,
                "regret_ms": regret_ms,
                "regret_pct": regret_ms / oracle_ms if oracle_ms > 0.0 else None,
            }
        )
        previous_active = actual_active

    regrets = [row["regret_pct"] for row in rows if row.get("regret_pct") is not None]
    return {
        "plan_mode": "prev_token_plan",
        "steps": rows,
        "decode_steps": args.decode_steps,
        "mean_regret_pct": sum(regrets) / len(regrets) if regrets else None,
        "max_regret_pct": max(regrets) if regrets else None,
        "final_plan_backend": rows[-1]["plan_backend"] if rows else None,
    }


def _profile_one(args, *, query_len: int, kv_len: int, heads: int, kv_heads: int, active_fraction: float) -> list[dict]:
    q, k, v, logical_kv_heads, block_active = _make_decode_tensors(
        args,
        query_len=query_len,
        kv_len=kv_len,
        heads=heads,
        kv_heads=kv_heads,
        active_fraction=active_fraction,
    )
    metadata = StreamAttnMetadataCache.from_value(v, block_size=args.block_size, use_triton=True)
    update_tokens = min(query_len, kv_len)
    new_v = v[:, kv_len - update_tokens : kv_len, :, :].contiguous()
    update_start = kv_len - update_tokens
    metadata_update_wall_ms = _time_wall_cuda(
        lambda: metadata.update_value_bounds_(
            new_v,
            start_pos=update_start,
            use_triton=_metadata_update_uses_triton(args.metadata_update_backend),
        ),
        warmup=args.metadata_warmup,
        iters=args.metadata_iters,
    )

    dense_ms = _time_cuda(
        lambda: dense_attention_forward(q, k, v, causal=False),
        warmup=args.warmup,
        iters=args.iters,
    )
    dense_out = dense_attention_forward(q, k, v, causal=False)
    mass_ms = _time_cuda(
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
            num_warps=args.num_warps,
            num_stages=args.num_stages,
            telemetry=False,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    _, mass_info = stream_attn_gate1(
        q,
        k,
        v,
        causal=False,
        mode="gate1",
        skip_predicate="mass",
        error_budget=args.error_budget,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        telemetry=False,
        return_info=True,
    )
    actual_active = _active_fraction(mass_info, block_active)
    value_ms = None
    if not args.disable_value_bound:
        value_ms = _time_cuda(
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
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                telemetry=False,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

    oracle_backend, oracle_ms = _winner(
        {"dense": dense_ms, "gate1_mass": mass_ms, "gate1_value_bound": value_ms}
    )
    cost_model = DecodeCostModel.from_json(args.decode_cost_json)
    key = _cost_key(args, q, k, logical_kv_heads=logical_kv_heads)
    cost_model_hit = cost_model.lookup(key) is not None

    base = {
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
            "expansion_factor": heads / logical_kv_heads,
        },
        "block_size": args.block_size,
        "tile_size_q": args.tile_size_q,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "pattern": args.pattern,
        "requested_active_fraction": active_fraction,
        "block_quantized_active_fraction": block_active,
        "actual_active_fraction": actual_active,
        "metadata_update_backend": args.metadata_update_backend,
        "metadata_update_wall_ms": metadata_update_wall_ms,
        "cost_model_hit": cost_model_hit,
    }

    rows = []
    for plan_mode in args.plan_modes:
        if plan_mode == "prev_token_plan":
            continue
        row = _profile_plan_mode(
            args,
            q=q,
            k=k,
            v=v,
            metadata=metadata,
            logical_kv_heads=logical_kv_heads,
            block_active=block_active,
            actual_active=actual_active,
            plan_mode=plan_mode,
            dense_ms=dense_ms,
            mass_ms=mass_ms,
            value_ms=value_ms,
            oracle_backend=oracle_backend,
            oracle_ms=oracle_ms,
            dense_out=dense_out,
            cost_model=cost_model,
        )
        rows.append({**base, **row})

    if "prev_token_plan" in args.plan_modes and query_len == 1:
        prev = _profile_prev_token_plan(
            args,
            k=k,
            v=v,
            metadata=metadata,
            logical_kv_heads=logical_kv_heads,
            block_active=block_active,
            cost_model=cost_model,
        )
        if prev is not None:
            rows.append(
                {
                    **base,
                    "plan_mode": "prev_token_plan",
                    "plan_backend": prev.get("final_plan_backend"),
                    "plan_reason": "previous_token_simulation",
                    "plan_predicted_ms": None,
                    "plan_wall_ms": None,
                    "plan_run_ms": None,
                    "oracle_backend": oracle_backend,
                    "oracle_ms": oracle_ms,
                    "regret_pct": prev.get("mean_regret_pct"),
                    "prev_token_step_summary": prev,
                    "dense_decode_ms": dense_ms,
                    "gate1_mass_ms": mass_ms,
                    "gate1_value_bound_ms": value_ms,
                }
            )

    torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-cost-json", required=True)
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
    parser.add_argument("--plan-modes", nargs="+", default=["cold_plan", "hint_plan", "prev_token_plan"])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--tile-size-q", type=int, default=16)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--plan-warmup", type=int, default=10)
    parser.add_argument("--plan-iters", type=int, default=100)
    parser.add_argument("--metadata-warmup", type=int, default=3)
    parser.add_argument("--metadata-iters", type=int, default=8)
    parser.add_argument("--metadata-update-backend", choices=["auto", "triton", "torch"], default="auto")
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--step-warmup", type=int, default=1)
    parser.add_argument("--step-iters", type=int, default=3)
    parser.add_argument("--safety-margin", type=float, default=1.10)
    parser.add_argument("--max-router-regret-pct", type=float, default=0.05)
    parser.add_argument("--prefer-value-bound-if-within", type=float, default=1.10)
    parser.add_argument("--collect-telemetry-every", type=int, default=0)
    parser.add_argument("--min-kv-len-for-gate1", type=int, default=4096)
    parser.add_argument("--max-active-fraction-mass", type=float, default=0.35)
    parser.add_argument("--max-active-fraction-value-bound", type=float, default=0.30)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--disable-mass", action="store_true")
    parser.add_argument("--disable-value-bound", action="store_true")
    parser.add_argument("--allow-value-bound-without-metadata", action="store_true")
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

    valid_plan_modes = {"cold_plan", "hint_plan", "prev_token_plan"}
    args.plan_modes = _parse_values(args.plan_modes, str)
    unknown = set(args.plan_modes) - valid_plan_modes
    if unknown:
        raise ValueError(f"unknown plan modes: {sorted(unknown)}")

    rows = []
    for query_len, kv_len, heads, kv_heads, active_fraction in itertools.product(
        query_lens,
        kv_lens,
        heads_values,
        kv_head_values,
        active_fractions,
    ):
        if query_len > kv_len:
            continue
        try:
            rows.extend(
                _profile_one(
                    args,
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
                    "block_size": args.block_size,
                    "tile_size_q": args.tile_size_q,
                    "pattern": args.pattern,
                    "requested_active_fraction": active_fraction,
                }
            )
        finally:
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
