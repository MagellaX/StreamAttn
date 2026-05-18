"""Profile the stateful StreamAttn decode wrapper.

This benchmark validates the FlashInfer-style wrapper layer over the existing
planned decode functions. It remains scoped to contiguous KV cache tensors and
``causal=False`` decode mechanics.
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Optional

import torch

from stream_attention import StreamAttnMetadataCache
from stream_attention.decode import (
    DecodeCostKey,
    DecodeCostModel,
    StreamAttnDecodePolicy,
    StreamAttnDecodeWorkspace,
    StreamAttnDecodeWrapper,
)
from stream_attention.gate1 import dense_attention_forward, stream_attn_gate1

from benchmarks.profile_gate1_decode import (
    _make_decode_tensors,
    _metadata_update_uses_triton,
    _parse_values,
    _time_cuda,
    _time_wall_cuda,
)
from benchmarks.profile_stream_attn_decode_plan import _active_fraction, _error_metrics, _winner


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


def _cost_key(args, q: torch.Tensor, k: torch.Tensor, *, logical_kv_heads: int) -> DecodeCostKey:
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


def _profile_one(
    args,
    *,
    query_len: int,
    kv_len: int,
    heads: int,
    kv_heads: int,
    active_fraction: float,
) -> dict:
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
    metadata_update_wall_ms = _time_wall_cuda(
        lambda: metadata.update_value_bounds_(
            new_v,
            start_pos=kv_len - update_tokens,
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
    value_ms: Optional[float] = None
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
    workspace = StreamAttnDecodeWorkspace.allocate(
        device=q.device,
        max_batch=args.batch,
        max_query_len=max(query_len, args.tile_size_q),
        max_kv_len=kv_len,
        max_heads=heads,
        head_dim=args.dim,
        block_size=args.block_size,
        dtype=q.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(
        workspace,
        policy=_policy(args),
        decode_cost_model=cost_model,
    )
    wrapper.plan(
        query_shape=q.shape,
        kv_shape=k.shape,
        attention_type=args.attention_type,
        kv_heads=logical_kv_heads,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        error_budget=args.error_budget,
    )
    wrapper.observe_active_fraction(actual_active)
    wrapper_plan = wrapper.plan_step(q, k, metadata=metadata)
    wrapper_run_ms = _time_cuda(
        lambda: wrapper.run(q, k, v, metadata=metadata),
        warmup=args.warmup,
        iters=args.iters,
    )
    wrapper_out = wrapper.run(q, k, v, metadata=metadata)
    wrapper_backend = wrapper.last_plan.backend if wrapper.last_plan is not None else None
    wrapper_reason = wrapper.last_plan.reason if wrapper.last_plan is not None else None
    previous_token_regret_ms = max(0.0, wrapper_run_ms - oracle_ms)

    key = _cost_key(args, q, k, logical_kv_heads=logical_kv_heads)
    cost_model_hit = cost_model.lookup(key) is not None
    return {
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
        "actual_active_fraction": actual_active,
        "block_quantized_active_fraction": block_active,
        "metadata_update_backend": args.metadata_update_backend,
        "metadata_update_wall_ms": metadata_update_wall_ms,
        "cost_model_hit": cost_model_hit,
        "dense_ms": dense_ms,
        "gate1_mass_ms": mass_ms,
        "gate1_value_bound_ms": value_ms,
        "oracle_backend": oracle_backend,
        "oracle_ms": oracle_ms,
        "wrapper_backend": wrapper_backend,
        "wrapper_reason": wrapper_reason,
        "wrapper_plan_backend": wrapper_plan.backend,
        "wrapper_plan_reason": wrapper_plan.reason,
        "wrapper_predicted_ms": wrapper_plan.predicted_ms,
        "predicted_active_fraction": wrapper_plan.predicted_active_fraction,
        "wrapper_run_ms": wrapper_run_ms,
        "regret_ms": previous_token_regret_ms,
        "regret_pct": previous_token_regret_ms / oracle_ms if oracle_ms > 0.0 else None,
        "previous_token_regret_pct": previous_token_regret_ms / oracle_ms if oracle_ms > 0.0 else None,
        "wrapper_error": _error_metrics(wrapper_out, dense_out),
        "last_active_fraction": wrapper.last_active_fraction,
        "workspace_step_index": wrapper.workspace.step_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-cost-json", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-lens", nargs="+", default=["1"])
    parser.add_argument("--kv-lens", nargs="+", default=["8192", "16384"])
    parser.add_argument("--heads", nargs="+", default=["16"])
    parser.add_argument("--kv-heads", nargs="+", default=["16"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--attention-type", choices=["mha", "gqa", "mqa"], default="mha")
    parser.add_argument("--pattern", choices=["random", "peaked", "sink_local", "sliding_recent"], default="peaked")
    parser.add_argument("--active-fraction", nargs="+", default=["0.0625", "0.25", "1.0"])
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
    parser.add_argument("--metadata-warmup", type=int, default=3)
    parser.add_argument("--metadata-iters", type=int, default=8)
    parser.add_argument("--metadata-update-backend", choices=["auto", "triton", "torch"], default="auto")
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

    rows = []
    for query_len, kv_len, heads, kv_heads, active_fraction in itertools.product(
        _parse_values(args.query_lens, int),
        _parse_values(args.kv_lens, int),
        _parse_values(args.heads, int),
        _parse_values(args.kv_heads, int),
        _parse_values(args.active_fraction, float),
    ):
        if query_len > kv_len:
            continue
        try:
            rows.append(
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
