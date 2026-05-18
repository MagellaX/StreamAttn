"""Compare FlashInfer single decode against StreamAttn decode paths.

This benchmark is deliberately narrow: batch size 1, query length 1, contiguous
KV cache, and FlashInfer's ``single_decode_with_kv_cache`` API. It is the first
direct baseline for the StreamAttn decode thesis: exact decode vs metadata-aware
StreamAttn sparse routing when synthetic sparsity exists.
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
    _dtype,
    _make_pattern,
    _metadata_update_uses_triton,
    _parse_values,
    _time_cuda,
    _time_wall_cuda,
)
from benchmarks.profile_stream_attn_decode_plan import _active_fraction, _error_metrics, _winner

try:
    import flashinfer

    HAS_FLASHINFER = True
except ImportError:
    flashinfer = None
    HAS_FLASHINFER = False


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


def _make_comparison_tensors(
    args,
    *,
    kv_len: int,
    heads: int,
    kv_heads: int,
    active_fraction: float,
):
    dtype = _dtype(args.dtype)
    if args.attention_type == "mha":
        logical_kv_heads = heads
    elif args.attention_type == "mqa":
        logical_kv_heads = 1
    else:
        logical_kv_heads = kv_heads
    if heads % logical_kv_heads != 0:
        raise ValueError("heads must be divisible by kv_heads for GQA/MQA")

    _, k_logical, v_logical, block_active = _make_pattern(
        batch=1,
        query_len=1,
        kv_len=kv_len,
        heads=logical_kv_heads,
        dim=args.dim,
        dtype=dtype,
        pattern=args.pattern,
        active_fraction=active_fraction,
        block_size=args.block_size,
        peak=args.peak,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )
    if args.pattern == "random":
        q = torch.randn(1, 1, heads, args.dim, device="cuda", dtype=dtype)
    else:
        q = torch.zeros(1, 1, heads, args.dim, device="cuda", dtype=dtype)
        q[..., 0] = args.peak

    if args.attention_type == "mha":
        k_stream = k_logical
        v_stream = v_logical
    else:
        group = heads // logical_kv_heads
        k_stream = k_logical.repeat_interleave(group, dim=2)
        v_stream = v_logical.repeat_interleave(group, dim=2)
    return q, k_logical, v_logical, k_stream, v_stream, logical_kv_heads, block_active


def _flashinfer_single_decode(q, k, v, *, use_tensor_cores: bool):
    q_fi = q[0, 0].contiguous()
    k_fi = k[0].contiguous()
    v_fi = v[0].contiguous()
    return flashinfer.decode.single_decode_with_kv_cache(
        q_fi,
        k_fi,
        v_fi,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        use_tensor_cores=use_tensor_cores,
    )


def _make_wrapper(args, q, k, *, logical_kv_heads: int) -> Optional[StreamAttnDecodeWrapper]:
    if not args.decode_cost_json:
        return None
    cost_model = DecodeCostModel.from_json(args.decode_cost_json)
    workspace = StreamAttnDecodeWorkspace.allocate(
        device=q.device,
        max_batch=1,
        max_query_len=max(1, args.tile_size_q),
        max_kv_len=k.shape[1],
        max_heads=q.shape[2],
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
    return wrapper


def _cost_model_hit(args, q, k, *, logical_kv_heads: int) -> Optional[bool]:
    if not args.decode_cost_json:
        return None
    key = DecodeCostKey.from_tensors(
        q,
        k,
        kv_heads=logical_kv_heads,
        attention_type=args.attention_type,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    return DecodeCostModel.from_json(args.decode_cost_json).lookup(key) is not None


def _profile_one(args, *, kv_len: int, heads: int, kv_heads: int, active_fraction: float) -> dict:
    if not HAS_FLASHINFER:
        raise RuntimeError(
            "FlashInfer is not installed. Use a CUDA 12.8 image and install "
            "flashinfer-python flashinfer-cubin."
        )
    q, k_logical, v_logical, k_stream, v_stream, logical_kv_heads, block_active = (
        _make_comparison_tensors(
            args,
            kv_len=kv_len,
            heads=heads,
            kv_heads=kv_heads,
            active_fraction=active_fraction,
        )
    )
    metadata = StreamAttnMetadataCache.from_value(v_stream, block_size=args.block_size, use_triton=True)
    new_v = v_stream[:, kv_len - 1 : kv_len, :, :].contiguous()
    metadata_update_wall_ms = _time_wall_cuda(
        lambda: metadata.update_value_bounds_(
            new_v,
            start_pos=kv_len - 1,
            use_triton=_metadata_update_uses_triton(args.metadata_update_backend),
        ),
        warmup=args.metadata_warmup,
        iters=args.metadata_iters,
    )

    dense_ms = _time_cuda(
        lambda: dense_attention_forward(q, k_stream, v_stream, causal=False),
        warmup=args.warmup,
        iters=args.iters,
    )
    dense_out = dense_attention_forward(q, k_stream, v_stream, causal=False)
    mass_ms = _time_cuda(
        lambda: stream_attn_gate1(
            q,
            k_stream,
            v_stream,
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
    mass_out, mass_info = stream_attn_gate1(
        q,
        k_stream,
        v_stream,
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
    value_out = None
    if not args.disable_value_bound:
        value_ms = _time_cuda(
            lambda: stream_attn_gate1(
                q,
                k_stream,
                v_stream,
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
        value_out = stream_attn_gate1(
            q,
            k_stream,
            v_stream,
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
        )

    flashinfer_ms = _time_cuda(
        lambda: _flashinfer_single_decode(q, k_logical, v_logical, use_tensor_cores=False),
        warmup=args.warmup,
        iters=args.iters,
    )
    flashinfer_out = _flashinfer_single_decode(
        q,
        k_logical,
        v_logical,
        use_tensor_cores=False,
    ).view(1, 1, heads, args.dim)
    flashinfer_tc_ms = None
    flashinfer_tc_out = None
    if args.flashinfer_tensor_cores:
        flashinfer_tc_ms = _time_cuda(
            lambda: _flashinfer_single_decode(q, k_logical, v_logical, use_tensor_cores=True),
            warmup=args.warmup,
            iters=args.iters,
        )
        flashinfer_tc_out = _flashinfer_single_decode(
            q,
            k_logical,
            v_logical,
            use_tensor_cores=True,
        ).view(1, 1, heads, args.dim)

    wrapper = _make_wrapper(args, q, k_stream, logical_kv_heads=logical_kv_heads)
    wrapper_ms = None
    wrapper_backend = None
    wrapper_reason = None
    wrapper_error = None
    if wrapper is not None:
        wrapper.observe_active_fraction(actual_active)
        wrapper_plan = wrapper.plan_step(q, k_stream, metadata=metadata)
        wrapper_ms = _time_cuda(
            lambda: wrapper.run(q, k_stream, v_stream, metadata=metadata),
            warmup=args.warmup,
            iters=args.iters,
        )
        wrapper_out = wrapper.run(q, k_stream, v_stream, metadata=metadata)
        wrapper_backend = wrapper_plan.backend
        wrapper_reason = wrapper_plan.reason
        wrapper_error = _error_metrics(wrapper_out, dense_out)

    stream_oracle_backend, stream_oracle_ms = _winner(
        {
            "streamattn_dense": dense_ms,
            "streamattn_mass": mass_ms,
            "streamattn_value_bound": value_ms,
            "streamattn_wrapper": wrapper_ms,
        }
    )
    best_flashinfer_ms = min(
        value for value in [flashinfer_ms, flashinfer_tc_ms] if value is not None
    )
    best_flashinfer_backend = (
        "flashinfer_single_tc"
        if flashinfer_tc_ms is not None and flashinfer_tc_ms <= flashinfer_ms
        else "flashinfer_single"
    )
    streamattn_oracle_wins = stream_oracle_ms < best_flashinfer_ms
    streamattn_wrapper_wins = (
        wrapper_ms is not None and wrapper_ms < best_flashinfer_ms
    )
    fair_mha_comparison = args.attention_type == "mha" and logical_kv_heads == heads
    return {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer_available": True,
        "shape": {
            "batch": 1,
            "query_len": 1,
            "kv_len": kv_len,
            "heads": heads,
            "kv_heads": logical_kv_heads,
            "dim": args.dim,
            "dtype": args.dtype,
            "attention_type": args.attention_type,
            "streamattn_kv_layout": (
                "expanded" if logical_kv_heads != heads else "mha"
            ),
            "flashinfer_kv_layout": "true_gqa" if logical_kv_heads != heads else "mha",
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
        "cost_model_hit": _cost_model_hit(args, q, k_stream, logical_kv_heads=logical_kv_heads),
        "flashinfer_single_ms": flashinfer_ms,
        "flashinfer_single_tc_ms": flashinfer_tc_ms,
        "best_flashinfer_backend": best_flashinfer_backend,
        "best_flashinfer_ms": best_flashinfer_ms,
        "streamattn_dense_ms": dense_ms,
        "streamattn_mass_ms": mass_ms,
        "streamattn_value_bound_ms": value_ms,
        "streamattn_wrapper_ms": wrapper_ms,
        "streamattn_wrapper_backend": wrapper_backend,
        "streamattn_wrapper_reason": wrapper_reason,
        "streamattn_oracle_backend": stream_oracle_backend,
        "streamattn_oracle_ms": stream_oracle_ms,
        "streamattn_oracle_wins": streamattn_oracle_wins,
        "streamattn_oracle_vs_flashinfer_speedup": (
            best_flashinfer_ms / stream_oracle_ms if stream_oracle_ms > 0.0 else None
        ),
        "streamattn_wrapper_wins": streamattn_wrapper_wins,
        "streamattn_wrapper_vs_flashinfer_speedup": (
            best_flashinfer_ms / wrapper_ms if wrapper_ms is not None and wrapper_ms > 0.0 else None
        ),
        "streamattn_wrapper_regret_pct": (
            max(0.0, wrapper_ms - stream_oracle_ms) / stream_oracle_ms
            if wrapper_ms is not None and stream_oracle_ms > 0.0
            else None
        ),
        "mass_error": _error_metrics(mass_out, dense_out),
        "value_bound_error": _error_metrics(value_out, dense_out) if value_out is not None else None,
        "wrapper_error": wrapper_error,
        "flashinfer_error": _error_metrics(flashinfer_out, dense_out),
        "flashinfer_tc_error": (
            _error_metrics(flashinfer_tc_out, dense_out)
            if flashinfer_tc_out is not None
            else None
        ),
        "fair_mha_comparison": fair_mha_comparison,
        "comparison_note": (
            "fair_mha"
            if fair_mha_comparison
            else "gqa_mqa_not_fair_streamattn_expands_kv"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-cost-json", default="")
    parser.add_argument("--kv-lens", nargs="+", default=["8192", "16384", "32768"])
    parser.add_argument("--heads", nargs="+", default=["16", "32"])
    parser.add_argument("--kv-heads", nargs="+", default=["16", "8", "1"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--attention-type", choices=["mha", "gqa", "mqa"], default="mha")
    parser.add_argument("--pattern", choices=["random", "peaked", "sink_local", "sliding_recent"], default="peaked")
    parser.add_argument("--active-fraction", nargs="+", default=["0.0625", "0.125", "0.25", "1.0"])
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
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
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
    if not HAS_FLASHINFER:
        raise RuntimeError(
            "FlashInfer is not installed. Use a CUDA 12.8 image and install "
            "flashinfer-python flashinfer-cubin."
        )
    torch.manual_seed(0)

    rows = []
    for kv_len, heads, kv_heads, active_fraction in itertools.product(
        _parse_values(args.kv_lens, int),
        _parse_values(args.heads, int),
        _parse_values(args.kv_heads, int),
        _parse_values(args.active_fraction, float),
    ):
        if args.attention_type == "mqa" and kv_heads != 1:
            continue
        try:
            rows.append(
                _profile_one(
                    args,
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
                        "batch": 1,
                        "query_len": 1,
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
                    "flashinfer_available": HAS_FLASHINFER,
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
