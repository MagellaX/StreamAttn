"""GPU benchmark for dense vs Gate-1 vs router-auto regret."""

import argparse
import json

import torch

from stream_attention import StreamAttnMetadataCache
from stream_attention.gate1 import make_route_request, stream_attn_gate1
from stream_attention.router import (
    CostEntry,
    CostKey,
    Gate1CostModel,
    StreamAttnPolicy,
    StreamAttnRouter,
    router_regret,
)
from stream_attention.telemetry import ActiveFractionTelemetry


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


def _make_peaked(args):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda")
    q = torch.zeros(args.batch, args.seq, args.heads, args.dim, device=device, dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.randn_like(q)
    num_blocks = (args.seq + args.block_size - 1) // args.block_size
    active_blocks = max(0, min(num_blocks, round(args.active_fraction * num_blocks)))
    active_tokens = min(args.seq, active_blocks * args.block_size)
    q[..., 0] = args.peak
    k[:, :active_tokens, :, 0] = args.peak
    k[:, active_tokens:, :, 0] = -args.peak
    return q, k, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--active-fraction", type=float, default=0.0625)
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--error-budget", type=float, default=1e-3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    q, k, v = _make_peaked(args)
    metadata = StreamAttnMetadataCache.from_value(v, block_size=args.block_size)
    request = make_route_request(
        q,
        k,
        causal=False,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        model_id="router-profile",
        layer_id=0,
        head_id=-1,
    )

    dense_ms = _time_cuda(
        lambda: stream_attn_gate1(
            q,
            k,
            v,
            causal=False,
            mode="dense",
            telemetry=False,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    gate1_ms = _time_cuda(
        lambda: stream_attn_gate1(
            q,
            k,
            v,
            causal=False,
            mode="gate1",
            metadata=metadata,
            error_budget=args.error_budget,
            block_size=args.block_size,
            tile_size_q=args.tile_size_q,
            telemetry=False,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )

    cost_model = Gate1CostModel()
    cost_model.update(
        CostKey.from_request(request),
        CostEntry(dense_ms=dense_ms, qk_only_ms=min(dense_ms, gate1_ms)),
    )
    router = StreamAttnRouter(
        policy=StreamAttnPolicy(min_confidence=0.7, history_min_observations=4),
        telemetry=ActiveFractionTelemetry(min_observations=4),
        cost_model=cost_model,
    )
    # Warm history with the known active pattern. A probe predictor will replace
    # this seeded path for cold-start benchmarking later.
    _, warm_info = stream_attn_gate1(
        q,
        k,
        v,
        causal=False,
        mode="gate1",
        metadata=metadata,
        error_budget=args.error_budget,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        router=router,
        request=request,
        telemetry=True,
        return_info=True,
    )
    for _ in range(3):
        router.observe(
            request,
            cta_pv_executed=warm_info.stats.cta_pv_executed,
            cta_tiles_total=warm_info.stats.cta_tiles_total,
        )

    auto_ms = _time_cuda(
        lambda: stream_attn_gate1(
            q,
            k,
            v,
            causal=False,
            mode="auto",
            router=router,
            metadata=metadata,
            request=request,
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
        metadata=metadata,
        request=request,
        error_budget=args.error_budget,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        telemetry=True,
        return_info=True,
    )
    regret_ms, regret_relative = router_regret(
        dense_ms=dense_ms,
        gate1_ms=gate1_ms,
        chosen_backend=auto_info.decision.backend,
    )

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "active_fraction_requested": args.active_fraction,
                "actual_active_fraction": auto_info.active_pv_fraction,
                "dense_ms": dense_ms,
                "gate1_ms": gate1_ms,
                "auto_ms": auto_ms,
                "oracle_ms": min(dense_ms, gate1_ms),
                "router_backend": auto_info.decision.backend,
                "router_reason": auto_info.decision.reason,
                "regret_ms": regret_ms,
                "regret_relative": regret_relative,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
