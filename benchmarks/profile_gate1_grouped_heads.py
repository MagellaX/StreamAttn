"""Simulate aggregate versus grouped-head Gate-1 routing.

This is intentionally a simulation, not a launch-splitting implementation. It
uses per-head active PV fractions and the current cost model to estimate whether
grouped sparse/dense head launches are worth building.
"""

import argparse
import json

import torch

from stream_attention import StreamAttnMetadataCache
from stream_attention.gate1 import stream_attn_gate1
from stream_attention.router import CostEntry


def _parse_floats(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def _head_costs(entry: CostEntry, active_fraction: float, heads: int):
    dense_h = entry.dense_ms / heads
    qk_h = entry.qk_only_ms / heads
    pv_h = entry.pv_ms / heads
    overhead_h = entry.predicate_overhead_ms / heads
    gate1_h = qk_h + active_fraction * pv_h + overhead_h
    return dense_h, gate1_h


def simulate_grouped_heads(
    *,
    entry: CostEntry,
    per_head_active: list[float],
    safety_margin: float,
    aggregate_threshold: float,
    gather_tax_ms: float = 0.0,
):
    heads = len(per_head_active)
    aggregate_active = sum(per_head_active) / heads
    dense_all_ms = entry.dense_ms
    gate1_all_ms = entry.predict_gate1_ms(aggregate_active)

    aggregate_auto_backend = (
        "gate1"
        if aggregate_active <= aggregate_threshold
        and entry.profitable(aggregate_active, safety_margin=safety_margin)
        else "dense"
    )
    aggregate_auto_ms = gate1_all_ms if aggregate_auto_backend == "gate1" else dense_all_ms

    sparse_heads = []
    dense_heads = []
    grouped_oracle_ms = 0.0
    for head_idx, active in enumerate(per_head_active):
        dense_h, gate1_h = _head_costs(entry, active, heads)
        if gate1_h * safety_margin < dense_h:
            sparse_heads.append(head_idx)
            grouped_oracle_ms += gate1_h
        else:
            dense_heads.append(head_idx)
            grouped_oracle_ms += dense_h

    return {
        "heads": heads,
        "per_head_active": per_head_active,
        "aggregate_active": aggregate_active,
        "dense_all_ms": dense_all_ms,
        "gate1_all_ms": gate1_all_ms,
        "aggregate_auto_backend": aggregate_auto_backend,
        "aggregate_auto_ms": aggregate_auto_ms,
        "grouped_oracle_ms": grouped_oracle_ms,
        "grouped_oracle_with_tax_ms": grouped_oracle_ms + gather_tax_ms,
        "grouped_oracle_speedup_vs_dense": dense_all_ms / grouped_oracle_ms,
        "grouped_oracle_speedup_vs_aggregate_auto": aggregate_auto_ms
        / grouped_oracle_ms,
        "grouped_oracle_with_tax_speedup_vs_dense": dense_all_ms
        / (grouped_oracle_ms + gather_tax_ms),
        "grouped_oracle_with_tax_speedup_vs_aggregate_auto": aggregate_auto_ms
        / (grouped_oracle_ms + gather_tax_ms),
        "gather_tax_ms": gather_tax_ms,
        "sparse_heads": sparse_heads,
        "dense_heads": dense_heads,
        "num_sparse_heads": len(sparse_heads),
        "num_dense_heads": len(dense_heads),
    }


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


def _make_mixed_head_tensors(args, per_head_active: list[float]):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda")
    heads = len(per_head_active)
    q = torch.zeros(args.batch, args.seq, heads, args.dim, device=device, dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.randn_like(q)
    num_blocks = (args.seq + args.block_size - 1) // args.block_size

    actual_requested = []
    for head_idx, active_fraction in enumerate(per_head_active):
        active_blocks = max(0, min(num_blocks, round(active_fraction * num_blocks)))
        active_tokens = min(args.seq, active_blocks * args.block_size)
        q[:, :, head_idx, 0] = args.peak
        k[:, :active_tokens, head_idx, 0] = args.peak
        k[:, active_tokens:, head_idx, 0] = -args.peak
        actual_requested.append(active_blocks / num_blocks if num_blocks else 0.0)
    return q, k, v, actual_requested


def _measure_gather_scatter_tax(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sparse_heads: list[int],
    dense_heads: list[int],
    warmup: int,
    iters: int,
) -> float:
    groups = [group for group in (sparse_heads, dense_heads) if group]
    if not groups:
        return 0.0
    indices = [
        torch.tensor(group, device=q.device, dtype=torch.long)
        for group in groups
    ]
    out = torch.empty_like(q)

    def run_once():
        for index in indices:
            q_group = q.index_select(2, index).contiguous()
            k_group = k.index_select(2, index).contiguous()
            v_group = v.index_select(2, index).contiguous()
            tmp = torch.empty_like(q_group)
            out.index_copy_(2, index, tmp)
            # Keep tensors live through the timed region.
            _ = k_group, v_group

    return _time_cuda(run_once, warmup=warmup, iters=iters)


def _actual_raw_stats(args, entry: CostEntry, per_head_active: list[float]):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --source actual_raw_stats")
    torch.manual_seed(args.seed)
    q, k, v, block_quantized_active = _make_mixed_head_tensors(args, per_head_active)
    metadata = StreamAttnMetadataCache.from_value(v, block_size=args.block_size)
    _, info = stream_attn_gate1(
        q,
        k,
        v,
        causal=args.causal,
        mode="gate1",
        metadata=metadata,
        error_budget=args.error_budget,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        telemetry=True,
        return_info=True,
    )
    if info.stats is None or info.per_head_stats is None:
        raise RuntimeError("Gate-1 did not return raw stats")

    actual_per_head = [
        stats.active_pv_fraction for stats in info.per_head_stats
    ]
    gather_tax_ms = args.gather_tax_ms
    initial = simulate_grouped_heads(
        entry=entry,
        per_head_active=actual_per_head,
        safety_margin=args.safety_margin,
        aggregate_threshold=args.aggregate_threshold,
        gather_tax_ms=gather_tax_ms,
    )
    if args.measure_gather_tax:
        gather_tax_ms = _measure_gather_scatter_tax(
            q,
            k,
            v,
            sparse_heads=initial["sparse_heads"],
            dense_heads=initial["dense_heads"],
            warmup=args.warmup,
            iters=args.iters,
        )
        initial = simulate_grouped_heads(
            entry=entry,
            per_head_active=actual_per_head,
            safety_margin=args.safety_margin,
            aggregate_threshold=args.aggregate_threshold,
            gather_tax_ms=gather_tax_ms,
        )

    return {
        **initial,
        "source": "actual_raw_stats",
        "device": torch.cuda.get_device_name(0),
        "requested_per_head_active": per_head_active,
        "block_quantized_requested_per_head_active": block_quantized_active,
        "actual_per_head_active": actual_per_head,
        "actual_aggregate_active": info.stats.active_pv_fraction,
        "stats": {
            "cta_tiles_total": info.stats.cta_tiles_total,
            "cta_pv_executed": info.stats.cta_pv_executed,
            "cta_pv_skipped": info.stats.cta_pv_skipped,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["manual", "actual_raw_stats"], default="manual")
    parser.add_argument(
        "--per-head-active",
        default="0.05,0.08,0.80,0.92",
        help="Comma-separated active PV fraction per head.",
    )
    parser.add_argument("--dense-ms", type=float, default=0.10)
    parser.add_argument("--qk-only-ms", type=float, default=0.05)
    parser.add_argument("--predicate-overhead-ms", type=float, default=0.0)
    parser.add_argument("--safety-margin", type=float, default=1.10)
    parser.add_argument("--aggregate-threshold", type=float, default=0.30)
    parser.add_argument("--gather-tax-ms", type=float, default=0.0)
    parser.add_argument("--measure-gather-tax", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    entry = CostEntry(
        dense_ms=args.dense_ms,
        qk_only_ms=args.qk_only_ms,
        predicate_overhead_ms=args.predicate_overhead_ms,
    )
    per_head_active = _parse_floats(args.per_head_active)
    if args.source == "actual_raw_stats":
        result = _actual_raw_stats(args, entry, per_head_active)
    else:
        result = simulate_grouped_heads(
            entry=entry,
            per_head_active=per_head_active,
            safety_margin=args.safety_margin,
            aggregate_threshold=args.aggregate_threshold,
            gather_tax_ms=args.gather_tax_ms,
        )
        result["source"] = "manual"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
