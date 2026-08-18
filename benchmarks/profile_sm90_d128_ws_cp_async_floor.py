"""Compare cooperative and warp-specialized cp.async QK+PV floors on H100."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stream_attention.backends.sm90.transposed_gqa_exact import (  # noqa: E402
    compile_transposed_gqa_exact_extension,
)


SCHEMA = "streamattn.sm90_d128_ws_cp_async_qkpv_floor.v1"
RESOURCE_FIELDS = (
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "blocks_per_sm",
    "max_threads_per_block",
)


def _parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _measure(
    fn: Callable[[], None], *, warmup: int, iters: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return samples


def _stats(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _graph_callable(fn: Callable[[], None]) -> Callable[[], None]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def _decode_resources(values: torch.Tensor) -> dict[str, object]:
    raw = [int(value) for value in values.cpu().tolist()]
    if len(raw) != 12:
        raise ValueError(f"expected 12 resource values, got {len(raw)}")
    return {
        "cooperative": dict(zip(RESOURCE_FIELDS, raw[:5])),
        "warp_specialized": dict(zip(RESOURCE_FIELDS, raw[5:10])),
        "cooperative_storage_bytes": raw[10],
        "warp_specialized_storage_bytes": raw[11],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--splits", default="4,8,16")
    parser.add_argument("--consumer-registers", default="96,112,128,160")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("an SM90 GPU is required")
    if args.kv_len % 64:
        raise ValueError("kv_len must be divisible by 64")

    torch.manual_seed(args.seed)
    q_group = torch.zeros(
        (args.batch, args.kv_heads, 8, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_group[:, :, :4].normal_()
    k = torch.randn(
        (args.batch, args.kv_heads, args.kv_len, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    extension = compile_transposed_gqa_exact_extension(
        cutlass_root=args.cutlass_root,
        build_dir=args.build_dir,
        head_dim=128,
        verbose=args.verbose_build,
    )

    cells: list[dict[str, object]] = []
    groups = args.batch * args.kv_heads
    register_options = _parse_ints(args.consumer_registers)
    for splits in _parse_ints(args.splits):
        cooperative_out = torch.empty(
            (groups, splits), device="cuda", dtype=torch.float32
        )

        def cooperative() -> None:
            extension.qkpv_async_checksum_out(
                q_group, k, v, cooperative_out, splits
            )

        cooperative()
        eager_coop = _stats(
            _measure(
                cooperative,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
        )
        graph_coop = _stats(
            _measure(
                _graph_callable(cooperative),
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
        )
        for consumer_registers in register_options:
            specialized_out = torch.empty_like(cooperative_out)

            def specialized(consumer_registers: int = consumer_registers) -> None:
                extension.qkpv_ws_cp_async_checksum_out(
                    q_group,
                    k,
                    v,
                    specialized_out,
                    splits,
                    consumer_registers,
                )

            specialized()
            torch.cuda.synchronize()
            error = {
                "max_abs": float(
                    (cooperative_out - specialized_out).abs().max().item()
                ),
                "max_rel": float(
                    (
                        (cooperative_out - specialized_out).abs()
                        / cooperative_out.abs().clamp_min(1.0e-6)
                    )
                    .max()
                    .item()
                ),
            }
            eager_ws = _stats(
                _measure(
                    specialized,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
            )
            graph_ws = _stats(
                _measure(
                    _graph_callable(specialized),
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
            )
            cell = {
                "splits": splits,
                "consumer_registers": consumer_registers,
                "producer_ctas": groups * splits,
                "tiles_per_cta": (args.kv_len // 64 + splits - 1) // splits,
                "correctness": error,
                "cooperative": {
                    "eager": eager_coop,
                    "graph_device_floor": graph_coop,
                },
                "warp_specialized": {
                    "eager": eager_ws,
                    "graph_device_floor": graph_ws,
                },
                "speedup": {
                    "eager": eager_coop["median_ms"] / eager_ws["median_ms"],
                    "graph_device_floor": (
                        graph_coop["median_ms"] / graph_ws["median_ms"]
                    ),
                },
            }
            cells.append(cell)
            print(
                f"[ws-cp] splits={splits} regs={consumer_registers} "
                f"ctas={groups * splits} "
                f"speedup={cell['speedup']['graph_device_floor']:.4f}x "
                f"max_abs={error['max_abs']:.6g}",
                flush=True,
            )

    result = {
        "schema": SCHEMA,
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "q_heads": args.kv_heads * 4,
            "kv_heads": args.kv_heads,
            "group_size": 4,
            "kv_len": args.kv_len,
            "head_dim": 128,
            "dtype": "bfloat16",
        },
        "resources": {
            str(registers): _decode_resources(
                extension.qkpv_floor_resource_info(registers)
            )
            for registers in register_options
        },
        "cells": cells,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
