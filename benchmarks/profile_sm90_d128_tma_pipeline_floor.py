"""Profile SM90 D128 cp.async and TMA data-movement floors.

The default workload matches the logical K/V tiles in the H100 D128/G4
B4/32K anchor: B * Hkv * (N / 64) = 4 * 8 * 512 = 16384 tiles.
"""

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

from stream_attention.backends.sm90.tma_pipeline_floor import (
    compile_tma_pipeline_floor_extension,
    decode_resource_info,
)


SCHEMA = "streamattn.sm90_d128_tma_pipeline_floor.v1"
TILE_BYTES = 64 * 128 * 2


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _measure(
    fn: Callable[[], None], *, warmup: int, iters: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / iters)
    return values


def _stats(values: list[float], logical_bytes: int) -> dict[str, object]:
    median = statistics.median(values)
    return {
        "samples_ms": values,
        "median_ms": median,
        "min_ms": min(values),
        "max_ms": max(values),
        "logical_tb_per_s": logical_bytes / median / 1.0e9,
    }


def _graph_callable(fn: Callable[[], None]) -> Callable[[], None]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tiles", type=int, default=16384)
    parser.add_argument("--tiles-per-cta", default="32,64,128")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--cutlass-root", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise RuntimeError(f"SM90 is required, got sm_{major}{minor}")
    if args.total_tiles <= 0:
        raise ValueError("total_tiles must be positive")

    torch.manual_seed(args.seed)
    rows = args.total_tiles * 64
    k = torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    extension = compile_tma_pipeline_floor_extension(
        cutlass_root=args.cutlass_root,
        build_dir=args.build_dir,
        verbose=args.verbose_build,
    )

    cells: list[dict[str, object]] = []
    for tiles_per_cta in _parse_ints(args.tiles_per_cta):
        if tiles_per_cta <= 0 or args.total_tiles % tiles_per_cta:
            raise ValueError("tiles_per_cta must divide total_tiles")
        ctas = args.total_tiles // tiles_per_cta
        outputs = {
            name: torch.empty((ctas,), device="cuda", dtype=torch.float32)
            for name in ("cp_async_k", "cp_async_kv", "tma_k", "tma_kv")
        }
        eager_fns: dict[str, Callable[[], None]] = {
            "cp_async_k": lambda: extension.cp_async_k_out(
                k, outputs["cp_async_k"], tiles_per_cta
            ),
            "cp_async_kv": lambda: extension.cp_async_kv_out(
                k, v, outputs["cp_async_kv"], tiles_per_cta
            ),
            "tma_k": lambda: extension.tma_k_out(
                k, outputs["tma_k"], tiles_per_cta
            ),
            "tma_kv": lambda: extension.tma_kv_out(
                k, v, outputs["tma_kv"], tiles_per_cta
            ),
        }
        for fn in eager_fns.values():
            fn()
        torch.cuda.synchronize()
        errors = {
            "k_max_abs": float(
                (outputs["cp_async_k"] - outputs["tma_k"]).abs().max().item()
            ),
            "kv_max_abs": float(
                (outputs["cp_async_kv"] - outputs["tma_kv"]).abs().max().item()
            ),
        }

        modes: dict[str, object] = {}
        graph_fns = {name: _graph_callable(fn) for name, fn in eager_fns.items()}
        for name, fn in eager_fns.items():
            logical_bytes = args.total_tiles * TILE_BYTES * (2 if "kv" in name else 1)
            eager = _stats(
                _measure(fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats),
                logical_bytes,
            )
            graph = _stats(
                _measure(
                    graph_fns[name],
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                ),
                logical_bytes,
            )
            modes[name] = {"eager": eager, "graph_device_floor": graph}

        cp_k = modes["cp_async_k"]["graph_device_floor"]["median_ms"]
        cp_kv = modes["cp_async_kv"]["graph_device_floor"]["median_ms"]
        tma_k = modes["tma_k"]["graph_device_floor"]["median_ms"]
        tma_kv = modes["tma_kv"]["graph_device_floor"]["median_ms"]
        cell = {
            "tiles_per_cta": tiles_per_cta,
            "producer_ctas": ctas,
            "logical_k_bytes": args.total_tiles * TILE_BYTES,
            "logical_kv_bytes": args.total_tiles * TILE_BYTES * 2,
            "correctness": errors,
            "modes": modes,
            "ratios": {
                "tma_vs_cp_k": cp_k / tma_k,
                "tma_vs_cp_kv": cp_kv / tma_kv,
                "independent_v_tma_cost_ratio": tma_kv / tma_k,
                "independent_v_cp_cost_ratio": cp_kv / cp_k,
            },
        }
        cells.append(cell)
        print(
            "[tma-floor] "
            f"tiles/cta={tiles_per_cta} ctas={ctas} "
            f"K={cell['ratios']['tma_vs_cp_k']:.4f}x "
            f"KV={cell['ratios']['tma_vs_cp_kv']:.4f}x "
            f"err={max(errors.values()):.6g}",
            flush=True,
        )

    resources = decode_resource_info(extension.resource_info(k, v, 64))
    result = {
        "schema": SCHEMA,
        "device": torch.cuda.get_device_name(),
        "compute_capability": f"{major}.{minor}",
        "dtype": "bfloat16",
        "head_dim": 128,
        "tile_shape": [64, 128],
        "total_tiles": args.total_tiles,
        "workload_mapping": {
            "batch": 4,
            "kv_heads": 8,
            "kv_len": 32768,
            "formula": "B * Hkv * (N / 64)",
        },
        "resources": resources,
        "cells": cells,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
