"""Profile a serial RS epoch against a TMA producer/consumer RS epoch."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stream_attention.backends.sm90.grouped_prefill_epoch_floor import (  # noqa: E402
    compile_grouped_prefill_epoch_floor_extension,
    decode_resource_info,
    decode_tma_resource_info,
)


SCHEMA = "streamattn.sm90_grouped_prefill_tma_epoch_floor.v1"
METHODS = ("serial_rs", "tma_rs")


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


def _reference_checksums(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tiles_per_cta: int,
) -> torch.Tensor:
    ctas = q.size(0)
    k_tiles = k.reshape(ctas, tiles_per_cta, 64, 128).float()
    v_tiles = v.reshape(ctas, tiles_per_cta, 64, 128).float()
    scores = torch.matmul(q.float().unsqueeze(1), k_tiles.transpose(-1, -2))
    probability = torch.softmax(scores / math.sqrt(128.0), dim=-1).to(
        torch.bfloat16
    )
    output = torch.matmul(probability.float(), v_tiles)
    return output.sum(dim=(1, 2, 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tiles", type=int, default=4096)
    parser.add_argument("--tiles-per-cta", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("an SM90 GPU is required")
    if args.total_tiles <= 0:
        raise ValueError("total_tiles must be positive")

    torch.manual_seed(args.seed)
    k = torch.randn(
        (args.total_tiles, 64, 128), device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        (args.total_tiles, 64, 128), device="cuda", dtype=torch.bfloat16
    )
    extension = compile_grouped_prefill_epoch_floor_extension(
        cutlass_root=args.cutlass_root,
        build_dir=args.build_dir,
        verbose=args.verbose_build,
    )

    cells: list[dict[str, object]] = []
    for tiles_per_cta in _parse_ints(args.tiles_per_cta):
        if tiles_per_cta <= 0 or args.total_tiles % tiles_per_cta:
            raise ValueError("each tiles_per_cta must divide total_tiles")
        ctas = args.total_tiles // tiles_per_cta
        q = torch.randn((ctas, 64, 128), device="cuda", dtype=torch.bfloat16)
        outputs = {
            name: torch.empty((ctas,), device="cuda", dtype=torch.float32)
            for name in METHODS
        }
        functions: dict[str, Callable[[], None]] = {
            "serial_rs": lambda: extension.epoch_rs_reuse_q_out(
                q, k, v, outputs["serial_rs"], tiles_per_cta
            ),
            "tma_rs": lambda: extension.epoch_rs_tma_out(
                q, k, v, outputs["tma_rs"], tiles_per_cta
            ),
        }
        for fn in functions.values():
            fn()
        torch.cuda.synchronize()
        paired_max_abs = float(
            (outputs["serial_rs"] - outputs["tma_rs"]).abs().max().item()
        )
        reference_ctas = min(4, ctas)
        reference = _reference_checksums(
            q[:reference_ctas],
            k[: reference_ctas * tiles_per_cta],
            v[: reference_ctas * tiles_per_cta],
            tiles_per_cta=tiles_per_cta,
        )
        reference_scale = max(float(reference.abs().max().item()), 1.0e-6)
        serial_reference_abs = float(
            (outputs["serial_rs"][:reference_ctas] - reference).abs().max().item()
        )
        tma_reference_abs = float(
            (outputs["tma_rs"][:reference_ctas] - reference).abs().max().item()
        )
        graphs = {name: _graph_callable(fn) for name, fn in functions.items()}
        modes: dict[str, object] = {}
        for name in METHODS:
            modes[name] = {
                "eager": _stats(
                    _measure(
                        functions[name],
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                ),
                "graph_device_floor": _stats(
                    _measure(
                        graphs[name],
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                ),
            }
        serial_ms = modes["serial_rs"]["graph_device_floor"]["median_ms"]
        tma_ms = modes["tma_rs"]["graph_device_floor"]["median_ms"]
        cell = {
            "tiles_per_cta": tiles_per_cta,
            "ctas": ctas,
            "modes": modes,
            "ratios": {"tma_rs_vs_serial_rs": serial_ms / tma_ms},
            "correctness": {
                "serial_vs_tma_max_abs": paired_max_abs,
                "reference_max_abs": reference_scale,
                "serial_reference_max_abs": serial_reference_abs,
                "tma_reference_max_abs": tma_reference_abs,
                "serial_reference_max_rel": serial_reference_abs / reference_scale,
                "tma_reference_max_rel": tma_reference_abs / reference_scale,
            },
        }
        cells.append(cell)
        print(
            "[tma-epoch-floor] "
            f"tiles/cta={tiles_per_cta} ctas={ctas} "
            f"TMA_RS={cell['ratios']['tma_rs_vs_serial_rs']:.4f}x "
            f"paired_abs={paired_max_abs:.6g}",
            flush=True,
        )

    resources = decode_resource_info(extension.resource_info())
    tma_resources = decode_tma_resource_info(
        extension.epoch_rs_tma_resource_info(
            torch.empty((1, 64, 128), device="cuda", dtype=torch.bfloat16),
            k,
            v,
        )
    )
    result = {
        "schema": SCHEMA,
        "device": torch.cuda.get_device_name(),
        "compute_capability": "%d.%d" % torch.cuda.get_device_capability(),
        "geometry": {
            "qk": [64, 64, 128],
            "pv": [64, 128, 64],
            "dtype": "bfloat16",
            "topology": "one_producer_warpgroup_one_consumer_warpgroup",
            "k_stages": 2,
            "v_stages": 2,
        },
        "total_tiles": args.total_tiles,
        "resources": {
            "serial_rs": resources["kernels"]["epoch_rs_reuse_q"],
            "tma_rs": tma_resources,
            "serial_shared_bytes": resources["epoch_rs_shared_bytes"],
            "tma_shared_bytes": resources["tma_epoch_shared_bytes"],
        },
        "cells": cells,
        "interpretation_contract": {
            "positive_tma_gate": (
                "paired correctness, zero local bytes, no CTA-wide steady-state "
                "barrier, and tma_rs_vs_serial_rs > 1"
            ),
            "scope": "attention-epoch component floor; not a complete attention kernel",
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
