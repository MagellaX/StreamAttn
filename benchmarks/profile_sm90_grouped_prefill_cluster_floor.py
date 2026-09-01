"""Profile independent TMA loads against two-CTA TMA multicast."""

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

from stream_attention.backends.sm90.grouped_prefill_cluster_floor import (  # noqa: E402
    compile_grouped_prefill_cluster_floor_extension,
    decode_cluster_resource_info,
)


SCHEMA = "streamattn.sm90_grouped_prefill_cluster_floor.v1"
METHODS = ("independent", "multicast")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tiles", type=int, default=4096)
    parser.add_argument("--tiles-per-group", default="1,2,4,8")
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
    extension = compile_grouped_prefill_cluster_floor_extension(
        cutlass_root=args.cutlass_root,
        build_dir=args.build_dir,
        verbose=args.verbose_build,
    )

    cells: list[dict[str, object]] = []
    for tiles_per_group in _parse_ints(args.tiles_per_group):
        if tiles_per_group <= 0 or args.total_tiles % tiles_per_group:
            raise ValueError("each tiles_per_group must divide total_tiles")
        groups = args.total_tiles // tiles_per_group
        outputs = {
            name: torch.empty((groups * 2,), device="cuda", dtype=torch.float32)
            for name in METHODS
        }
        functions: dict[str, Callable[[], None]] = {
            "independent": lambda: extension.independent_out(
                k, v, outputs["independent"], tiles_per_group
            ),
            "multicast": lambda: extension.multicast_out(
                k, v, outputs["multicast"], tiles_per_group
            ),
        }
        for fn in functions.values():
            fn()
        torch.cuda.synchronize()
        paired_max_abs = float(
            (outputs["independent"] - outputs["multicast"]).abs().max().item()
        )
        pair_internal_max_abs = max(
            float((output[0::2] - output[1::2]).abs().max().item())
            for output in outputs.values()
        )
        reference = (k[:tiles_per_group].float().sum() + v[:tiles_per_group].float().sum())
        reference_abs = float(reference.abs().item())
        reference_scale = max(reference_abs, 1.0e-6)
        reference_errors = {
            name: float((output[0] - reference).abs().item()) for name, output in outputs.items()
        }
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
        independent_ms = modes["independent"]["graph_device_floor"]["median_ms"]
        multicast_ms = modes["multicast"]["graph_device_floor"]["median_ms"]
        cell = {
            "tiles_per_group": tiles_per_group,
            "groups": groups,
            "ctas": groups * 2,
            "modes": modes,
            "ratios": {"multicast_vs_independent": independent_ms / multicast_ms},
            "correctness": {
                "independent_vs_multicast_max_abs": paired_max_abs,
                "pair_internal_max_abs": pair_internal_max_abs,
                "reference_max_abs": reference_scale,
                "independent_reference_max_abs": reference_errors["independent"],
                "multicast_reference_max_abs": reference_errors["multicast"],
                "independent_reference_max_rel": (
                    reference_errors["independent"] / reference_scale
                ),
                "multicast_reference_max_rel": (
                    reference_errors["multicast"] / reference_scale
                ),
            },
        }
        cells.append(cell)
        print(
            "[cluster-floor] "
            f"tiles/group={tiles_per_group} groups={groups} "
            f"multicast/independent={cell['ratios']['multicast_vs_independent']:.4f}x "
            f"paired_abs={paired_max_abs:.6g}",
            flush=True,
        )

    resources = decode_cluster_resource_info(extension.resource_info(k, v))
    ratios = [cell["ratios"]["multicast_vs_independent"] for cell in cells]
    paired_correct = all(
        cell["correctness"]["independent_vs_multicast_max_abs"] == 0.0
        for cell in cells
    )
    zero_local_bytes = all(
        values["local_bytes_per_thread"] == 0 for values in resources.values()
    )
    viable = paired_correct and zero_local_bytes and min(ratios) >= 0.9
    result = {
        "schema": SCHEMA,
        "device": torch.cuda.get_device_name(),
        "compute_capability": "%d.%d" % torch.cuda.get_device_capability(),
        "geometry": {
            "dtype": "bfloat16",
            "kv_tile": [64, 128],
            "ctas_per_kv_group": 2,
            "independent_cluster_shape": [1, 1, 1],
            "multicast_cluster_shape": [2, 1, 1],
            "stages": 2,
            "threads_per_cta": 256,
            "consumer_threads_per_cta": 128,
        },
        "total_tiles": args.total_tiles,
        "resources": resources,
        "cells": cells,
        "gate": {
            "paired_correct": paired_correct,
            "zero_local_bytes": zero_local_bytes,
            "minimum_ratio": min(ratios),
            "within_ten_percent_all_cells": min(ratios) >= 0.9,
            "decision": (
                "build_cluster_attention_epoch" if viable else "reject_cluster_multicast"
            ),
        },
        "interpretation_contract": {
            "positive_gate": (
                "paired correctness, zero local bytes, and multicast at least 0.90x "
                "independent TMA in every tested cell"
            ),
            "scope": "K/V transport and shared-memory consumption floor only",
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
