"""Profile paired serial RS CTAs against one dual-consumer TMA RS CTA."""

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
    decode_grouped2_resource_info,
)


SCHEMA = "streamattn.sm90_grouped_prefill_dual_consumer_floor.v1"
METHODS = ("serial_grouped2", "tma_grouped2")


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
    tiles_per_group: int,
) -> torch.Tensor:
    groups = q.size(0) // 2
    q_grouped = q.reshape(groups, 2, 64, 128).float()
    k_grouped = k.reshape(groups, tiles_per_group, 64, 128).float()
    v_grouped = v.reshape(groups, tiles_per_group, 64, 128).float()
    scores = torch.matmul(
        q_grouped.unsqueeze(2), k_grouped.unsqueeze(1).transpose(-1, -2)
    )
    probability = torch.softmax(scores / math.sqrt(128.0), dim=-1).to(
        torch.bfloat16
    )
    output = torch.matmul(probability.float(), v_grouped.unsqueeze(1))
    return output.sum(dim=(2, 3, 4)).flatten()


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
    extension = compile_grouped_prefill_epoch_floor_extension(
        cutlass_root=args.cutlass_root,
        build_dir=args.build_dir,
        verbose=args.verbose_build,
    )

    cells: list[dict[str, object]] = []
    first_q: torch.Tensor | None = None
    for tiles_per_group in _parse_ints(args.tiles_per_group):
        if tiles_per_group <= 0 or args.total_tiles % tiles_per_group:
            raise ValueError("each tiles_per_group must divide total_tiles")
        groups = args.total_tiles // tiles_per_group
        q = torch.randn((groups * 2, 64, 128), device="cuda", dtype=torch.bfloat16)
        first_q = q if first_q is None else first_q
        outputs = {
            name: torch.empty((groups * 2,), device="cuda", dtype=torch.float32)
            for name in METHODS
        }
        functions: dict[str, Callable[[], None]] = {
            "serial_grouped2": lambda: extension.epoch_rs_grouped2_serial_out(
                q, k, v, outputs["serial_grouped2"], tiles_per_group
            ),
            "tma_grouped2": lambda: extension.epoch_rs_grouped2_tma_out(
                q, k, v, outputs["tma_grouped2"], tiles_per_group
            ),
        }
        for fn in functions.values():
            fn()
        torch.cuda.synchronize()
        paired_max_abs = float(
            (outputs["serial_grouped2"] - outputs["tma_grouped2"])
            .abs()
            .max()
            .item()
        )
        reference_groups = min(2, groups)
        reference = _reference_checksums(
            q[: reference_groups * 2],
            k[: reference_groups * tiles_per_group],
            v[: reference_groups * tiles_per_group],
            tiles_per_group=tiles_per_group,
        )
        reference_scale = max(float(reference.abs().max().item()), 1.0e-6)
        reference_count = reference_groups * 2
        serial_reference_abs = float(
            (outputs["serial_grouped2"][:reference_count] - reference)
            .abs()
            .max()
            .item()
        )
        tma_reference_abs = float(
            (outputs["tma_grouped2"][:reference_count] - reference)
            .abs()
            .max()
            .item()
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
        serial_ms = modes["serial_grouped2"]["graph_device_floor"]["median_ms"]
        tma_ms = modes["tma_grouped2"]["graph_device_floor"]["median_ms"]
        cell = {
            "tiles_per_group": tiles_per_group,
            "groups": groups,
            "query_tiles": groups * 2,
            "modes": modes,
            "ratios": {"tma_grouped2_vs_serial_grouped2": serial_ms / tma_ms},
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
            "[dual-consumer-floor] "
            f"tiles/group={tiles_per_group} groups={groups} "
            f"TMA2/serial2={cell['ratios']['tma_grouped2_vs_serial_grouped2']:.4f}x "
            f"paired_abs={paired_max_abs:.6g}",
            flush=True,
        )

    assert first_q is not None
    resources = decode_grouped2_resource_info(
        extension.epoch_rs_grouped2_tma_resource_info(first_q, k, v)
    )
    zero_local_bytes = all(
        values["local_bytes_per_thread"] == 0 for values in resources.values()
    )
    all_paired = all(
        cell["correctness"]["serial_vs_tma_max_abs"] == 0.0 for cell in cells
    )
    ratios = [
        cell["ratios"]["tma_grouped2_vs_serial_grouped2"] for cell in cells
    ]
    result = {
        "schema": SCHEMA,
        "device": torch.cuda.get_device_name(),
        "compute_capability": "%d.%d" % torch.cuda.get_device_capability(),
        "geometry": {
            "qk": [64, 64, 128],
            "pv": [64, 128, 64],
            "dtype": "bfloat16",
            "serial_topology": "two_independent_128_thread_rs_ctas",
            "tma_topology": "one_producer_warpgroup_two_consumer_warpgroups",
            "queries_per_kv_group": 2,
            "k_stages": 2,
            "v_stages": 2,
        },
        "total_tiles": args.total_tiles,
        "resources": resources,
        "cells": cells,
        "gate": {
            "paired_correct": all_paired,
            "zero_local_bytes": zero_local_bytes,
            "all_cells_faster": min(ratios) > 1.0,
            "minimum_ratio": min(ratios),
            "decision": (
                "retain_same_cta_grouped_candidate"
                if all_paired and zero_local_bytes and min(ratios) > 1.0
                else "reject_same_cta_grouped_candidate"
            ),
        },
        "interpretation_contract": {
            "positive_gate": (
                "paired correctness, zero local bytes, and grouped TMA faster than "
                "two independent vectorized serial RS CTAs in every tested cell"
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
