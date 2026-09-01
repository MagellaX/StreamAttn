"""Profile serial, independent-TMA, and cluster-multicast attention epochs."""

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
    decode_cluster2_epoch_resource_info,
    decode_grouped2_resource_info,
)


SCHEMA = "streamattn.sm90_grouped_prefill_cluster_epoch_floor.v1"
METHODS = ("serial_grouped2", "independent_tma", "multicast_cluster")


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
            "independent_tma": lambda: extension.epoch_rs_cluster2_independent_out(
                q, k, v, outputs["independent_tma"], tiles_per_group
            ),
            "multicast_cluster": lambda: extension.epoch_rs_cluster2_multicast_out(
                q, k, v, outputs["multicast_cluster"], tiles_per_group
            ),
        }
        for fn in functions.values():
            fn()
        torch.cuda.synchronize()
        paired_max_abs = max(
            float((outputs[METHODS[0]] - outputs[name]).abs().max().item())
            for name in METHODS[1:]
        )
        reference_groups = min(2, groups)
        reference = _reference_checksums(
            q[: reference_groups * 2],
            k[: reference_groups * tiles_per_group],
            v[: reference_groups * tiles_per_group],
            tiles_per_group=tiles_per_group,
        )
        reference_count = reference_groups * 2
        reference_scale = max(float(reference.abs().max().item()), 1.0e-6)
        reference_errors = {
            name: float((output[:reference_count] - reference).abs().max().item())
            for name, output in outputs.items()
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
        serial_ms = modes["serial_grouped2"]["graph_device_floor"]["median_ms"]
        independent_ms = modes["independent_tma"]["graph_device_floor"]["median_ms"]
        multicast_ms = modes["multicast_cluster"]["graph_device_floor"]["median_ms"]
        cell = {
            "tiles_per_group": tiles_per_group,
            "groups": groups,
            "query_tiles": groups * 2,
            "modes": modes,
            "ratios": {
                "independent_tma_vs_serial": serial_ms / independent_ms,
                "multicast_vs_independent_tma": independent_ms / multicast_ms,
                "multicast_vs_serial": serial_ms / multicast_ms,
            },
            "correctness": {
                "all_methods_max_abs": paired_max_abs,
                "reference_max_abs": reference_scale,
                **{
                    f"{name}_reference_max_abs": error
                    for name, error in reference_errors.items()
                },
                **{
                    f"{name}_reference_max_rel": error / reference_scale
                    for name, error in reference_errors.items()
                },
            },
        }
        cells.append(cell)
        print(
            "[cluster-epoch-floor] "
            f"tiles/group={tiles_per_group} groups={groups} "
            f"mcast/ind={cell['ratios']['multicast_vs_independent_tma']:.4f}x "
            f"mcast/serial={cell['ratios']['multicast_vs_serial']:.4f}x "
            f"paired_abs={paired_max_abs:.6g}",
            flush=True,
        )

    assert first_q is not None
    grouped_resources = decode_grouped2_resource_info(
        extension.epoch_rs_grouped2_tma_resource_info(first_q, k, v)
    )
    cluster_resources = decode_cluster2_epoch_resource_info(
        extension.epoch_rs_cluster2_resource_info(first_q, k, v)
    )
    resources = {
        "serial_grouped2": grouped_resources["serial_grouped2"],
        **cluster_resources,
    }
    paired_correct = all(
        cell["correctness"]["all_methods_max_abs"] == 0.0 for cell in cells
    )
    zero_local_bytes = all(
        values["local_bytes_per_thread"] == 0 for values in resources.values()
    )
    serial_ratios = [cell["ratios"]["multicast_vs_serial"] for cell in cells]
    transport_ratios = [
        cell["ratios"]["multicast_vs_independent_tma"] for cell in cells
    ]
    positive = paired_correct and zero_local_bytes and min(serial_ratios) > 1.0
    result = {
        "schema": SCHEMA,
        "device": torch.cuda.get_device_name(),
        "compute_capability": "%d.%d" % torch.cuda.get_device_capability(),
        "geometry": {
            "qk": [64, 64, 128],
            "pv": [64, 128, 64],
            "dtype": "bfloat16",
            "queries_per_kv_group": 2,
            "cluster_shape": [2, 1, 1],
            "k_stages": 2,
            "v_stages": 2,
            "consumer_warpgroups_per_cta": 1,
        },
        "total_tiles": args.total_tiles,
        "resources": resources,
        "cells": cells,
        "gate": {
            "paired_correct": paired_correct,
            "zero_local_bytes": zero_local_bytes,
            "minimum_multicast_vs_independent_tma": min(transport_ratios),
            "minimum_multicast_vs_serial": min(serial_ratios),
            "all_cells_faster_than_serial": min(serial_ratios) > 1.0,
            "decision": (
                "build_complete_cluster_prefill" if positive else "reject_cluster_attention_epoch"
            ),
        },
        "interpretation_contract": {
            "positive_gate": (
                "paired correctness, zero local bytes, and cluster multicast "
                "faster than two vectorized serial RS CTAs in every tested cell"
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
