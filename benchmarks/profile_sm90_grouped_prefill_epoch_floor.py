"""Profile SS-PV and RS-PV grouped-prefill attention epochs on H100."""

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
)


SCHEMA = "streamattn.sm90_grouped_prefill_epoch_floor.v2"
METHODS = ("qk", "qk_softmax", "pv_ss", "pv_rs", "epoch_ss", "epoch_rs")


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
    p: torch.Tensor,
) -> dict[str, torch.Tensor]:
    score = torch.matmul(q.float(), k.float().transpose(-1, -2))
    probability = torch.softmax(score / math.sqrt(128.0), dim=-1).to(
        torch.bfloat16
    )
    epoch = torch.matmul(probability.float(), v.float().transpose(-1, -2))
    pv = torch.matmul(p.to(torch.bfloat16).float(), v.float().transpose(-1, -2))
    return {
        "pv": pv.sum(dim=(-2, -1)),
        "epoch": epoch.sum(dim=(-2, -1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tiles", type=int, default=4096)
    parser.add_argument("--tiles-per-cta", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
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
    q = torch.randn(
        (args.total_tiles, 64, 128), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn(
        (args.total_tiles, 128, 64), device="cuda", dtype=torch.bfloat16
    )
    p = torch.softmax(
        torch.randn((args.total_tiles, 64, 64), device="cuda"), dim=-1
    )
    extension = compile_grouped_prefill_epoch_floor_extension(
        cutlass_root=args.cutlass_root,
        build_dir=args.build_dir,
        verbose=args.verbose_build,
    )

    reference_tiles = min(4, args.total_tiles)
    reference = _reference_checksums(
        q[:reference_tiles], k[:reference_tiles], v[:reference_tiles], p[:reference_tiles]
    )
    reference_outputs = {
        name: torch.empty((reference_tiles,), device="cuda", dtype=torch.float32)
        for name in METHODS
    }
    extension.pv_ss_out(
        p[:reference_tiles], v[:reference_tiles], reference_outputs["pv_ss"], 1
    )
    extension.pv_rs_out(
        p[:reference_tiles], v[:reference_tiles], reference_outputs["pv_rs"], 1
    )
    extension.epoch_ss_out(
        q[:reference_tiles],
        k[:reference_tiles],
        v[:reference_tiles],
        reference_outputs["epoch_ss"],
        1,
    )
    extension.epoch_rs_out(
        q[:reference_tiles],
        k[:reference_tiles],
        v[:reference_tiles],
        reference_outputs["epoch_rs"],
        1,
    )
    torch.cuda.synchronize()
    pv_reference_scale = max(float(reference["pv"].abs().max().item()), 1.0e-6)
    epoch_reference_scale = max(
        float(reference["epoch"].abs().max().item()), 1.0e-6
    )
    pv_ss_reference_max_abs = float(
        (reference_outputs["pv_ss"] - reference["pv"]).abs().max().item()
    )
    pv_rs_reference_max_abs = float(
        (reference_outputs["pv_rs"] - reference["pv"]).abs().max().item()
    )
    epoch_ss_reference_max_abs = float(
        (reference_outputs["epoch_ss"] - reference["epoch"]).abs().max().item()
    )
    epoch_rs_reference_max_abs = float(
        (reference_outputs["epoch_rs"] - reference["epoch"]).abs().max().item()
    )
    correctness = {
        "pv_ss_vs_rs_max_abs": float(
            (reference_outputs["pv_ss"] - reference_outputs["pv_rs"])
            .abs()
            .max()
            .item()
        ),
        "epoch_ss_vs_rs_max_abs": float(
            (reference_outputs["epoch_ss"] - reference_outputs["epoch_rs"])
            .abs()
            .max()
            .item()
        ),
        "pv_reference_max_abs": pv_reference_scale,
        "epoch_reference_max_abs": epoch_reference_scale,
        "pv_ss_reference_max_abs": pv_ss_reference_max_abs,
        "pv_rs_reference_max_abs": pv_rs_reference_max_abs,
        "epoch_ss_reference_max_abs": epoch_ss_reference_max_abs,
        "epoch_rs_reference_max_abs": epoch_rs_reference_max_abs,
        "pv_ss_reference_max_rel": pv_ss_reference_max_abs / pv_reference_scale,
        "pv_rs_reference_max_rel": pv_rs_reference_max_abs / pv_reference_scale,
        "epoch_ss_reference_max_rel": (
            epoch_ss_reference_max_abs / epoch_reference_scale
        ),
        "epoch_rs_reference_max_rel": (
            epoch_rs_reference_max_abs / epoch_reference_scale
        ),
    }

    cells: list[dict[str, object]] = []
    for tiles_per_cta in _parse_ints(args.tiles_per_cta):
        if tiles_per_cta <= 0 or args.total_tiles % tiles_per_cta:
            raise ValueError("each tiles_per_cta must divide total_tiles")
        ctas = args.total_tiles // tiles_per_cta
        outputs = {
            name: torch.empty((ctas,), device="cuda", dtype=torch.float32)
            for name in METHODS
        }
        functions: dict[str, Callable[[], None]] = {
            "qk": lambda: extension.qk_out(q, k, outputs["qk"], tiles_per_cta),
            "qk_softmax": lambda: extension.qk_softmax_out(
                q, k, outputs["qk_softmax"], tiles_per_cta
            ),
            "pv_ss": lambda: extension.pv_ss_out(
                p, v, outputs["pv_ss"], tiles_per_cta
            ),
            "pv_rs": lambda: extension.pv_rs_out(
                p, v, outputs["pv_rs"], tiles_per_cta
            ),
            "epoch_ss": lambda: extension.epoch_ss_out(
                q, k, v, outputs["epoch_ss"], tiles_per_cta
            ),
            "epoch_rs": lambda: extension.epoch_rs_out(
                q, k, v, outputs["epoch_rs"], tiles_per_cta
            ),
        }
        for fn in functions.values():
            fn()
        torch.cuda.synchronize()
        graph_functions = {
            name: _graph_callable(fn) for name, fn in functions.items()
        }
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
                        graph_functions[name],
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                ),
            }
        medians = {
            name: modes[name]["graph_device_floor"]["median_ms"]
            for name in METHODS
        }
        softmax_increment = max(0.0, medians["qk_softmax"] - medians["qk"])
        cell = {
            "tiles_per_cta": tiles_per_cta,
            "producer_ctas": ctas,
            "modes": modes,
            "ratios": {
                "pv_rs_vs_ss": medians["pv_ss"] / medians["pv_rs"],
                "epoch_rs_vs_ss": medians["epoch_ss"] / medians["epoch_rs"],
                "qk_softmax_vs_qk": medians["qk_softmax"] / medians["qk"],
                "isolated_components_vs_epoch_rs": (
                    (medians["qk_softmax"] + medians["pv_rs"])
                    / medians["epoch_rs"]
                ),
            },
            "derived_ms": {
                "softmax_increment": softmax_increment,
                "isolated_qk_plus_pv": medians["qk"] + medians["pv_rs"],
                "isolated_qk_softmax_plus_pv": (
                    medians["qk_softmax"] + medians["pv_rs"]
                ),
            },
        }
        cells.append(cell)
        print(
            "[epoch-floor] "
            f"tiles/cta={tiles_per_cta} ctas={ctas} "
            f"PV_RS={cell['ratios']['pv_rs_vs_ss']:.4f}x "
            f"epoch_RS={cell['ratios']['epoch_rs_vs_ss']:.4f}x",
            flush=True,
        )

    resources = decode_resource_info(extension.resource_info())
    result = {
        "schema": SCHEMA,
        "device": torch.cuda.get_device_name(),
        "compute_capability": "%d.%d" % torch.cuda.get_device_capability(),
        "geometry": {
            "qk": [64, 64, 128],
            "pv": [64, 128, 64],
            "dtype": "bfloat16",
        },
        "total_tiles": args.total_tiles,
        "correctness": correctness,
        "resources": resources,
        "cells": cells,
        "interpretation_contract": {
            "pv_rs_positive": "pv_rs_vs_ss > 1 with zero local bytes",
            "epoch_rs_positive": "epoch_rs_vs_ss > 1 with paired correctness",
            "isolated_component_sum": (
                "diagnostic sum of separately launched kernels; not a theoretical "
                "overlap lower bound"
            ),
            "next_gate": "TMA producer/consumer overlap only if both are positive",
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
