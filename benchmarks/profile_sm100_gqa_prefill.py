"""Profile native Blackwell causal GQA prefill against Flash SDPA."""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from stream_attention.backends.sm100.gqa_prefill import (
    Sm100GqaPrefillPlan,
    TILE_VARIANTS,
)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None


def _elapsed_ms(fn: Callable[[], None], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _stabilize_tensor_core_clocks(iterations: int) -> None:
    if iterations <= 0:
        return
    lhs = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    rhs = torch.randn_like(lhs)
    output = torch.empty_like(lhs)
    for _ in range(iterations):
        torch.mm(lhs, rhs, out=output)
    torch.cuda.synchronize()


def _flash_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
        dropout_p=0.0,
        enable_gqa=True,
    ).transpose(1, 2)


def _flash_backend():
    if sdpa_kernel is None or SDPBackend is None:
        return contextlib.nullcontext()
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--tiles", nargs="+", default=list(TILE_VARIANTS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--clock-warmup-iters", type=int, default=256)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("an SM100a B200-class GPU is required")
    unknown = sorted(set(args.tiles) - set(TILE_VARIANTS))
    if unknown:
        raise ValueError(f"unknown tile variants: {unknown}")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    _stabilize_tensor_core_clocks(args.clock_warmup_iters)

    rows: list[dict[str, object]] = []
    for batch in args.batches:
        for seq_len in args.seq_lens:
            q = torch.randn(
                batch, seq_len, 16, 128, device="cuda", dtype=torch.bfloat16
            )
            k = torch.randn(
                batch, seq_len, 2, 128, device="cuda", dtype=torch.bfloat16
            )
            v = torch.randn_like(k)
            with torch.no_grad():
                with _flash_backend():
                    flash_output = _flash_sdpa(q, k, v)
                for tile in args.tiles:
                    row: dict[str, object] = {
                        "batch": batch,
                        "seq_len": seq_len,
                        "q_heads": 16,
                        "kv_heads": 2,
                        "group_size": 8,
                        "head_dim": 128,
                        "tile": tile,
                    }
                    try:
                        plan = Sm100GqaPrefillPlan.build(
                            q,
                            k,
                            v,
                            tile=tile,
                            build_dir=args.build_dir,
                        )
                        output = plan.run()
                        torch.cuda.synchronize()
                        flash_samples: list[float] = []
                        candidate_samples: list[float] = []
                        for repeat in range(args.repeats):
                            if repeat % 2 == 0:
                                with _flash_backend():
                                    flash_samples.append(
                                        _elapsed_ms(
                                            lambda: _flash_sdpa(q, k, v),
                                            warmup=args.warmup,
                                            iterations=args.iterations,
                                        )
                                    )
                                candidate_samples.append(
                                    _elapsed_ms(
                                        plan.run,
                                        warmup=args.warmup,
                                        iterations=args.iterations,
                                    )
                                )
                            else:
                                candidate_samples.append(
                                    _elapsed_ms(
                                        plan.run,
                                        warmup=args.warmup,
                                        iterations=args.iterations,
                                    )
                                )
                                with _flash_backend():
                                    flash_samples.append(
                                        _elapsed_ms(
                                            lambda: _flash_sdpa(q, k, v),
                                            warmup=args.warmup,
                                            iterations=args.iterations,
                                        )
                                    )
                        paired_speedups = [
                            flash / candidate
                            for flash, candidate in zip(
                                flash_samples, candidate_samples
                            )
                        ]
                        flash_ms = statistics.median(flash_samples)
                        candidate_ms = statistics.median(candidate_samples)
                        row.update(
                            status="ok",
                            flash_ms=flash_ms,
                            candidate_ms=candidate_ms,
                            speedup_vs_flash=statistics.median(paired_speedups),
                            flash_samples_ms=flash_samples,
                            candidate_samples_ms=candidate_samples,
                            paired_speedups=paired_speedups,
                            max_abs_error=float(
                                (output - flash_output).abs().max().item()
                            ),
                        )
                    except Exception as exc:
                        row.update(
                            status="error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    rows.append(row)
                    print(json.dumps(row), flush=True)

    result = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "rows": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
