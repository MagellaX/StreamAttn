"""Strict paired gate for the SM80 phased-K/V exact D128 decode canary."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from benchmarks.profile_gate0_true_gqa import _dense_true_gqa
from benchmarks.profile_stream_attn_gate0_wrapper import _error, _time_cuda
from benchmarks.profile_tk_tensor_core_exact_decode import (
    _compile_extension,
    _find_or_clone_tk,
    _make_flashinfer_batched_exact_runner,
    _pack_kv_head_major,
)


RESULT_SCHEMA = "streamattn.sm80_d128_phased_kv_gate.v1"
MATRIX_SCHEMA = "streamattn.sm80_d128_phased_kv_gate.matrix.v1"


def _paired_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [float(trial["speedup_vs_flashinfer"]) for trial in trials]
    return {
        "trials": trials,
        "speedup_median": float(statistics.median(speedups)),
        "speedup_min": float(min(speedups)),
        "speedup_max": float(max(speedups)),
        "wins": sum(speedup > 1.0 for speedup in speedups),
        "trial_count": len(speedups),
    }


def _matrix_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.matrix_specs:
        return [{}]
    parsed = json.loads(args.matrix_specs)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--matrix-specs must be a non-empty JSON list")
    if not all(isinstance(spec, dict) for spec in parsed):
        raise ValueError("every matrix spec must be a JSON object")
    return parsed


def _cell_config(args: argparse.Namespace, spec: dict[str, Any]) -> SimpleNamespace:
    fields = (
        "batch",
        "q_heads",
        "kv_heads",
        "head_dim",
        "kv_len",
        "dtype",
        "seed",
        "num_chunks",
        "warmup",
        "paired_trials",
        "paired_iters",
        "flashinfer_page_size",
        "flashinfer_workspace_mb",
    )
    values = {field: spec.get(field, getattr(args, field)) for field in fields}
    values["name"] = str(spec.get("name", "cell"))
    values["production_plan"] = bool(args.production_plan)
    values["tk_root"] = str(args.resolved_tk_root)
    return SimpleNamespace(**values)


def _run_cell(config: SimpleNamespace, ext: Any) -> dict[str, Any]:
    if config.dtype != "bf16" or config.head_dim != 128:
        raise ValueError("the phased-K/V gate requires BF16 D128")
    if config.q_heads % config.kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    group_size = config.q_heads // config.kv_heads
    if group_size not in (4, 8):
        raise ValueError("the phased-K/V gate supports G4 or G8")
    if config.num_chunks <= 0 or config.num_chunks % 4:
        raise ValueError("num_chunks must be positive and divisible by four")
    if (config.kv_len // 16) % config.num_chunks:
        raise ValueError("num_chunks must divide kv_len/16")

    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    q = torch.randn(
        config.batch,
        config.q_heads,
        config.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        config.batch,
        config.kv_len,
        config.kv_heads,
        config.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    k_group = _pack_kv_head_major(k)
    v_group = _pack_kv_head_major(v)
    if config.production_plan:
        from stream_attention.backends.sm80.tk_grouped_exact import ExactDecodePlan

        query_4d = q[:, None]
        output_4d = torch.empty_like(query_4d)
        plan = ExactDecodePlan.build(
            query_4d,
            k_group,
            v_group,
            output=output_4d,
            tk_root=Path(config.tk_root),
            promoted_only=True,
        )

        def run_streamattn() -> torch.Tensor:
            return plan.run()[:, 0]

        workspace_bytes = plan.workspace_bytes
    else:
        grouped_chunks = config.num_chunks // 4
        partial_out = torch.empty(
            config.batch,
            config.kv_heads,
            grouped_chunks * 16,
            config.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        partial_lse = torch.empty(
            config.batch,
            config.kv_heads,
            grouped_chunks,
            16,
            device=device,
            dtype=torch.float32,
        )
        out = torch.empty_like(q)

        def run_streamattn() -> torch.Tensor:
            return ext.exact_decode_chunk_merged_staged_grouped_direct_out(
                q,
                k_group,
                v_group,
                partial_out,
                partial_lse,
                out,
                config.num_chunks,
            )

        workspace_bytes = partial_out.nbytes + partial_lse.nbytes

    run_flashinfer = _make_flashinfer_batched_exact_runner(
        q,
        k,
        v,
        page_size=config.flashinfer_page_size,
        workspace_mb=config.flashinfer_workspace_mb,
    )
    stream_output = run_streamattn().clone()
    flashinfer_output = run_flashinfer().clone()
    dense_reference = _dense_true_gqa(q[:, None], k, v)[:, 0]
    torch.cuda.synchronize(device)

    for _ in range(config.warmup):
        run_streamattn()
        run_flashinfer()
    torch.cuda.synchronize(device)

    trials: list[dict[str, Any]] = []
    for trial in range(config.paired_trials):
        if trial % 2 == 0:
            stream_ms = _time_cuda(
                run_streamattn,
                device=device,
                warmup=0,
                iters=config.paired_iters,
            )
            flashinfer_ms = _time_cuda(
                run_flashinfer,
                device=device,
                warmup=0,
                iters=config.paired_iters,
            )
            order = "streamattn_first"
        else:
            flashinfer_ms = _time_cuda(
                run_flashinfer,
                device=device,
                warmup=0,
                iters=config.paired_iters,
            )
            stream_ms = _time_cuda(
                run_streamattn,
                device=device,
                warmup=0,
                iters=config.paired_iters,
            )
            order = "flashinfer_first"
        trials.append(
            {
                "trial": trial,
                "order": order,
                "streamattn_ms": float(stream_ms),
                "flashinfer_ms": float(flashinfer_ms),
                "speedup_vs_flashinfer": float(flashinfer_ms / stream_ms),
            }
        )

    paired = _paired_summary(trials)
    stream_quality = _error(stream_output, dense_reference)
    flashinfer_quality = _error(flashinfer_output, dense_reference)
    quality_pass = float(stream_quality["max_abs_error"]) <= 5e-4
    strict_pass = bool(
        quality_pass
        and paired["wins"] == paired["trial_count"]
        and paired["speedup_min"] > 1.0
    )
    return {
        "schema": RESULT_SCHEMA,
        "name": config.name,
        "shape": {
            "batch": config.batch,
            "q_heads": config.q_heads,
            "kv_heads": config.kv_heads,
            "group_size": group_size,
            "head_dim": config.head_dim,
            "kv_len": config.kv_len,
            "dtype": config.dtype,
            "num_chunks": config.num_chunks,
            "producer_warps": 4,
            "producer_pipeline": "register_resident_kv_pair",
            "kv_layout": "B,Hkv,N,D",
        },
        "execution": {
            "path": "ExactDecodePlan.run" if config.production_plan else "extension_out",
            "workspace_bytes": int(workspace_bytes),
        },
        "quality": {
            "streamattn_vs_dense": stream_quality,
            "flashinfer_vs_dense": flashinfer_quality,
            "pass": quality_pass,
        },
        "paired": paired,
        "decision": "strict_pass" if strict_pass else "fallback",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--paired-trials", type=int, default=15)
    parser.add_argument("--paired-iters", type=int, default=100)
    parser.add_argument("--flashinfer-page-size", type=int, default=16)
    parser.add_argument("--flashinfer-workspace-mb", type=int, default=128)
    parser.add_argument("--matrix-specs", default="")
    parser.add_argument("--tk-root", default="")
    parser.add_argument("--checkout-dir", default="")
    parser.add_argument("--cuda-arch", default="sm_80")
    parser.add_argument("--torch-cuda-arch-list", default="8.0")
    parser.add_argument("--compile-verbose", action="store_true")
    parser.add_argument("--production-plan", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    tk_root = _find_or_clone_tk(args)
    args.resolved_tk_root = tk_root
    compile_start = time.perf_counter()
    if args.production_plan:
        from stream_attention.backends.sm80.tk_grouped_exact import (
            compile_grouped_exact_extension,
        )

        ext = compile_grouped_exact_extension(
            tk_root=tk_root,
            verbose=args.compile_verbose,
        )
    else:
        ext = _compile_extension(
            tk_root=tk_root,
            cuda_arch=args.cuda_arch,
            torch_cuda_arch_list=args.torch_cuda_arch_list,
            verbose=args.compile_verbose,
        )
    compile_s = time.perf_counter() - compile_start
    cells = [_run_cell(_cell_config(args, spec), ext) for spec in _matrix_specs(args)]
    output: dict[str, Any] = {
        "schema": MATRIX_SCHEMA,
        "timestamp_unix": time.time(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "compile_s": compile_s,
        "cells": cells,
        "strict_pass_count": sum(cell["decision"] == "strict_pass" for cell in cells),
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
