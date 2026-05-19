"""Microbenchmark one Gate-0 projection scan kernel.

This target is intentionally narrower than ``profile_gate0_candidate_filters``:
it performs all setup first, then runs only one projection scan variant in the
timed/profiled region. It is meant to be launched directly or under Nsight
Compute directly or under Nsight Compute. The NCU wrapper profiles from process
start and filters to the Triton projection kernel names.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_candidate_filters import (
    _actual_skip_labels,
    _block_order,
    _block_lengths,
    _candidate_metrics,
    _candidate_metrics_from_prediction,
    _project_query,
    _projection_matrix,
    _projection_metadata,
    _projection_metadata_dtype,
    _projection_scores,
    _scan_block_range,
    _static_seed_threshold,
    _unpack_bitmask,
)
from benchmarks.profile_gate0_summary_bounds import (
    _dtype,
    _load_real_tensors,
    _scan_full_qk,
    _sync,
    _time_call,
)
from stream_attention.kernels.gate0_projection_bitmask_triton import (
    gate0_projection_bitmask_static_threshold_triton,
    gate0_projection_bitmask_triton,
)
from stream_attention.kernels.gate0_projection_mask_triton import (
    gate0_projection_mask_static_threshold_triton,
    gate0_projection_mask_triton,
)
from stream_attention.kernels.gate0_projection_scan_triton import gate0_projection_scan_triton


def _slice_head(q: torch.Tensor, k: torch.Tensor, head_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if head_index < 0:
        return q, k
    if head_index >= q.shape[2]:
        raise ValueError(f"head index {head_index} is outside [0, {q.shape[2]})")
    return (
        q[:, :, head_index : head_index + 1, :].contiguous(),
        k[:, :, head_index : head_index + 1, :].contiguous(),
    )


def _cuda_profiler_start() -> None:
    try:
        torch.cuda.profiler.start()
    except Exception:
        # Older/stripped CUDA runtime builds may not expose profiler controls.
        pass


def _cuda_profiler_stop() -> None:
    try:
        torch.cuda.profiler.stop()
    except Exception:
        pass


def _prepare(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype(args.dtype)
    q, k, _v = _load_real_tensors(args, device=device, dtype=dtype)
    q, k = _slice_head(q, k, args.head_index)

    batch, query_len, heads, dim = q.shape
    seq_k = k.shape[1]
    num_blocks = (seq_k + args.block_size - 1) // args.block_size
    block_lengths = _block_lengths(seq_k, args.block_size, device=device)
    block_log_lengths = block_lengths.float().clamp_min(1).log()
    order = _block_order(
        args.block_order,
        num_blocks=num_blocks,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )
    actual_skip, has_state, thresholds, _tile_max = _actual_skip_labels(
        q,
        k,
        block_size=args.block_size,
        error_budget=args.error_budget,
        block_order=order,
    )
    static_thresholds = _static_seed_threshold(
        q,
        k,
        block_size=args.block_size,
        error_budget=args.error_budget,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )
    scan_start, scan_end = _scan_block_range(
        args.scan_region,
        num_blocks=num_blocks,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )

    projection = _projection_matrix(
        args.projection_kind,
        dim=dim,
        rank=args.projection_dim,
        seed=args.seed,
        device=device,
    )
    metadata_dtype = _projection_metadata_dtype(args.projection_metadata_dtype)
    proj_min, proj_max = _projection_metadata(
        k,
        block_size=args.block_size,
        projection=projection,
        metadata_dtype=metadata_dtype,
    )
    q_proj = _project_query(q, projection)

    q_bhsd = q.permute(0, 2, 1, 3).contiguous().float()
    k_bhnd = k.permute(0, 2, 1, 3).contiguous().float()
    full_qk_scan_ms = _time_call(
        lambda: _scan_full_qk(q_bhsd, k_bhnd, block_size=args.block_size, scale=1.0 / math.sqrt(dim)),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    q_projection_ms = _time_call(
        lambda: _project_query(q, projection),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    state: Dict[str, Any] = {
        "device": device,
        "q": q,
        "projection": projection,
        "q_proj": q_proj,
        "proj_min": proj_min,
        "proj_max": proj_max,
        "thresholds": thresholds,
        "has_state": has_state,
        "static_thresholds": static_thresholds,
        "block_lengths": block_lengths,
        "block_log_lengths": block_log_lengths,
        "actual_skip": actual_skip,
        "scan_start": scan_start,
        "scan_end": scan_end,
        "num_blocks": num_blocks,
        "shape": {
            "batch": batch,
            "query_len": query_len,
            "kv_len": seq_k,
            "heads": heads,
            "dim": dim,
            "dtype": args.dtype,
        },
        "full_qk_scan_ms": full_qk_scan_ms,
        "q_projection_ms": q_projection_ms,
    }
    return state


def _build_run_fn(args, state):
    q_proj = state["q_proj"]
    proj_min = state["proj_min"]
    proj_max = state["proj_max"]
    dim = state["shape"]["dim"]
    scan_start = state["scan_start"]
    scan_end = state["scan_end"]
    num_blocks = state["num_blocks"]
    device = state["device"]

    if args.backend == "triton_score":
        output = torch.full(
            (state["shape"]["batch"], state["shape"]["heads"], state["shape"]["query_len"], num_blocks),
            float("inf"),
            device=device,
            dtype=torch.float32,
        )
        return lambda: gate0_projection_scan_triton(
            q_proj,
            proj_min,
            proj_max,
            dim=dim,
            scan_start=scan_start,
            scan_end=scan_end,
            blocks_per_program=args.blocks_per_program,
            output=output,
            clear_output=False,
        )

    if args.backend == "triton_mask":
        output = torch.zeros(
            (state["shape"]["batch"], state["shape"]["heads"], state["shape"]["query_len"], num_blocks),
            device=device,
            dtype=torch.uint8,
        )
        if args.threshold_mode == "static":
            return lambda: gate0_projection_mask_static_threshold_triton(
                q_proj,
                proj_min,
                proj_max,
                state["static_thresholds"],
                state["block_log_lengths"],
                dim=dim,
                filter_margin=args.filter_margin,
                scan_start=scan_start,
                scan_end=scan_end,
                blocks_per_program=args.blocks_per_program,
                output=output,
                clear_output=False,
            )
        return lambda: gate0_projection_mask_triton(
            q_proj,
            proj_min,
            proj_max,
            state["thresholds"],
            state["has_state"],
            state["block_log_lengths"],
            dim=dim,
            filter_margin=args.filter_margin,
            scan_start=scan_start,
            scan_end=scan_end,
            blocks_per_program=args.blocks_per_program,
            output=output,
            clear_output=False,
        )

    if args.backend == "triton_bitmask":
        num_words = (num_blocks + 31) // 32
        output = torch.zeros(
            (state["shape"]["batch"], state["shape"]["heads"], state["shape"]["query_len"], num_words),
            device=device,
            dtype=torch.int32,
        )
        if args.threshold_mode == "static":
            return lambda: gate0_projection_bitmask_static_threshold_triton(
                q_proj,
                proj_min,
                proj_max,
                state["static_thresholds"],
                state["block_log_lengths"],
                dim=dim,
                filter_margin=args.filter_margin,
                scan_start=scan_start,
                scan_end=scan_end,
                words_per_program=args.words_per_program,
                output=output,
                clear_output=False,
            )
        return lambda: gate0_projection_bitmask_triton(
            q_proj,
            proj_min,
            proj_max,
            state["thresholds"],
            state["has_state"],
            state["block_log_lengths"],
            dim=dim,
            filter_margin=args.filter_margin,
            scan_start=scan_start,
            scan_end=scan_end,
            words_per_program=args.words_per_program,
            output=output,
            clear_output=False,
        )

    raise ValueError(f"unsupported backend: {args.backend}")


def _metrics_from_output(args, output, state):
    if args.backend == "triton_score":
        return _candidate_metrics(
            scores=output,
            actual_skip=state["actual_skip"],
            has_state=state["has_state"],
            thresholds=state["thresholds"],
            block_lengths=state["block_lengths"],
            filter_margin=args.filter_margin,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
        )
    predicted = (
        _unpack_bitmask(output, num_blocks=state["num_blocks"])
        if args.backend == "triton_bitmask"
        else output.bool()
    )
    return _candidate_metrics_from_prediction(
        predicted=predicted,
        actual_skip=state["actual_skip"],
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", default="")
    parser.add_argument("--tensor-format", choices=["pt"], default="pt")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--head-index", type=int, default=-1)
    parser.add_argument("--backend", choices=["triton_score", "triton_mask", "triton_bitmask"], default="triton_mask")
    parser.add_argument("--threshold-mode", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--projection-kind", choices=["random", "hadamard"], default="random")
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--projection-metadata-dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--scan-region", choices=["all", "middle_only", "middle_plus_old"], default="middle_only")
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--filter-margin", type=float, default=32.0)
    parser.add_argument("--error-budget", type=float, default=1e-2)
    parser.add_argument("--blocks-per-program", type=int, default=32)
    parser.add_argument("--words-per-program", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--cuda-profiler-api", action="store_true")
    args = parser.parse_args()

    if args.backend == "triton_score" and args.threshold_mode != "dynamic":
        raise ValueError("triton_score only supports dynamic threshold evaluation")

    with torch.no_grad():
        state = _prepare(args)
        run_fn = _build_run_fn(args, state)
        device = state["device"]
        candidate_scan_ms = _time_call(
            lambda: run_fn(),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        output = run_fn()
        _sync(device)
        metrics = _metrics_from_output(args, output, state)

        if args.cuda_profiler_api and device.type == "cuda":
            _sync(device)
            _cuda_profiler_start()
            for _ in range(args.profile_iters):
                run_fn()
            _sync(device)
            _cuda_profiler_stop()

    full_qk_scan_ms = state["full_qk_scan_ms"]
    predicted_compute_fraction = 1.0 - float(metrics["predicted_skip_fraction"])
    estimated_qk_path_ms = state["q_projection_ms"] + candidate_scan_ms + predicted_compute_fraction * full_qk_scan_ms
    payload = {
        "backend": args.backend,
        "threshold_mode": args.threshold_mode,
        "projection_kind": args.projection_kind,
        "projection_dim": args.projection_dim,
        "projection_metadata_dtype": args.projection_metadata_dtype,
        "block_size": args.block_size,
        "scan_region": args.scan_region,
        "block_order": args.block_order,
        "scan_start": state["scan_start"],
        "scan_end": state["scan_end"],
        "shape": state["shape"],
        "candidate_scan_ms": candidate_scan_ms,
        "full_qk_scan_ms": full_qk_scan_ms,
        "scan_over_qk": candidate_scan_ms / full_qk_scan_ms if full_qk_scan_ms > 0 else None,
        "q_projection_ms": state["q_projection_ms"],
        "estimated_qk_path_ms": estimated_qk_path_ms,
        "estimated_speedup_vs_qk": full_qk_scan_ms / estimated_qk_path_ms if estimated_qk_path_ms > 0 else None,
        "words_per_program": args.words_per_program if args.backend == "triton_bitmask" else None,
        "blocks_per_program": args.blocks_per_program if args.backend != "triton_bitmask" else None,
        **metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
