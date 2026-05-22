"""Tune the Triton direct-output seed-only backend on captured true-GQA tensors.

This is a narrow backend-economics profiler.  It does not test new policies; it
only asks whether the existing logit-safe seed-only runtime can be pushed below
FlashInfer TC exact by tuning the Triton launch shape.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_true_gqa import _flashinfer_single_decode  # noqa: E402
from benchmarks.profile_gate0_true_gqa import _true_gqa_kv  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _dtype,
    _error,
    _load_tensor,
    _sync,
    _time_cuda,
)
from stream_attention.kernels.gate0_launch_floor_triton import (  # noqa: E402
    gate0_launch_floor_empty_triton_forward,
)
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward_out,
)


def _parse_ints(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty integer list: {raw!r}")
    return vals


def _configs(args: argparse.Namespace) -> Iterable[dict[str, int]]:
    for block_size in _parse_ints(args.block_sizes):
        for middle_seed_blocks in _parse_ints(args.middle_seed_blocks_list):
            for num_warps in _parse_ints(args.num_warps_list):
                for num_stages in _parse_ints(args.num_stages_list):
                    yield {
                        "block_size": block_size,
                        "middle_seed_blocks": middle_seed_blocks,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    }


def _time_or_error(fn, *, device: torch.device, warmup: int, iters: int) -> tuple[float | None, str | None]:
    try:
        return _time_cuda(fn, device=device, warmup=warmup, iters=iters), None
    except Exception as exc:  # pragma: no cover - backend dependent
        _sync(device)
        return None, f"{type(exc).__name__}: {exc}"


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)

    q = _load_tensor(args.q_path, key="q", device=device, dtype=dtype).contiguous()
    k_expanded = _load_tensor(args.k_path, key="k", device=device, dtype=dtype)
    v_expanded = _load_tensor(args.v_path, key="v", device=device, dtype=dtype)
    k_true = _true_gqa_kv(k_expanded, true_kv_heads=args.true_kv_heads).contiguous()
    v_true = _true_gqa_kv(v_expanded, true_kv_heads=args.true_kv_heads).contiguous()
    selected_heads = torch.arange(q.shape[2], device=device, dtype=torch.int32)

    def flashinfer_exact() -> torch.Tensor:
        return _flashinfer_single_decode(q, k_true, v_true, use_tensor_cores=args.flashinfer_tensor_cores)

    def empty_launch() -> torch.Tensor:
        return gate0_launch_floor_empty_triton_forward(q, selected_heads)

    flash_ref = flashinfer_exact().clone()
    _sync(device)
    flash_ms = _time_cuda(flashinfer_exact, device=device, warmup=args.warmup, iters=args.iters)
    empty_ms = _time_cuda(empty_launch, device=device, warmup=args.warmup, iters=args.iters)

    rows: list[dict[str, Any]] = []
    for cfg in _configs(args):
        out = torch.empty_like(q)

        def seed_direct() -> torch.Tensor:
            return gate0_seed_only_attention_triton_forward_out(
                q,
                k_true,
                v_true,
                out,
                block_size=cfg["block_size"],
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=cfg["middle_seed_blocks"],
                block_order=args.block_order,
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
            )

        seed_ms, failure = _time_or_error(seed_direct, device=device, warmup=args.warmup, iters=args.iters)
        row: dict[str, Any] = {
            **cfg,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "block_order": args.block_order,
            "seed_blocks": args.sink_blocks + args.recent_blocks + cfg["middle_seed_blocks"],
            "seed_tokens": (args.sink_blocks + args.recent_blocks + cfg["middle_seed_blocks"])
            * cfg["block_size"],
            "flashinfer_tc_exact_ms": flash_ms,
            "empty_launch_ms": empty_ms,
            "seed_direct_full_prealloc_ms": seed_ms,
            "failure": failure,
        }
        if seed_ms is not None:
            seed_ref = seed_direct().clone()
            _sync(device)
            row["speedup_vs_flashinfer"] = flash_ms / seed_ms
            row["seed_vs_flashinfer_exact"] = _error(seed_ref, flash_ref)
        rows.append(row)

    valid_rows = [r for r in rows if r["seed_direct_full_prealloc_ms"] is not None]
    best = min(valid_rows, key=lambda r: r["seed_direct_full_prealloc_ms"]) if valid_rows else None
    return {
        "schema": "streamattn.gate0.seed_only_triton_tune.v1",
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "q_heads": int(q.shape[2]),
            "true_kv_heads": int(k_true.shape[2]),
            "group_size": int(q.shape[2] // k_true.shape[2]),
            "kv_len": int(k_true.shape[1]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "timing_floor": {
            "flashinfer_tc_exact_ms": flash_ms,
            "empty_launch_ms": empty_ms,
            "empty_launch_fraction_of_flashinfer": empty_ms / flash_ms,
        },
        "search": {
            "block_sizes": _parse_ints(args.block_sizes),
            "middle_seed_blocks": _parse_ints(args.middle_seed_blocks_list),
            "num_warps": _parse_ints(args.num_warps_list),
            "num_stages": _parse_ints(args.num_stages_list),
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "best": best,
        "rows": sorted(
            rows,
            key=lambda r: (
                r["seed_direct_full_prealloc_ms"] is None,
                r["seed_direct_full_prealloc_ms"] or math.inf,
            ),
        ),
        "decision": (
            "triton_seed_tune_beats_flashinfer"
            if best and best["seed_direct_full_prealloc_ms"] < flash_ms
            else "triton_seed_tune_still_slower_than_flashinfer"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-sizes", default="16,32,64")
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks-list", default="0,2,4,8")
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first", "sink_recent_first"])
    parser.add_argument("--num-warps-list", default="1,2,4,8")
    parser.add_argument("--num-stages-list", default="2,3,4")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
