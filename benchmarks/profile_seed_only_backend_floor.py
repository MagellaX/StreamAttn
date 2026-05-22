"""Profile seed-only backend floors on captured true-GQA tensors.

This benchmark is deliberately narrower than the earlier policy sweeps.  It
answers the backend question directly:

    For a logit-safe seed-only policy, where is the time going?

It times launch-only, Q-only, seed K/V load, full direct-output seed-only, and
selected compact-output seed-only paths against FlashInfer tensor-core exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_true_gqa import (  # noqa: E402
    _flashinfer_single_decode,
    _parse_heads,
)
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
    gate0_launch_floor_q_only_triton_forward,
    gate0_launch_floor_qkv_no_softmax_triton_forward,
)
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward,
    gate0_seed_only_attention_triton_forward_out,
    gate0_seed_only_selected_attention_kv_major_triton_forward,
    gate0_seed_only_selected_attention_triton_forward,
)


def _selected_tensor(heads: List[int], *, device: torch.device) -> torch.Tensor:
    if not heads:
        raise ValueError("seed-heads must not be empty")
    return torch.tensor(heads, device=device, dtype=torch.int32)


def _time_optional(fn, *, device: torch.device, warmup: int, iters: int) -> tuple[float | None, str | None]:
    try:
        return _time_cuda(fn, device=device, warmup=warmup, iters=iters), None
    except Exception as exc:  # pragma: no cover - backend dependent
        _sync(device)
        return None, f"{type(exc).__name__}: {exc}"


def _fmt_speedup(reference_ms: float | None, candidate_ms: float | None) -> float | None:
    if reference_ms is None or candidate_ms is None or candidate_ms <= 0:
        return None
    return float(reference_ms) / float(candidate_ms)


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)

    q = _load_tensor(args.q_path, key="q", device=device, dtype=dtype)
    k_expanded = _load_tensor(args.k_path, key="k", device=device, dtype=dtype)
    v_expanded = _load_tensor(args.v_path, key="v", device=device, dtype=dtype)
    k_true = _true_gqa_kv(k_expanded, true_kv_heads=args.true_kv_heads)
    v_true = _true_gqa_kv(v_expanded, true_kv_heads=args.true_kv_heads)
    k_true_kv_major = k_true.permute(0, 2, 1, 3).contiguous()
    v_true_kv_major = v_true.permute(0, 2, 1, 3).contiguous()

    seed_heads = _parse_heads(args.seed_heads)
    seed_head_set = set(seed_heads)
    all_heads = list(range(int(q.shape[2])))
    seed_heads_tensor = _selected_tensor(seed_heads, device=device)
    seed_heads_index = torch.tensor(seed_heads, device=device, dtype=torch.long)
    seed_is_all_heads = seed_heads == all_heads
    full_out_buffer = torch.empty_like(q)
    prealloc_seed_out = torch.empty_like(q)

    def flashinfer_exact() -> torch.Tensor:
        return _flashinfer_single_decode(
            q,
            k_true,
            v_true,
            use_tensor_cores=args.flashinfer_tensor_cores,
        )

    def empty_launch() -> torch.Tensor:
        return gate0_launch_floor_empty_triton_forward(q, seed_heads_tensor)

    def q_only() -> torch.Tensor:
        return gate0_launch_floor_q_only_triton_forward(
            q,
            seed_heads_tensor,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    def qkv_no_softmax() -> torch.Tensor:
        return gate0_launch_floor_qkv_no_softmax_triton_forward(
            q,
            k_true,
            v_true,
            seed_heads_tensor,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    def seed_direct_full() -> torch.Tensor:
        return gate0_seed_only_attention_triton_forward(
            q,
            k_true,
            v_true,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            return_raw_stats=False,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )[0]

    def seed_direct_full_prealloc() -> torch.Tensor:
        return gate0_seed_only_attention_triton_forward_out(
            q,
            k_true,
            v_true,
            prealloc_seed_out,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    def seed_selected_compact() -> torch.Tensor:
        return gate0_seed_only_selected_attention_triton_forward(
            q,
            k_true,
            v_true,
            seed_heads_tensor,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            return_raw_stats=False,
            validate_heads=False,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )[0]

    def seed_selected_kv_major() -> torch.Tensor:
        return gate0_seed_only_selected_attention_kv_major_triton_forward(
            q,
            k_true_kv_major,
            v_true_kv_major,
            seed_heads_tensor,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            return_raw_stats=False,
            validate_heads=False,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )[0]

    selected_seed_cache = seed_selected_compact()
    full_seed_cache = seed_direct_full()

    def scatter_selected_only() -> torch.Tensor:
        full_out_buffer.index_copy_(2, seed_heads_index, selected_seed_cache)
        return full_out_buffer

    def copy_full_only() -> torch.Tensor:
        full_out_buffer.copy_(full_seed_cache)
        return full_out_buffer

    timings: Dict[str, float | None] = {}
    failures: Dict[str, str] = {}
    for name, fn in (
        ("flashinfer_tc_exact_ms", flashinfer_exact),
        ("empty_launch_ms", empty_launch),
        ("q_only_ms", q_only),
        ("qkv_seed_load_no_softmax_ms", qkv_no_softmax),
        ("seed_direct_full_ms", seed_direct_full),
        ("seed_direct_full_prealloc_ms", seed_direct_full_prealloc),
        ("seed_selected_compact_ms", seed_selected_compact),
        ("seed_selected_kv_major_ms", seed_selected_kv_major),
        ("scatter_selected_only_ms", scatter_selected_only),
        ("copy_full_only_ms", copy_full_only),
    ):
        value, failure = _time_optional(fn, device=device, warmup=args.warmup, iters=args.iters)
        timings[name] = value
        if failure is not None:
            failures[name] = failure

    flash_out = flashinfer_exact()
    direct_full_out = seed_direct_full()
    selected_out = seed_selected_compact()
    selected_kv_major_out = seed_selected_kv_major()
    direct_selected = direct_full_out.index_select(2, seed_heads_index)
    prealloc_out = seed_direct_full_prealloc().clone()
    _sync(device)

    seed_blocks = args.sink_blocks + args.recent_blocks + args.middle_seed_blocks
    result = {
        "schema": "streamattn.gate0.seed_only_backend_floor.v1",
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "q_heads": int(q.shape[2]),
            "true_kv_heads": int(k_true.shape[2]),
            "group_size": int(q.shape[2] // k_true.shape[2]),
            "kv_len": int(k_true.shape[1]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
            "seed_heads": seed_heads,
            "seed_is_all_heads": seed_is_all_heads,
        },
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "seed_blocks": seed_blocks,
            "seed_tokens_upper_bound": seed_blocks * args.block_size,
            "block_order": args.block_order,
        },
        "timing": {
            **timings,
            "seed_direct_full_speedup_vs_flashinfer": _fmt_speedup(
                timings.get("flashinfer_tc_exact_ms"),
                timings.get("seed_direct_full_ms"),
            ),
            "seed_direct_full_prealloc_speedup_vs_flashinfer": _fmt_speedup(
                timings.get("flashinfer_tc_exact_ms"),
                timings.get("seed_direct_full_prealloc_ms"),
            ),
            "seed_selected_compact_speedup_vs_flashinfer": _fmt_speedup(
                timings.get("flashinfer_tc_exact_ms"),
                timings.get("seed_selected_compact_ms"),
            ),
            "seed_selected_kv_major_speedup_vs_flashinfer": _fmt_speedup(
                timings.get("flashinfer_tc_exact_ms"),
                timings.get("seed_selected_kv_major_ms"),
            ),
            "empty_launch_fraction_of_flashinfer": (
                timings["empty_launch_ms"] / timings["flashinfer_tc_exact_ms"]
                if timings.get("empty_launch_ms") and timings.get("flashinfer_tc_exact_ms")
                else None
            ),
        },
        "quality": {
            "seed_direct_full_vs_flashinfer": _error(direct_full_out, flash_out),
            "seed_direct_full_prealloc_vs_direct_full": _error(prealloc_out, direct_full_out),
            "seed_selected_compact_vs_direct_full_selected_heads": _error(selected_out, direct_selected),
            "seed_selected_kv_major_vs_direct_full_selected_heads": _error(
                selected_kv_major_out,
                direct_selected,
            ),
            "selected_seed_head_count": len(seed_heads),
            "selected_exact_head_count": len([head for head in all_heads if head not in seed_head_set]),
        },
        "failures": failures,
        "decision": (
            "direct_full_seed_backend_beats_flashinfer"
            if timings.get("seed_direct_full_ms")
            and timings.get("flashinfer_tc_exact_ms")
            and float(timings["seed_direct_full_ms"]) < float(timings["flashinfer_tc_exact_ms"])
            else "direct_full_seed_backend_needs_lower_overhead_scheduler"
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--seed-heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first", "sink_recent_first"])
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
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
