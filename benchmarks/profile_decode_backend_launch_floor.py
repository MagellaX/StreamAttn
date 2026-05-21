"""Measure the decode-backend launch/composition floor.

This benchmark is deliberately diagnostic.  It compares FlashInfer tensor-core
exact decode against progressively heavier standalone StreamAttn selected-head
kernels, then times serial composition in a single callable:

* FlashInfer TC exact all heads;
* empty selected-head Triton launch;
* selected-head Q-only launch;
* selected-head Q/K/V load plus QK reductions, no softmax/PV;
* normal selected-head seed-only Gate-0;
* FlashInfer TC plus each standalone kernel.

If FlashInfer plus even an empty/small StreamAttn launch erases the oracle
margin, StreamAttn needs to enter the exact decode backend rather than compose
beside it.
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
from benchmarks.profile_gate0_true_gqa import _dense_true_gqa, _true_gqa_kv  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _dtype,
    _error,
    _load_tensor,
    _time_cuda,
)
from stream_attention.kernels.gate0_launch_floor_triton import (  # noqa: E402
    gate0_launch_floor_empty_triton_forward,
    gate0_launch_floor_q_only_triton_forward,
    gate0_launch_floor_qkv_no_softmax_triton_forward,
)
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_selected_attention_kv_major_triton_forward,
    gate0_seed_only_selected_attention_triton_forward,
)


try:
    import flashinfer  # noqa: F401

    HAS_FLASHINFER = True
except Exception:  # pragma: no cover - optional dependency
    HAS_FLASHINFER = False


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0.0:
        return None
    return num / den


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _time_optional(fn, *, device: torch.device, warmup: int, iters: int) -> float | None:
    try:
        return _time_cuda(fn, device=device, warmup=warmup, iters=iters)
    except Exception:
        return None


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
    seed_heads_tensor = torch.tensor(seed_heads, device=device, dtype=torch.int32)

    def dense_all() -> torch.Tensor:
        return _dense_true_gqa(q, k_true, v_true)

    def flashinfer_all() -> torch.Tensor:
        return _flashinfer_single_decode(
            q,
            k_true,
            v_true,
            use_tensor_cores=args.flashinfer_tensor_cores,
        )

    def empty_seed() -> torch.Tensor:
        return gate0_launch_floor_empty_triton_forward(q, seed_heads_tensor)

    def q_only_seed() -> torch.Tensor:
        return gate0_launch_floor_q_only_triton_forward(
            q,
            seed_heads_tensor,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    def qkv_no_softmax_seed() -> torch.Tensor:
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

    def normal_seed() -> torch.Tensor:
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
            validate_heads=False,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )[0]

    def normal_seed_kv_major() -> torch.Tensor:
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
            validate_heads=False,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )[0]

    def compose(first, second):
        def run():
            first()
            return second()

        return run

    dense_ms = _time_cuda(dense_all, device=device, warmup=args.warmup, iters=args.iters)
    dense_out = dense_all()

    flashinfer_ms = None
    flashinfer_error = None
    flashinfer_available = HAS_FLASHINFER
    flashinfer_failure = None
    if args.measure_flashinfer:
        if not HAS_FLASHINFER:
            flashinfer_available = False
            flashinfer_failure = "FlashInfer is not available"
        else:
            try:
                flashinfer_ms = _time_cuda(
                    flashinfer_all,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                flashinfer_error = _error(flashinfer_all(), dense_out)
            except Exception as exc:
                flashinfer_failure = repr(exc)

    empty_ms = _time_cuda(empty_seed, device=device, warmup=args.warmup, iters=args.iters)
    q_only_ms = _time_cuda(q_only_seed, device=device, warmup=args.warmup, iters=args.iters)
    qkv_no_softmax_ms = _time_cuda(
        qkv_no_softmax_seed,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    seed_ms = _time_cuda(normal_seed, device=device, warmup=args.warmup, iters=args.iters)
    seed_kv_major_ms = (
        _time_cuda(normal_seed_kv_major, device=device, warmup=args.warmup, iters=args.iters)
        if args.measure_kv_major_seed
        else None
    )

    composed: Dict[str, float | None] = {}
    if flashinfer_ms is not None:
        composed["flashinfer_plus_empty_triton_ms"] = _time_cuda(
            compose(flashinfer_all, empty_seed),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        composed["flashinfer_plus_q_only_triton_ms"] = _time_cuda(
            compose(flashinfer_all, q_only_seed),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        composed["flashinfer_plus_qkv_no_softmax_triton_ms"] = _time_cuda(
            compose(flashinfer_all, qkv_no_softmax_seed),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        composed["flashinfer_plus_seed_triton_ms"] = _time_cuda(
            compose(flashinfer_all, normal_seed),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        if args.measure_kv_major_seed:
            composed["flashinfer_plus_seed_kv_major_triton_ms"] = _time_cuda(
                compose(flashinfer_all, normal_seed_kv_major),
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )

    seed_out = normal_seed()
    seed_kv_major_error = (
        _error(normal_seed_kv_major(), seed_out) if args.measure_kv_major_seed else None
    )
    reference_ms = flashinfer_ms if flashinfer_ms is not None else dense_ms
    composition_overheads = {
        key.replace("_ms", "_over_reference_ms"): _delta(value, reference_ms)
        for key, value in composed.items()
    }
    separate_sparse_launch_viable = (
        composed.get("flashinfer_plus_seed_triton_ms") is not None
        and reference_ms is not None
        and composed["flashinfer_plus_seed_triton_ms"] < reference_ms
    )

    return {
        "shape": {
            "batch": int(q.shape[0]),
            "q_heads": int(q.shape[2]),
            "kv_heads": int(k_true.shape[2]),
            "head_dim": int(q.shape[3]),
            "kv_len": int(k_true.shape[1]),
            "seed_heads": seed_heads,
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "timing": {
            "dense_torch_sdpa_true_gqa_ms": dense_ms,
            "flashinfer_tc_ms": flashinfer_ms,
            "empty_seed_kernel_ms": empty_ms,
            "seed_q_only_ms": q_only_ms,
            "seed_qkv_no_softmax_ms": qkv_no_softmax_ms,
            "normal_seed_only_ms": seed_ms,
            "normal_seed_only_kv_major_ms": seed_kv_major_ms,
            **composed,
            **composition_overheads,
            "empty_over_flashinfer_tc": _ratio(empty_ms, flashinfer_ms),
            "q_only_over_flashinfer_tc": _ratio(q_only_ms, flashinfer_ms),
            "qkv_no_softmax_over_flashinfer_tc": _ratio(qkv_no_softmax_ms, flashinfer_ms),
            "seed_only_over_flashinfer_tc": _ratio(seed_ms, flashinfer_ms),
            "flashinfer_plus_seed_over_flashinfer_tc": _ratio(
                composed.get("flashinfer_plus_seed_triton_ms"),
                flashinfer_ms,
            ),
            "separate_sparse_launch_viable": bool(separate_sparse_launch_viable),
        },
        "quality": {
            "flashinfer_error_vs_torch_sdpa": flashinfer_error,
            "seed_kv_major_error_vs_interleaved_seed": seed_kv_major_error,
        },
        "backend": {
            "flashinfer_available": flashinfer_available,
            "flashinfer_failure": flashinfer_failure,
            "flashinfer_tensor_cores": args.flashinfer_tensor_cores,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--seed-heads", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument(
        "--block-order",
        choices=["sequential", "recent_first", "sink_recent_first"],
        default="recent_first",
    )
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--measure-flashinfer", action="store_true")
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--measure-kv-major-seed", action="store_true")
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()

    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

