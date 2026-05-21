"""Profile exact selected-head true-GQA decode backend candidates.

This benchmark answers the backend question after the seed-only Gate-0 path:
can the exact remaining heads run near dense quality without Python composition?
It uses real captured post-RoPE Q/K/V tensors and compares:

* dense all-head PyTorch SDPA true-GQA;
* PyTorch SDPA on exact remaining heads grouped by KV head;
* FlashInfer single-decode on all heads and grouped exact remaining heads;
* current Triton split-K exact branch on all heads and grouped exact heads.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_true_gqa import (  # noqa: E402
    _dense_selected_heads,
    _head_groups_by_kv,
    _parse_heads,
    _select_kv_head,
    _select_q_heads,
)
from benchmarks.profile_gate0_true_gqa import _dense_true_gqa, _true_gqa_kv  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _dtype,
    _error,
    _load_tensor,
    _sync,
    _time_cuda,
)
from stream_attention.kernels.gate1_inline_projection_splitk_triton import (  # noqa: E402
    TRITON_AVAILABLE,
    gate1_inline_projection_splitk_attention_triton_forward,
    make_splitk_workspace,
)
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_selected_attention_triton_forward,
)

try:
    import flashinfer

    HAS_FLASHINFER = True
except Exception:  # pragma: no cover - optional dependency
    flashinfer = None
    HAS_FLASHINFER = False


def _time_group_oracle(
    items: Sequence[Dict[str, Any]],
    *,
    key: str,
) -> Dict[str, Any]:
    values = [float(item[key]) for item in items]
    return {
        "parallel_oracle_ms": max(values) if values else 0.0,
        "serial_sum_ms": sum(values),
    }


def _flashinfer_single_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    use_tensor_cores: bool,
) -> torch.Tensor:
    if not HAS_FLASHINFER:
        raise RuntimeError("FlashInfer is not available")
    out = flashinfer.decode.single_decode_with_kv_cache(
        q[0, 0].contiguous(),
        k[0].contiguous(),
        v[0].contiguous(),
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        use_tensor_cores=use_tensor_cores,
    )
    return out.view(1, 1, q.shape[2], q.shape[3])


def _make_dummy_projection_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    rank: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks = math.ceil(k.shape[1] / block_size)
    q_proj = torch.empty(q.shape[0], q.shape[2], 1, rank, device=q.device, dtype=torch.float32)
    proj_min = torch.empty(q.shape[0], k.shape[2], blocks, rank, device=q.device, dtype=q.dtype)
    proj_max = torch.empty_like(proj_min)
    return q_proj, proj_min, proj_max


def _triton_splitk_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rank: int,
    block_size: int,
    num_chunks: int,
    num_warps: int,
    num_stages: int,
    workspace: Dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    q_proj, proj_min, proj_max = _make_dummy_projection_inputs(
        q,
        k,
        rank=rank,
        block_size=block_size,
    )
    head_modes = torch.ones(q.shape[2], device=q.device, dtype=torch.int32)
    return gate1_inline_projection_splitk_attention_triton_forward(
        q,
        k,
        v,
        q_proj,
        proj_min,
        proj_max,
        compute_qproj=False,
        num_chunks=num_chunks,
        error_budget=0.0,
        filter_margin=0.0,
        block_size=block_size,
        sink_blocks=0,
        recent_blocks=0,
        middle_seed_blocks=0,
        block_order="sequential",
        seed_strategy="recompute_seed",
        head_modes=head_modes,
        return_raw_stats=False,
        workspace=workspace,
        num_warps=num_warps,
        num_stages=num_stages,
    )[0]


def _seed_kernel_exact_selected(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_heads: torch.Tensor,
    *,
    block_size: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Use the selected-head seed kernel as a simple serial exact backend."""

    middle_seed_blocks = math.ceil(k.shape[1] / block_size)
    return gate0_seed_only_selected_attention_triton_forward(
        q,
        k,
        v,
        selected_heads,
        block_size=block_size,
        sink_blocks=0,
        recent_blocks=0,
        middle_seed_blocks=middle_seed_blocks,
        block_order="sequential",
        return_raw_stats=False,
        validate_heads=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )[0]


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

    q_heads = int(q.shape[2])
    seed_heads = _parse_heads(args.seed_heads)
    exact_heads = [head for head in range(q_heads) if head not in set(seed_heads)]
    exact_groups = _head_groups_by_kv(exact_heads, q_heads=q_heads, kv_heads=k_true.shape[2])

    exact_items = []
    for kv_head, heads in exact_groups.items():
        exact_items.append(
            {
                "kv_head": kv_head,
                "q_heads": heads,
                "q": _select_q_heads(q, heads),
                "k": _select_kv_head(k_true, kv_head),
                "v": _select_kv_head(v_true, kv_head),
            }
        )

    def dense_all() -> torch.Tensor:
        return _dense_true_gqa(q, k_true, v_true)

    def torch_exact_selected() -> torch.Tensor:
        return _dense_selected_heads(q, k_true, v_true, exact_heads)

    dense_ms = _time_cuda(dense_all, device=device, warmup=args.warmup, iters=args.iters)
    torch_selected_ms = _time_cuda(
        torch_exact_selected,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    torch_group_timings = []
    for item in exact_items:
        ms = _time_cuda(
            lambda item=item: _dense_true_gqa(item["q"], item["k"], item["v"]),
            device=device,
            warmup=args.group_warmup,
            iters=args.group_iters,
        )
        torch_group_timings.append(
            {"kv_head": item["kv_head"], "q_heads": item["q_heads"], "torch_sdpa_ms": ms}
        )

    flashinfer_all_ms = None
    flashinfer_all_error = None
    flashinfer_group_timings: List[Dict[str, Any]] = []
    flashinfer_error = None
    flashinfer_errors: List[Dict[str, Any]] = []
    if HAS_FLASHINFER and args.measure_flashinfer:
        dense_out = dense_all()
        try:
            flashinfer_all_ms = _time_cuda(
                lambda: _flashinfer_single_decode(
                    q,
                    k_true,
                    v_true,
                    use_tensor_cores=args.flashinfer_tensor_cores,
                ),
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            flashinfer_all_error = _error(
                _flashinfer_single_decode(
                    q,
                    k_true,
                    v_true,
                    use_tensor_cores=args.flashinfer_tensor_cores,
                ),
                dense_out,
            )
        except Exception as exc:
            flashinfer_errors.append(
                {
                    "scope": "all_heads",
                    "q_heads": q_heads,
                    "kv_heads": int(k_true.shape[2]),
                    "use_tensor_cores": args.flashinfer_tensor_cores,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        flashinfer_parts = []
        for item in exact_items:
            try:
                ms = _time_cuda(
                    lambda item=item: _flashinfer_single_decode(
                        item["q"],
                        item["k"],
                        item["v"],
                        use_tensor_cores=args.flashinfer_tensor_cores,
                    ),
                    device=device,
                    warmup=args.group_warmup,
                    iters=args.group_iters,
                )
                out = _flashinfer_single_decode(
                    item["q"],
                    item["k"],
                    item["v"],
                    use_tensor_cores=args.flashinfer_tensor_cores,
                )
                flashinfer_parts.append((item, out))
                flashinfer_group_timings.append(
                    {"kv_head": item["kv_head"], "q_heads": item["q_heads"], "flashinfer_ms": ms}
                )
            except Exception as exc:
                flashinfer_errors.append(
                    {
                        "scope": "exact_group",
                        "kv_head": item["kv_head"],
                        "q_heads": item["q_heads"],
                        "kv_heads": 1,
                        "use_tensor_cores": args.flashinfer_tensor_cores,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if len(flashinfer_parts) == len(exact_items):
            exact_dense_out = torch_exact_selected()
            exact_flashinfer_out = torch.empty_like(exact_dense_out)
            local = 0
            for item, out in flashinfer_parts:
                width = out.shape[2]
                exact_flashinfer_out[:, :, local : local + width, :] = out
                local += width
            flashinfer_error = _error(exact_flashinfer_out, exact_dense_out)

    triton_all_ms = None
    triton_group_timings: List[Dict[str, Any]] = []
    triton_all_error = None
    triton_exact_error = None
    seed_kernel_selected_ms = None
    seed_kernel_group_timings: List[Dict[str, Any]] = []
    seed_kernel_exact_error = None
    if TRITON_AVAILABLE and args.measure_triton_splitk:
        selected_exact_heads = torch.tensor(exact_heads, device=device, dtype=torch.int32)
        seed_kernel_selected_ms = _time_cuda(
            lambda: _seed_kernel_exact_selected(
                q,
                k_true,
                v_true,
                selected_exact_heads,
                block_size=args.block_size,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            ),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        seed_exact_out = _seed_kernel_exact_selected(
            q,
            k_true,
            v_true,
            selected_exact_heads,
            block_size=args.block_size,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        seed_kernel_exact_error = _error(seed_exact_out, torch_exact_selected())
        for item in exact_items:
            local_heads = torch.arange(
                item["q"].shape[2],
                device=device,
                dtype=torch.int32,
            )
            ms = _time_cuda(
                lambda item=item, local_heads=local_heads: _seed_kernel_exact_selected(
                    item["q"],
                    item["k"],
                    item["v"],
                    local_heads,
                    block_size=args.block_size,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                ),
                device=device,
                warmup=args.group_warmup,
                iters=args.group_iters,
            )
            seed_kernel_group_timings.append(
                {
                    "kv_head": item["kv_head"],
                    "q_heads": item["q_heads"],
                    "seed_kernel_full_exact_ms": ms,
                }
            )

        all_workspace = make_splitk_workspace(
            q,
            rank=args.projection_dim,
            num_chunks=args.num_chunks,
            seed_strategy="recompute_seed",
        )
        triton_all_ms = _time_cuda(
            lambda: _triton_splitk_exact(
                q,
                k_true,
                v_true,
                rank=args.projection_dim,
                block_size=args.block_size,
                num_chunks=args.num_chunks,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                workspace=all_workspace,
            ),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        triton_all_error = _error(
            _triton_splitk_exact(
                q,
                k_true,
                v_true,
                rank=args.projection_dim,
                block_size=args.block_size,
                num_chunks=args.num_chunks,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                workspace=all_workspace,
            ),
            dense_all(),
        )
        triton_parts = []
        for item in exact_items:
            workspace = make_splitk_workspace(
                item["q"],
                rank=args.projection_dim,
                num_chunks=args.num_chunks,
                seed_strategy="recompute_seed",
            )
            ms = _time_cuda(
                lambda item=item, workspace=workspace: _triton_splitk_exact(
                    item["q"],
                    item["k"],
                    item["v"],
                    rank=args.projection_dim,
                    block_size=args.block_size,
                    num_chunks=args.num_chunks,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                    workspace=workspace,
                ),
                device=device,
                warmup=args.group_warmup,
                iters=args.group_iters,
            )
            out = _triton_splitk_exact(
                item["q"],
                item["k"],
                item["v"],
                rank=args.projection_dim,
                block_size=args.block_size,
                num_chunks=args.num_chunks,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                workspace=workspace,
            )
            triton_parts.append((item, out))
            triton_group_timings.append(
                {"kv_head": item["kv_head"], "q_heads": item["q_heads"], "triton_splitk_ms": ms}
            )
        exact_dense_out = torch_exact_selected()
        exact_triton_out = torch.empty_like(exact_dense_out)
        local = 0
        for item, out in triton_parts:
            width = out.shape[2]
            exact_triton_out[:, :, local : local + width, :] = out
            local += width
        triton_exact_error = _error(exact_triton_out, exact_dense_out)

    torch_oracle = _time_group_oracle(torch_group_timings, key="torch_sdpa_ms")
    flashinfer_oracle = (
        _time_group_oracle(flashinfer_group_timings, key="flashinfer_ms")
        if flashinfer_group_timings
        else None
    )
    triton_oracle = (
        _time_group_oracle(triton_group_timings, key="triton_splitk_ms")
        if triton_group_timings
        else None
    )
    seed_kernel_oracle = (
        _time_group_oracle(seed_kernel_group_timings, key="seed_kernel_full_exact_ms")
        if seed_kernel_group_timings
        else None
    )

    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "flashinfer_available": HAS_FLASHINFER,
        "flashinfer_tensor_cores": args.flashinfer_tensor_cores,
        "flashinfer_errors": flashinfer_errors,
        "triton_available": TRITON_AVAILABLE,
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "kv_len": int(k_true.shape[1]),
            "q_heads": q_heads,
            "true_kv_heads": int(k_true.shape[2]),
            "group_size": q_heads // int(k_true.shape[2]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "policy": {
            "seed_heads": seed_heads,
            "exact_heads": exact_heads,
            "exact_groups": [
                {"kv_head": item["kv_head"], "q_heads": item["q_heads"]}
                for item in exact_items
            ],
        },
        "timing": {
            "dense_all_true_gqa_ms": dense_ms,
            "torch_sdpa_exact_selected_serial_ms": torch_selected_ms,
            "torch_sdpa_exact_group_parallel_oracle_ms": torch_oracle["parallel_oracle_ms"],
            "torch_sdpa_exact_group_serial_sum_ms": torch_oracle["serial_sum_ms"],
            "flashinfer_all_true_gqa_ms": flashinfer_all_ms,
            "flashinfer_exact_group_parallel_oracle_ms": (
                flashinfer_oracle["parallel_oracle_ms"] if flashinfer_oracle else None
            ),
            "flashinfer_exact_group_serial_sum_ms": (
                flashinfer_oracle["serial_sum_ms"] if flashinfer_oracle else None
            ),
            "seed_kernel_exact_selected_ms": seed_kernel_selected_ms,
            "seed_kernel_exact_group_parallel_oracle_ms": (
                seed_kernel_oracle["parallel_oracle_ms"] if seed_kernel_oracle else None
            ),
            "seed_kernel_exact_group_serial_sum_ms": (
                seed_kernel_oracle["serial_sum_ms"] if seed_kernel_oracle else None
            ),
            "triton_splitk_all_exact_ms": triton_all_ms,
            "triton_splitk_exact_group_parallel_oracle_ms": (
                triton_oracle["parallel_oracle_ms"] if triton_oracle else None
            ),
            "triton_splitk_exact_group_serial_sum_ms": (
                triton_oracle["serial_sum_ms"] if triton_oracle else None
            ),
        },
        "group_timings": {
            "torch_sdpa": torch_group_timings,
            "flashinfer": flashinfer_group_timings,
            "seed_kernel_full_exact": seed_kernel_group_timings,
            "triton_splitk": triton_group_timings,
        },
        "quality": {
            "flashinfer_all_vs_dense": flashinfer_all_error,
            "flashinfer_exact_selected_vs_torch_selected": flashinfer_error,
            "seed_kernel_exact_selected_vs_torch_selected": seed_kernel_exact_error,
            "triton_splitk_all_vs_dense": triton_all_error,
            "triton_splitk_exact_selected_vs_torch_selected": triton_exact_error,
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
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--group-warmup", type=int, default=3)
    parser.add_argument("--group-iters", type=int, default=10)
    parser.add_argument("--measure-flashinfer", action="store_true")
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--measure-triton-splitk", action="store_true")
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
