"""Profile calibrated seed-only Gate-0 on true-GQA captured tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_true_gqa import _dense_true_gqa, _true_gqa_kv  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _dtype,
    _error,
    _load_tensor,
    _sync,
    _time_cuda,
)
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward,
    gate0_seed_only_selected_attention_triton_forward,
)


def _parse_heads(raw: str) -> List[int]:
    return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))


def _head_groups_by_kv(heads: Sequence[int], *, q_heads: int, kv_heads: int) -> Dict[int, List[int]]:
    group_size = q_heads // kv_heads
    groups: Dict[int, List[int]] = {kv_head: [] for kv_head in range(kv_heads)}
    for head in heads:
        if head < 0 or head >= q_heads:
            raise ValueError(f"head {head} outside [0, {q_heads})")
        groups[head // group_size].append(head)
    return {kv_head: group for kv_head, group in groups.items() if group}


def _select_q_heads(q: torch.Tensor, heads: Sequence[int]) -> torch.Tensor:
    return q.index_select(2, torch.tensor(list(heads), device=q.device, dtype=torch.long)).contiguous()


def _select_kv_head(tensor: torch.Tensor, kv_head: int) -> torch.Tensor:
    return tensor[:, :, kv_head : kv_head + 1, :].contiguous()


def _scatter_heads(out: torch.Tensor, heads: Sequence[int], values: torch.Tensor) -> None:
    out.index_copy_(2, torch.tensor(list(heads), device=out.device, dtype=torch.long), values)


def _scatter_selected(out: torch.Tensor, heads: Sequence[int], values: torch.Tensor) -> None:
    _scatter_heads(out, heads, values)


def _dense_selected_heads(
    q: torch.Tensor,
    k_true: torch.Tensor,
    v_true: torch.Tensor,
    heads: Sequence[int],
) -> torch.Tensor:
    if not heads:
        return torch.empty(q.shape[0], q.shape[1], 0, q.shape[3], device=q.device, dtype=q.dtype)
    groups = _head_groups_by_kv(heads, q_heads=q.shape[2], kv_heads=k_true.shape[2])
    local_index = {int(head): idx for idx, head in enumerate(heads)}
    out = torch.empty(q.shape[0], q.shape[1], len(heads), q.shape[3], device=q.device, dtype=q.dtype)
    for kv_head, group_heads in groups.items():
        group_out = _dense_true_gqa(
            _select_q_heads(q, group_heads),
            _select_kv_head(k_true, kv_head),
            _select_kv_head(v_true, kv_head),
        )
        local_positions = [local_index[head] for head in group_heads]
        out.index_copy_(
            2,
            torch.tensor(local_positions, device=out.device, dtype=torch.long),
            group_out,
        )
    return out


def _per_head_error(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, Any]:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
    diff = (actual - expected).detach().abs().float()
    rows = []
    max_values = []
    mean_values = []
    for head in range(diff.shape[2]):
        head_diff = diff[:, :, head, :]
        max_error = float(head_diff.max().item())
        mean_error = float(head_diff.mean().item())
        max_values.append(max_error)
        mean_values.append(mean_error)
        rows.append({"head": head, "max_abs_error": max_error, "mean_abs_error": mean_error})
    worst_head = max(range(len(max_values)), key=lambda idx: max_values[idx]) if max_values else None
    return {
        "per_head": rows,
        "worst_head": worst_head,
        "max_abs_error": max(max_values) if max_values else 0.0,
        "mean_abs_error": sum(mean_values) / len(mean_values) if mean_values else 0.0,
    }


def _time_parallel_groups(
    fns: Sequence[Callable[[], torch.Tensor]],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float | None:
    if device.type != "cuda" or not fns:
        return None
    streams = [torch.cuda.Stream(device=device) for _ in fns]
    current = torch.cuda.current_stream(device)
    for _ in range(max(0, warmup)):
        gate = torch.cuda.Event()
        gate.record(current)
        for stream, fn in zip(streams, fns):
            stream.wait_event(gate)
            with torch.cuda.stream(stream):
                fn()
        for stream in streams:
            current.wait_stream(stream)
    _sync(device)

    timings = []
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(current)
        for stream, fn in zip(streams, fns):
            stream.wait_event(start)
            with torch.cuda.stream(stream):
                fn()
        for stream in streams:
            current.wait_stream(stream)
        end.record(current)
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return float(sum(timings) / len(timings))


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
    seed_heads_tensor = torch.tensor(seed_heads, device=device, dtype=torch.int32)
    exact_heads = [head for head in range(q_heads) if head not in set(seed_heads)]
    seed_groups = _head_groups_by_kv(seed_heads, q_heads=q_heads, kv_heads=k_true.shape[2])
    exact_groups = _head_groups_by_kv(exact_heads, q_heads=q_heads, kv_heads=k_true.shape[2])

    def dense_all() -> torch.Tensor:
        return _dense_true_gqa(q, k_true, v_true)

    def seed_only_groups() -> List[torch.Tensor]:
        outs = []
        for kv_head, heads in seed_groups.items():
            outs.append(
                gate0_seed_only_attention_triton_forward(
                    _select_q_heads(q, heads),
                    _select_kv_head(k_true, kv_head),
                    _select_kv_head(v_true, kv_head),
                    block_size=args.block_size,
                    sink_blocks=args.sink_blocks,
                    recent_blocks=args.recent_blocks,
                    middle_seed_blocks=args.middle_seed_blocks,
                    block_order=args.block_order,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                )[0]
            )
        return outs

    def seed_only_selected() -> torch.Tensor:
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

    def hybrid_seed_serial() -> torch.Tensor:
        out = torch.empty_like(q)
        for kv_head, heads in exact_groups.items():
            group_out = _dense_true_gqa(
                _select_q_heads(q, heads),
                _select_kv_head(k_true, kv_head),
                _select_kv_head(v_true, kv_head),
            )
            _scatter_heads(out, heads, group_out)
        for kv_head, heads in seed_groups.items():
            group_out = gate0_seed_only_attention_triton_forward(
                _select_q_heads(q, heads),
                _select_kv_head(k_true, kv_head),
                _select_kv_head(v_true, kv_head),
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )[0]
            _scatter_heads(out, heads, group_out)
        return out

    def hybrid_seed_selected_serial() -> torch.Tensor:
        out = torch.empty_like(q)
        for kv_head, heads in exact_groups.items():
            group_out = _dense_true_gqa(
                _select_q_heads(q, heads),
                _select_kv_head(k_true, kv_head),
                _select_kv_head(v_true, kv_head),
            )
            _scatter_heads(out, heads, group_out)
        selected_out = seed_only_selected()
        _scatter_selected(out, seed_heads, selected_out)
        return out

    parallel_fns: List[Callable[[], torch.Tensor]] = []
    for kv_head, heads in exact_groups.items():
        q_group = _select_q_heads(q, heads)
        k_group = _select_kv_head(k_true, kv_head)
        v_group = _select_kv_head(v_true, kv_head)
        parallel_fns.append(
            lambda qg=q_group, kg=k_group, vg=v_group: _dense_true_gqa(qg, kg, vg)
        )
    for kv_head, heads in seed_groups.items():
        q_group = _select_q_heads(q, heads)
        k_group = _select_kv_head(k_true, kv_head)
        v_group = _select_kv_head(v_true, kv_head)
        parallel_fns.append(
            lambda qg=q_group, kg=k_group, vg=v_group: gate0_seed_only_attention_triton_forward(
                qg,
                kg,
                vg,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )[0]
        )

    dense_ms = _time_cuda(dense_all, device=device, warmup=args.warmup, iters=args.iters)
    dense_selected_ms = _time_cuda(
        lambda: _dense_selected_heads(q, k_true, v_true, seed_heads),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    seed_groups_ms = _time_cuda(seed_only_groups, device=device, warmup=args.warmup, iters=args.iters)
    seed_selected_ms = _time_cuda(seed_only_selected, device=device, warmup=args.warmup, iters=args.iters)
    hybrid_ms = _time_cuda(hybrid_seed_serial, device=device, warmup=args.warmup, iters=args.iters)
    hybrid_selected_ms = _time_cuda(
        hybrid_seed_selected_serial,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    parallel_stream_ms = (
        _time_parallel_groups(parallel_fns, device=device, warmup=args.warmup, iters=args.iters)
        if args.measure_parallel_streams
        else None
    )
    group_timings = []
    for kv_head, heads in seed_groups.items():
        q_group = _select_q_heads(q, heads)
        k_group = _select_kv_head(k_true, kv_head)
        v_group = _select_kv_head(v_true, kv_head)
        ms = _time_cuda(
            lambda qg=q_group, kg=k_group, vg=v_group: gate0_seed_only_attention_triton_forward(
                qg,
                kg,
                vg,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )[0],
            device=device,
            warmup=args.group_warmup,
            iters=args.group_iters,
        )
        group_timings.append({"kv_head": kv_head, "q_heads": heads, "seed_only_ms": ms})

    dense_out = dense_all()
    dense_selected_out = _dense_selected_heads(q, k_true, v_true, seed_heads)
    selected_out = seed_only_selected()
    hybrid_out = hybrid_seed_serial()
    hybrid_selected_out = hybrid_seed_selected_serial()
    _sync(device)
    group_parallel_oracle = max([item["seed_only_ms"] for item in group_timings] + [0.0])
    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
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
            "seed_groups": [{"kv_head": kv, "q_heads": heads} for kv, heads in seed_groups.items()],
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "timing": {
            "true_gqa_dense_all_ms": dense_ms,
            "true_gqa_dense_selected_heads_ms": dense_selected_ms,
            "seed_only_groups_ms": seed_groups_ms,
            "seed_only_selected_ms": seed_selected_ms,
            "hybrid_seed_serial_ms": hybrid_ms,
            "hybrid_seed_selected_serial_ms": hybrid_selected_ms,
            "hybrid_seed_parallel_stream_ms": parallel_stream_ms,
            "seed_only_group_parallel_oracle_ms": group_parallel_oracle,
            "seed_only_selected_speedup_vs_dense_selected": (
                dense_selected_ms / seed_selected_ms if seed_selected_ms else None
            ),
            "seed_only_selected_speedup_vs_true_dense": dense_ms / seed_selected_ms if seed_selected_ms else None,
            "seed_only_groups_speedup_vs_true_dense": dense_ms / seed_groups_ms if seed_groups_ms else None,
            "hybrid_seed_serial_speedup_vs_true_dense": dense_ms / hybrid_ms if hybrid_ms else None,
            "hybrid_seed_selected_serial_speedup_vs_true_dense": (
                dense_ms / hybrid_selected_ms if hybrid_selected_ms else None
            ),
            "hybrid_seed_parallel_stream_speedup_vs_true_dense": (
                dense_ms / parallel_stream_ms if parallel_stream_ms else None
            ),
            "seed_group_parallel_oracle_speedup_vs_true_dense": (
                dense_ms / group_parallel_oracle if group_parallel_oracle else None
            ),
        },
        "group_timings": group_timings,
        "quality": {
            "hybrid_seed_error_vs_true_dense": _error(hybrid_out, dense_out),
            "hybrid_seed_error_vs_true_dense_per_head": _per_head_error(hybrid_out, dense_out),
            "hybrid_seed_selected_error_vs_true_dense": _error(hybrid_selected_out, dense_out),
            "hybrid_seed_selected_error_vs_true_dense_per_head": _per_head_error(
                hybrid_selected_out, dense_out
            ),
            "seed_selected_error_vs_dense_selected": _error(selected_out, dense_selected_out),
            "seed_selected_error_vs_dense_selected_per_head": _per_head_error(
                selected_out, dense_selected_out
            ),
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
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--group-warmup", type=int, default=3)
    parser.add_argument("--group-iters", type=int, default=10)
    parser.add_argument("--measure-parallel-streams", action="store_true")
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
