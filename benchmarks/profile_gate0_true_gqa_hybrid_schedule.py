"""Profile true-GQA hybrid scheduling for calibrated Gate-0.

The true-GQA fused-hybrid kernel proved correctness, but it loses to exact
true-GQA dense because it runs custom split-K exact mode for every non-sparse
Q head. This benchmark tests the next scheduling question:

* run exact SDPA for non-sparse Q heads grouped by KV head;
* run Gate-0 fused-hybrid only for trusted sparse Q heads;
* assemble the full head output and compare against true-GQA dense.

It is a science benchmark, not a production runtime. The decisive metric is the
overlap lower bound, ``max(dense_exact_groups_ms, sparse_trusted_groups_ms)``.
If that cannot beat true-GQA dense, a fused/concurrent scheduler is unlikely to
save this path.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_true_gqa import _dense_true_gqa, _true_gqa_kv  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _dtype,
    _error,
    _load_policy,
    _load_tensor,
    _stats_to_dict,
    _sync,
    _time_cuda,
    _time_wall_cuda,
)
from stream_attention.gate0_fused_hybrid import (  # noqa: E402
    Gate0FusedHybridPolicy,
    build_gate0_projection_metadata,
    make_gate0_fused_hybrid_workspace,
    stream_attn_gate0_fused_hybrid,
)


def _parse_heads(raw: str) -> List[int]:
    if not raw:
        return []
    return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))


def _validate_heads(heads: Sequence[int], *, q_heads: int, name: str) -> None:
    bad = [head for head in heads if head < 0 or head >= q_heads]
    if bad:
        raise ValueError(f"{name} contains heads outside [0, {q_heads}): {bad}")


def _head_groups_by_kv(heads: Sequence[int], *, q_heads: int, kv_heads: int) -> Dict[int, List[int]]:
    if q_heads % kv_heads != 0:
        raise ValueError("Q heads must be a multiple of KV heads")
    group_size = q_heads // kv_heads
    groups: Dict[int, List[int]] = {kv_head: [] for kv_head in range(kv_heads)}
    for head in heads:
        groups[int(head) // group_size].append(int(head))
    return {kv_head: group for kv_head, group in groups.items() if group}


def _select_q_heads(q: torch.Tensor, heads: Sequence[int]) -> torch.Tensor:
    if not heads:
        shape = list(q.shape)
        shape[2] = 0
        return torch.empty(shape, device=q.device, dtype=q.dtype)
    index = torch.tensor(list(heads), device=q.device, dtype=torch.long)
    return q.index_select(2, index).contiguous()


def _select_kv_head(tensor: torch.Tensor, kv_head: int) -> torch.Tensor:
    return tensor[:, :, int(kv_head) : int(kv_head) + 1, :].contiguous()


def _scatter_heads(out: torch.Tensor, heads: Sequence[int], values: torch.Tensor) -> None:
    if not heads:
        return
    index = torch.tensor(list(heads), device=out.device, dtype=torch.long)
    out.index_copy_(2, index, values)


def _local_sparse_policy(base: Gate0FusedHybridPolicy, *, local_q_heads: int) -> Gate0FusedHybridPolicy:
    if local_q_heads <= 0:
        raise ValueError("local_q_heads must be positive")
    return replace(
        base,
        head_modes=tuple(0 for _ in range(local_q_heads)),
        trusted_sparse_heads=tuple(range(local_q_heads)),
        exact_heads=(),
        kv_heads=1,
    )


def _metadata_bytes(metadata) -> int:
    return (
        metadata.proj_min.numel() * metadata.proj_min.element_size()
        + metadata.proj_max.numel() * metadata.proj_max.element_size()
        + metadata.projection.numel() * metadata.projection.element_size()
    )


def _safe_div(num: float, den: float) -> float | None:
    return None if den <= 0 else num / den


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    policy = _load_policy(
        args.policy_json,
        section=args.policy_section,
        entry_index=args.policy_entry_index,
    )
    true_policy = replace(policy, kv_heads=args.true_kv_heads)
    q = _load_tensor(args.q_path, key="q", device=device, dtype=dtype)
    k_expanded = _load_tensor(args.k_path, key="k", device=device, dtype=dtype)
    v_expanded = _load_tensor(args.v_path, key="v", device=device, dtype=dtype)
    k_true = _true_gqa_kv(k_expanded, true_kv_heads=args.true_kv_heads)
    v_true = _true_gqa_kv(v_expanded, true_kv_heads=args.true_kv_heads)
    if q.shape[1] != 1:
        raise ValueError("hybrid scheduling benchmark requires query_len == 1")
    if q.shape[2] != policy.heads:
        raise ValueError("query head count does not match policy")
    if q.shape[2] % k_true.shape[2] != 0:
        raise ValueError("query head count must be a multiple of true KV heads")

    q_heads = int(q.shape[2])
    trusted_sparse_heads = (
        _parse_heads(args.trusted_heads)
        if args.trusted_heads
        else list(int(head) for head in policy.trusted_sparse_heads)
    )
    _validate_heads(trusted_sparse_heads, q_heads=q_heads, name="trusted_heads")
    exact_heads = [head for head in range(q_heads) if head not in set(trusted_sparse_heads)]
    dense_groups = _head_groups_by_kv(exact_heads, q_heads=q_heads, kv_heads=k_true.shape[2])
    sparse_groups = _head_groups_by_kv(trusted_sparse_heads, q_heads=q_heads, kv_heads=k_true.shape[2])

    dense_group_items = []
    for kv_head, heads in dense_groups.items():
        dense_group_items.append(
            {
                "kv_head": kv_head,
                "q_heads": heads,
                "q": _select_q_heads(q, heads),
                "k": _select_kv_head(k_true, kv_head),
                "v": _select_kv_head(v_true, kv_head),
            }
        )

    sparse_group_items = []
    sparse_metadata_build_ms = 0.0
    sparse_metadata_bytes = 0
    for kv_head, heads in sparse_groups.items():
        q_group = _select_q_heads(q, heads)
        k_group = _select_kv_head(k_true, kv_head)
        v_group = _select_kv_head(v_true, kv_head)
        local_policy = _local_sparse_policy(true_policy, local_q_heads=len(heads))
        build_ms = _time_wall_cuda(
            lambda kg=k_group, lp=local_policy: build_gate0_projection_metadata(kg, lp),
            device=device,
            warmup=args.metadata_warmup,
            iters=args.metadata_iters,
        )
        metadata = build_gate0_projection_metadata(k_group, local_policy)
        sparse_metadata_build_ms += build_ms
        sparse_metadata_bytes += _metadata_bytes(metadata)
        workspace = make_gate0_fused_hybrid_workspace(q_group, local_policy) if q_group.is_cuda else None
        sparse_group_items.append(
            {
                "kv_head": kv_head,
                "q_heads": heads,
                "q": q_group,
                "k": k_group,
                "v": v_group,
                "policy": local_policy,
                "metadata": metadata,
                "workspace": workspace,
                "metadata_build_ms": build_ms,
                "metadata_bytes": _metadata_bytes(metadata),
            }
        )

    def dense_all() -> torch.Tensor:
        return _dense_true_gqa(q, k_true, v_true)

    def dense_exact_groups() -> torch.Tensor:
        out = torch.empty_like(q)
        for item in dense_group_items:
            group_out = _dense_true_gqa(item["q"], item["k"], item["v"])
            _scatter_heads(out, item["q_heads"], group_out)
        return out

    def sparse_trusted_groups() -> List[torch.Tensor]:
        outs = []
        for item in sparse_group_items:
            outs.append(
                stream_attn_gate0_fused_hybrid(
                    item["q"],
                    item["k"],
                    item["v"],
                    policy=item["policy"],
                    metadata=item["metadata"],
                    workspace=item["workspace"],
                    fallback="dense",
                    return_info=False,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                )
            )
        return outs

    def hybrid_serial() -> torch.Tensor:
        out = torch.empty_like(q)
        for item in dense_group_items:
            group_out = _dense_true_gqa(item["q"], item["k"], item["v"])
            _scatter_heads(out, item["q_heads"], group_out)
        for item in sparse_group_items:
            group_out = stream_attn_gate0_fused_hybrid(
                item["q"],
                item["k"],
                item["v"],
                policy=item["policy"],
                metadata=item["metadata"],
                workspace=item["workspace"],
                fallback="dense",
                return_info=False,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )
            _scatter_heads(out, item["q_heads"], group_out)
        return out

    true_dense_ms = _time_cuda(dense_all, device=device, warmup=args.warmup, iters=args.iters)
    dense_exact_groups_ms = (
        _time_cuda(dense_exact_groups, device=device, warmup=args.warmup, iters=args.iters)
        if dense_group_items
        else 0.0
    )
    sparse_trusted_groups_ms = (
        _time_cuda(sparse_trusted_groups, device=device, warmup=args.warmup, iters=args.iters)
        if sparse_group_items
        else 0.0
    )
    hybrid_serial_ms = _time_cuda(hybrid_serial, device=device, warmup=args.warmup, iters=args.iters)

    dense_group_timings = []
    for item in dense_group_items:
        ms = _time_cuda(
            lambda item=item: _dense_true_gqa(item["q"], item["k"], item["v"]),
            device=device,
            warmup=args.group_warmup,
            iters=args.group_iters,
        )
        dense_group_timings.append(
            {
                "kv_head": item["kv_head"],
                "q_heads": item["q_heads"],
                "head_count": len(item["q_heads"]),
                "dense_ms": ms,
            }
        )

    sparse_group_timings = []
    sparse_infos = []
    for item in sparse_group_items:
        ms = _time_cuda(
            lambda item=item: stream_attn_gate0_fused_hybrid(
                item["q"],
                item["k"],
                item["v"],
                policy=item["policy"],
                metadata=item["metadata"],
                workspace=item["workspace"],
                fallback="dense",
                return_info=False,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            ),
            device=device,
            warmup=args.group_warmup,
            iters=args.group_iters,
        )
        group_out, info = stream_attn_gate0_fused_hybrid(
            item["q"],
            item["k"],
            item["v"],
            policy=item["policy"],
            metadata=item["metadata"],
            workspace=item["workspace"],
            fallback="dense",
            return_info=True,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        _ = group_out
        sparse_infos.append(info)
        sparse_group_timings.append(
            {
                "kv_head": item["kv_head"],
                "q_heads": item["q_heads"],
                "head_count": len(item["q_heads"]),
                "gate0_ms": ms,
                "metadata_build_ms": item["metadata_build_ms"],
                "metadata_bytes": item["metadata_bytes"],
                "stats": _stats_to_dict(info.stats if info else None),
                "per_head_stats": [
                    _stats_to_dict(stat) for stat in (info.per_head_stats or ())
                ]
                if info
                else None,
            }
        )

    dense_out = dense_all()
    exact_out = dense_exact_groups() if dense_group_items else torch.empty_like(q)
    hybrid_out = hybrid_serial()
    _sync(device)

    oracle_max_ms = max(dense_exact_groups_ms, sparse_trusted_groups_ms)
    group_parallel_oracle_ms = max(
        [item["dense_ms"] for item in dense_group_timings]
        + [item["gate0_ms"] for item in sparse_group_timings]
        + [0.0]
    )
    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "kv_len": int(k_expanded.shape[1]),
            "q_heads": q_heads,
            "expanded_kv_heads": int(k_expanded.shape[2]),
            "true_kv_heads": int(k_true.shape[2]),
            "group_size": q_heads // int(k_true.shape[2]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "policy": {
            "trusted_sparse_heads": trusted_sparse_heads,
            "exact_heads": exact_heads,
            "dense_groups": [
                {"kv_head": item["kv_head"], "q_heads": item["q_heads"]}
                for item in dense_group_items
            ],
            "sparse_groups": [
                {"kv_head": item["kv_head"], "q_heads": item["q_heads"]}
                for item in sparse_group_items
            ],
            "block_size": true_policy.block_size,
            "num_chunks": true_policy.num_chunks,
            "filter_margin": true_policy.filter_margin,
            "projection_dim": true_policy.projection_dim,
            "projection_metadata_dtype": true_policy.projection_metadata_dtype,
            "true_policy_kv_heads": true_policy.kv_heads,
        },
        "timing": {
            "true_gqa_dense_all_ms": true_dense_ms,
            "dense_exact_groups_ms": dense_exact_groups_ms,
            "sparse_trusted_groups_ms": sparse_trusted_groups_ms,
            "hybrid_serial_ms": hybrid_serial_ms,
            "hybrid_overlap_oracle_ms": oracle_max_ms,
            "hybrid_group_parallel_oracle_ms": group_parallel_oracle_ms,
            "hybrid_serial_speedup_vs_true_dense": _safe_div(true_dense_ms, hybrid_serial_ms),
            "hybrid_overlap_oracle_speedup_vs_true_dense": _safe_div(true_dense_ms, oracle_max_ms),
            "hybrid_group_parallel_oracle_speedup_vs_true_dense": _safe_div(
                true_dense_ms, group_parallel_oracle_ms
            ),
            "dense_exact_groups_fraction_of_dense": _safe_div(dense_exact_groups_ms, true_dense_ms),
            "sparse_trusted_groups_fraction_of_dense": _safe_div(sparse_trusted_groups_ms, true_dense_ms),
        },
        "metadata": {
            "sparse_metadata_build_wall_ms_sum": sparse_metadata_build_ms,
            "sparse_metadata_bytes_sum": sparse_metadata_bytes,
        },
        "group_timings": {
            "dense_exact": dense_group_timings,
            "sparse_trusted": sparse_group_timings,
        },
        "quality": {
            "hybrid_error_vs_true_dense": _error(hybrid_out, dense_out),
            "dense_exact_groups_error_on_exact_heads": _error(
                exact_out.index_select(
                    2,
                    torch.tensor(exact_heads, device=exact_out.device, dtype=torch.long),
                ).contiguous()
                if exact_heads
                else torch.empty_like(q[:, :, :0, :]),
                dense_out.index_select(
                    2,
                    torch.tensor(exact_heads, device=dense_out.device, dtype=torch.long),
                ).contiguous()
                if exact_heads
                else torch.empty_like(q[:, :, :0, :]),
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--trusted-heads", default="")
    parser.add_argument("--policy-section", choices=["auto", "stable_entries", "entries"], default="auto")
    parser.add_argument("--policy-entry-index", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--group-warmup", type=int, default=3)
    parser.add_argument("--group-iters", type=int, default=10)
    parser.add_argument("--metadata-warmup", type=int, default=1)
    parser.add_argument("--metadata-iters", type=int, default=2)
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
