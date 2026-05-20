"""Compare expanded-GQA and true-GQA fused-hybrid Gate-0 runtime.

The capture path historically saves expanded K/V for MHA-shaped Gate-0
profiling. This benchmark can unexpand those tensors back to true GQA by
selecting one KV head per Q-head group, then exercises the StreamAttn wrapper
on both layouts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _error,
    _load_policy,
    _load_tensor,
    _make_cost_model,
    _stats_to_dict,
    _sync,
    _time_cuda,
    _time_wall_cuda,
)
from stream_attention.decode import (  # noqa: E402
    StreamAttnDecodePolicy,
    StreamAttnDecodeWorkspace,
    StreamAttnDecodeWrapper,
)
from stream_attention.gate0_fused_hybrid import (  # noqa: E402
    Gate0FusedHybridPolicy,
    Gate0ProjectionMetadata,
    build_gate0_projection_metadata,
)
from stream_attention.gate1 import dense_attention_forward  # noqa: E402


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _true_gqa_kv(tensor: torch.Tensor, *, true_kv_heads: int) -> torch.Tensor:
    if tensor.shape[2] == true_kv_heads:
        return tensor.contiguous()
    if true_kv_heads <= 0 or tensor.shape[2] % true_kv_heads != 0:
        raise ValueError("expanded head count must be divisible by true_kv_heads")
    group_size = tensor.shape[2] // true_kv_heads
    return tensor[:, :, torch.arange(true_kv_heads, device=tensor.device) * group_size, :].contiguous()


def _dense_true_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_bh = q.permute(0, 2, 1, 3).contiguous()
    k_bh = k.permute(0, 2, 1, 3).contiguous()
    v_bh = v.permute(0, 2, 1, 3).contiguous()
    try:
        out = F.scaled_dot_product_attention(
            q_bh,
            k_bh,
            v_bh,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=q_bh.shape[1] != k_bh.shape[1],
        )
    except TypeError:
        if q_bh.shape[1] != k_bh.shape[1]:
            group_size = q_bh.shape[1] // k_bh.shape[1]
            k_bh = k_bh.repeat_interleave(group_size, dim=1)
            v_bh = v_bh.repeat_interleave(group_size, dim=1)
        out = F.scaled_dot_product_attention(
            q_bh,
            k_bh,
            v_bh,
            dropout_p=0.0,
            is_causal=False,
        )
    return out.permute(0, 2, 1, 3).contiguous()


def _metadata_bytes(metadata: Gate0ProjectionMetadata) -> int:
    return (
        metadata.proj_min.numel() * metadata.proj_min.element_size()
        + metadata.proj_max.numel() * metadata.proj_max.element_size()
        + metadata.projection.numel() * metadata.projection.element_size()
    )


def _profile_wrapper(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    policy: Gate0FusedHybridPolicy,
    metadata: Gate0ProjectionMetadata,
    dense_ms: float,
    error_budget: float,
    safety_margin: float,
    tile_size_q: int,
    num_warps: int,
    num_stages: int,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor, Any, Any]:
    device = q.device
    cost_model = _make_cost_model(
        q,
        k,
        dense_ms=dense_ms,
        attention_type="gqa" if q.shape[2] != k.shape[2] else "mha",
        block_size=policy.block_size,
        tile_size_q=tile_size_q,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    decode_policy = StreamAttnDecodePolicy(
        safety_margin=safety_margin,
        allow_mass=False,
        allow_value_bound=False,
        collect_telemetry_every=0,
        min_kv_len_for_gate0=1,
    )
    workspace = StreamAttnDecodeWorkspace.allocate(
        device=device,
        max_batch=q.shape[0],
        max_query_len=max(q.shape[1], tile_size_q),
        max_kv_len=k.shape[1],
        max_heads=q.shape[2],
        head_dim=q.shape[3],
        block_size=policy.block_size,
        dtype=q.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(
        workspace,
        policy=decode_policy,
        decode_cost_model=cost_model,
        gate0_fused_hybrid_policy=policy,
        gate0_projection_metadata=metadata,
    )
    wrapper.plan(
        query_shape=q.shape,
        kv_shape=k.shape,
        attention_type="gqa" if q.shape[2] != k.shape[2] else "mha",
        kv_heads=k.shape[2],
        block_size=policy.block_size,
        tile_size_q=tile_size_q,
        num_warps=num_warps,
        num_stages=num_stages,
        error_budget=error_budget,
    )
    plan = wrapper.plan_step(q, k)
    if plan.backend != "gate0_fused_hybrid":
        raise RuntimeError(f"wrapper did not select gate0_fused_hybrid: {plan}")
    ms = _time_cuda(
        lambda: wrapper.run(q, k, v, return_info=False),
        device=device,
        warmup=warmup,
        iters=iters,
    )
    out, info = wrapper.run(q, k, v, return_info=True)
    return ms, out, info, plan


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
    error_budget = policy.error_budget if args.error_budget is None else float(args.error_budget)

    expanded_dense_ms = _time_cuda(
        lambda: dense_attention_forward(q, k_expanded, v_expanded, causal=False),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    true_dense_ms = _time_cuda(
        lambda: _dense_true_gqa(q, k_true, v_true),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    expanded_dense_out = dense_attention_forward(q, k_expanded, v_expanded, causal=False)
    true_dense_out = _dense_true_gqa(q, k_true, v_true)

    expanded_metadata_build_ms = _time_wall_cuda(
        lambda: build_gate0_projection_metadata(k_expanded, policy),
        device=device,
        warmup=args.metadata_warmup,
        iters=args.metadata_iters,
    )
    true_metadata_build_ms = _time_wall_cuda(
        lambda: build_gate0_projection_metadata(k_true, true_policy),
        device=device,
        warmup=args.metadata_warmup,
        iters=args.metadata_iters,
    )
    expanded_metadata = build_gate0_projection_metadata(k_expanded, policy)
    true_metadata = build_gate0_projection_metadata(k_true, true_policy)

    expanded_gate0_ms, expanded_gate0_out, expanded_info, expanded_plan = _profile_wrapper(
        q=q,
        k=k_expanded,
        v=v_expanded,
        policy=policy,
        metadata=expanded_metadata,
        dense_ms=expanded_dense_ms,
        error_budget=error_budget,
        safety_margin=args.safety_margin,
        tile_size_q=args.tile_size_q,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        warmup=args.warmup,
        iters=args.iters,
    )
    true_gate0_ms, true_gate0_out, true_info, true_plan = _profile_wrapper(
        q=q,
        k=k_true,
        v=v_true,
        policy=true_policy,
        metadata=true_metadata,
        dense_ms=true_dense_ms,
        error_budget=error_budget,
        safety_margin=args.safety_margin,
        tile_size_q=args.tile_size_q,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        warmup=args.warmup,
        iters=args.iters,
    )

    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "kv_len": int(k_expanded.shape[1]),
            "q_heads": int(q.shape[2]),
            "expanded_kv_heads": int(k_expanded.shape[2]),
            "true_kv_heads": int(k_true.shape[2]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "policy": {
            "trusted_sparse_heads": list(policy.trusted_sparse_heads),
            "exact_heads": list(policy.exact_heads),
            "block_size": policy.block_size,
            "num_chunks": policy.num_chunks,
            "filter_margin": policy.filter_margin,
            "projection_dim": policy.projection_dim,
            "expected_max_abs_error": policy.expected_max_abs_error,
            "expected_speedup_vs_dense": policy.expected_speedup_vs_dense,
            "expanded_policy_kv_heads": policy.kv_heads,
            "true_policy_kv_heads": true_policy.kv_heads,
        },
        "timing": {
            "expanded_dense_ms": expanded_dense_ms,
            "true_gqa_dense_ms": true_dense_ms,
            "expanded_gate0_ms": expanded_gate0_ms,
            "true_gqa_gate0_ms": true_gate0_ms,
            "expanded_speedup_vs_expanded_dense": expanded_dense_ms / expanded_gate0_ms,
            "true_gqa_speedup_vs_true_dense": true_dense_ms / true_gate0_ms,
            "true_gqa_speedup_vs_expanded_dense": expanded_dense_ms / true_gate0_ms,
            "expanded_metadata_build_wall_ms": expanded_metadata_build_ms,
            "true_gqa_metadata_build_wall_ms": true_metadata_build_ms,
        },
        "metadata": {
            "expanded_bytes": _metadata_bytes(expanded_metadata),
            "true_gqa_bytes": _metadata_bytes(true_metadata),
            "byte_ratio_true_over_expanded": _metadata_bytes(true_metadata) / _metadata_bytes(expanded_metadata),
            "expanded_proj_shape": list(expanded_metadata.proj_min.shape),
            "true_gqa_proj_shape": list(true_metadata.proj_min.shape),
        },
        "quality": {
            "true_dense_vs_expanded_dense": _error(true_dense_out, expanded_dense_out),
            "expanded_gate0_vs_expanded_dense": _error(expanded_gate0_out, expanded_dense_out),
            "true_gqa_gate0_vs_true_dense": _error(true_gate0_out, true_dense_out),
            "true_gqa_gate0_vs_expanded_gate0": _error(true_gate0_out, expanded_gate0_out),
        },
        "plans": {
            "expanded": {
                "backend": expanded_plan.backend,
                "reason": expanded_plan.reason,
                "predicted_ms": expanded_plan.predicted_ms,
            },
            "true_gqa": {
                "backend": true_plan.backend,
                "reason": true_plan.reason,
                "predicted_ms": true_plan.predicted_ms,
            },
        },
        "stats": {
            "expanded": _stats_to_dict(expanded_info.stats if expanded_info else None),
            "true_gqa": _stats_to_dict(true_info.stats if true_info else None),
            "expanded_per_head": [
                _stats_to_dict(item) for item in (expanded_info.per_head_stats or ())
            ]
            if expanded_info
            else None,
            "true_gqa_per_head": [
                _stats_to_dict(item) for item in (true_info.per_head_stats or ())
            ]
            if true_info
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--policy-section", choices=["auto", "stable_entries", "entries"], default="auto")
    parser.add_argument("--policy-entry-index", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--tile-size-q", type=int, default=16)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--safety-margin", type=float, default=1.10)
    parser.add_argument("--error-budget", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--metadata-warmup", type=int, default=1)
    parser.add_argument("--metadata-iters", type=int, default=3)
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
