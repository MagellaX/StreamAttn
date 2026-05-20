"""Profile StreamAttnDecodeWrapper with calibrated fused-hybrid Gate-0.

This benchmark is intentionally runtime-facing: it exercises the same
``StreamAttnDecodeWrapper`` path that callers use, rather than calling the
Gate-0 split-K kernel directly as a research microbenchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stream_attention.decode import (  # noqa: E402
    DecodeCostEntry,
    DecodeCostKey,
    DecodeCostModel,
    StreamAttnDecodePolicy,
    StreamAttnDecodeWorkspace,
    StreamAttnDecodeWrapper,
)
from stream_attention.gate0_fused_hybrid import (  # noqa: E402
    Gate0FusedHybridPolicy,
    build_gate0_projection_metadata,
    make_gate0_fused_hybrid_workspace,
    stream_attn_gate0_fused_hybrid,
)
from stream_attention.gate1 import dense_attention_forward  # noqa: E402


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _load_tensor(path: str, *, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        tensor = payload.get(key)
        if tensor is None:
            if len(payload) != 1:
                raise ValueError(f"{path} does not contain key '{key}'")
            tensor = next(iter(payload.values()))
    else:
        tensor = payload
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} did not contain a tensor")
    return tensor.to(device=device, dtype=dtype).contiguous()


def _load_policy(path: str, *, section: str, entry_index: int) -> Gate0FusedHybridPolicy:
    if section == "auto":
        return Gate0FusedHybridPolicy.from_json(path, entry_index=entry_index)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get(section) or []
    if not entries:
        raise ValueError(f"policy JSON has no '{section}' entries")
    return Gate0FusedHybridPolicy.from_entry(entries[entry_index])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_cuda(fn, *, device: torch.device, warmup: int, iters: int) -> float:
    _sync(device)
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type != "cuda":
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - start) * 1000.0 / max(1, iters)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize(device)
    return start_event.elapsed_time(end_event) / max(1, iters)


def _time_wall_cuda(fn, *, device: torch.device, warmup: int, iters: int) -> float:
    _sync(device)
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


def _error(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    diff = (actual - expected).detach().abs().float()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
    }


def _stats_to_dict(value: Any):
    if value is None:
        return None
    if is_dataclass(value):
        data = asdict(value)
        if hasattr(value, "projection_skip_fraction"):
            data["projection_skip_fraction"] = float(value.projection_skip_fraction)
        if hasattr(value, "pv_executed_fraction"):
            data["pv_executed_fraction"] = float(value.pv_executed_fraction)
        return data
    return value


def _make_cost_model(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    dense_ms: float,
    block_size: int,
    tile_size_q: int,
    num_warps: int,
    num_stages: int,
) -> DecodeCostModel:
    model = DecodeCostModel()
    key = DecodeCostKey.from_tensors(
        q,
        k,
        kv_heads=k.shape[2],
        attention_type="mha",
        block_size=block_size,
        tile_size_q=tile_size_q,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    model.update(
        key,
        DecodeCostEntry.from_measurement(
            dense_ms=dense_ms,
            qk_scan_ms=max(dense_ms, 1.0e-6),
            gate1_mass_ms=max(dense_ms, 1.0e-6),
            active_fraction=1.0,
        ),
    )
    return model


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    q = _load_tensor(args.q_path, key="q", device=device, dtype=dtype)
    k = _load_tensor(args.k_path, key="k", device=device, dtype=dtype)
    v = _load_tensor(args.v_path, key="v", device=device, dtype=dtype)
    policy = _load_policy(
        args.policy_json,
        section=args.policy_section,
        entry_index=args.policy_entry_index,
    )
    if args.error_budget is None:
        error_budget = float(policy.error_budget)
    else:
        error_budget = float(args.error_budget)

    dense_ms = _time_cuda(
        lambda: dense_attention_forward(q, k, v, causal=False),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    dense_out = dense_attention_forward(q, k, v, causal=False)

    metadata_build_wall_ms = _time_wall_cuda(
        lambda: build_gate0_projection_metadata(k, policy),
        device=device,
        warmup=args.metadata_warmup,
        iters=args.metadata_iters,
    )
    metadata = build_gate0_projection_metadata(k, policy)

    direct_workspace = make_gate0_fused_hybrid_workspace(q, policy) if q.is_cuda else None
    direct_ms = _time_cuda(
        lambda: stream_attn_gate0_fused_hybrid(
            q,
            k,
            v,
            policy=policy,
            metadata=metadata,
            workspace=direct_workspace,
            return_info=False,
            fallback="dense",
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        ),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    cost_model = _make_cost_model(
        q,
        k,
        dense_ms=dense_ms,
        block_size=policy.block_size,
        tile_size_q=args.tile_size_q,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    decode_policy = StreamAttnDecodePolicy(
        safety_margin=args.safety_margin,
        allow_mass=False,
        allow_value_bound=False,
        collect_telemetry_every=0,
        min_kv_len_for_gate0=args.min_kv_len_for_gate0,
    )
    workspace = StreamAttnDecodeWorkspace.allocate(
        device=device,
        max_batch=q.shape[0],
        max_query_len=max(q.shape[1], args.tile_size_q),
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
        attention_type="mha",
        kv_heads=k.shape[2],
        block_size=policy.block_size,
        tile_size_q=args.tile_size_q,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        error_budget=error_budget,
    )
    plan = wrapper.plan_step(q, k)
    if args.require_gate0 and plan.backend != "gate0_fused_hybrid":
        raise RuntimeError(f"wrapper did not select gate0_fused_hybrid: {plan}")

    wrapper_ms = _time_cuda(
        lambda: wrapper.run(q, k, v, return_info=False),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    wrapper_out, wrapper_info = wrapper.run(q, k, v, return_info=True)
    direct_out, direct_info = stream_attn_gate0_fused_hybrid(
        q,
        k,
        v,
        policy=policy,
        metadata=metadata,
        workspace=direct_workspace,
        return_info=True,
        fallback="dense",
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )

    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "kv_len": int(k.shape[1]),
            "heads": int(q.shape[2]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "policy": {
            "model_id": policy.model_id,
            "layer_id": policy.layer_id,
            "kv_len_bucket": policy.kv_len_bucket,
            "trusted_sparse_heads": list(policy.trusted_sparse_heads),
            "exact_heads": list(policy.exact_heads),
            "expected_speedup_vs_dense": policy.expected_speedup_vs_dense,
            "expected_max_abs_error": policy.expected_max_abs_error,
            "block_size": policy.block_size,
            "num_chunks": policy.num_chunks,
            "filter_margin": policy.filter_margin,
            "projection_dim": policy.projection_dim,
            "safety_budget_name": policy.safety_budget_name,
        },
        "plan": {
            "backend": plan.backend,
            "reason": plan.reason,
            "predicted_ms": plan.predicted_ms,
            "dense_ms": plan.dense_ms,
            "metadata_update_required": plan.metadata_update_required,
            "projection_metadata_required": plan.projection_metadata_required,
        },
        "timing": {
            "dense_ms": dense_ms,
            "direct_gate0_ms": direct_ms,
            "wrapper_gate0_ms": wrapper_ms,
            "metadata_build_wall_ms": metadata_build_wall_ms,
            "direct_speedup_vs_dense": dense_ms / direct_ms if direct_ms > 0 else None,
            "wrapper_speedup_vs_dense": dense_ms / wrapper_ms if wrapper_ms > 0 else None,
            "wrapper_over_direct": wrapper_ms / direct_ms if direct_ms > 0 else None,
        },
        "quality": {
            "wrapper_error_vs_dense": _error(wrapper_out, dense_out),
            "direct_error_vs_dense": _error(direct_out, dense_out),
            "wrapper_error_vs_direct": _error(wrapper_out, direct_out),
        },
        "stats": {
            "wrapper": _stats_to_dict(wrapper_info.stats if wrapper_info else None),
            "direct": _stats_to_dict(direct_info.stats if direct_info else None),
            "wrapper_per_head": [
                _stats_to_dict(item) for item in (wrapper_info.per_head_stats or ())
            ]
            if wrapper_info
            else None,
            "direct_per_head": [
                _stats_to_dict(item) for item in (direct_info.per_head_stats or ())
            ]
            if direct_info
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--policy-section", choices=["auto", "stable_entries", "entries"], default="auto")
    parser.add_argument("--policy-entry-index", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--tile-size-q", type=int, default=16)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--safety-margin", type=float, default=1.10)
    parser.add_argument("--min-kv-len-for-gate0", type=int, default=16384)
    parser.add_argument("--error-budget", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--metadata-warmup", type=int, default=1)
    parser.add_argument("--metadata-iters", type=int, default=3)
    parser.add_argument("--no-require-gate0", dest="require_gate0", action="store_false")
    parser.add_argument("--summary-json-out", default="")
    parser.set_defaults(require_gate0=True)
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
