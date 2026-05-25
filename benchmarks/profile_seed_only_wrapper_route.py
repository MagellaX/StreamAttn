"""Benchmark the policy-aware StreamAttnDecodeWrapper seed-only route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_true_gqa import _true_gqa_kv  # noqa: E402
from benchmarks.profile_seed_only_batch_scaling import (  # noqa: E402
    _make_flashinfer_wrapper,
    _make_paged_kv_cache,
)
from benchmarks.profile_seed_only_captured_batch import _rows_for_layer  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype, _error, _load_tensor, _sync, _time_cuda  # noqa: E402
from stream_attention.decode import (  # noqa: E402
    Gate0SeedOnlyBatchedPolicy,
    StreamAttnDecodePolicy,
    StreamAttnDecodeWorkspace,
    StreamAttnDecodeWrapper,
)
from stream_attention.kernels.gate0_seed_only_triton import gate0_seed_only_attention_triton_forward_out  # noqa: E402


def _stack_captures(rows: list[dict[str, Any]], *, device: torch.device, dtype: torch.dtype):
    q_parts = []
    k_parts = []
    v_parts = []
    true_kv_heads = None
    prompt_ids = []
    for row in rows:
        row_true_kv_heads = int((row.get("meta") or {}).get("logical_num_kv_heads") or row["shape"]["heads"])
        true_kv_heads = row_true_kv_heads if true_kv_heads is None else true_kv_heads
        if row_true_kv_heads != true_kv_heads:
            raise ValueError("all rows must have the same true KV head count")
        q_parts.append(_load_tensor(row["q_path"], key="q", device=device, dtype=dtype))
        k_expanded = _load_tensor(row["k_path"], key="k", device=device, dtype=dtype)
        v_expanded = _load_tensor(row["v_path"], key="v", device=device, dtype=dtype)
        k_parts.append(_true_gqa_kv(k_expanded, true_kv_heads=true_kv_heads).contiguous())
        v_parts.append(_true_gqa_kv(v_expanded, true_kv_heads=true_kv_heads).contiguous())
        prompt_ids.append(row.get("prompt_id"))
    return torch.cat(q_parts, dim=0), torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0), true_kv_heads, prompt_ids


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    rows = _rows_for_layer(metadata, args.layer_id)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    q, k, v, true_kv_heads, prompt_ids = _stack_captures(rows, device=device, dtype=dtype)

    if args.policy_json:
        seed_policy = Gate0SeedOnlyBatchedPolicy.from_json(args.policy_json)
    else:
        seed_policy = Gate0SeedOnlyBatchedPolicy(
            model_id=metadata.get("model_id") or "unknown",
            layer_id=args.layer_id,
            dtype=args.dtype,
            kv_len_bucket=k.shape[1],
            min_batch=max(1, int(args.min_batch)),
            heads=q.shape[2],
            kv_heads=k.shape[2],
            dim=q.shape[3],
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
            expected_dense_ms=args.expected_dense_ms,
            expected_seed_only_ms=args.expected_seed_only_ms,
            expected_speedup_vs_dense=args.expected_speedup_vs_dense,
        )
    if seed_policy.min_batch > q.shape[0]:
        seed_policy = Gate0SeedOnlyBatchedPolicy(
            **{**seed_policy.__dict__, "min_batch": q.shape[0]}
        )

    out = torch.empty_like(q)
    workspace = StreamAttnDecodeWorkspace.allocate(
        device=device,
        max_batch=q.shape[0],
        max_query_len=q.shape[1],
        max_kv_len=k.shape[1],
        max_heads=q.shape[2],
        head_dim=q.shape[3],
        block_size=seed_policy.block_size,
        dtype=q.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(
        workspace,
        policy=StreamAttnDecodePolicy(
            collect_telemetry_every=0,
            min_kv_len_for_gate0_seed_only=1,
            safety_margin=args.safety_margin,
        ),
        gate0_seed_only_batched_policy=seed_policy,
    )
    wrapper.plan(
        query_shape=q.shape,
        kv_shape=k.shape,
        block_size=seed_policy.block_size,
        tile_size_q=16,
        num_warps=seed_policy.num_warps,
        num_stages=seed_policy.num_stages,
    )

    kv_cache, indptr, indices, last_page_len = _make_paged_kv_cache(k, v, page_size=args.page_size)
    flash_workspace = torch.empty(args.workspace_mb * 1024 * 1024, device=device, dtype=torch.uint8)
    flash_wrapper = _make_flashinfer_wrapper(
        workspace=flash_workspace,
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        q_heads=q.shape[2],
        kv_heads=k.shape[2],
        dim=q.shape[3],
        page_size=args.page_size,
        dtype=dtype,
        backend=args.flashinfer_backend,
        use_tensor_cores=args.flashinfer_tensor_cores,
        disable_split_kv=args.disable_split_kv,
    )

    def direct_seed_run() -> torch.Tensor:
        return gate0_seed_only_attention_triton_forward_out(
            q,
            k,
            v,
            out,
            block_size=seed_policy.block_size,
            sink_blocks=seed_policy.sink_blocks,
            recent_blocks=seed_policy.recent_blocks,
            middle_seed_blocks=seed_policy.middle_seed_blocks,
            block_order=seed_policy.block_order,
            num_warps=seed_policy.num_warps,
            num_stages=seed_policy.num_stages,
        )

    def wrapper_run() -> torch.Tensor:
        return wrapper.run(q, k, v, active_fraction_hint=1.0)

    def flashinfer_run() -> torch.Tensor:
        return flash_wrapper.run(q[:, 0].contiguous(), kv_cache).view_as(q)

    wrapper_ref, info = wrapper.run(q, k, v, active_fraction_hint=1.0, return_info=True)
    direct_ref = direct_seed_run().clone()
    flash_ref = flashinfer_run().clone()
    _sync(device)
    wrapper_ms = _time_cuda(wrapper_run, device=device, warmup=args.warmup, iters=args.iters)
    direct_ms = _time_cuda(direct_seed_run, device=device, warmup=args.warmup, iters=args.iters)
    flash_ms = _time_cuda(flashinfer_run, device=device, warmup=args.warmup, iters=args.iters)
    batch = int(q.shape[0])
    return {
        "schema": "streamattn.gate0.seed_only_wrapper_route.v1",
        "capture": {
            "metadata_json": args.metadata_json,
            "layer_id": args.layer_id,
            "prompt_ids": prompt_ids,
            "row_count": batch,
        },
        "shape": {
            "batch": batch,
            "query_len": int(q.shape[1]),
            "q_heads": int(q.shape[2]),
            "true_kv_heads": int(true_kv_heads),
            "group_size": int(q.shape[2] // true_kv_heads),
            "kv_len": int(k.shape[1]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "policy": {
            "policy_id": seed_policy.policy_id,
            "mode": seed_policy.mode,
            "min_batch": seed_policy.min_batch,
            "block_size": seed_policy.block_size,
            "sink_blocks": seed_policy.sink_blocks,
            "recent_blocks": seed_policy.recent_blocks,
            "middle_seed_blocks": seed_policy.middle_seed_blocks,
            "block_order": seed_policy.block_order,
            "expected_seed_only_ms": seed_policy.expected_seed_only_ms,
            "expected_dense_ms": seed_policy.expected_dense_ms,
            "expected_speedup_vs_dense": seed_policy.expected_speedup_vs_dense,
        },
        "route": {
            "plan_backend": wrapper.last_plan.backend if wrapper.last_plan is not None else None,
            "plan_reason": wrapper.last_plan.reason if wrapper.last_plan is not None else None,
            "backend_used": info.backend_used,
            "fallback_reason": info.fallback_reason,
        },
        "timing": {
            "wrapper_seed_only_ms": wrapper_ms,
            "direct_seed_only_ms": direct_ms,
            "flashinfer_batch_tc_exact_ms": flash_ms,
            "wrapper_over_direct_ms": wrapper_ms - direct_ms,
            "wrapper_speedup_vs_flashinfer": flash_ms / wrapper_ms,
            "direct_speedup_vs_flashinfer": flash_ms / direct_ms,
            "wrapper_per_row_us": wrapper_ms * 1000.0 / batch,
            "flashinfer_per_row_us": flash_ms * 1000.0 / batch,
        },
        "quality": {
            "wrapper_vs_direct_seed": _error(wrapper_ref, direct_ref),
            "wrapper_vs_flashinfer_exact": _error(wrapper_ref, flash_ref),
        },
        "decision": "wrapper_seed_only_beats_flashinfer"
        if wrapper_ms < flash_ms and info.backend_used == "gate0_seed_only_batched"
        else "wrapper_seed_only_not_viable",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--policy-json", default="")
    parser.add_argument("--layer-id", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-batch", type=int, default=8)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first", "sink_recent_first"])
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--expected-seed-only-ms", type=float, default=0.03559)
    parser.add_argument("--expected-dense-ms", type=float, default=0.05698)
    parser.add_argument("--expected-speedup-vs-dense", type=float, default=1.60)
    parser.add_argument("--safety-margin", type=float, default=1.10)
    parser.add_argument("--flashinfer-backend", default="auto")
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--disable-split-kv", action="store_true")
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
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
