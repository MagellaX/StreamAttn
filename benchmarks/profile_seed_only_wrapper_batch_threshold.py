"""Measure the batched seed-only wrapper threshold on captured Qwen rows."""

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

from benchmarks.profile_seed_only_captured_batch import _rows_for_layer  # noqa: E402
from benchmarks.profile_seed_only_wrapper_route import _stack_captures  # noqa: E402
from benchmarks.profile_seed_only_batch_scaling import (  # noqa: E402
    _make_flashinfer_wrapper,
    _make_paged_kv_cache,
)
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype, _error, _sync, _time_cuda  # noqa: E402
from stream_attention.decode import (  # noqa: E402
    Gate0SeedOnlyBatchedPolicy,
    StreamAttnDecodePolicy,
    StreamAttnSeedOnlyDecodeService,
    StreamAttnDecodeWorkspace,
    StreamAttnDecodeWrapper,
    stream_attn_decode_plan,
)
from stream_attention.kernels.gate0_seed_only_triton import gate0_seed_only_attention_triton_forward_out  # noqa: E402


def _clone_policy(policy: Gate0SeedOnlyBatchedPolicy, **updates) -> Gate0SeedOnlyBatchedPolicy:
    payload = {**policy.__dict__, **updates}
    return Gate0SeedOnlyBatchedPolicy(**payload)


def _parse_batch_sizes(text: str) -> List[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("batch-sizes must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError("batch sizes must be positive")
    return sorted(set(values))


def _make_policy(args: argparse.Namespace, metadata: Dict[str, Any], q, k) -> Gate0SeedOnlyBatchedPolicy:
    if args.policy_json:
        return Gate0SeedOnlyBatchedPolicy.from_json(args.policy_json)
    return Gate0SeedOnlyBatchedPolicy(
        policy_id="ad_hoc_seed_only_threshold",
        model_id=metadata.get("model_id") or "unknown",
        layer_id=args.layer_id,
        dtype=args.dtype,
        kv_len_bucket=k.shape[1],
        min_batch=args.product_min_batch,
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
        expected_seed_only_ms=args.expected_seed_only_ms,
        expected_dense_ms=args.expected_dense_ms,
        expected_speedup_vs_dense=args.expected_speedup_vs_dense,
        max_kl=args.max_kl,
        max_logprob_delta=args.max_logprob_delta,
    )


def _profile_batch(
    *,
    args: argparse.Namespace,
    product_policy: Gate0SeedOnlyBatchedPolicy,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    true_kv_heads: int,
    prompt_ids: List[Any],
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    q_b = q[:batch].contiguous()
    k_b = k[:batch].contiguous()
    v_b = v[:batch].contiguous()
    prompt_b = prompt_ids[:batch]
    product_plan = stream_attn_decode_plan(
        q_b,
        k_b,
        gate0_seed_only_batched_policy=product_policy,
        policy=StreamAttnDecodePolicy(
            collect_telemetry_every=0,
            min_kv_len_for_gate0_seed_only=1,
            safety_margin=args.safety_margin,
        ),
        active_fraction_hint=1.0,
        block_size=product_policy.block_size,
        tile_size_q=16,
        num_warps=product_policy.num_warps,
        num_stages=product_policy.num_stages,
    )

    forced_policy = _clone_policy(product_policy, min_batch=args.forced_min_batch)
    out = torch.empty_like(q_b)
    workspace = StreamAttnDecodeWorkspace.allocate(
        device=device,
        max_batch=batch,
        max_query_len=q_b.shape[1],
        max_kv_len=k_b.shape[1],
        max_heads=q_b.shape[2],
        head_dim=q_b.shape[3],
        block_size=forced_policy.block_size,
        dtype=q_b.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(
        workspace,
        policy=StreamAttnDecodePolicy(
            collect_telemetry_every=0,
            min_kv_len_for_gate0_seed_only=1,
            safety_margin=args.safety_margin,
        ),
        gate0_seed_only_batched_policy=forced_policy,
    )
    wrapper.plan(
        query_shape=q_b.shape,
        kv_shape=k_b.shape,
        block_size=forced_policy.block_size,
        tile_size_q=16,
        num_warps=forced_policy.num_warps,
        num_stages=forced_policy.num_stages,
    )

    kv_cache, indptr, indices, last_page_len = _make_paged_kv_cache(
        k_b,
        v_b,
        page_size=args.page_size,
    )
    flash_workspace = torch.empty(args.workspace_mb * 1024 * 1024, device=device, dtype=torch.uint8)
    flash_wrapper = _make_flashinfer_wrapper(
        workspace=flash_workspace,
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        q_heads=q_b.shape[2],
        kv_heads=k_b.shape[2],
        dim=q_b.shape[3],
        page_size=args.page_size,
        dtype=dtype,
        backend=args.flashinfer_backend,
        use_tensor_cores=args.flashinfer_tensor_cores,
        disable_split_kv=args.disable_split_kv,
    )

    def direct_seed_run() -> torch.Tensor:
        return gate0_seed_only_attention_triton_forward_out(
            q_b,
            k_b,
            v_b,
            out,
            block_size=forced_policy.block_size,
            sink_blocks=forced_policy.sink_blocks,
            recent_blocks=forced_policy.recent_blocks,
            middle_seed_blocks=forced_policy.middle_seed_blocks,
            block_order=forced_policy.block_order,
            num_warps=forced_policy.num_warps,
            num_stages=forced_policy.num_stages,
        )

    def forced_wrapper_run() -> torch.Tensor:
        return wrapper.run(q_b, k_b, v_b, active_fraction_hint=1.0)

    def flashinfer_run() -> torch.Tensor:
        return flash_wrapper.run(q_b[:, 0].contiguous(), kv_cache).view_as(q_b)

    product_workspace = StreamAttnDecodeWorkspace.allocate(
        device=device,
        max_batch=batch,
        max_query_len=q_b.shape[1],
        max_kv_len=k_b.shape[1],
        max_heads=q_b.shape[2],
        head_dim=q_b.shape[3],
        block_size=product_policy.block_size,
        dtype=q_b.dtype,
    )
    product_wrapper = StreamAttnDecodeWrapper(
        product_workspace,
        policy=StreamAttnDecodePolicy(
            collect_telemetry_every=0,
            min_kv_len_for_gate0_seed_only=1,
            safety_margin=args.safety_margin,
        ),
        gate0_seed_only_batched_policy=product_policy,
        dense_fallback=lambda query, key, value: flash_wrapper.run(
            query[:, 0].contiguous(),
            kv_cache,
        ).view_as(query),
        dense_fallback_backend="flashinfer_dense",
    )
    product_wrapper.plan(
        query_shape=q_b.shape,
        kv_shape=k_b.shape,
        block_size=product_policy.block_size,
        tile_size_q=16,
        num_warps=product_policy.num_warps,
        num_stages=product_policy.num_stages,
    )

    def product_wrapper_run() -> torch.Tensor:
        return product_wrapper.run(q_b, k_b, v_b, active_fraction_hint=1.0)

    service = StreamAttnSeedOnlyDecodeService(
        policy=product_policy,
        decode_policy=StreamAttnDecodePolicy(
            collect_telemetry_every=0,
            min_kv_len_for_gate0_seed_only=1,
            safety_margin=args.safety_margin,
        ),
        dense_fallback=lambda query, key, value: flash_wrapper.run(
            query[:, 0].contiguous(),
            kv_cache,
        ).view_as(query),
        dense_fallback_backend="flashinfer_dense",
    )
    planned_direct_service = StreamAttnSeedOnlyDecodeService(
        policy=forced_policy,
        decode_policy=StreamAttnDecodePolicy(
            collect_telemetry_every=0,
            min_kv_len_for_gate0_seed_only=1,
            safety_margin=args.safety_margin,
        ),
        dense_fallback=lambda query, key, value: flash_wrapper.run(
            query[:, 0].contiguous(),
            kv_cache,
        ).view_as(query),
        dense_fallback_backend="flashinfer_dense",
    )
    planned_direct_runner = planned_direct_service.plan_direct_seed_only(q_b, k_b, v_b)

    def service_run() -> torch.Tensor:
        return service.run(q_b, k_b, v_b, active_fraction_hint=1.0, return_info=False)

    def planned_direct_run() -> torch.Tensor:
        return planned_direct_runner.run()

    def planned_direct_mutation_check() -> Dict[str, Any]:
        q_mut = q_b.clone()
        mutation_service = StreamAttnSeedOnlyDecodeService(
            policy=forced_policy,
            decode_policy=StreamAttnDecodePolicy(
                collect_telemetry_every=0,
                min_kv_len_for_gate0_seed_only=1,
                safety_margin=args.safety_margin,
            ),
            dense_fallback=lambda query, key, value: flash_wrapper.run(
                query[:, 0].contiguous(),
                kv_cache,
            ).view_as(query),
            dense_fallback_backend="flashinfer_dense",
        )
        runner = mutation_service.plan_direct_seed_only(q_mut, k_b, v_b)
        before = runner.run().clone()
        q_mut[:, :, :, 0].add_(1.0)
        after = runner.run().clone()
        delta = (after - before).detach().abs().float()
        max_change = float(delta.max().item())
        return {
            "planned_direct_mutation_max_change": max_change,
            "planned_direct_mutation_changed_output": bool(max_change > 0.0),
        }

    wrapper_ref, info = wrapper.run(q_b, k_b, v_b, active_fraction_hint=1.0, return_info=True)
    product_ref, product_info = product_wrapper.run(
        q_b,
        k_b,
        v_b,
        active_fraction_hint=1.0,
        return_info=True,
    )
    planned_direct_ref, planned_direct_info = planned_direct_runner.run_with_info()
    service_ref, service_info = service.run(
        q_b,
        k_b,
        v_b,
        active_fraction_hint=1.0,
        return_info=True,
    )
    mutation = planned_direct_mutation_check()
    direct_ref = direct_seed_run().clone()
    flash_ref = flashinfer_run().clone()
    _sync(device)
    planned_direct_ms = _time_cuda(
        planned_direct_run,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    service_ms = _time_cuda(service_run, device=device, warmup=args.warmup, iters=args.iters)
    product_wrapper_ms = _time_cuda(
        product_wrapper_run,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    wrapper_ms = _time_cuda(forced_wrapper_run, device=device, warmup=args.warmup, iters=args.iters)
    direct_ms = _time_cuda(direct_seed_run, device=device, warmup=args.warmup, iters=args.iters)
    flash_ms = _time_cuda(flashinfer_run, device=device, warmup=args.warmup, iters=args.iters)

    return {
        "batch": batch,
        "prompt_ids": prompt_b,
        "product_route": {
            "backend": product_plan.backend,
            "reason": product_plan.reason,
            "min_batch": product_policy.min_batch,
        },
        "forced_route": {
            "backend_used": info.backend_used,
            "fallback_reason": info.fallback_reason,
            "runtime_counters": wrapper.runtime_counters(),
        },
        "product_wrapper_route": {
            "backend_used": getattr(product_info, "backend_used", None),
            "fallback_reason": getattr(product_info, "fallback_reason", None),
            "runtime_counters": product_wrapper.runtime_counters(),
        },
        "service_route": {
            "backend_used": getattr(service_info, "backend_used", None),
            "fallback_reason": getattr(service_info, "fallback_reason", None),
            "plan_reason": getattr(service_info, "plan_reason", None),
            "runtime_counters": getattr(service_info, "runtime_counters", None),
        },
        "planned_direct_route": {
            "backend_used": getattr(planned_direct_info, "backend_used", None),
            "fallback_reason": getattr(planned_direct_info, "fallback_reason", None),
            "plan_reason": getattr(planned_direct_info, "plan_reason", None),
            "runtime_counters": getattr(planned_direct_info, "runtime_counters", None),
        },
        "shape": {
            "query_len": int(q_b.shape[1]),
            "q_heads": int(q_b.shape[2]),
            "true_kv_heads": int(true_kv_heads),
            "group_size": int(q_b.shape[2] // true_kv_heads),
            "kv_len": int(k_b.shape[1]),
            "dim": int(q_b.shape[3]),
        },
        "timing": {
            "forced_wrapper_seed_only_ms": wrapper_ms,
            "product_wrapper_ms": product_wrapper_ms,
            "service_ms": service_ms,
            "planned_direct_seed_only_ms": planned_direct_ms,
            "direct_seed_only_ms": direct_ms,
            "flashinfer_batch_tc_exact_ms": flash_ms,
            "product_wrapper_over_flashinfer_ms": product_wrapper_ms - flash_ms,
            "service_over_flashinfer_ms": service_ms - flash_ms,
            "service_over_direct_ms": service_ms - direct_ms,
            "planned_direct_over_flashinfer_ms": planned_direct_ms - flash_ms,
            "planned_direct_over_direct_ms": planned_direct_ms - direct_ms,
            "wrapper_over_direct_ms": wrapper_ms - direct_ms,
            "product_wrapper_speedup_vs_flashinfer": flash_ms / product_wrapper_ms,
            "service_speedup_vs_flashinfer": flash_ms / service_ms,
            "planned_direct_speedup_vs_flashinfer": flash_ms / planned_direct_ms,
            "forced_wrapper_speedup_vs_flashinfer": flash_ms / wrapper_ms,
            "direct_speedup_vs_flashinfer": flash_ms / direct_ms,
            "product_wrapper_per_row_us": product_wrapper_ms * 1000.0 / batch,
            "service_per_row_us": service_ms * 1000.0 / batch,
            "planned_direct_per_row_us": planned_direct_ms * 1000.0 / batch,
            "forced_wrapper_per_row_us": wrapper_ms * 1000.0 / batch,
            "flashinfer_per_row_us": flash_ms * 1000.0 / batch,
        },
        "quality": {
            "product_wrapper_vs_flashinfer_exact": _error(product_ref, flash_ref),
            "planned_direct_vs_direct_seed": _error(planned_direct_ref, direct_ref),
            "planned_direct_vs_flashinfer_exact": _error(planned_direct_ref, flash_ref),
            "service_vs_direct_seed": _error(service_ref, direct_ref),
            "service_vs_flashinfer_exact": _error(service_ref, flash_ref),
            "wrapper_vs_direct_seed": _error(wrapper_ref, direct_ref),
            "wrapper_vs_flashinfer_exact": _error(wrapper_ref, flash_ref),
        },
        "diagnostics": mutation,
        "decision": {
            "product_wrapper_beats_flashinfer": bool(product_wrapper_ms < flash_ms),
            "planned_direct_beats_flashinfer": bool(
                planned_direct_ms < flash_ms
                and getattr(planned_direct_info, "backend_used", None) == "gate0_seed_only_batched"
            ),
            "service_beats_flashinfer": bool(
                service_ms < flash_ms
                and getattr(service_info, "backend_used", None) == "gate0_seed_only_batched"
            ),
            "forced_wrapper_beats_flashinfer": bool(
                wrapper_ms < flash_ms and info.backend_used == "gate0_seed_only_batched"
            ),
            "direct_seed_beats_flashinfer": bool(direct_ms < flash_ms),
            "product_route_uses_seed_only": product_plan.backend == "gate0_seed_only_batched",
        },
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    rows = _rows_for_layer(metadata, args.layer_id)
    max_batch = max(batch_sizes)
    if len(rows) < max_batch:
        raise ValueError(f"metadata has {len(rows)} rows, but max batch size is {max_batch}")
    rows = rows[:max_batch]
    q, k, v, true_kv_heads, prompt_ids = _stack_captures(rows, device=device, dtype=dtype)
    product_policy = _make_policy(args, metadata, q, k)
    if product_policy.min_batch != args.product_min_batch:
        product_policy = _clone_policy(product_policy, min_batch=args.product_min_batch)
    entries = [
        _profile_batch(
            args=args,
            product_policy=product_policy,
            q=q,
            k=k,
            v=v,
            true_kv_heads=true_kv_heads,
            prompt_ids=prompt_ids,
            batch=batch,
            device=device,
            dtype=dtype,
        )
        for batch in batch_sizes
    ]

    def _first_batch(predicate):
        for entry in entries:
            if predicate(entry):
                return entry["batch"]
        return None

    return {
        "schema": "streamattn.gate0.seed_only_wrapper_batch_threshold.v1",
        "capture": {
            "metadata_json": args.metadata_json,
            "layer_id": args.layer_id,
            "row_count": len(rows),
            "prompt_ids": prompt_ids,
        },
        "policy": {
            "policy_id": product_policy.policy_id,
            "mode": product_policy.mode,
            "product_min_batch": product_policy.min_batch,
            "forced_min_batch": args.forced_min_batch,
            "block_size": product_policy.block_size,
            "sink_blocks": product_policy.sink_blocks,
            "recent_blocks": product_policy.recent_blocks,
            "middle_seed_blocks": product_policy.middle_seed_blocks,
            "block_order": product_policy.block_order,
            "expected_seed_only_ms": product_policy.expected_seed_only_ms,
            "expected_dense_ms": product_policy.expected_dense_ms,
            "expected_speedup_vs_dense": product_policy.expected_speedup_vs_dense,
        },
        "thresholds": {
            "first_direct_seed_beats_flashinfer_batch": _first_batch(
                lambda row: row["decision"]["direct_seed_beats_flashinfer"]
            ),
            "first_forced_wrapper_beats_flashinfer_batch": _first_batch(
                lambda row: row["decision"]["forced_wrapper_beats_flashinfer"]
            ),
            "first_product_route_seed_only_batch": _first_batch(
                lambda row: row["decision"]["product_route_uses_seed_only"]
            ),
            "first_product_wrapper_beats_flashinfer_batch": _first_batch(
                lambda row: row["decision"]["product_wrapper_beats_flashinfer"]
            ),
            "first_planned_direct_beats_flashinfer_batch": _first_batch(
                lambda row: row["decision"]["planned_direct_beats_flashinfer"]
            ),
            "first_service_beats_flashinfer_batch": _first_batch(
                lambda row: row["decision"]["service_beats_flashinfer"]
            ),
        },
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--policy-json", default="")
    parser.add_argument("--layer-id", type=int, default=8)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--product-min-batch", type=int, default=8)
    parser.add_argument("--forced-min-batch", type=int, default=1)
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
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    parser.add_argument("--max-logprob-delta", type=float, default=1.0e-3)
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
