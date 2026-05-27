"""Closed-loop safety replay for a multi-layer seed-only route bundle.

The single-layer rollout benchmark proves an isolated layer is logit-safe.  The
product question is stricter: if all green layers in a route bundle are patched
in the same forward pass, do logits and generation remain stable?

This benchmark keeps latency interpretation separate from safety.  The patched
model path still runs the dense attention module and overwrites the current-token
attention output via hooks, so full-model timings here are replay diagnostics.
The reported attention-bundle timing estimate is built from packaged per-layer
H100 route measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_closed_loop_rollout import (  # noqa: E402
    _aggregate_step_rows,
    _append_token,
    _batched_tokens,
    _coupled_sample_tokens,
    _forward_last_logits,
    _logit_row_metrics,
    _make_seed_only_patch_hook,
    _prompts_from_args,
)
from benchmarks.profile_real_llm_gate1_heads import _attention_modules, _import_transformers  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype  # noqa: E402
from stream_attention.decode import (  # noqa: E402
    Gate0SeedOnlyBatchedPolicy,
    find_packaged_gate0_seed_only_batched_policies,
    load_packaged_gate0_seed_only_batched_policy,
    packaged_gate0_seed_only_batched_policy_registry,
)


def _parse_layer_ids(text: str) -> List[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("layers must contain at least one layer id")
    if any(value < 0 for value in values):
        raise ValueError("layer ids must be non-negative")
    return sorted(set(values))


def _parse_policy_names(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


@dataclass(frozen=True)
class RouteBundle:
    policy_names: List[str]
    policies: List[Gate0SeedOnlyBatchedPolicy]
    artifacts: List[Dict[str, Any]]
    layer_ids: List[int]


def _artifact_for_packaged_policy(name: str) -> Dict[str, Any]:
    registry = packaged_gate0_seed_only_batched_policy_registry()
    for entry in registry.get("policies") or []:
        if entry.get("name") != name and name not in (entry.get("aliases") or []):
            continue
        path = REPO_ROOT / "stream_attention" / str(entry["path"])
        return json.loads(path.read_text(encoding="utf-8"))
    raise KeyError(f"unknown packaged policy artifact {name!r}")


def _route_bundle_from_args(args: argparse.Namespace) -> RouteBundle:
    policy_names = _parse_policy_names(args.policy_names)
    requested_layers = _parse_layer_ids(args.layers) if args.layers else []
    if args.use_packaged_policies:
        if policy_names:
            names = policy_names
        else:
            names = find_packaged_gate0_seed_only_batched_policies(
                model_id=args.model,
                dtype=args.dtype,
                kv_len_bucket=args.max_seq,
                min_batch=args.batch_size,
            )
            if requested_layers:
                requested = set(requested_layers)
                names = [
                    name
                    for name in names
                    if load_packaged_gate0_seed_only_batched_policy(name).layer_id in requested
                ]
        if not names:
            raise ValueError("no packaged policies matched the requested multi-layer route")
        policies = [load_packaged_gate0_seed_only_batched_policy(name) for name in names]
        artifacts = [_artifact_for_packaged_policy(name) for name in names]
    else:
        if not requested_layers:
            raise ValueError("layers are required when packaged policies are disabled")
        policies = [
            Gate0SeedOnlyBatchedPolicy(
                policy_id=f"ad_hoc_multi_layer_l{layer_id}",
                model_id=args.model,
                layer_id=layer_id,
                dtype=args.dtype,
                kv_len_bucket=args.max_seq,
                min_batch=args.batch_size,
                heads=args.q_heads,
                kv_heads=args.kv_heads,
                dim=args.head_dim,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                max_kl=args.max_kl,
                min_topk_overlap=args.min_topk_overlap,
                max_logprob_delta=args.max_logprob_delta,
            )
            for layer_id in requested_layers
        ]
        artifacts = [
            {
                "policy_id": policy.policy_id,
                "layer_id": policy.layer_id,
                "timing": {
                    "flashinfer_batch_tc_exact_ms": policy.expected_dense_ms,
                    "streamattn_seed_only_ms": policy.expected_seed_only_ms,
                    "speedup_vs_flashinfer_batch": policy.expected_speedup_vs_dense,
                },
            }
            for policy in policies
        ]
        names = [policy.policy_id for policy in policies]
    triples = sorted(
        zip(policies, names, artifacts),
        key=lambda item: item[0].layer_id,
    )
    policies = [item[0] for item in triples]
    sorted_names = [item[1] for item in triples]
    artifacts = [item[2] for item in triples]
    _validate_route_bundle(policies, args=args)
    return RouteBundle(
        policy_names=sorted_names,
        policies=policies,
        artifacts=artifacts,
        layer_ids=[int(policy.layer_id) for policy in policies],
    )


def _validate_route_bundle(policies: Sequence[Gate0SeedOnlyBatchedPolicy], *, args: argparse.Namespace) -> None:
    if not policies:
        raise ValueError("route bundle is empty")
    seen = set()
    first = policies[0]
    for policy in policies:
        if policy.layer_id in seen:
            raise ValueError(f"duplicate policy layer {policy.layer_id}")
        seen.add(policy.layer_id)
        mismatches = []
        if policy.model_id != args.model:
            mismatches.append("model_id")
        if policy.dtype != args.dtype:
            mismatches.append("dtype")
        if int(policy.kv_len_bucket) != int(args.max_seq):
            mismatches.append("kv_len_bucket")
        if int(policy.min_batch) > int(args.batch_size):
            mismatches.append("batch_below_min")
        if mismatches:
            raise ValueError(f"policy {policy.policy_id} mismatches route: {','.join(mismatches)}")
        seed_fields = (
            "block_size",
            "sink_blocks",
            "recent_blocks",
            "middle_seed_blocks",
            "block_order",
            "num_warps",
            "num_stages",
        )
        for field in seed_fields:
            if getattr(policy, field) != getattr(first, field):
                raise ValueError(f"policy {policy.policy_id} has inconsistent seed field {field}")


def _apply_policy_defaults(args: argparse.Namespace, bundle: RouteBundle) -> None:
    first = bundle.policies[0]
    args.block_size = int(first.block_size)
    args.sink_blocks = int(first.sink_blocks)
    args.recent_blocks = int(first.recent_blocks)
    args.middle_seed_blocks = int(first.middle_seed_blocks)
    args.block_order = str(first.block_order)
    args.num_warps = int(first.num_warps)
    args.num_stages = int(first.num_stages)


def _timing_bundle(
    policies: Sequence[Gate0SeedOnlyBatchedPolicy],
    artifacts: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
) -> Dict[str, Any]:
    suffix = "b4" if batch_size <= 4 else "b8"
    flash = 0.0
    service = 0.0
    planned = 0.0
    complete = True
    per_layer = []
    for policy, artifact in zip(policies, artifacts):
        timing = artifact.get("timing", {})
        flash_ms = timing.get(f"h100_flashinfer_{suffix}_ms")
        service_ms = timing.get(f"h100_service_{suffix}_ms")
        planned_ms = timing.get(f"h100_planned_direct_{suffix}_ms")
        if flash_ms is None or service_ms is None or planned_ms is None:
            complete = False
            continue
        flash += float(flash_ms)
        service += float(service_ms)
        planned += float(planned_ms)
        per_layer.append(
            {
                "layer_id": int(policy.layer_id),
                "flashinfer_ms": float(flash_ms),
                "service_ms": float(service_ms),
                "planned_direct_ms": float(planned_ms),
                "service_speedup": float(flash_ms) / max(float(service_ms), 1.0e-12),
                "planned_direct_speedup": float(flash_ms) / max(float(planned_ms), 1.0e-12),
            }
        )
    return {
        "batch": int(batch_size),
        "source": "packaged_policy_h100_thresholds",
        "complete": complete,
        "layer_count": len(per_layer),
        "flashinfer_selected_layers_ms": flash,
        "streamattn_service_selected_layers_ms": service,
        "streamattn_planned_direct_selected_layers_ms": planned,
        "service_speedup_vs_flashinfer_selected_layers": flash / max(service, 1.0e-12),
        "planned_direct_speedup_vs_flashinfer_selected_layers": flash / max(planned, 1.0e-12),
        "per_layer": per_layer,
    }


@dataclass
class _LayerPatchState:
    layer_id: int
    module_name: str
    capture: Optional[Any] = None
    call_count: int = 0


def _patched_last_logits_multi(
    model: torch.nn.Module,
    modules: Sequence[tuple[int, str, torch.nn.Module]],
    *,
    tokens: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, int]]:
    states = {
        int(layer_id): _LayerPatchState(layer_id=int(layer_id), module_name=module_name)
        for layer_id, module_name, _module in modules
    }
    handles = []
    for layer_id, module_name, module in modules:
        state = states[int(layer_id)]
        pre_hook, output_hook = _make_seed_only_patch_hook(
            module=module,
            layer_id=int(layer_id),
            module_name=module_name,
            state=state,
            args=args,
        )
        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.register_forward_hook(output_hook))
    try:
        logits = _forward_last_logits(model, tokens)
    finally:
        for handle in handles:
            handle.remove()
    return logits, {str(layer_id): int(state.call_count) for layer_id, state in states.items()}


def _merge_call_counts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def _run_teacher_forced_multi(
    *,
    model: torch.nn.Module,
    modules: Sequence[tuple[int, str, torch.nn.Module]],
    tokens: Dict[str, torch.Tensor],
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    current = {key: value.clone() for key, value in tokens.items()}
    steps = []
    patch_calls: Dict[str, int] = {}
    dense_time = 0.0
    stream_time = 0.0
    for step in range(args.steps):
        start = time.perf_counter()
        dense_logits = _forward_last_logits(model, current)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dense_time += time.perf_counter() - start
        start = time.perf_counter()
        seed_logits, calls = _patched_last_logits_multi(
            model,
            modules,
            tokens=current,
            args=args,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stream_time += time.perf_counter() - start
        patch_calls = _merge_call_counts(patch_calls, calls)
        row_metrics = _logit_row_metrics(seed_logits, dense_logits, top_k=args.top_k)
        next_token = torch.argmax(dense_logits, dim=-1)
        steps.append(
            {
                "step": step,
                "mode": "teacher_forced_dense_tokens_multi_layer",
                "next_tokens": next_token.tolist(),
                "rows": [
                    {**row, "prompt_kind": prompt_rows[int(row["row"])]["kind"]}
                    for row in row_metrics
                ],
            }
        )
        current = _append_token(current, next_token)
    return {
        "summary": _aggregate_step_rows(steps, batch_size=tokens["input_ids"].shape[0]),
        "steps": steps,
        "runtime_replay": {
            "dense_full_forward_total_ms": dense_time * 1000.0,
            "streamattn_patched_full_forward_total_ms": stream_time * 1000.0,
            "patch_call_count_by_layer": patch_calls,
            "patch_call_count": sum(patch_calls.values()),
        },
    }


def _run_greedy_multi(
    *,
    model: torch.nn.Module,
    modules: Sequence[tuple[int, str, torch.nn.Module]],
    tokens: Dict[str, torch.Tensor],
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    force_dense_after_divergence: bool,
) -> Dict[str, Any]:
    dense_current = {key: value.clone() for key, value in tokens.items()}
    seed_current = {key: value.clone() for key, value in tokens.items()}
    steps = []
    patch_calls: Dict[str, int] = {}
    diverged = torch.zeros(tokens["input_ids"].shape[0], device=tokens["input_ids"].device, dtype=torch.bool)
    for step in range(args.steps):
        dense_logits = _forward_last_logits(model, dense_current)
        seed_logits, calls = _patched_last_logits_multi(
            model,
            modules,
            tokens=seed_current,
            args=args,
        )
        patch_calls = _merge_call_counts(patch_calls, calls)
        row_metrics = _logit_row_metrics(seed_logits, dense_logits, top_k=args.top_k)
        dense_next = torch.argmax(dense_logits, dim=-1)
        seed_next = torch.argmax(seed_logits, dim=-1)
        changed = dense_next != seed_next
        diverged = diverged | changed
        next_for_seed = dense_next if force_dense_after_divergence else seed_next
        steps.append(
            {
                "step": step,
                "mode": (
                    "forced_same_token_after_divergence_multi_layer"
                    if force_dense_after_divergence
                    else "greedy_closed_loop_multi_layer"
                ),
                "dense_next_tokens": dense_next.tolist(),
                "streamattn_next_tokens": seed_next.tolist(),
                "diverged_rows_so_far": torch.nonzero(diverged, as_tuple=False).flatten().tolist(),
                "rows": [
                    {**row, "prompt_kind": prompt_rows[int(row["row"])]["kind"]}
                    for row in row_metrics
                ],
            }
        )
        dense_current = _append_token(dense_current, dense_next)
        seed_current = _append_token(seed_current, next_for_seed)
    summary = _aggregate_step_rows(steps, batch_size=tokens["input_ids"].shape[0])
    summary["diverged_row_count"] = int(diverged.sum().item())
    summary["sequence_exact_match_rate"] = 1.0 - (float(diverged.sum().item()) / max(1, int(diverged.numel())))
    return {
        "summary": summary,
        "steps": steps,
        "runtime_replay": {
            "patch_call_count_by_layer": patch_calls,
            "patch_call_count": sum(patch_calls.values()),
        },
    }


def _run_sampling_multi(
    *,
    model: torch.nn.Module,
    modules: Sequence[tuple[int, str, torch.nn.Module]],
    tokens: Dict[str, torch.Tensor],
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    dense_current = {key: value.clone() for key, value in tokens.items()}
    seed_current = {key: value.clone() for key, value in tokens.items()}
    steps = []
    patch_calls: Dict[str, int] = {}
    diverged = torch.zeros(tokens["input_ids"].shape[0], device=tokens["input_ids"].device, dtype=torch.bool)
    for step in range(args.steps):
        dense_logits = _forward_last_logits(model, dense_current)
        seed_logits, calls = _patched_last_logits_multi(
            model,
            modules,
            tokens=seed_current,
            args=args,
        )
        patch_calls = _merge_call_counts(patch_calls, calls)
        row_metrics = _logit_row_metrics(seed_logits, dense_logits, top_k=args.top_k)
        dense_next, seed_next = _coupled_sample_tokens(
            dense_logits,
            seed_logits,
            step=step,
            args=args,
        )
        changed = dense_next != seed_next
        diverged = diverged | changed
        steps.append(
            {
                "step": step,
                "mode": "coupled_sampling_closed_loop_multi_layer",
                "dense_next_tokens": dense_next.tolist(),
                "streamattn_next_tokens": seed_next.tolist(),
                "diverged_rows_so_far": torch.nonzero(diverged, as_tuple=False).flatten().tolist(),
                "rows": [
                    {
                        **row,
                        "sample_token_changed": bool(changed[int(row["row"])].item()),
                        "prompt_kind": prompt_rows[int(row["row"])]["kind"],
                    }
                    for row in row_metrics
                ],
            }
        )
        dense_current = _append_token(dense_current, dense_next)
        seed_current = _append_token(seed_current, seed_next)
    summary = _aggregate_step_rows(steps, batch_size=tokens["input_ids"].shape[0])
    sample_changed = [
        row
        for step in steps
        for row in step["rows"]
        if row.get("sample_token_changed")
    ]
    diverged_rows = {int(row["row"]) for row in sample_changed}
    first_sample_divergence = None
    for step in steps:
        rows = [int(row["row"]) for row in step["rows"] if row.get("sample_token_changed")]
        if rows:
            first_sample_divergence = {"step": int(step["step"]), "rows": rows}
            break
    summary["sample_token_changed_count"] = len(sample_changed)
    summary["sample_agreement_rate"] = 1.0 - (len(sample_changed) / max(1, int(summary["case_count"])))
    summary["sample_diverged_row_count"] = len(diverged_rows)
    summary["sample_sequence_exact_match_rate"] = 1.0 - (len(diverged_rows) / max(1, tokens["input_ids"].shape[0]))
    summary["first_sample_divergence"] = first_sample_divergence
    summary["sampling"] = {
        "temperature": args.sample_temperature,
        "top_p": args.sample_top_p,
        "top_k": args.sample_top_k,
        "seed": args.sample_seed,
    }
    return {
        "summary": summary,
        "steps": steps,
        "runtime_replay": {
            "patch_call_count_by_layer": patch_calls,
            "patch_call_count": sum(patch_calls.values()),
        },
    }


def _merge_chunk_rollouts_multi(
    chunks: Sequence[Dict[str, Any]],
    *,
    mode_key: str,
    total_batch: int,
    step_count: int,
) -> Dict[str, Any]:
    combined_steps = []
    patch_calls: Dict[str, int] = {}
    dense_total = 0.0
    stream_total = 0.0
    for chunk in chunks:
        offset = int(chunk["row_offset"])
        result = chunk[mode_key]
        runtime = result.get("runtime_replay") or {}
        patch_calls = _merge_call_counts(
            patch_calls,
            {str(k): int(v) for k, v in (runtime.get("patch_call_count_by_layer") or {}).items()},
        )
        dense_total += float(runtime.get("dense_full_forward_total_ms") or 0.0)
        stream_total += float(runtime.get("streamattn_patched_full_forward_total_ms") or 0.0)
        for step in result["steps"]:
            rows = []
            for row in step["rows"]:
                local = int(row["row"])
                adjusted = dict(row)
                adjusted["local_row"] = local
                adjusted["row"] = offset + local
                rows.append(adjusted)
            combined_steps.append({**step, "rows": rows})
    summary = _aggregate_step_rows(combined_steps, batch_size=total_batch)
    summary["step_count"] = step_count
    if mode_key == "greedy":
        diverged_rows = {
            int(row["row"])
            for step in combined_steps
            for row in step["rows"]
            if row["top1_changed"]
        }
        summary["diverged_row_count"] = len(diverged_rows)
        summary["sequence_exact_match_rate"] = 1.0 - (len(diverged_rows) / max(1, total_batch))
    if mode_key == "sampling":
        sample_changed = [
            row
            for step in combined_steps
            for row in step["rows"]
            if row.get("sample_token_changed")
        ]
        diverged_rows = {int(row["row"]) for row in sample_changed}
        first_sample_divergence = None
        for step in combined_steps:
            rows = [int(row["row"]) for row in step["rows"] if row.get("sample_token_changed")]
            if rows:
                first_sample_divergence = {"step": int(step["step"]), "rows": rows}
                break
        summary["sample_token_changed_count"] = len(sample_changed)
        summary["sample_agreement_rate"] = 1.0 - (len(sample_changed) / max(1, int(summary["case_count"])))
        summary["sample_diverged_row_count"] = len(diverged_rows)
        summary["sample_sequence_exact_match_rate"] = 1.0 - (len(diverged_rows) / max(1, total_batch))
        summary["first_sample_divergence"] = first_sample_divergence
        summary["sampling"] = (
            chunks[0].get("sampling", {}).get("summary", {}).get("sampling")
            if chunks
            else None
        )
    return {
        "summary": summary,
        "steps": combined_steps,
        "runtime_replay": {
            "patch_call_count_by_layer": patch_calls,
            "patch_call_count": sum(patch_calls.values()),
            "dense_full_forward_total_ms": dense_total,
            "streamattn_patched_full_forward_total_ms": stream_total,
        },
    }


def _safety_passed(result: Dict[str, Any], *, args: argparse.Namespace) -> Dict[str, Any]:
    checks = []
    for key in ("teacher_forced", "greedy", "sampling"):
        if key not in result:
            continue
        summary = result[key]["summary"]
        checks.append(
            {
                "mode": key,
                "kl_passed": float(summary.get("kl_max", 0.0)) <= float(args.max_kl),
                "top1_passed": int(summary.get("top1_changed_count", 0)) == 0,
                "topk_passed": int(summary.get("topk_overlap_min", 0)) >= int(args.min_topk_overlap),
                "logprob_passed": float(summary.get("reference_top1_logprob_delta_max_abs", 0.0))
                <= float(args.max_logprob_delta),
                "sample_passed": (
                    key != "sampling"
                    or int(summary.get("sample_token_changed_count", 0)) == 0
                ),
                "greedy_passed": (
                    key != "greedy"
                    or int(summary.get("diverged_row_count", 0)) == 0
                ),
            }
        )
    for check in checks:
        check["passed"] = all(value for key, value in check.items() if key.endswith("_passed"))
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "gates": {
            "max_kl": args.max_kl,
            "min_topk_overlap": args.min_topk_overlap,
            "max_logprob_delta": args.max_logprob_delta,
            "require_zero_top1_changes": True,
            "require_zero_sample_changes": True,
            "require_zero_greedy_divergences": True,
        },
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    bundle = _route_bundle_from_args(args)
    _apply_policy_defaults(args, bundle)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    prompt_rows = _prompts_from_args(args)

    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=args.use_safetensors,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    modules_by_layer = {int(layer_id): (int(layer_id), name, module) for layer_id, name, module in _attention_modules(model)}
    missing = [layer_id for layer_id in bundle.layer_ids if layer_id not in modules_by_layer]
    if missing:
        raise ValueError(f"model is missing requested attention layers: {missing}")
    modules = [modules_by_layer[layer_id] for layer_id in bundle.layer_ids]
    chunk_results = []
    model_batch = max(1, int(args.model_batch_size))
    for row_offset in range(0, len(prompt_rows), model_batch):
        chunk_prompts = prompt_rows[row_offset : row_offset + model_batch]
        print(
            "[multi-layer-rollout] preparing model replay rows "
            f"{row_offset}..{row_offset + len(chunk_prompts) - 1}",
            flush=True,
        )
        tokens = _batched_tokens(tokenizer, chunk_prompts, max_seq=args.max_seq, device=device)
        chunk: Dict[str, Any] = {
            "row_offset": row_offset,
            "rows": chunk_prompts,
            "shape": {
                "batch": int(tokens["input_ids"].shape[0]),
                "prompt_seq_len": int(tokens["input_ids"].shape[1]),
            },
        }
        if args.mode in {"teacher_forced", "all"}:
            print("[multi-layer-rollout] running teacher-forced dense-token rollout", flush=True)
            chunk["teacher_forced"] = _run_teacher_forced_multi(
                model=model,
                modules=modules,
                tokens=tokens,
                prompt_rows=chunk_prompts,
                args=args,
            )
        if args.mode in {"greedy", "all"}:
            print("[multi-layer-rollout] running greedy closed-loop rollout", flush=True)
            chunk["greedy"] = _run_greedy_multi(
                model=model,
                modules=modules,
                tokens=tokens,
                prompt_rows=chunk_prompts,
                args=args,
                force_dense_after_divergence=False,
            )
        if args.mode in {"sampling", "all"}:
            print("[multi-layer-rollout] running coupled sampling closed-loop rollout", flush=True)
            chunk["sampling"] = _run_sampling_multi(
                model=model,
                modules=modules,
                tokens=tokens,
                prompt_rows=chunk_prompts,
                args=args,
            )
        if args.mode in {"forced_after_divergence", "all"}:
            print("[multi-layer-rollout] running forced-same-token diagnostic rollout", flush=True)
            chunk["forced_same_token_after_divergence"] = _run_greedy_multi(
                model=model,
                modules=modules,
                tokens=tokens,
                prompt_rows=chunk_prompts,
                args=args,
                force_dense_after_divergence=True,
            )
        chunk_results.append(chunk)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result: Dict[str, Any] = {
        "schema": "streamattn.gate0.seed_only_multi_layer_rollout.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model": {
            "model_id": args.model,
            "layer_ids": bundle.layer_ids,
            "attention_modules": [
                {"layer_id": int(layer_id), "module_name": module_name}
                for layer_id, module_name, _module in modules
            ],
        },
        "route_bundle": {
            "policy_names": bundle.policy_names,
            "policy_ids": [policy.policy_id for policy in bundle.policies],
            "mode": "all_seed_only",
            "layer_count": len(bundle.layer_ids),
        },
        "shape": {
            "batch": len(prompt_rows),
            "model_replay_batch_size": model_batch,
            "prompt_seq_len": int(chunk_results[0]["shape"]["prompt_seq_len"]) if chunk_results else None,
            "dtype": args.dtype,
        },
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        },
        "attention_runtime_estimate": _timing_bundle(
            bundle.policies,
            bundle.artifacts,
            batch_size=args.batch_size,
        ),
        "safety_gate": {
            "steps": args.steps,
            "top_k": args.top_k,
            "max_kl": args.max_kl,
            "min_topk_overlap": args.min_topk_overlap,
            "max_logprob_delta": args.max_logprob_delta,
        },
        "prompts": [
            {"row": idx, "kind": row["kind"]}
            for idx, row in enumerate(prompt_rows)
        ],
        "chunks": [
            {
                "row_offset": chunk["row_offset"],
                "batch": chunk["shape"]["batch"],
                "prompt_seq_len": chunk["shape"]["prompt_seq_len"],
            }
            for chunk in chunk_results
        ],
    }
    if args.mode in {"teacher_forced", "all"}:
        result["teacher_forced"] = _merge_chunk_rollouts_multi(
            chunk_results,
            mode_key="teacher_forced",
            total_batch=len(prompt_rows),
            step_count=args.steps,
        )
    if args.mode in {"greedy", "all"}:
        result["greedy"] = _merge_chunk_rollouts_multi(
            chunk_results,
            mode_key="greedy",
            total_batch=len(prompt_rows),
            step_count=args.steps,
        )
    if args.mode in {"sampling", "all"}:
        result["sampling"] = _merge_chunk_rollouts_multi(
            chunk_results,
            mode_key="sampling",
            total_batch=len(prompt_rows),
            step_count=args.steps,
        )
    if args.mode in {"forced_after_divergence", "all"}:
        result["forced_same_token_after_divergence"] = _merge_chunk_rollouts_multi(
            chunk_results,
            mode_key="forced_same_token_after_divergence",
            total_batch=len(prompt_rows),
            step_count=args.steps,
        )
    result["decision"] = _safety_passed(result, args=args)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--policy-names", default="")
    parser.add_argument("--use-packaged-policies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-kinds", default="needle,code,long_doc,chat_doc")
    parser.add_argument("--prompt-repeat", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--mode", choices=["teacher_forced", "greedy", "sampling", "forced_after_divergence", "all"], default="all")
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    parser.add_argument("--min-topk-overlap", type=int, default=4)
    parser.add_argument("--max-logprob-delta", type=float, default=2.0e-3)
    parser.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
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
