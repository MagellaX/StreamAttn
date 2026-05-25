"""Closed-loop safety replay for the batched seed-only Gate-0 policy.

This benchmark answers the safety question left after teacher-forced logit
replay:

    If L8 seed-only changes logits now, does generation drift over future
    decode steps?

The patched model path is intentionally a safety replay, not a latency path.
The target attention module still runs its normal dense implementation, then a
forward hook replaces the current-token attention output with StreamAttn
seed-only output. Runtime latency should be measured by the decode wrapper
benchmarks, not by this full-model hook.
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_batched_logit_safety import (  # noqa: E402
    _parse_prompt_kinds,
    _prompt_for_kind,
)
from benchmarks.profile_gate0_true_gqa import _true_gqa_kv  # noqa: E402
from benchmarks.profile_real_llm_gate1_heads import (  # noqa: E402
    CapturedAttentionInput,
    _attention_modules,
    _import_transformers,
    _shape_qkv,
)
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward_out,
)


def _prompts_from_args(args: argparse.Namespace) -> List[Dict[str, str]]:
    kinds = _parse_prompt_kinds(args.prompt_kinds)
    prompts = []
    for kind in kinds[: args.batch_size]:
        prompt = (_prompt_for_kind(kind).strip() + " ") * max(1, int(args.prompt_repeat))
        prompts.append({"kind": kind, "prompt": prompt.strip()})
    if not prompts:
        raise ValueError("no prompts were provided")
    return prompts


def _batched_tokens(tokenizer, prompts: Sequence[Dict[str, str]], *, max_seq: int, device: torch.device):
    encoded = tokenizer(
        [row["prompt"] for row in prompts],
        return_tensors="pt",
        truncation=True,
        max_length=max_seq,
        padding="max_length",
    ).to(device)
    lengths = encoded["attention_mask"].sum(dim=1)
    if int(lengths.min().item()) != int(lengths.max().item()):
        raise ValueError("closed-loop batch requires equal prompt lengths")
    return encoded


def _logit_row_metrics(candidate: torch.Tensor, reference: torch.Tensor, *, top_k: int) -> List[Dict[str, Any]]:
    cand = candidate.detach().float()
    ref = reference.detach().float()
    logp = F.log_softmax(ref, dim=-1)
    logq = F.log_softmax(cand, dim=-1)
    p = logp.exp()
    kl = torch.sum(p * (logp - logq), dim=-1)
    ref_top = torch.topk(ref, k=top_k, dim=-1)
    cand_top = torch.topk(cand, k=top_k, dim=-1)
    rows = []
    for row in range(ref.shape[0]):
        ref_top1 = int(ref_top.indices[row, 0].item())
        cand_top1 = int(cand_top.indices[row, 0].item())
        ref_set = set(int(x) for x in ref_top.indices[row].tolist())
        cand_set = set(int(x) for x in cand_top.indices[row].tolist())
        overlap = len(ref_set & cand_set)
        margin = float((ref_top.values[row, 0] - ref_top.values[row, 1]).item())
        rows.append(
            {
                "row": row,
                "kl_ref_to_candidate": float(kl[row].item()),
                "max_logit_delta": float((cand[row] - ref[row]).abs().max().item()),
                "reference_top1": ref_top1,
                "candidate_top1": cand_top1,
                "top1_changed": bool(ref_top1 != cand_top1),
                "topk_overlap": overlap,
                "reference_top1_margin": margin,
                "reference_top1_logprob_delta": float((logq[row, ref_top1] - logp[row, ref_top1]).item()),
                "reference_top_tokens": ref_top.indices[row].tolist(),
                "candidate_top_tokens": cand_top.indices[row].tolist(),
            }
        )
    return rows


def _aggregate_step_rows(step_rows: Sequence[Dict[str, Any]], *, batch_size: int) -> Dict[str, Any]:
    flat = [row for step in step_rows for row in step["rows"]]
    kl_values = [float(row["kl_ref_to_candidate"]) for row in flat]
    logit_values = [float(row["max_logit_delta"]) for row in flat]
    top1_changed = [row for row in flat if row["top1_changed"]]
    min_overlap = min((int(row["topk_overlap"]) for row in flat), default=0)
    margin_values = [float(row["reference_top1_margin"]) for row in flat]
    logprob_delta_values = [float(row["reference_top1_logprob_delta"]) for row in flat]
    worst = max(flat, key=lambda row: float(row["kl_ref_to_candidate"])) if flat else None
    first_divergence = None
    for step in step_rows:
        changed_rows = [row["row"] for row in step["rows"] if row["top1_changed"]]
        if changed_rows:
            first_divergence = {
                "step": int(step["step"]),
                "rows": changed_rows,
            }
            break
    return {
        "step_count": len(step_rows),
        "batch_size": batch_size,
        "case_count": len(flat),
        "kl_max": max(kl_values) if kl_values else 0.0,
        "kl_mean": float(torch.tensor(kl_values).mean().item()) if kl_values else 0.0,
        "max_logit_delta": max(logit_values) if logit_values else 0.0,
        "top1_changed_count": len(top1_changed),
        "top1_agreement_rate": 1.0 - (len(top1_changed) / max(1, len(flat))),
        "topk_overlap_min": min_overlap,
        "reference_top1_margin_min": min(margin_values) if margin_values else 0.0,
        "reference_top1_logprob_delta_max_abs": max(
            (abs(value) for value in logprob_delta_values),
            default=0.0,
        ),
        "first_divergence": first_divergence,
        "worst_case_by_kl": worst,
    }


@dataclass
class _SeedOnlyPatchState:
    capture: Optional[CapturedAttentionInput] = None
    call_count: int = 0


def _make_seed_only_patch_hook(
    *,
    module: torch.nn.Module,
    layer_id: int,
    module_name: str,
    state: _SeedOnlyPatchState,
    args: argparse.Namespace,
):
    def pre_hook(mod, hook_args, hook_kwargs):
        hidden_states = hook_args[0] if hook_args else hook_kwargs.get("hidden_states")
        if hidden_states is None:
            return
        state.capture = CapturedAttentionInput(
            layer_id=layer_id,
            module_name=module_name,
            module=mod,
            hidden_states=hidden_states.detach(),
            kwargs=dict(hook_kwargs),
        )

    def output_hook(_mod, _inputs, output):
        if state.capture is None:
            return output
        q, k, v, meta = _shape_qkv(state.capture, apply_rope=True)
        true_kv_heads = int(meta.get("num_kv_heads", k.shape[2]))
        k_true = _true_gqa_kv(k.contiguous(), true_kv_heads=true_kv_heads).contiguous()
        v_true = _true_gqa_kv(v.contiguous(), true_kv_heads=true_kv_heads).contiguous()
        q_last = q[:, -1:, :, :].contiguous()
        seed_heads = torch.empty_like(q_last)
        gate0_seed_only_attention_triton_forward_out(
            q_last,
            k_true,
            v_true,
            seed_heads,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        patch = module.o_proj(seed_heads.reshape(seed_heads.shape[0], 1, -1)).to(
            dtype=(output[0] if isinstance(output, tuple) else output).dtype
        )
        if isinstance(output, tuple):
            attn = output[0].clone()
            attn[:, -1:, :] = patch
            patched = (attn, *output[1:])
        else:
            patched = output.clone()
            patched[:, -1:, :] = patch
        state.call_count += 1
        state.capture = None
        return patched

    return pre_hook, output_hook


def _forward_last_logits(model: torch.nn.Module, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        out = model(**tokens, use_cache=False)
    return out.logits[:, -1, :].detach()


def _patched_last_logits(
    model: torch.nn.Module,
    module: torch.nn.Module,
    *,
    layer_id: int,
    module_name: str,
    tokens: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int]:
    state = _SeedOnlyPatchState()
    pre_hook, output_hook = _make_seed_only_patch_hook(
        module=module,
        layer_id=layer_id,
        module_name=module_name,
        state=state,
        args=args,
    )
    handles = [
        module.register_forward_pre_hook(pre_hook, with_kwargs=True),
        module.register_forward_hook(output_hook),
    ]
    try:
        logits = _forward_last_logits(model, tokens)
    finally:
        for handle in handles:
            handle.remove()
    return logits, state.call_count


def _append_token(tokens: Dict[str, torch.Tensor], next_token: torch.Tensor) -> Dict[str, torch.Tensor]:
    next_token = next_token.view(-1, 1)
    next_mask = torch.ones_like(next_token)
    out = {
        "input_ids": torch.cat([tokens["input_ids"], next_token], dim=1),
        "attention_mask": torch.cat([tokens["attention_mask"], next_mask], dim=1),
    }
    if "position_ids" in tokens:
        next_pos = tokens["position_ids"][:, -1:] + 1
        out["position_ids"] = torch.cat([tokens["position_ids"], next_pos], dim=1)
    return out


def _sampling_probs(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("sampling temperature must be positive")
    scores = logits.detach().float() / float(temperature)
    if top_k > 0 and top_k < scores.shape[-1]:
        top_values, _ = torch.topk(scores, k=top_k, dim=-1)
        cutoff = top_values[:, -1:].expand_as(scores)
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        remove = cdf > float(top_p)
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(dim=-1, index=sorted_idx, src=sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
    return probs


def _coupled_sample_tokens(
    dense_logits: torch.Tensor,
    seed_logits: torch.Tensor,
    *,
    step: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense_probs = _sampling_probs(
        dense_logits,
        temperature=args.sample_temperature,
        top_p=args.sample_top_p,
        top_k=args.sample_top_k,
    )
    seed_probs = _sampling_probs(
        seed_logits,
        temperature=args.sample_temperature,
        top_p=args.sample_top_p,
        top_k=args.sample_top_k,
    )
    generator = torch.Generator(device=dense_logits.device)
    generator.manual_seed(int(args.sample_seed) + int(step))
    uniforms = torch.rand(
        dense_logits.shape[0],
        1,
        device=dense_logits.device,
        generator=generator,
    )
    dense_next = torch.searchsorted(torch.cumsum(dense_probs, dim=-1), uniforms).squeeze(-1)
    seed_next = torch.searchsorted(torch.cumsum(seed_probs, dim=-1), uniforms).squeeze(-1)
    max_token = dense_logits.shape[-1] - 1
    return dense_next.clamp_max(max_token), seed_next.clamp_max(max_token)


def _run_teacher_forced(
    *,
    model: torch.nn.Module,
    module: torch.nn.Module,
    module_name: str,
    tokens: Dict[str, torch.Tensor],
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    current = {key: value.clone() for key, value in tokens.items()}
    steps = []
    patch_calls = 0
    dense_time = 0.0
    stream_time = 0.0
    for step in range(args.steps):
        start = time.perf_counter()
        dense_logits = _forward_last_logits(model, current)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dense_time += time.perf_counter() - start
        start = time.perf_counter()
        seed_logits, calls = _patched_last_logits(
            model,
            module,
            layer_id=args.layer_id,
            module_name=module_name,
            tokens=current,
            args=args,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stream_time += time.perf_counter() - start
        patch_calls += calls
        row_metrics = _logit_row_metrics(seed_logits, dense_logits, top_k=args.top_k)
        next_token = torch.argmax(dense_logits, dim=-1)
        steps.append(
            {
                "step": step,
                "mode": "teacher_forced_dense_tokens",
                "next_tokens": next_token.tolist(),
                "rows": [
                    {
                        **row,
                        "prompt_kind": prompt_rows[int(row["row"])]["kind"],
                    }
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
            "patch_call_count": patch_calls,
        },
    }


def _run_greedy(
    *,
    model: torch.nn.Module,
    module: torch.nn.Module,
    module_name: str,
    tokens: Dict[str, torch.Tensor],
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    force_dense_after_divergence: bool,
) -> Dict[str, Any]:
    dense_current = {key: value.clone() for key, value in tokens.items()}
    seed_current = {key: value.clone() for key, value in tokens.items()}
    steps = []
    patch_calls = 0
    diverged = torch.zeros(tokens["input_ids"].shape[0], device=tokens["input_ids"].device, dtype=torch.bool)
    for step in range(args.steps):
        dense_logits = _forward_last_logits(model, dense_current)
        seed_logits, calls = _patched_last_logits(
            model,
            module,
            layer_id=args.layer_id,
            module_name=module_name,
            tokens=seed_current,
            args=args,
        )
        patch_calls += calls
        row_metrics = _logit_row_metrics(seed_logits, dense_logits, top_k=args.top_k)
        dense_next = torch.argmax(dense_logits, dim=-1)
        seed_next = torch.argmax(seed_logits, dim=-1)
        changed = dense_next != seed_next
        diverged = diverged | changed
        next_for_seed = dense_next if force_dense_after_divergence else seed_next
        steps.append(
            {
                "step": step,
                "mode": "forced_same_token_after_divergence" if force_dense_after_divergence else "greedy_closed_loop",
                "dense_next_tokens": dense_next.tolist(),
                "streamattn_next_tokens": seed_next.tolist(),
                "diverged_rows_so_far": torch.nonzero(diverged, as_tuple=False).flatten().tolist(),
                "rows": [
                    {
                        **row,
                        "prompt_kind": prompt_rows[int(row["row"])]["kind"],
                    }
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
            "patch_call_count": patch_calls,
        },
    }


def _run_sampling(
    *,
    model: torch.nn.Module,
    module: torch.nn.Module,
    module_name: str,
    tokens: Dict[str, torch.Tensor],
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    dense_current = {key: value.clone() for key, value in tokens.items()}
    seed_current = {key: value.clone() for key, value in tokens.items()}
    steps = []
    patch_calls = 0
    diverged = torch.zeros(tokens["input_ids"].shape[0], device=tokens["input_ids"].device, dtype=torch.bool)
    for step in range(args.steps):
        dense_logits = _forward_last_logits(model, dense_current)
        seed_logits, calls = _patched_last_logits(
            model,
            module,
            layer_id=args.layer_id,
            module_name=module_name,
            tokens=seed_current,
            args=args,
        )
        patch_calls += calls
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
                "mode": "coupled_sampling_closed_loop",
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
    first_sample_divergence = None
    for step in steps:
        rows = [row["row"] for row in step["rows"] if row.get("sample_token_changed")]
        if rows:
            first_sample_divergence = {"step": int(step["step"]), "rows": rows}
            break
    summary["sample_token_changed_count"] = len(sample_changed)
    summary["sample_agreement_rate"] = 1.0 - (len(sample_changed) / max(1, int(summary["case_count"])))
    summary["sample_diverged_row_count"] = int(diverged.sum().item())
    summary["sample_sequence_exact_match_rate"] = 1.0 - (float(diverged.sum().item()) / max(1, int(diverged.numel())))
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
            "patch_call_count": patch_calls,
        },
    }


def _merge_chunk_rollouts(
    chunks: Sequence[Dict[str, Any]],
    *,
    mode_key: str,
    total_batch: int,
    step_count: int,
) -> Dict[str, Any]:
    combined_steps = []
    patch_calls = 0
    dense_total = 0.0
    stream_total = 0.0
    for chunk in chunks:
        offset = int(chunk["row_offset"])
        result = chunk[mode_key]
        runtime = result.get("runtime_replay") or {}
        patch_calls += int(runtime.get("patch_call_count") or 0)
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
            "patch_call_count": patch_calls,
            "dense_full_forward_total_ms": dense_total,
            "streamattn_patched_full_forward_total_ms": stream_total,
        },
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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
    modules = _attention_modules(model)
    _layer_id, module_name, module = modules[args.layer_id]
    chunk_results = []
    model_batch = max(1, int(args.model_batch_size))
    for row_offset in range(0, len(prompt_rows), model_batch):
        chunk_prompts = prompt_rows[row_offset : row_offset + model_batch]
        print(
            f"[closed-loop] preparing model replay rows {row_offset}..{row_offset + len(chunk_prompts) - 1}",
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
            print("[closed-loop] running teacher-forced dense-token rollout", flush=True)
            chunk["teacher_forced"] = _run_teacher_forced(
                model=model,
                module=module,
                module_name=module_name,
                tokens=tokens,
                prompt_rows=chunk_prompts,
                args=args,
            )
        if args.mode in {"greedy", "all"}:
            print("[closed-loop] running greedy closed-loop rollout", flush=True)
            chunk["greedy"] = _run_greedy(
                model=model,
                module=module,
                module_name=module_name,
                tokens=tokens,
                prompt_rows=chunk_prompts,
                args=args,
                force_dense_after_divergence=False,
            )
        if args.mode in {"sampling", "all"}:
            print("[closed-loop] running coupled sampling closed-loop rollout", flush=True)
            chunk["sampling"] = _run_sampling(
                model=model,
                module=module,
                module_name=module_name,
                tokens=tokens,
                prompt_rows=chunk_prompts,
                args=args,
            )
        if args.mode in {"forced_after_divergence", "all"}:
            print("[closed-loop] running forced-same-token diagnostic rollout", flush=True)
            chunk["forced_same_token_after_divergence"] = _run_greedy(
                model=model,
                module=module,
                module_name=module_name,
                tokens=tokens,
                prompt_rows=chunk_prompts,
                args=args,
                force_dense_after_divergence=True,
            )
        chunk_results.append(chunk)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result: Dict[str, Any] = {
        "schema": "streamattn.gate0.seed_only_closed_loop_rollout.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model": {
            "model_id": args.model,
            "layer_id": args.layer_id,
            "attention_module": module_name,
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
        "safety_gate": {
            "steps": args.steps,
            "top_k": args.top_k,
            "max_kl": args.max_kl,
            "min_topk_overlap": args.min_topk_overlap,
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
        result["teacher_forced"] = _merge_chunk_rollouts(
            chunk_results,
            mode_key="teacher_forced",
            total_batch=len(prompt_rows),
            step_count=args.steps,
        )
    if args.mode in {"greedy", "all"}:
        result["greedy"] = _merge_chunk_rollouts(
            chunk_results,
            mode_key="greedy",
            total_batch=len(prompt_rows),
            step_count=args.steps,
        )
    if args.mode in {"sampling", "all"}:
        result["sampling"] = _merge_chunk_rollouts(
            chunk_results,
            mode_key="sampling",
            total_batch=len(prompt_rows),
            step_count=args.steps,
        )
    if args.mode in {"forced_after_divergence", "all"}:
        result["forced_same_token_after_divergence"] = _merge_chunk_rollouts(
            chunk_results,
            mode_key="forced_same_token_after_divergence",
            total_batch=len(prompt_rows),
            step_count=args.steps,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-kinds", default="needle,code,long_doc,needle,code,long_doc,needle,code")
    parser.add_argument("--prompt-repeat", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-batch-size", type=int, default=1)
    parser.add_argument("--layer-id", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=8)
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    parser.add_argument("--min-topk-overlap", type=int, default=4)
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
