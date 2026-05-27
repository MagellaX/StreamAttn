"""End-to-end KV-cache decode benchmark for a seed-only route bundle.

This is the first benchmark that answers the serving question directly:

    Does StreamAttn speed up actual model decode, not just selected attention
    kernel calls?

It performs dense prefill, then times `use_cache=True` decode steps.  For the
candidate route, selected attention modules are patched so their decode forward
computes post-RoPE Q/K/V, updates the Hugging Face KV cache, runs the StreamAttn
seed-only kernel, and applies the module output projection.  Dense attention is
not computed for those selected layers during the measured decode loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_closed_loop_rollout import (  # noqa: E402
    _coupled_sample_tokens,
    _logit_row_metrics,
    _prompts_from_args,
)
from benchmarks.profile_gate0_seed_only_multi_layer_rollout import (  # noqa: E402
    _route_bundle_from_args,
)
from benchmarks.profile_real_llm_gate1_heads import _attention_modules, _import_transformers  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward_out_cachepos_bhnd,
)


def _import_qwen_rotary():
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except Exception as exc:  # pragma: no cover - optional dependency/version
        raise RuntimeError("Qwen route bundle decode requires transformers Qwen2 helpers") from exc
    return apply_rotary_pos_emb


def _batch_tokens(tokenizer, prompts: Sequence[Dict[str, str]], *, max_seq: int, device: torch.device):
    encoded = tokenizer(
        [row["prompt"] for row in prompts],
        return_tensors="pt",
        truncation=True,
        max_length=max_seq,
        padding="max_length",
    ).to(device)
    lengths = encoded["attention_mask"].sum(dim=1)
    if int(lengths.min().item()) != max_seq:
        raise ValueError(
            "route-bundle decode benchmark expects all prompt rows to fill max_seq; "
            f"min length was {int(lengths.min().item())}, max_seq={max_seq}"
        )
    return encoded


def _append_decode_attention_mask(mask: torch.Tensor, next_token: torch.Tensor) -> torch.Tensor:
    return torch.cat([mask, torch.ones_like(next_token)], dim=1)


def _summarize_logit_steps(step_rows: Sequence[Dict[str, Any]], *, batch_size: int) -> Dict[str, Any]:
    flat = [row for step in step_rows for row in step["rows"]]
    kl_values = [float(row["kl_ref_to_candidate"]) for row in flat]
    top1_changed = [row for row in flat if row["top1_changed"]]
    logit_values = [float(row["max_logit_delta"]) for row in flat]
    margin_values = [float(row["reference_top1_margin"]) for row in flat]
    logprob_delta_values = [float(row["reference_top1_logprob_delta"]) for row in flat]
    sample_changed = [row for row in flat if row.get("sample_token_changed")]
    first_divergence = None
    first_sample_divergence = None
    for step in step_rows:
        rows = [int(row["row"]) for row in step["rows"] if row["top1_changed"]]
        if rows and first_divergence is None:
            first_divergence = {"step": int(step["step"]), "rows": rows}
        sample_rows = [int(row["row"]) for row in step["rows"] if row.get("sample_token_changed")]
        if sample_rows and first_sample_divergence is None:
            first_sample_divergence = {"step": int(step["step"]), "rows": sample_rows}
    return {
        "step_count": len(step_rows),
        "batch_size": batch_size,
        "case_count": len(flat),
        "kl_max": max(kl_values) if kl_values else 0.0,
        "kl_mean": float(torch.tensor(kl_values).mean().item()) if kl_values else 0.0,
        "max_logit_delta": max(logit_values) if logit_values else 0.0,
        "top1_changed_count": len(top1_changed),
        "top1_agreement_rate": 1.0 - (len(top1_changed) / max(1, len(flat))),
        "topk_overlap_min": min((int(row["topk_overlap"]) for row in flat), default=0),
        "reference_top1_margin_min": min(margin_values) if margin_values else 0.0,
        "reference_top1_logprob_delta_max_abs": max(
            (abs(value) for value in logprob_delta_values),
            default=0.0,
        ),
        "sample_token_changed_count": len(sample_changed),
        "sample_agreement_rate": 1.0 - (len(sample_changed) / max(1, len(flat))),
        "first_divergence": first_divergence,
        "first_sample_divergence": first_sample_divergence,
        "worst_case_by_kl": max(flat, key=lambda row: float(row["kl_ref_to_candidate"])) if flat else None,
    }


class _SeedOnlyQwenDecodePatch:
    def __init__(self, *, policy, original_forward):
        self.policy = policy
        self.original_forward = original_forward
        self.call_count = 0
        self.seed_call_count = 0
        self.fallback_reasons: Counter[str] = Counter()
        self.fallback_samples: List[Dict[str, Any]] = []

    def forward(
        self,
        module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        self.call_count += 1
        if past_key_value is None:
            past_key_value = kwargs.get("past_key_value", kwargs.get("past_key_values"))
        if cache_position is None:
            position_ids = kwargs.get("cache_position", kwargs.get("position_ids"))
            if position_ids is not None:
                cache_position = position_ids.reshape(-1)
        fallback_reason = None
        if hidden_states.shape[1] != 1:
            fallback_reason = f"query_len_{hidden_states.shape[1]}"
        elif past_key_value is None:
            fallback_reason = "missing_past_key_value"
        elif cache_position is None:
            fallback_reason = "missing_cache_position"
        elif not cache_position.is_cuda:
            fallback_reason = "non_cuda_cache_position"
        elif not hidden_states.is_cuda:
            fallback_reason = "non_cuda_hidden_states"
        if fallback_reason is not None:
            self.fallback_reasons[fallback_reason] += 1
            if len(self.fallback_samples) < 5:
                self.fallback_samples.append(
                    {
                        "reason": fallback_reason,
                        "hidden_shape": list(hidden_states.shape),
                        "past_key_value_type": type(past_key_value).__name__ if past_key_value is not None else None,
                        "cache_position_shape": list(cache_position.shape) if cache_position is not None else None,
                        "attention_mask_shape": list(attention_mask.shape) if attention_mask is not None else None,
                        "kwargs_keys": sorted(kwargs.keys()),
                    }
                )
            return self.original_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )

        apply_rotary_pos_emb = _import_qwen_rotary()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            module.layer_idx,
            cache_kwargs,
        )

        q = query_states.transpose(1, 2).contiguous()
        k = key_states
        v = value_states
        out = torch.empty_like(q)
        gate0_seed_only_attention_triton_forward_out_cachepos_bhnd(
            q,
            k,
            v,
            out,
            cache_position,
            block_size=self.policy.block_size,
            sink_blocks=self.policy.sink_blocks,
            recent_blocks=self.policy.recent_blocks,
            middle_seed_blocks=self.policy.middle_seed_blocks,
            block_order=self.policy.block_order,
            num_warps=self.policy.num_warps,
            num_stages=self.policy.num_stages,
        )
        attn_output = out.reshape(*input_shape, -1).contiguous()
        attn_output = module.o_proj(attn_output)
        self.seed_call_count += 1
        return attn_output, None


def _cache_summary(cache: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "is_none": cache is None,
        "type": type(cache).__name__ if cache is not None else None,
    }
    if cache is None:
        return summary
    for name in ("get_seq_length", "get_max_cache_shape"):
        fn = getattr(cache, name, None)
        if fn is None:
            continue
        try:
            summary[name] = int(fn()) if name == "get_seq_length" else fn()
        except TypeError:
            try:
                summary[name] = int(fn(0)) if name == "get_seq_length" else fn(0)
            except Exception as exc:  # pragma: no cover - version-specific cache helper
                summary[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # pragma: no cover - version-specific cache helper
            summary[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    try:
        summary["len"] = len(cache)
    except Exception:
        pass
    try:
        layer0 = cache[0]
        if isinstance(layer0, (tuple, list)) and len(layer0) >= 2:
            summary["layer0_key_shape"] = list(layer0[0].shape)
            summary["layer0_value_shape"] = list(layer0[1].shape)
    except Exception:
        pass
    return summary


@contextmanager
def _patched_seed_only_decode_modules(
    model: torch.nn.Module,
    bundle,
) -> Iterator[Dict[str, _SeedOnlyQwenDecodePatch]]:
    modules_by_layer = {int(layer_id): module for layer_id, _name, module in _attention_modules(model)}
    patches: Dict[str, _SeedOnlyQwenDecodePatch] = {}
    originals = []
    for policy in bundle.policies:
        module = modules_by_layer.get(int(policy.layer_id))
        if module is None:
            raise ValueError(f"model is missing layer {policy.layer_id}")
        patch = _SeedOnlyQwenDecodePatch(policy=policy, original_forward=module.forward)
        originals.append((module, module.forward))
        module.forward = types.MethodType(patch.forward, module)
        patches[str(policy.layer_id)] = patch
    try:
        yield patches
    finally:
        for module, original in originals:
            module.forward = original


def _prefill(model, tokens: Dict[str, torch.Tensor]):
    with torch.inference_mode():
        return model(**tokens, use_cache=True, logits_to_keep=1)


def _args_with_steps(args: argparse.Namespace, steps: int) -> argparse.Namespace:
    clone = argparse.Namespace(**vars(args))
    clone.steps = int(steps)
    return clone


def _decode_loop(
    *,
    model,
    past_key_values,
    attention_mask: torch.Tensor,
    first_token: torch.Tensor,
    fixed_input_tokens: Optional[Sequence[torch.Tensor]],
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    input_token = first_token
    mask = attention_mask
    logits_by_step = []
    input_tokens = []
    generated_next_tokens = []
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = time.perf_counter()
    with torch.inference_mode():
        for step in range(args.steps):
            if fixed_input_tokens is not None:
                input_token = fixed_input_tokens[step]
            input_tokens.append(input_token.detach().clone())
            out = model(
                input_ids=input_token,
                attention_mask=_append_decode_attention_mask(mask, input_token),
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :].detach()
            logits_by_step.append(logits)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_next_tokens.append(next_token.detach().clone())
            mask = _append_decode_attention_mask(mask, input_token)
            input_token = next_token
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    return {
        "total_ms": total_ms,
        "ms_per_token": total_ms / max(1, args.steps),
        "logits_by_step": logits_by_step,
        "input_tokens": input_tokens,
        "generated_next_tokens": generated_next_tokens,
        "final_attention_mask": mask,
    }


def _warmup_decode(
    *,
    model,
    tokens: Dict[str, torch.Tensor],
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    bundle=None,
) -> Dict[str, Any]:
    warmup_args = _args_with_steps(args, args.warmup_steps)
    prefill = _prefill(model, tokens)
    first_token = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
    if bundle is None:
        result = _decode_loop(
            model=model,
            past_key_values=prefill.past_key_values,
            attention_mask=tokens["attention_mask"],
            first_token=first_token,
            fixed_input_tokens=None,
            prompt_rows=prompt_rows,
            args=warmup_args,
        )
        return {"decode": result, "patch_counts": None}
    with _patched_seed_only_decode_modules(model, bundle) as patches:
        result = _decode_loop(
            model=model,
            past_key_values=prefill.past_key_values,
            attention_mask=tokens["attention_mask"],
            first_token=first_token,
            fixed_input_tokens=None,
            prompt_rows=prompt_rows,
            args=warmup_args,
        )
    patch_counts = {
        layer_id: {
            "forward_calls": patch.call_count,
            "seed_only_decode_calls": patch.seed_call_count,
            "fallback_reasons": dict(patch.fallback_reasons),
            "fallback_samples": patch.fallback_samples,
        }
        for layer_id, patch in patches.items()
    }
    return {"decode": result, "patch_counts": patch_counts}


def _compare_decode_logits(
    dense_logits: Sequence[torch.Tensor],
    seed_logits: Sequence[torch.Tensor],
    *,
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    steps = []
    for step, (dense, seed) in enumerate(zip(dense_logits, seed_logits)):
        rows = _logit_row_metrics(seed, dense, top_k=args.top_k)
        dense_sample, seed_sample = _coupled_sample_tokens(dense, seed, step=step, args=args)
        changed = dense_sample != seed_sample
        steps.append(
            {
                "step": step,
                "rows": [
                    {
                        **row,
                        "prompt_kind": prompt_rows[int(row["row"])]["kind"],
                        "sample_token_changed": bool(changed[int(row["row"])].item()),
                    }
                    for row in rows
                ],
            }
        )
    return {
        "summary": _summarize_logit_steps(steps, batch_size=len(prompt_rows)),
        "steps": steps,
    }


def _safety_decision(summary: Dict[str, Any], *, args: argparse.Namespace) -> Dict[str, Any]:
    checks = {
        "kl_passed": float(summary.get("kl_max", 0.0)) <= float(args.max_kl),
        "top1_passed": int(summary.get("top1_changed_count", 0)) == 0,
        "topk_passed": int(summary.get("topk_overlap_min", 0)) >= int(args.min_topk_overlap),
        "logprob_passed": float(summary.get("reference_top1_logprob_delta_max_abs", 0.0))
        <= float(args.max_logprob_delta),
        "sample_passed": int(summary.get("sample_token_changed_count", 0)) == 0,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "gates": {
            "max_kl": args.max_kl,
            "min_topk_overlap": args.min_topk_overlap,
            "max_logprob_delta": args.max_logprob_delta,
            "require_zero_top1_changes": True,
            "require_zero_sample_changes": True,
        },
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    bundle = _route_bundle_from_args(args)
    prompt_rows = _prompts_from_args(args)

    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": args.use_safetensors,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()
    tokens = _batch_tokens(tokenizer, prompt_rows, max_seq=args.max_seq, device=device)
    warmup_summary: Dict[str, Any] = {"steps": int(args.warmup_steps)}

    if args.warmup_steps > 0:
        print("[route-bundle-decode] warmup dense exact decode", flush=True)
        dense_warmup = _warmup_decode(
            model=model,
            tokens=tokens,
            prompt_rows=prompt_rows,
            args=args,
            bundle=None,
        )
        warmup_summary["dense_ms_per_token"] = dense_warmup["decode"]["ms_per_token"]
        dense_warmup = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[route-bundle-decode] warmup StreamAttn seed-only bundle decode", flush=True)
        seed_warmup = _warmup_decode(
            model=model,
            tokens=tokens,
            prompt_rows=prompt_rows,
            args=args,
            bundle=bundle,
        )
        warmup_summary["streamattn_ms_per_token"] = seed_warmup["decode"]["ms_per_token"]
        warmup_summary["streamattn_patch_counts"] = seed_warmup["patch_counts"]
        seed_warmup = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[route-bundle-decode] dense prefill for baseline cache", flush=True)
    dense_prefill = _prefill(model, tokens)
    dense_prefill_cache = _cache_summary(dense_prefill.past_key_values)
    first_token = torch.argmax(dense_prefill.logits[:, -1, :], dim=-1, keepdim=True)
    print("[route-bundle-decode] timing dense exact decode", flush=True)
    dense = _decode_loop(
        model=model,
        past_key_values=dense_prefill.past_key_values,
        attention_mask=tokens["attention_mask"],
        first_token=first_token,
        fixed_input_tokens=None,
        prompt_rows=prompt_rows,
        args=args,
    )
    dense_prefill = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[route-bundle-decode] dense prefill for StreamAttn cache", flush=True)
    seed_prefill = _prefill(model, tokens)
    seed_prefill_cache = _cache_summary(seed_prefill.past_key_values)
    print("[route-bundle-decode] timing StreamAttn seed-only bundle decode", flush=True)
    with _patched_seed_only_decode_modules(model, bundle) as patches:
        seed = _decode_loop(
            model=model,
            past_key_values=seed_prefill.past_key_values,
            attention_mask=tokens["attention_mask"],
            first_token=first_token,
            fixed_input_tokens=dense["input_tokens"],
            prompt_rows=prompt_rows,
            args=args,
        )
    comparison = _compare_decode_logits(
        dense["logits_by_step"],
        seed["logits_by_step"],
        prompt_rows=prompt_rows,
        args=args,
    )
    patch_counts = {
        layer_id: {
            "forward_calls": patch.call_count,
            "seed_only_decode_calls": patch.seed_call_count,
            "fallback_reasons": dict(patch.fallback_reasons),
            "fallback_samples": patch.fallback_samples,
        }
        for layer_id, patch in patches.items()
    }
    return {
        "schema": "streamattn.seed_only_route_bundle_decode.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model": {
            "model_id": args.model,
            "attn_implementation": args.attn_implementation or "default",
        },
        "route_bundle": {
            "policy_names": bundle.policy_names,
            "policy_ids": [policy.policy_id for policy in bundle.policies],
            "layers": bundle.layer_ids,
        },
        "shape": {
            "batch": len(prompt_rows),
            "prompt_seq_len": int(tokens["input_ids"].shape[1]),
            "steps": int(args.steps),
            "dtype": args.dtype,
        },
        "timing": {
            "dense_decode_total_ms": dense["total_ms"],
            "streamattn_decode_total_ms": seed["total_ms"],
            "dense_decode_ms_per_token": dense["ms_per_token"],
            "streamattn_decode_ms_per_token": seed["ms_per_token"],
            "speedup_vs_dense_decode": dense["total_ms"] / max(seed["total_ms"], 1.0e-12),
        },
        "warmup": warmup_summary,
        "cache": {
            "dense_prefill": dense_prefill_cache,
            "seed_prefill": seed_prefill_cache,
        },
        "safety": comparison["summary"],
        "decision": _safety_decision(comparison["summary"], args=args),
        "patch_counts": patch_counts,
        "prompts": [{"row": idx, "kind": row["kind"]} for idx, row in enumerate(prompt_rows)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--policy-names", default="")
    parser.add_argument("--use-packaged-policies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-kinds", default="needle,code,long_doc,chat_doc")
    parser.add_argument("--prompt-repeat", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
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
