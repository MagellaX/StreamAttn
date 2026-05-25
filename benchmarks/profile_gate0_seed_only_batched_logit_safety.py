"""Distribution-aware safety replay for batched seed-only Gate-0 decode.

This profiler validates the current StreamAttn product wedge:

    mixed prompt rows -> batched true-GQA seed-only attention at one layer
    -> output projection -> remaining model -> logits.

It intentionally measures worst-row and worst-position safety.  The seed-only
attention computation is batched across prompt rows for each target position,
which matches the runtime direction better than one-prompt replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_logit_replay import (  # noqa: E402
    _attention_output_hook,
    _capture_qkv_and_logits,
    _error_summary,
    _final_logits_with_patch,
    _logit_metrics,
    _project_attention_output,
    _selected_logits,
    _target_positions,
)
from benchmarks.profile_gate0_true_gqa import _dense_true_gqa, _true_gqa_kv  # noqa: E402
from benchmarks.profile_real_llm_gate1_heads import (  # noqa: E402
    _attention_modules,
    _import_transformers,
    _load_prompts,
)
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward_out,
)


def _parse_prompt_kinds(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _prompt_for_kind(kind: str) -> str:
    if kind == "code":
        return (
            "def stream_attention_decode(q, k_cache, v_cache, policy):\n"
            "    kv_head = q_head // group_size\n"
            "    if policy.seed_only_group:\n"
            "        schedule_seed_blocks(sink, recent, middle_seed)\n"
            "    else:\n"
            "        schedule_exact_blocks(k_cache, v_cache)\n"
            "    return online_softmax_merge(partial_states)\n"
        )
    if kind == "long_doc":
        return (
            "StreamAttn long context technical memorandum. "
            "The system stores cached key and value tensors, maintains online softmax state, "
            "routes true grouped-query attention heads, verifies approximation error, and "
            "falls back to exact decode when calibration is stale. "
        )
    if kind == "chat_doc":
        return (
            "User: Summarize the implementation status.\n"
            "Assistant: StreamAttn has a seed-only batched route, wrapper telemetry, "
            "distribution-aware safety checks, and dense fallback for unsupported requests.\n"
        )
    return (
        "Needle retrieval context with cached KV metadata, online softmax, middle blocks, "
        "sink tokens, recent tokens, sparse decode routing, exact repair, and long-context retrieval. "
    )


def _prompts_from_args(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.prompt_file:
        prompts = _load_prompts(args)
        return [
            {"kind": f"file_{idx}", "prompt": prompt}
            for idx, prompt in enumerate(prompts[: args.max_prompts])
        ]
    kinds = _parse_prompt_kinds(args.prompt_kinds)
    prompts = []
    for kind in kinds[: args.max_prompts]:
        prompt = (_prompt_for_kind(kind).strip() + " ") * max(1, int(args.prompt_repeat))
        prompts.append({"kind": kind, "prompt": prompt.strip()})
    if not prompts:
        raise ValueError("no prompts were provided")
    return prompts


def _target_logprob_delta(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    target_positions: Sequence[int],
) -> Dict[str, Any]:
    cand_logp = F.log_softmax(candidate.detach().float(), dim=-1)
    ref_logp = F.log_softmax(reference.detach().float(), dim=-1)
    values = []
    rows = []
    seq_len = int(input_ids.shape[1])
    for local_idx, position in enumerate(target_positions):
        if int(position) + 1 >= seq_len:
            rows.append(
                {
                    "position": int(position),
                    "has_teacher_target": False,
                    "target_token": None,
                    "target_logprob_delta": None,
                }
            )
            continue
        target = int(input_ids[0, int(position) + 1].item())
        delta = float((cand_logp[0, local_idx, target] - ref_logp[0, local_idx, target]).item())
        values.append(delta)
        rows.append(
            {
                "position": int(position),
                "has_teacher_target": True,
                "target_token": target,
                "target_logprob_delta": delta,
            }
        )
    return {
        "max_abs_target_logprob_delta": max((abs(value) for value in values), default=0.0),
        "target_logprob_delta_values": values,
        "per_position": rows,
    }


def _stack_batched_attention(
    *,
    q_full_rows: Sequence[torch.Tensor],
    k_true_rows: Sequence[torch.Tensor],
    v_true_rows: Sequence[torch.Tensor],
    target_positions: Sequence[int],
    kv_len: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense_by_position = []
    seed_by_position = []
    for position in target_positions:
        prefix_len = int(position) + 1
        start = max(0, prefix_len - kv_len) if kv_len and kv_len > 0 else 0
        q = torch.cat(
            [q_full[:, position : position + 1, :, :].contiguous() for q_full in q_full_rows],
            dim=0,
        )
        k = torch.cat(
            [k_true[:, start:prefix_len, :, :].contiguous() for k_true in k_true_rows],
            dim=0,
        )
        v = torch.cat(
            [v_true[:, start:prefix_len, :, :].contiguous() for v_true in v_true_rows],
            dim=0,
        )
        dense_by_position.append(_dense_true_gqa(q, k, v))
        seed_out = torch.empty_like(q)
        gate0_seed_only_attention_triton_forward_out(
            q,
            k,
            v,
            seed_out,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        seed_by_position.append(seed_out.clone())
    return torch.cat(dense_by_position, dim=1), torch.cat(seed_by_position, dim=1)


def _aggregate_metric_rows(rows: Sequence[Dict[str, Any]], *, metric_key: str) -> Dict[str, Any]:
    kl_values = []
    max_logit_delta_values = []
    top1_changed = 0
    topk_overlap_min = None
    topk_changed = 0
    top1_logprob_delta_values = []
    target_delta_values = []
    worst = {
        "prompt_id": None,
        "prompt_kind": None,
        "position": None,
        "kl_ref_to_candidate": -1.0,
    }
    for row in rows:
        metrics = row[metric_key]
        kl_values.append(float(metrics["kl_ref_to_candidate"]))
        max_logit_delta_values.append(float(metrics["max_abs_error"]))
        top1_changed += int(metrics["top1_changed_count"])
        topk_changed += int(metrics["topk_changed_count"])
        overlap = int(metrics["topk_overlap_min"])
        topk_overlap_min = overlap if topk_overlap_min is None else min(topk_overlap_min, overlap)
        top1_logprob_delta_values.extend(
            float(value) for value in metrics.get("reference_top1_logprob_delta_values") or []
        )
        target_delta_values.extend(
            float(value)
            for value in row.get("target_next_token", {}).get("target_logprob_delta_values", [])
        )
        for position_row in metrics.get("per_position") or []:
            kl = float(position_row["kl_ref_to_candidate"])
            if kl > float(worst["kl_ref_to_candidate"]):
                worst = {
                    "prompt_id": row["prompt_id"],
                    "prompt_kind": row["prompt_kind"],
                    "position": int(position_row["position"]),
                    "kl_ref_to_candidate": kl,
                    "topk_overlap": int(position_row["topk_overlap"]),
                    "top1_changed": bool(position_row["top1_changed"]),
                }
    kl_tensor = torch.tensor(kl_values, dtype=torch.float32) if kl_values else torch.zeros(1)
    return {
        "row_count": len(rows),
        "position_count_per_row": rows[0]["position_count"] if rows else 0,
        "case_count": sum(int(row["position_count"]) for row in rows),
        "kl_max": max(kl_values) if kl_values else 0.0,
        "kl_mean": float(kl_tensor.mean().item()) if kl_values else 0.0,
        "kl_p95": float(torch.quantile(kl_tensor, 0.95).item()) if len(kl_values) > 1 else (kl_values[0] if kl_values else 0.0),
        "max_logit_delta": max(max_logit_delta_values) if max_logit_delta_values else 0.0,
        "top1_changed_count": top1_changed,
        "topk_changed_count": topk_changed,
        "topk_overlap_min": int(topk_overlap_min if topk_overlap_min is not None else 0),
        "reference_top1_logprob_delta_max_abs": max(
            (abs(value) for value in top1_logprob_delta_values),
            default=0.0,
        ),
        "target_next_token_logprob_delta_max_abs": max(
            (abs(value) for value in target_delta_values),
            default=0.0,
        ),
        "worst_case_by_kl": worst,
    }


def _passes_gate(summary: Dict[str, Any], args: argparse.Namespace) -> bool:
    if args.require_top1_match and int(summary["top1_changed_count"]) != 0:
        return False
    if int(summary["topk_overlap_min"]) < int(args.min_topk_overlap):
        return False
    if float(summary["kl_max"]) > float(args.max_kl):
        return False
    if float(summary["reference_top1_logprob_delta_max_abs"]) > float(args.max_top1_logprob_delta):
        return False
    if float(summary["target_next_token_logprob_delta_max_abs"]) > float(args.max_target_logprob_delta):
        return False
    if args.max_logit_delta > 0.0 and float(summary["max_logit_delta"]) > float(args.max_logit_delta):
        return False
    return True


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    prompts = _prompts_from_args(args)

    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=args.use_safetensors,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    modules = _attention_modules(model)
    _layer_id, module_name, module = modules[args.layer_id]

    q_full_rows: List[torch.Tensor] = []
    k_true_rows: List[torch.Tensor] = []
    v_true_rows: List[torch.Tensor] = []
    token_rows: List[Dict[str, torch.Tensor]] = []
    baseline_logits_rows: List[torch.Tensor] = []
    prompt_rows = []
    target_positions: List[int] | None = None
    seq_len = None
    q_heads = kv_heads = dim = None

    for prompt_id, prompt_row in enumerate(prompts):
        tokens = tokenizer(
            prompt_row["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq,
        ).to(device)
        row_seq_len = int(tokens["input_ids"].shape[1])
        row_positions = _target_positions(
            row_seq_len,
            count=args.position_count,
            stride=args.position_stride,
        )
        if target_positions is None:
            target_positions = row_positions
            seq_len = row_seq_len
        elif row_positions != target_positions:
            raise ValueError(
                "all prompts must produce the same target positions for batched replay; "
                f"prompt 0 positions={target_positions}, prompt {prompt_id} positions={row_positions}"
            )
        print(
            f"[batched-logit-safety] capturing prompt {prompt_id} kind={prompt_row['kind']} seq_len={row_seq_len}",
            flush=True,
        )
        (capture, q_full, k_full, v_full, meta), baseline_logits = _capture_qkv_and_logits(
            model=model,
            tokens=tokens,
            layer_id=args.layer_id,
            apply_rope=args.tensor_space == "post_rope",
            target_positions=target_positions,
        )
        true_kv_heads = int(meta.get("num_kv_heads", k_full.shape[2]))
        k_true = _true_gqa_kv(k_full.contiguous(), true_kv_heads=true_kv_heads)
        v_true = _true_gqa_kv(v_full.contiguous(), true_kv_heads=true_kv_heads)
        q_heads = int(q_full.shape[2])
        kv_heads = int(k_true.shape[2])
        dim = int(q_full.shape[3])
        q_full_rows.append(q_full.contiguous())
        k_true_rows.append(k_true.contiguous())
        v_true_rows.append(v_true.contiguous())
        token_rows.append(tokens)
        baseline_logits_rows.append(baseline_logits)
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_kind": prompt_row["kind"],
                "seq_len": row_seq_len,
                "module_name_from_capture": capture.module_name,
            }
        )

    assert target_positions is not None and seq_len is not None
    print(
        f"[batched-logit-safety] computing batched dense/seed attention for positions {target_positions}",
        flush=True,
    )
    with torch.no_grad():
        dense_heads, seed_heads = _stack_batched_attention(
            q_full_rows=q_full_rows,
            k_true_rows=k_true_rows,
            v_true_rows=v_true_rows,
            target_positions=target_positions,
            kv_len=args.kv_len,
            args=args,
        )

    policy_rows = []
    dense_validation_rows = []
    for prompt_id, prompt_row in enumerate(prompt_rows):
        prompt_target_positions = target_positions
        dense_prompt_heads = dense_heads[prompt_id : prompt_id + 1].contiguous()
        seed_prompt_heads = seed_heads[prompt_id : prompt_id + 1].contiguous()
        with torch.no_grad():
            dense_patch = _project_attention_output(module, dense_prompt_heads)
            seed_patch = _project_attention_output(module, seed_prompt_heads)
        dense_patch_logits = _final_logits_with_patch(
            model=model,
            module=module,
            tokens=token_rows[prompt_id],
            patch=dense_patch,
            target_positions=prompt_target_positions,
        )
        seed_logits = _final_logits_with_patch(
            model=model,
            module=module,
            tokens=token_rows[prompt_id],
            patch=seed_patch,
            target_positions=prompt_target_positions,
        )
        baseline_logits = baseline_logits_rows[prompt_id]
        dense_metrics = _logit_metrics(
            dense_patch_logits,
            baseline_logits,
            target_positions=prompt_target_positions,
            top_k=args.top_k,
        )
        policy_vs_baseline = _logit_metrics(
            seed_logits,
            baseline_logits,
            target_positions=prompt_target_positions,
            top_k=args.top_k,
        )
        policy_vs_dense = _logit_metrics(
            seed_logits,
            dense_patch_logits,
            target_positions=prompt_target_positions,
            top_k=args.top_k,
        )
        target_next = _target_logprob_delta(
            seed_logits,
            baseline_logits,
            input_ids=token_rows[prompt_id]["input_ids"],
            target_positions=prompt_target_positions,
        )
        common = {
            **prompt_row,
            "position_count": len(prompt_target_positions),
            "target_positions": list(prompt_target_positions),
        }
        dense_validation_rows.append(
            {
                **common,
                "logits_vs_model_baseline": dense_metrics,
            }
        )
        policy_rows.append(
            {
                **common,
                "policy_name": "all_seed_only",
                "post_o_proj_error_vs_dense_patch": _error_summary(seed_patch, dense_patch),
                "head_output_error_vs_dense": _error_summary(seed_prompt_heads, dense_prompt_heads),
                "logits_vs_model_baseline": policy_vs_baseline,
                "logits_vs_dense_patch": policy_vs_dense,
                "target_next_token": target_next,
            }
        )

    baseline_summary = _aggregate_metric_rows(
        dense_validation_rows,
        metric_key="logits_vs_model_baseline",
    )
    policy_summary = _aggregate_metric_rows(
        policy_rows,
        metric_key="logits_vs_model_baseline",
    )
    dense_reference_summary = _aggregate_metric_rows(
        policy_rows,
        metric_key="logits_vs_dense_patch",
    )
    passes = _passes_gate(policy_summary, args)
    return {
        "schema": "streamattn.gate0.seed_only_batched_logit_safety.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model": {
            "model_id": args.model,
            "layer_id": args.layer_id,
            "attention_module": module_name,
        },
        "shape": {
            "batch": len(prompts),
            "seq_len": seq_len,
            "kv_len": int(min(args.kv_len, seq_len) if args.kv_len and args.kv_len > 0 else seq_len),
            "q_heads": q_heads,
            "true_kv_heads": kv_heads,
            "group_size": int(q_heads // kv_heads) if q_heads and kv_heads else None,
            "dim": dim,
            "dtype": args.dtype,
        },
        "target_positions": target_positions,
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "seed_blocks": args.sink_blocks + args.recent_blocks + args.middle_seed_blocks,
            "seed_tokens": (args.sink_blocks + args.recent_blocks + args.middle_seed_blocks)
            * args.block_size,
            "block_order": args.block_order,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        },
        "safety_gate": {
            "require_top1_match": args.require_top1_match,
            "min_topk_overlap": args.min_topk_overlap,
            "max_kl": args.max_kl,
            "max_logit_delta": args.max_logit_delta,
            "max_top1_logprob_delta": args.max_top1_logprob_delta,
            "max_target_logprob_delta": args.max_target_logprob_delta,
        },
        "baseline": {
            "dense_patch_logits_vs_model_baseline": baseline_summary,
        },
        "policy": {
            "name": "all_seed_only",
            "summary_vs_model_baseline": policy_summary,
            "summary_vs_dense_patch": dense_reference_summary,
            "passes_distribution_gate": passes,
            "fallback_recommendation": "use_streamattn_seed_only"
            if passes
            else "fallback_to_dense",
        },
        "rows": policy_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--prompt-kinds", default="needle,code,long_doc,needle,code,long_doc,needle,code")
    parser.add_argument("--prompt-repeat", type=int, default=3000)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--layer-id", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--tensor-space", choices=["post_rope"], default="post_rope")
    parser.add_argument("--position-count", type=int, default=8)
    parser.add_argument("--position-stride", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--require-top1-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-topk-overlap", type=int, default=4)
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    parser.add_argument("--max-logit-delta", type=float, default=0.0)
    parser.add_argument("--max-top1-logprob-delta", type=float, default=0.10)
    parser.add_argument("--max-target-logprob-delta", type=float, default=0.10)
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
