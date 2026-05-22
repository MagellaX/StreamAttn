"""Replay Gate-0 downstream policies through the remaining model to logits.

This profiler tests the strongest downstream-safety question:

    If we replace one layer's final-token attention output with a seed-only or
    seed+repair variant, do final next-token logits actually change?

It captures the target layer's post-RoPE Q/K/V during a normal model forward,
computes dense and seed-only attention outputs for the final token, projects
candidate head outputs through ``o_proj``, then re-runs the base transformer
with a forward hook that replaces only the target layer's final-token attention
output.  It compares final-token logits against the unmodified model and
against a dense-patch validation replay.

This is intentionally a research profiler, not a runtime path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_kv_group_repair_real import q_heads_for_kv_group  # noqa: E402
from benchmarks.profile_gate0_true_gqa import _dense_true_gqa, _true_gqa_kv  # noqa: E402
from benchmarks.profile_real_llm_gate1_heads import (  # noqa: E402
    _attention_modules,
    _capture_attention_inputs,
    _import_transformers,
    _load_prompts,
    _parse_int_list,
    _shape_qkv,
)
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward,
)


def _parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def _error_summary(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    diff = (actual - expected).detach().float()
    abs_diff = diff.abs()
    diff_l2 = torch.linalg.vector_norm(diff)
    expected_l2 = torch.linalg.vector_norm(expected.detach().float())
    return {
        "max_abs_error": float(abs_diff.max().item()),
        "mean_abs_error": float(abs_diff.mean().item()),
        "l2_error": float(diff_l2.item()),
        "relative_l2_error": float(diff_l2.item() / max(expected_l2.item(), 1.0e-6)),
    }


def _logit_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    target_positions: Sequence[int],
    top_k: int,
) -> Dict[str, Any]:
    cand = candidate.detach().float()
    ref = reference.detach().float()
    logp = F.log_softmax(ref, dim=-1)
    logq = F.log_softmax(cand, dim=-1)
    p = logp.exp()
    kl = torch.sum(p * (logp - logq), dim=-1)
    ref_top = torch.topk(ref, k=top_k, dim=-1)
    cand_top = torch.topk(cand, k=top_k, dim=-1)
    top1_changed_count = 0
    topk_changed_count = 0
    min_topk_overlap = top_k
    top1_logprob_delta_values = []
    per_position = []
    for local_idx, position in enumerate(target_positions):
        ref_top1 = int(ref_top.indices[0, local_idx, 0].item())
        cand_top1 = int(cand_top.indices[0, local_idx, 0].item())
        ref_top_set = set(int(x) for x in ref_top.indices[0, local_idx].tolist())
        cand_top_set = set(int(x) for x in cand_top.indices[0, local_idx].tolist())
        overlap = len(ref_top_set & cand_top_set)
        top1_changed = ref_top1 != cand_top1
        topk_changed = ref_top_set != cand_top_set
        top1_changed_count += int(top1_changed)
        topk_changed_count += int(topk_changed)
        min_topk_overlap = min(min_topk_overlap, overlap)
        top1_delta = float((logq[0, local_idx, ref_top1] - logp[0, local_idx, ref_top1]).item())
        top1_logprob_delta_values.append(top1_delta)
        per_position.append(
            {
                "position": int(position),
                "kl_ref_to_candidate": float(kl[0, local_idx].item()),
                "reference_top1": ref_top1,
                "candidate_top1": cand_top1,
                "top1_changed": bool(top1_changed),
                "topk_changed": bool(topk_changed),
                "topk_overlap": overlap,
                "reference_top_tokens": ref_top.indices[0, local_idx].tolist(),
                "candidate_top_tokens": cand_top.indices[0, local_idx].tolist(),
                "reference_top_scores": ref_top.values[0, local_idx].tolist(),
                "candidate_top_scores": cand_top.values[0, local_idx].tolist(),
                "reference_top1_logprob_delta": top1_delta,
            }
        )
    max_abs_top1_delta = max((abs(value) for value in top1_logprob_delta_values), default=0.0)
    return {
        **_error_summary(cand, ref),
        "kl_ref_to_candidate": float(kl.max().item()),
        "kl_abs_max": float(kl.abs().max().item()),
        "top1_changed": bool(top1_changed_count > 0),
        "top1_changed_count": top1_changed_count,
        "topk": top_k,
        "target_position_count": len(target_positions),
        "topk_changed": bool(topk_changed_count > 0),
        "topk_changed_count": topk_changed_count,
        "topk_overlap_min": min_topk_overlap,
        "reference_top1_logprob_delta": max_abs_top1_delta,
        "reference_top1_logprob_delta_values": top1_logprob_delta_values,
        "per_position": per_position,
    }


def _flatten_heads(heads: torch.Tensor) -> torch.Tensor:
    return heads.contiguous().view(heads.shape[0], heads.shape[1], heads.shape[2] * heads.shape[3])


def _project_attention_output(module: torch.nn.Module, heads: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return module.o_proj(_flatten_heads(heads).detach().clone())


def _replace_heads(base: torch.Tensor, source: torch.Tensor, heads: Sequence[int]) -> torch.Tensor:
    if not heads:
        return base.clone()
    out = base.clone()
    index = torch.tensor(list(heads), device=base.device, dtype=torch.long)
    out.index_copy_(2, index, source.index_select(2, index))
    return out


def _wo_projected_per_head(
    *,
    module: torch.nn.Module,
    candidate: torch.Tensor,
    dense: torch.Tensor,
) -> List[Dict[str, Any]]:
    weight = module.o_proj.weight.detach().float()
    _batch, _query, heads, dim = dense.shape
    rows = []
    for head in range(heads):
        delta = (candidate[:, :, head, :] - dense[:, :, head, :]).detach().float().reshape(-1, dim)
        start = head * dim
        end = start + dim
        projected = delta @ weight[:, start:end].T
        rows.append(
            {
                "head": head,
                "raw_l2_error": float(torch.linalg.vector_norm(delta).item()),
                "raw_max_abs_error": float(delta.abs().max().item()),
                "wo_projected_l2_error": float(torch.linalg.vector_norm(projected).item()),
                "wo_projected_max_abs_error": float(projected.abs().max().item()),
            }
        )
    return rows


def _rank_heads(rows: Sequence[Dict[str, Any]], metric: str, candidates: Iterable[int]) -> List[int]:
    allowed = set(int(head) for head in candidates)
    return [
        int(row["head"])
        for row in sorted(rows, key=lambda item: float(item.get(metric) or 0.0), reverse=True)
        if int(row["head"]) in allowed
    ]


def _attention_output_hook(patch: torch.Tensor, target_positions: Sequence[int]):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            attn = output[0]
            patched = attn.clone()
            index = torch.tensor(target_positions, device=patched.device, dtype=torch.long)
            patched.index_copy_(1, index, patch.to(device=patched.device, dtype=patched.dtype))
            return (patched, *output[1:])
        patched = output.clone()
        index = torch.tensor(target_positions, device=patched.device, dtype=torch.long)
        patched.index_copy_(1, index, patch.to(device=patched.device, dtype=patched.dtype))
        return patched

    return hook


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "model", model)


def _selected_logits(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    *,
    target_positions: Sequence[int],
) -> torch.Tensor:
    base = _base_model(model)
    with torch.no_grad():
        outputs = base(**tokens, use_cache=False)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        index = torch.tensor(target_positions, device=hidden.device, dtype=torch.long)
        logits = model.lm_head(hidden.index_select(1, index))
    return logits.detach()


def _final_logits_with_patch(
    *,
    model: torch.nn.Module,
    module: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    patch: torch.Tensor,
    target_positions: Sequence[int],
) -> torch.Tensor:
    handle = module.register_forward_hook(_attention_output_hook(patch, target_positions))
    try:
        return _selected_logits(model, tokens, target_positions=target_positions)
    finally:
        handle.remove()


def _make_policy_heads(
    *,
    dense_heads: torch.Tensor,
    seed_heads_all: torch.Tensor,
    seed_heads: Sequence[int],
) -> torch.Tensor:
    return _replace_heads(dense_heads, seed_heads_all, seed_heads)


def _candidate_specs(
    *,
    q_heads: int,
    kv_heads: int,
    dense_heads: torch.Tensor,
    seed_heads_all: torch.Tensor,
    module: torch.nn.Module,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    group_heads = {
        kv_head: q_heads_for_kv_group(kv_head, q_heads=q_heads, kv_heads=kv_heads)
        for kv_head in range(kv_heads)
    }
    all_heads = list(range(q_heads))
    seed_kv_groups = _parse_ints(args.seed_kv_groups)
    seed_group_heads = sorted({head for kv_head in seed_kv_groups for head in group_heads[kv_head]})
    trusted_heads = _parse_ints(args.trusted_seed_heads)

    specs: List[Dict[str, Any]] = [
        {"name": "dense_patch_validation", "seed_heads": []},
        {"name": f"seed_kv_groups_{'_'.join(str(x) for x in seed_kv_groups)}", "seed_heads": seed_group_heads},
        {"name": "all_seed_only", "seed_heads": all_heads},
        {"name": "trusted_cross_prompt_policy", "seed_heads": trusted_heads},
    ]

    repair_counts = _parse_ints(args.repair_counts)
    for base_name, base_seed_heads in (
        ("kv_group_0_seed", group_heads.get(0, [])),
        ("kv_group_1_seed", group_heads.get(1, [])),
        ("all_seed", all_heads),
    ):
        if not base_seed_heads:
            continue
        base = _make_policy_heads(dense_heads=dense_heads, seed_heads_all=seed_heads_all, seed_heads=base_seed_heads)
        projected = _wo_projected_per_head(module=module, candidate=base, dense=dense_heads)
        order = _rank_heads(projected, "wo_projected_l2_error", base_seed_heads)
        for count in repair_counts:
            repair = order[: min(int(count), len(order))]
            remaining_seed = [head for head in base_seed_heads if head not in set(repair)]
            specs.append(
                {
                    "name": f"{base_name}_repair_wo_l2_top{len(repair)}",
                    "seed_heads": remaining_seed,
                    "repair_heads": repair,
                }
            )

    # Deduplicate equivalent policies; repeated repair counts can saturate at group size.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for spec in specs:
        key = (spec["name"], tuple(spec.get("seed_heads") or ()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped[: max(1, args.max_policies)]


def _capture_qkv_and_logits(
    *,
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    layer_id: int,
    apply_rope: bool,
    target_positions: Sequence[int],
) -> tuple[Any, torch.Tensor]:
    captured, handles = _capture_attention_inputs(model, {layer_id})
    try:
        logits = _selected_logits(model, tokens, target_positions=target_positions)
    finally:
        for handle in handles:
            handle.remove()
    if not captured:
        raise RuntimeError(f"failed to capture attention inputs for layer {layer_id}")
    capture = captured[0]
    q, k, v, meta = _shape_qkv(capture, apply_rope=apply_rope)
    return (capture, q, k, v, meta), logits


def _target_positions(seq_len: int, *, count: int, stride: int) -> List[int]:
    if count <= 0:
        raise ValueError("--position-count must be positive")
    if stride <= 0:
        raise ValueError("--position-stride must be positive")
    positions = [seq_len - 1 - idx * stride for idx in range(count)]
    positions = [pos for pos in positions if pos >= 0]
    return sorted(set(positions))


def _compute_head_stacks(
    *,
    q_full: torch.Tensor,
    k_true_full: torch.Tensor,
    v_true_full: torch.Tensor,
    target_positions: Sequence[int],
    kv_len: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense_rows = []
    seed_rows = []
    for position in target_positions:
        prefix_len = int(position) + 1
        start = max(0, prefix_len - kv_len) if kv_len and kv_len > 0 else 0
        q = q_full[:, position : position + 1, :, :].contiguous()
        k = k_true_full[:, start:prefix_len, :, :].contiguous()
        v = v_true_full[:, start:prefix_len, :, :].contiguous()
        dense_rows.append(_dense_true_gqa(q, k, v))
        seed, _ = gate0_seed_only_attention_triton_forward(
            q,
            k,
            v,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            return_raw_stats=False,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        seed_rows.append(seed)
    return torch.cat(dense_rows, dim=1), torch.cat(seed_rows, dim=1)


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)

    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=args.use_safetensors,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    prompt = _load_prompts(args)[0]
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq,
    ).to(device)
    seq_len = int(tokens["input_ids"].shape[1])
    target_positions = _target_positions(
        seq_len,
        count=args.position_count,
        stride=args.position_stride,
    )
    if args.kv_len and seq_len != args.kv_len:
        print(
            f"[logit-replay] token count {seq_len} differs from kv_len {args.kv_len}; using captured sequence length",
            flush=True,
        )

    print("[logit-replay] capturing baseline Q/K/V and target-position logits", flush=True)
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
    modules = _attention_modules(model)
    _layer_id, module_name, module = modules[args.layer_id]

    print(
        f"[logit-replay] computing dense and seed-only attention patches for positions {target_positions}",
        flush=True,
    )
    with torch.no_grad():
        dense_heads, seed_heads_all = _compute_head_stacks(
            q_full=q_full,
            k_true_full=k_true,
            v_true_full=v_true,
            target_positions=target_positions,
            kv_len=args.kv_len,
            args=args,
        )
        dense_patch = _project_attention_output(module, dense_heads)

    print("[logit-replay] validating dense patch replay", flush=True)
    dense_patch_logits = _final_logits_with_patch(
        model=model,
        module=module,
        tokens=tokens,
        patch=dense_patch,
        target_positions=target_positions,
    )

    specs = _candidate_specs(
        q_heads=q_heads,
        kv_heads=kv_heads,
        dense_heads=dense_heads,
        seed_heads_all=seed_heads_all,
        module=module,
        args=args,
    )
    rows = []
    for spec in specs:
        name = spec["name"]
        seed_heads = list(spec.get("seed_heads") or [])
        print(f"[logit-replay] replaying {name} seed_heads={seed_heads}", flush=True)
        candidate_heads = _make_policy_heads(
            dense_heads=dense_heads,
            seed_heads_all=seed_heads_all,
            seed_heads=seed_heads,
        )
        patch = _project_attention_output(module, candidate_heads)
        logits = (
            dense_patch_logits
            if name == "dense_patch_validation"
            else _final_logits_with_patch(
                model=model,
                module=module,
                tokens=tokens,
                patch=patch,
                target_positions=target_positions,
            )
        )
        rows.append(
            {
                **spec,
                "post_o_proj_error_vs_dense_patch": _error_summary(patch, dense_patch),
                "logits_vs_model_baseline": _logit_metrics(
                    logits,
                    baseline_logits,
                    target_positions=target_positions,
                    top_k=args.top_k,
                ),
                "logits_vs_dense_patch": _logit_metrics(
                    logits,
                    dense_patch_logits,
                    target_positions=target_positions,
                    top_k=args.top_k,
                ),
            }
        )

    return {
        "schema": "streamattn.gate0.logit_replay.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model": {
            "model_id": args.model,
            "layer_id": args.layer_id,
            "attention_module": module_name,
            "module_name_from_capture": capture.module_name,
        },
        "shape": {
            "seq_len": seq_len,
            "kv_len": int(min(args.kv_len, seq_len) if args.kv_len and args.kv_len > 0 else seq_len),
            "q_heads": q_heads,
            "true_kv_heads": kv_heads,
            "group_size": q_heads // kv_heads,
            "dim": int(q_full.shape[3]),
            "dtype": args.dtype,
        },
        "target_positions": target_positions,
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "baseline": {
            "dense_patch_logits_vs_model_baseline": _logit_metrics(
                dense_patch_logits,
                baseline_logits,
                target_positions=target_positions,
                top_k=args.top_k,
            ),
        },
        "policies": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--layer-id", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--tensor-space", choices=["post_rope"], default="post_rope")
    parser.add_argument("--seed-kv-groups", default="0")
    parser.add_argument("--trusted-seed-heads", default="2,3,4")
    parser.add_argument("--repair-counts", default="0,2,4,7,11")
    parser.add_argument("--max-policies", type=int, default=18)
    parser.add_argument("--position-count", type=int, default=1)
    parser.add_argument("--position-stride", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
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
