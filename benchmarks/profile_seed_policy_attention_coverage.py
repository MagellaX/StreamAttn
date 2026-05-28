"""Prompt-aware attention coverage diagnostics for stressed seed policies.

This profiler answers a specific robustness question:

    Did a routed layer fail because the seed schedule missed important attention
    mass, or because the route is compositionally/logit sensitive even when seed
    coverage is reasonable?

It captures exact attention distributions at selected layers during decode under
two conditions:

* dense_conditioned: all upstream layers are exact;
* route_conditioned: the current routed bundle runs, while the capture computes
  exact attention at the target layer on the routed hidden/cache state.

The output is diagnostic and intentionally slower than production decode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_closed_loop_rollout import _prompts_from_args  # noqa: E402
from benchmarks.profile_gate0_seed_only_multi_layer_rollout import RouteBundle  # noqa: E402
from benchmarks.profile_seed_only_route_bundle_decode import (  # noqa: E402
    _append_decode_attention_mask,
    _batch_tokens,
    _cache_layer_tensors,
    _import_qwen_rotary,
    _native_cache_from_hf_cache,
    _native_cache_mask_bookkeeping,
    _patched_seed_only_decode_modules,
    _prefill,
)
from benchmarks.profile_seed_only_stress_attribution import QWEN3B_POLICY_BY_LAYER  # noqa: E402
from benchmarks.profile_real_llm_gate1_heads import _attention_modules, _import_transformers  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype  # noqa: E402
from stream_attention.decode import load_packaged_gate0_seed_only_batched_policy  # noqa: E402


DEFAULT_TARGET_BUCKETS = ("chat_instruction", "noisy_neartie", "json_tool", "needle_rag")
DEFAULT_SELECTOR_PROFILES = ("fixed_policy",)
SELECTOR_PROFILES = frozenset(
    {
        "fixed_policy",
        "support_block_oracle",
        "qk_block_max",
        "exact_mass_oracle",
        "support_mass_oracle",
        "value_residual_oracle",
    }
)


@dataclass
class CaptureState:
    condition: str
    steps: set[int]
    current_step: int = -1
    hook_calls: Dict[str, int] = None
    selected_calls: Dict[str, int] = None
    captured_heads: Dict[str, int] = None
    miss_reasons: Dict[str, int] = None

    def __post_init__(self):
        self.hook_calls = {}
        self.selected_calls = {}
        self.captured_heads = {}
        self.miss_reasons = {}

    def bump(self, field: str, key: str, amount: int = 1) -> None:
        target = getattr(self, field)
        target[key] = int(target.get(key, 0)) + int(amount)


def _parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]


def _parse_selector_profiles(text: str) -> List[str]:
    profiles = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if not profiles:
        profiles = list(DEFAULT_SELECTOR_PROFILES)
    unknown = sorted(set(profiles) - SELECTOR_PROFILES)
    if unknown:
        raise ValueError(f"unknown selector profiles: {unknown}; supported={sorted(SELECTOR_PROFILES)}")
    return profiles


def _policy_bundle_for_layers(layers: Sequence[int]) -> RouteBundle:
    names = [QWEN3B_POLICY_BY_LAYER[int(layer)] for layer in layers]
    policies = [load_packaged_gate0_seed_only_batched_policy(name) for name in names]
    return RouteBundle(
        policy_names=names,
        policies=policies,
        artifacts=[{} for _ in policies],
        layer_ids=[int(policy.layer_id) for policy in policies],
    )


def _seed_indices(
    *,
    seq_len: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
) -> torch.Tensor:
    num_blocks = math.ceil(seq_len / block_size)
    recent_start = num_blocks - recent_blocks
    blocks: List[int] = []
    blocks.extend(range(0, sink_blocks))
    blocks.extend(range(recent_start, num_blocks))
    if block_order == "sequential":
        blocks.extend(range(sink_blocks, sink_blocks + middle_seed_blocks))
    else:
        blocks.extend(range(recent_start - 1, recent_start - 1 - middle_seed_blocks, -1))
    indices: List[int] = []
    seen = set()
    for block in blocks:
        if block < 0 or block >= num_blocks:
            continue
        start = block * block_size
        end = min(seq_len, start + block_size)
        for idx in range(start, end):
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return torch.tensor(sorted(indices), dtype=torch.long)


def _policy_seed_blocks(
    *,
    seq_len: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    include_middle: bool = True,
) -> List[int]:
    num_blocks = math.ceil(seq_len / block_size)
    recent_start = num_blocks - recent_blocks
    blocks: List[int] = []
    blocks.extend(range(0, sink_blocks))
    blocks.extend(range(recent_start, num_blocks))
    if include_middle:
        if block_order == "sequential":
            blocks.extend(range(sink_blocks, sink_blocks + middle_seed_blocks))
        else:
            blocks.extend(range(recent_start - 1, recent_start - 1 - middle_seed_blocks, -1))
    seen = set()
    valid = []
    for block in blocks:
        if block < 0 or block >= num_blocks or block in seen:
            continue
        seen.add(block)
        valid.append(int(block))
    return valid


def _seed_mask_from_blocks(
    *,
    seq_len: int,
    block_size: int,
    blocks: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
    for block in blocks:
        start = int(block) * block_size
        end = min(seq_len, start + block_size)
        if start < seq_len:
            mask[start:end] = True
    return mask


def _block_scores_from_values(values: torch.Tensor, *, block_size: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    seq_len = int(values.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        scores[block] = float(values[start:end].float().sum().item())
    return scores


def _block_max_scores(values: torch.Tensor, *, block_size: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    seq_len = int(values.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        scores[block] = float(values[start:end].float().max().item())
    return scores


def _block_value_scores(probs: torch.Tensor, v: torch.Tensor, *, block_size: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    seq_len = int(probs.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    weighted_v = probs[:, None].float() * v.float()
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        scores[block] = float(torch.linalg.vector_norm(weighted_v[start:end].sum(dim=0)).item())
    return scores


def _select_middle_blocks(
    scores: Dict[int, float],
    *,
    base_blocks: Sequence[int],
    fixed_middle_blocks: Sequence[int],
    middle_seed_blocks: int,
) -> List[int]:
    base = set(int(block) for block in base_blocks)
    selected: List[int] = []
    for block, score in sorted(scores.items(), key=lambda item: (-float(item[1]), int(item[0]))):
        if block in base or block in selected:
            continue
        if score <= 0.0:
            continue
        selected.append(int(block))
        if len(selected) >= middle_seed_blocks:
            return selected
    for block in fixed_middle_blocks:
        if block in base or block in selected:
            continue
        selected.append(int(block))
        if len(selected) >= middle_seed_blocks:
            break
    return selected


def _selector_seed_masks(
    *,
    scores: torch.Tensor,
    probs: torch.Tensor,
    v: torch.Tensor,
    support_mask: torch.Tensor,
    distractor_mask: torch.Tensor,
    policy: Any,
    selector_profiles: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    seq_len = int(probs.shape[0])
    block_size = int(policy.block_size)
    base_blocks = _policy_seed_blocks(
        seq_len=seq_len,
        block_size=block_size,
        sink_blocks=int(policy.sink_blocks),
        recent_blocks=int(policy.recent_blocks),
        middle_seed_blocks=int(policy.middle_seed_blocks),
        block_order=str(policy.block_order),
        include_middle=False,
    )
    fixed_blocks = _policy_seed_blocks(
        seq_len=seq_len,
        block_size=block_size,
        sink_blocks=int(policy.sink_blocks),
        recent_blocks=int(policy.recent_blocks),
        middle_seed_blocks=int(policy.middle_seed_blocks),
        block_order=str(policy.block_order),
        include_middle=True,
    )
    fixed_middle_blocks = [block for block in fixed_blocks if block not in set(base_blocks)]
    support_values = support_mask.float()
    distractor_values = distractor_mask.float()
    support_count_scores = _block_scores_from_values(
        support_values - 0.25 * distractor_values,
        block_size=block_size,
    )
    qk_scores = _block_max_scores(scores, block_size=block_size)
    exact_mass_scores = _block_scores_from_values(probs, block_size=block_size)
    support_mass_scores = _block_scores_from_values(
        probs * support_values - 0.25 * probs * distractor_values,
        block_size=block_size,
    )
    value_scores = _block_value_scores(probs, v, block_size=block_size)
    score_by_selector = {
        "support_block_oracle": support_count_scores,
        "qk_block_max": qk_scores,
        "exact_mass_oracle": exact_mass_scores,
        "support_mass_oracle": support_mass_scores,
        "value_residual_oracle": value_scores,
    }
    out: Dict[str, Dict[str, Any]] = {}
    for selector in selector_profiles:
        if selector == "fixed_policy":
            blocks = fixed_blocks
        else:
            middle_blocks = _select_middle_blocks(
                score_by_selector[selector],
                base_blocks=base_blocks,
                fixed_middle_blocks=fixed_middle_blocks,
                middle_seed_blocks=int(policy.middle_seed_blocks),
            )
            blocks = list(base_blocks) + middle_blocks
        mask = _seed_mask_from_blocks(
            seq_len=seq_len,
            block_size=block_size,
            blocks=blocks,
            device=probs.device,
        )
        out[selector] = {
            "mask": mask,
            "selected_blocks": [int(block) for block in blocks],
            "middle_blocks": [int(block) for block in blocks if block not in set(base_blocks)],
            "overlap_with_fixed_blocks": len(set(blocks) & set(fixed_blocks)),
            "fixed_block_count": len(fixed_blocks),
        }
    return out


def _find_spans(text: str, terms: Iterable[str]) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    for term in terms:
        if not term:
            continue
        start = 0
        while True:
            found = text.find(term, start)
            if found < 0:
                break
            spans.append((found, found + len(term)))
            start = found + max(1, len(term))
    return spans


def _stress_terms(row: Dict[str, str]) -> tuple[List[str], List[str]]:
    bucket = row.get("bucket") or row.get("kind", "")
    prompt_id = row.get("id", "")
    try:
        idx = int(prompt_id.rsplit("_", 1)[-1])
    except Exception:
        idx = int(row.get("row", 0) or 0)
    if bucket == "code":
        return (
            [f"critical_window_token_{idx:03d}_alpha"],
            [
                f"critical_window_token_{idx:03d}_beta",
                f"critical_window_token_{idx:03d}_gamma",
                f"stale_window_token_{idx:03d}",
            ],
        )
    if bucket == "chat_instruction":
        return (
            [f"STYLE_LOCK_{idx:03d}_terse_lowercase"],
            ["verbose explanations", "uppercase headings", "apologies"],
        )
    if bucket == "needle_rag":
        return (
            [f"NOLIMA_ASSOC_{idx:03d}_cobalt", "ocean-paint association"],
            ["unrelated labels", "many projects"],
        )
    if bucket == "json_tool":
        return (
            [f"route_{idx:03d}_strict", "retries 2", "enabled true", "valid JSON"],
            ["malformed examples", "wrong", "distractor"],
        )
    if bucket == "noisy_neartie":
        answer = "amber" if idx % 2 == 0 else "azure"
        distractor = "azure" if answer == "amber" else "amber"
        return (
            [f"choose {answer}", f"only {answer}", "official tie-break"],
            [f"prefer {distractor}", f"choose {distractor}", "informal notes"],
        )
    if bucket == "math":
        answer = str(3 * (17 + idx) + 2 * (23 + idx))
        return ([f"only {answer}", "parity flag"], ["distractor recurrences", "otherwise"])
    if bucket == "multilingual":
        return ([f"CROSS_LINGUAL_LOCK_{idx:03d}"], ["wrong markers", "distractor"])
    if bucket == "long_doc":
        stance = "approve with monitoring" if idx % 2 == 0 else "reject pending audit"
        other = "reject pending audit" if stance == "approve with monitoring" else "approve with monitoring"
        return ([stance, "final committee stance"], [other, "old stances"])
    return ([], [])


def _token_role_masks(
    tokenizer,
    prompt_rows: Sequence[Dict[str, str]],
    *,
    max_seq: int,
    truncation_side: str,
) -> List[Dict[str, torch.Tensor]]:
    old_truncation_side = getattr(tokenizer, "truncation_side", None)
    tokenizer.truncation_side = truncation_side
    masks = []
    try:
        for row in prompt_rows:
            encoded = tokenizer(
                row["prompt"],
                truncation=True,
                max_length=max_seq,
                return_offsets_mapping=True,
            )
            offsets = encoded.get("offset_mapping")
            if offsets is None:
                raise RuntimeError("tokenizer must provide offset_mapping for prompt-aware coverage")
            support_terms, distractor_terms = _stress_terms(row)
            support_spans = _find_spans(row["prompt"], support_terms)
            distractor_spans = _find_spans(row["prompt"], distractor_terms)
            support = torch.zeros(len(offsets), dtype=torch.bool)
            distractor = torch.zeros(len(offsets), dtype=torch.bool)
            for token_idx, (start, end) in enumerate(offsets):
                if end <= start:
                    continue
                support[token_idx] = any(start < span_end and end > span_start for span_start, span_end in support_spans)
                distractor[token_idx] = any(
                    start < span_end and end > span_start for span_start, span_end in distractor_spans
                )
            masks.append(
                {
                    "support": support[:max_seq],
                    "distractor": distractor[:max_seq],
                }
            )
    finally:
        if old_truncation_side is not None:
            tokenizer.truncation_side = old_truncation_side
    return masks


def _selected_rows_from_artifact(
    path: str,
    *,
    target_buckets: set[str],
    include_step0: bool,
) -> Dict[int, set[int]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    safety = payload.get("safety") or {}
    row_steps: Dict[int, set[int]] = defaultdict(set)
    if include_step0:
        for prompt in payload.get("prompts") or []:
            if str(prompt.get("bucket") or prompt.get("kind")) in target_buckets:
                row_steps[int(prompt["row"])].add(0)
    for bucket, bucket_summary in (safety.get("by_prompt_bucket") or {}).items():
        if bucket not in target_buckets:
            continue
        for key in ("first_divergence", "first_sample_divergence"):
            event = bucket_summary.get(key)
            if not event:
                continue
            step = int(event.get("step", 0))
            for row in event.get("rows") or []:
                row_steps[int(row)].add(step)
        worst = bucket_summary.get("worst_case_by_kl") or {}
        if "row" in worst:
            row_steps[int(worst["row"])].add(0)
    return row_steps


def _role_mask_for_seq(row_mask: torch.Tensor, *, seq_len: int, device: torch.device) -> torch.Tensor:
    if seq_len <= row_mask.numel():
        return row_mask[:seq_len].to(device=device)
    pad = torch.zeros(seq_len - row_mask.numel(), dtype=torch.bool)
    return torch.cat([row_mask, pad], dim=0).to(device=device)


def _capture_metrics_for_head(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seed_mask: torch.Tensor,
    support_mask: torch.Tensor,
    distractor_mask: torch.Tensor,
) -> tuple[Dict[str, float], torch.Tensor]:
    scores = torch.matmul(k.float(), q.float()) / math.sqrt(q.shape[-1])
    probs = torch.softmax(scores, dim=0)
    return _capture_metrics_from_probs(
        probs=probs,
        v=v,
        seed_mask=seed_mask,
        support_mask=support_mask,
        distractor_mask=distractor_mask,
    ), probs.detach().cpu()


def _capture_metrics_from_probs(
    *,
    probs: torch.Tensor,
    v: torch.Tensor,
    seed_mask: torch.Tensor,
    support_mask: torch.Tensor,
    distractor_mask: torch.Tensor,
) -> Dict[str, float]:
    omitted_mask = ~seed_mask
    support_in = probs[seed_mask & support_mask].sum()
    support_out = probs[omitted_mask & support_mask].sum()
    distractor_in = probs[seed_mask & distractor_mask].sum()
    distractor_out = probs[omitted_mask & distractor_mask].sum()
    full_value = torch.sum(probs[:, None] * v.float(), dim=0)
    omitted_value = torch.sum(probs[omitted_mask, None] * v[omitted_mask].float(), dim=0)
    full_norm = torch.linalg.vector_norm(full_value).clamp_min(1.0e-6)
    value_residual_ratio = torch.linalg.vector_norm(omitted_value) / full_norm
    delta_full = (support_in + support_out) - (distractor_in + distractor_out)
    delta_seed = support_in - distractor_in
    metrics = {
        "mass_seed": float(probs[seed_mask].sum().item()),
        "mass_omitted": float(probs[omitted_mask].sum().item()),
        "support_in_seed": float(support_in.item()),
        "support_out_seed": float(support_out.item()),
        "distractor_in_seed": float(distractor_in.item()),
        "distractor_out_seed": float(distractor_out.item()),
        "delta_full": float(delta_full.item()),
        "delta_seed": float(delta_seed.item()),
        "delta_collapse": float((delta_full - delta_seed).item()),
        "value_residual_ratio": float(value_residual_ratio.item()),
    }
    return metrics


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.float().clamp_min(1.0e-30)
    q = q.float().clamp_min(1.0e-30)
    m = 0.5 * (p + q)
    return float((0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()).item())


def _first_positional_tensor_pair(args: Sequence[Any]) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    for item in args:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        first, second = item
        if torch.is_tensor(first) and torch.is_tensor(second):
            return first, second
    return None


def _first_cache_like(args: Sequence[Any]):
    for item in args:
        if item is None or torch.is_tensor(item) or isinstance(item, (tuple, list, dict)):
            continue
        if hasattr(item, "layers") or "Cache" in type(item).__name__:
            return item
    return None


def _first_cache_position_like(args: Sequence[Any]) -> Optional[torch.Tensor]:
    for item in args:
        if not torch.is_tensor(item):
            continue
        if item.dtype not in (torch.int32, torch.int64, torch.long):
            continue
        if item.dim() <= 1:
            return item
    return None


def _row_recommendation(row: Dict[str, Any]) -> str:
    support_out = float(row.get("support_out_seed", 0.0) or 0.0)
    omitted = float(row.get("mass_omitted", 0.0) or 0.0)
    collapse = float(row.get("delta_collapse", 0.0) or 0.0)
    value_residual = float(row.get("value_residual_ratio", 0.0) or 0.0)
    js = float(row.get("dense_vs_route_attention_js", 0.0) or 0.0)
    if support_out >= 0.05 or (omitted >= 0.25 and collapse >= 0.02):
        return "coverage_repair"
    if value_residual >= 0.35:
        return "value_sensitive_repair"
    if js >= 0.02:
        return "composition_repair"
    return "exact_or_margin_gate"


def _register_coverage_hooks(
    *,
    model,
    target_layers: Sequence[int],
    state: CaptureState,
    row_steps: Dict[int, set[int]],
    prompt_rows: Sequence[Dict[str, str]],
    role_masks: Sequence[Dict[str, torch.Tensor]],
    policies_by_layer: Dict[int, Any],
    selector_profiles: Sequence[str],
    native_cache,
    rows_out: List[Dict[str, Any]],
    probs_out: Dict[tuple, torch.Tensor],
):
    modules_by_layer = {int(layer_id): module for layer_id, _name, module in _attention_modules(model)}
    apply_rotary_pos_emb = _import_qwen_rotary()
    handles = []

    for layer_id in target_layers:
        module = modules_by_layer[int(layer_id)]
        policy = policies_by_layer[int(layer_id)]

        def hook(_mod, hook_args, hook_kwargs, *, _layer_id=int(layer_id), _module=module, _policy=policy):
            step = int(state.current_step)
            state.bump("hook_calls", str(_layer_id))
            selected_rows = [row for row, steps in row_steps.items() if step in steps]
            if not selected_rows:
                state.bump("miss_reasons", f"{_layer_id}:no_selected_rows")
                return
            state.bump("selected_calls", str(_layer_id))
            hidden_states = hook_args[0] if hook_args else hook_kwargs.get("hidden_states")
            if hidden_states is None:
                state.bump("miss_reasons", f"{_layer_id}:missing_hidden_states")
                return
            position_embeddings = hook_kwargs.get("position_embeddings")
            if position_embeddings is None:
                position_embeddings = _first_positional_tensor_pair(hook_args[1:])
            if position_embeddings is None:
                state.bump("miss_reasons", f"{_layer_id}:missing_position_embeddings")
                return
            past_key_value = hook_kwargs.get("past_key_value", hook_kwargs.get("past_key_values"))
            if past_key_value is None:
                past_key_value = _first_cache_like(hook_args[1:])
            if past_key_value is None:
                state.bump("miss_reasons", f"{_layer_id}:missing_past_key_value")
                return
            cache_position = hook_kwargs.get("cache_position")
            if cache_position is None:
                cache_position = _first_cache_position_like(hook_args[1:])
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, _module.head_dim)
            q = _module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k_current = _module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v_current = _module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            q, k_current = apply_rotary_pos_emb(q, k_current, cos, sin)
            pos = int(cache_position.reshape(-1)[0].item()) if cache_position is not None else None
            if state.condition == "route_conditioned" and native_cache is not None and _layer_id in native_cache.layer_to_index:
                k_past, v_past = native_cache.layer_cache(_layer_id)
                if pos is None:
                    pos = int(native_cache.prefill_len + step)
                k_past = k_past[:, :, :pos, :]
                v_past = v_past[:, :, :pos, :]
            else:
                k_past, v_past = _cache_layer_tensors(past_key_value, _layer_id)
                if pos is None:
                    pos = int(k_past.shape[2])
                k_past = k_past[:, :, :pos, :]
                v_past = v_past[:, :, :pos, :]
            k_all = torch.cat([k_past, k_current], dim=2)
            v_all = torch.cat([v_past, v_current], dim=2)
            seq_len = int(k_all.shape[2])
            seed_indices = _seed_indices(
                seq_len=seq_len,
                block_size=int(_policy.block_size),
                sink_blocks=int(_policy.sink_blocks),
                recent_blocks=int(_policy.recent_blocks),
                middle_seed_blocks=int(_policy.middle_seed_blocks),
                block_order=str(_policy.block_order),
            ).to(device=hidden_states.device)
            seed_mask = torch.zeros(seq_len, device=hidden_states.device, dtype=torch.bool)
            seed_mask[seed_indices] = True
            group_size = int(q.shape[1] // k_all.shape[1])
            for row in selected_rows:
                if row >= q.shape[0]:
                    continue
                support_mask = _role_mask_for_seq(
                    role_masks[row]["support"],
                    seq_len=seq_len,
                    device=hidden_states.device,
                )
                distractor_mask = _role_mask_for_seq(
                    role_masks[row]["distractor"],
                    seq_len=seq_len,
                    device=hidden_states.device,
                )
                for head in range(q.shape[1]):
                    kv_head = int(head // group_size)
                    q_row = q[row, head, 0, :]
                    k_row = k_all[row, kv_head, :, :]
                    v_row = v_all[row, kv_head, :, :]
                    scores = torch.matmul(k_row.float(), q_row.float()) / math.sqrt(q_row.shape[-1])
                    probs = torch.softmax(scores, dim=0)
                    key = (step, _layer_id, row, head)
                    probs_out[(state.condition, *key)] = probs.detach().cpu()
                    state.bump("captured_heads", str(_layer_id))
                    selector_masks = _selector_seed_masks(
                        scores=scores,
                        probs=probs,
                        v=v_row,
                        support_mask=support_mask,
                        distractor_mask=distractor_mask,
                        policy=_policy,
                        selector_profiles=selector_profiles,
                    )
                    for selector, selector_payload in selector_masks.items():
                        selector_mask = selector_payload["mask"]
                        metrics = _capture_metrics_from_probs(
                            probs=probs,
                            v=v_row,
                            seed_mask=selector_mask,
                            support_mask=support_mask,
                            distractor_mask=distractor_mask,
                        )
                        rows_out.append(
                            {
                                "condition": state.condition,
                                "selector": selector,
                                "step": step,
                                "layer": _layer_id,
                                "row": int(row),
                                "head": int(head),
                                "kv_head": kv_head,
                                "bucket": prompt_rows[row].get("bucket", prompt_rows[row].get("kind", "")),
                                "prompt_id": prompt_rows[row].get("id", ""),
                                "seq_len": seq_len,
                                "seed_tokens": int(selector_mask.sum().item()),
                                "selected_blocks": selector_payload["selected_blocks"],
                                "middle_blocks": selector_payload["middle_blocks"],
                                "overlap_with_fixed_blocks": selector_payload["overlap_with_fixed_blocks"],
                                "fixed_block_count": selector_payload["fixed_block_count"],
                                **metrics,
                            }
                        )

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


def _decode_capture_loop(
    *,
    model,
    past_key_values,
    attention_mask: torch.Tensor,
    first_token: torch.Tensor,
    fixed_input_tokens: Optional[Sequence[torch.Tensor]],
    state: CaptureState,
    max_step: int,
    native_routed_cache: bool,
):
    input_token = first_token
    mask = attention_mask
    logits_by_step = []
    input_tokens = []
    with torch.inference_mode():
        for step in range(max_step + 1):
            if fixed_input_tokens is not None:
                input_token = fixed_input_tokens[step]
            state.current_step = step
            input_tokens.append(input_token.detach().clone())
            model_kwargs = {
                "input_ids": input_token,
                "attention_mask": _append_decode_attention_mask(mask, input_token),
                "past_key_values": past_key_values,
                "use_cache": True,
                "logits_to_keep": 1,
            }
            if native_routed_cache:
                model_kwargs["cache_position"] = torch.tensor(
                    [mask.shape[1]],
                    device=input_token.device,
                    dtype=torch.long,
                )
            if hasattr(past_key_values, "_streamattn_mask_kv_length"):
                past_key_values._streamattn_past_kv_length = int(mask.shape[1])
                past_key_values._streamattn_mask_kv_length = int(mask.shape[1] + input_token.shape[1])
            out = model(**model_kwargs)
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :].detach()
            logits_by_step.append(logits)
            mask = _append_decode_attention_mask(mask, input_token)
            input_token = torch.argmax(logits, dim=-1, keepdim=True)
    return {
        "logits_by_step": logits_by_step,
        "input_tokens": input_tokens,
    }


def _dense_tokens(
    *,
    model,
    tokens: Dict[str, torch.Tensor],
    max_step: int,
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    prefill = _prefill(model, tokens)
    first_token = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
    state = CaptureState(condition="dense_token_path", steps=set())
    dense = _decode_capture_loop(
        model=model,
        past_key_values=prefill.past_key_values,
        attention_mask=tokens["attention_mask"],
        first_token=first_token,
        fixed_input_tokens=None,
        state=state,
        max_step=max_step,
        native_routed_cache=False,
    )
    return first_token, dense["input_tokens"]


def _aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for key_name in ("condition", "bucket", "layer", "selector"):
        values = sorted({str(row[key_name]) for row in rows})
        grouped[f"by_{key_name}"] = {}
        for value in values:
            subset = [row for row in rows if str(row[key_name]) == value]
            grouped[f"by_{key_name}"][value] = _metric_summary(subset)
    grouped["overall"] = _metric_summary(rows)
    return grouped


def _metric_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def vals(name: str) -> List[float]:
        return [float(row.get(name, 0.0) or 0.0) for row in rows]

    def p95(name: str) -> float:
        values = sorted(vals(name))
        if not values:
            return 0.0
        return values[min(len(values) - 1, int(round((len(values) - 1) * 0.95)))]

    return {
        "count": len(rows),
        "mass_omitted_mean": sum(vals("mass_omitted")) / max(1, len(rows)),
        "mass_omitted_p95": p95("mass_omitted"),
        "support_out_seed_p95": p95("support_out_seed"),
        "delta_collapse_p95": p95("delta_collapse"),
        "value_residual_ratio_p95": p95("value_residual_ratio"),
        "dense_vs_route_attention_js_p95": p95("dense_vs_route_attention_js"),
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    target_layers = _parse_ints(args.target_layers)
    routed_layers = _parse_ints(args.routed_layers)
    target_buckets = {part.strip() for part in args.target_buckets.split(",") if part.strip()}
    selector_profiles = _parse_selector_profiles(args.selector_profiles)

    prompt_rows = _prompts_from_args(args)
    for idx, row in enumerate(prompt_rows):
        row.setdefault("row", str(idx))
    row_steps = _selected_rows_from_artifact(
        args.failure_artifact,
        target_buckets=target_buckets,
        include_step0=args.include_step0,
    )
    if not row_steps:
        row_steps = {idx: {0} for idx, row in enumerate(prompt_rows) if row.get("bucket", row.get("kind")) in target_buckets}
    if args.max_rows > 0:
        keep_rows = set(sorted(row_steps)[: args.max_rows])
        row_steps = {row: steps for row, steps in row_steps.items() if row in keep_rows}
    capture_steps = sorted({step for steps in row_steps.values() for step in steps})
    max_step = max(capture_steps)

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
    tokens = _batch_tokens(
        tokenizer,
        prompt_rows,
        max_seq=args.max_seq,
        device=device,
        truncation_side=args.prompt_truncation_side,
    )
    role_masks = _token_role_masks(
        tokenizer,
        prompt_rows,
        max_seq=args.max_seq,
        truncation_side=args.prompt_truncation_side,
    )

    first_token, dense_input_tokens = _dense_tokens(model=model, tokens=tokens, max_step=max_step)
    policies_by_layer = {
        int(layer): load_packaged_gate0_seed_only_batched_policy(QWEN3B_POLICY_BY_LAYER[int(layer)])
        for layer in sorted(set(target_layers) | set(routed_layers))
    }

    rows: List[Dict[str, Any]] = []
    probs: Dict[tuple, torch.Tensor] = {}

    dense_prefill = _prefill(model, tokens)
    dense_state = CaptureState(condition="dense_conditioned", steps=set(capture_steps))
    dense_handles = _register_coverage_hooks(
        model=model,
        target_layers=target_layers,
        state=dense_state,
        row_steps=row_steps,
        prompt_rows=prompt_rows,
        role_masks=role_masks,
        policies_by_layer=policies_by_layer,
        selector_profiles=selector_profiles,
        native_cache=None,
        rows_out=rows,
        probs_out=probs,
    )
    try:
        _decode_capture_loop(
            model=model,
            past_key_values=dense_prefill.past_key_values,
            attention_mask=tokens["attention_mask"],
            first_token=first_token,
            fixed_input_tokens=dense_input_tokens,
            state=dense_state,
            max_step=max_step,
            native_routed_cache=False,
        )
    finally:
        for handle in dense_handles:
            handle.remove()

    routed_bundle = _policy_bundle_for_layers(routed_layers)
    route_prefill = _prefill(model, tokens)
    native_cache = _native_cache_from_hf_cache(
        route_prefill.past_key_values,
        routed_bundle,
        max_len=int(tokens["input_ids"].shape[1]) + max_step + 2,
        attach_hf_views=False,
    )
    route_state = CaptureState(condition="route_conditioned", steps=set(capture_steps))
    route_handles = _register_coverage_hooks(
        model=model,
        target_layers=target_layers,
        state=route_state,
        row_steps=row_steps,
        prompt_rows=prompt_rows,
        role_masks=role_masks,
        policies_by_layer=policies_by_layer,
        selector_profiles=selector_profiles,
        native_cache=native_cache,
        rows_out=rows,
        probs_out=probs,
    )
    try:
        with _native_cache_mask_bookkeeping(route_prefill.past_key_values, enabled=True):
            with _patched_seed_only_decode_modules(
                model,
                routed_bundle,
                native_cache=native_cache,
                native_cache_hf_sync_layers=set(),
                fused_rope_append_seed=True,
                packed_qkv_projection=True,
            ):
                _decode_capture_loop(
                    model=model,
                    past_key_values=route_prefill.past_key_values,
                    attention_mask=tokens["attention_mask"],
                    first_token=first_token,
                    fixed_input_tokens=dense_input_tokens,
                    state=route_state,
                    max_step=max_step,
                    native_routed_cache=True,
                )
    finally:
        for handle in route_handles:
            handle.remove()

    for row in rows:
        key = (int(row["step"]), int(row["layer"]), int(row["row"]), int(row["head"]))
        dense_p = probs.get(("dense_conditioned", *key))
        route_p = probs.get(("route_conditioned", *key))
        if dense_p is not None and route_p is not None:
            row["dense_vs_route_attention_js"] = _js_divergence(dense_p, route_p)
        else:
            row["dense_vs_route_attention_js"] = 0.0
        row["recommendation"] = _row_recommendation(row)

    result = {
        "schema": "streamattn.seed_policy_attention_coverage.v1",
        "model": args.model,
        "target_layers": target_layers,
        "routed_layers": routed_layers,
        "target_buckets": sorted(target_buckets),
        "selector_profiles": selector_profiles,
        "row_steps": {str(row): sorted(steps) for row, steps in row_steps.items()},
        "capture_steps": capture_steps,
        "condition_diagnostics": {
            "dense_conditioned": {
                "hook_calls": dense_state.hook_calls,
                "selected_calls": dense_state.selected_calls,
                "captured_heads": dense_state.captured_heads,
                "miss_reasons": dense_state.miss_reasons,
            },
            "route_conditioned": {
                "hook_calls": route_state.hook_calls,
                "selected_calls": route_state.selected_calls,
                "captured_heads": route_state.captured_heads,
                "miss_reasons": route_state.miss_reasons,
            },
        },
        "prompt_truncation_side": args.prompt_truncation_side,
        "rows": rows,
        "summary": _aggregate_rows(rows),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt-file", default="benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl")
    parser.add_argument("--prompt-truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--prompt-kinds", default="needle,code,long_doc,chat_doc")
    parser.add_argument("--prompt-repeat", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--target-layers", default="24,26,27")
    parser.add_argument("--routed-layers", default="0,14,16,24,26,27,35")
    parser.add_argument("--target-buckets", default="chat_instruction,noisy_neartie,json_tool,needle_rag")
    parser.add_argument(
        "--selector-profiles",
        default="fixed_policy",
        help=(
            "Comma-separated seed selector profiles to simulate: fixed_policy, "
            "support_block_oracle, qk_block_max, exact_mass_oracle, "
            "support_mass_oracle, value_residual_oracle."
        ),
    )
    parser.add_argument("--failure-artifact", default="")
    parser.add_argument("--include-step0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-rows", type=int, default=0)
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
