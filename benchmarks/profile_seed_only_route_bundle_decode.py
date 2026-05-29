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
import math
import sys
import time
import types
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set

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
from stream_attention.bucket_policy import qwen25_3b_bucket_route_decision  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_rope_append_packed_qkv_triton_forward_out_cachepos_bhnd,
    gate0_seed_only_attention_triton_forward_out_cachepos_bhnd,
    gate0_seed_only_rope_append_triton_forward_out_cachepos_bhnd,
)
from stream_attention.kernels.qwen_o_proj_triton import qwen_o_proj_triton_forward  # noqa: E402
from stream_attention.seed_selectors import (  # noqa: E402
    SELECTOR_PROFILES,
    select_seed_blocks_by_profile,
    seed_token_indices_from_blocks,
)

_SEED_OVERRIDE_ALIASES = {
    "block": "block_size",
    "block_size": "block_size",
    "sink": "sink_blocks",
    "sink_blocks": "sink_blocks",
    "recent": "recent_blocks",
    "recent_blocks": "recent_blocks",
    "middle": "middle_seed_blocks",
    "middle_seed_blocks": "middle_seed_blocks",
}


def parse_layer_seed_overrides(text: str) -> Dict[int, Dict[str, int]]:
    """Parse per-layer seed configs.

    Format examples:

        2:sink=2,recent=4,middle=10,block=32
        2:2,4,10
        2:32,2,4,10;18:sink=2,recent=6,middle=12
    """

    overrides: Dict[int, Dict[str, int]] = {}
    if not text.strip():
        return overrides
    for spec in text.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError(f"invalid layer seed override {spec!r}; expected LAYER:CONFIG")
        layer_text, config_text = spec.split(":", 1)
        layer_id = int(layer_text.strip())
        if layer_id < 0:
            raise ValueError("layer seed override ids must be non-negative")
        config_text = config_text.strip()
        if not config_text:
            raise ValueError(f"missing seed config for layer {layer_id}")
        config: Dict[str, int] = {}
        if "=" not in config_text:
            values = [int(part.strip()) for part in config_text.split(",") if part.strip()]
            if len(values) == 3:
                config = {
                    "sink_blocks": values[0],
                    "recent_blocks": values[1],
                    "middle_seed_blocks": values[2],
                }
            elif len(values) == 4:
                config = {
                    "block_size": values[0],
                    "sink_blocks": values[1],
                    "recent_blocks": values[2],
                    "middle_seed_blocks": values[3],
                }
            else:
                raise ValueError(
                    f"positional seed override for layer {layer_id} needs 3 or 4 integers"
                )
        else:
            for part in config_text.split(","):
                if not part.strip():
                    continue
                if "=" not in part:
                    raise ValueError(f"invalid seed override field {part!r}")
                key, value = part.split("=", 1)
                field = _SEED_OVERRIDE_ALIASES.get(key.strip())
                if field is None:
                    raise ValueError(f"unknown seed override field {key.strip()!r}")
                config[field] = int(value.strip())
        for field, value in config.items():
            if value <= 0:
                raise ValueError(f"{field} override for layer {layer_id} must be positive")
        overrides[layer_id] = config
    return overrides


def parse_layer_id_set(text: str) -> Set[int]:
    layers: Set[int] = set()
    if not text.strip():
        return layers
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        layer_id = int(part)
        if layer_id < 0:
            raise ValueError("layer ids must be non-negative")
        layers.add(layer_id)
    return layers


def _policy_seed_config(policy) -> Dict[str, Any]:
    return {
        "layer_id": int(policy.layer_id),
        "policy_id": policy.policy_id,
        "block_size": int(policy.block_size),
        "sink_blocks": int(policy.sink_blocks),
        "recent_blocks": int(policy.recent_blocks),
        "middle_seed_blocks": int(policy.middle_seed_blocks),
        "seed_tokens": int(policy.block_size)
        * int(policy.sink_blocks + policy.recent_blocks + policy.middle_seed_blocks),
        "block_order": policy.block_order,
        "num_warps": int(policy.num_warps),
        "num_stages": int(policy.num_stages),
    }


def _dynamic_seed_attention_reference_bhnd(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    seq_len: int,
    policy,
    selector_profile: str,
    out: torch.Tensor,
) -> torch.Tensor:
    """Reference selected-block seed attention for real decode replay.

    This is intentionally not a production kernel.  It lets the model-decode
    gate test whether dynamic block selection fixes stress safety before we
    spend time on a selected-block Triton/CUDA implementation.
    """

    batch, q_heads, query_len, head_dim = query_states.shape
    if query_len != 1:
        raise ValueError(f"dynamic selector reference only supports decode M=1, got {query_len}")
    kv_heads = int(key_cache.shape[1])
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")
    group_size = q_heads // kv_heads
    scale = 1.0 / math.sqrt(float(head_dim))
    seq_len = min(int(seq_len), int(key_cache.shape[2]))
    if seq_len <= 0:
        out.zero_()
        return out
    for b in range(int(batch)):
        for h in range(int(q_heads)):
            kv_head = h // group_size
            q_row = query_states[b, h, 0]
            k_row = key_cache[b, kv_head, :seq_len]
            v_row = value_cache[b, kv_head, :seq_len]
            selection = select_seed_blocks_by_profile(
                q=q_row,
                k=k_row,
                policy=policy,
                selector=selector_profile,
            )
            token_idx = seed_token_indices_from_blocks(
                seq_len=seq_len,
                block_size=int(policy.block_size),
                blocks=selection.selected_blocks,
                device=q_row.device,
            )
            if token_idx.numel() == 0:
                out[b, 0, h].zero_()
                continue
            k_sel = k_row.index_select(0, token_idx)
            v_sel = v_row.index_select(0, token_idx)
            scores = torch.matmul(k_sel.float(), q_row.float()) * scale
            probs = torch.softmax(scores, dim=0)
            acc = torch.matmul(probs, v_sel.float())
            out[b, 0, h].copy_(acc.to(out.dtype))
    return out


def apply_layer_seed_overrides(bundle, override_text: str):
    overrides = parse_layer_seed_overrides(override_text)
    if not overrides:
        return bundle, []
    policies_by_layer = {int(policy.layer_id): policy for policy in bundle.policies}
    unknown_layers = sorted(set(overrides) - set(policies_by_layer))
    if unknown_layers:
        raise ValueError(f"seed overrides target non-routed layers: {unknown_layers}")

    policies = []
    summaries = []
    for policy in bundle.policies:
        config = overrides.get(int(policy.layer_id))
        if not config:
            policies.append(policy)
            continue
        suffix = (
            f"s{config.get('block_size', policy.block_size)}_"
            f"{config.get('sink_blocks', policy.sink_blocks)}_"
            f"{config.get('recent_blocks', policy.recent_blocks)}_"
            f"{config.get('middle_seed_blocks', policy.middle_seed_blocks)}"
        )
        updated = replace(policy, **config, policy_id=f"{policy.policy_id}_{suffix}")
        policies.append(updated)
        summaries.append(
            {
                "layer_id": int(updated.layer_id),
                "old": _policy_seed_config(policy),
                "new": _policy_seed_config(updated),
            }
        )
    return type(bundle)(
        policy_names=[policy.policy_id for policy in policies],
        policies=policies,
        artifacts=bundle.artifacts,
        layer_ids=[int(policy.layer_id) for policy in policies],
    ), summaries


_PATCH_TIMING_STAGES = (
    ("qkv_ms", "start", "after_qkv"),
    ("rope_ms", "after_qkv", "after_rope"),
    ("cache_update_ms", "after_rope", "after_cache_update"),
    ("layout_ms", "after_cache_update", "after_layout"),
    ("seed_kernel_ms", "after_layout", "after_seed_kernel"),
    ("output_proj_ms", "after_seed_kernel", "end"),
)


class StreamAttnNativeKVCache:
    """Preallocated BHND K/V cache for routed seed-only decode layers."""

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        k: torch.Tensor,
        v: torch.Tensor,
        prefill_len: int,
        hf_cache: Any = None,
    ):
        if k.shape != v.shape:
            raise ValueError("native K/V cache tensors must have matching shapes")
        if k.dim() != 5:
            raise ValueError("native K/V cache must be [layers,batch,kv_heads,max_len,dim]")
        self.layer_ids = [int(layer_id) for layer_id in layer_ids]
        self.layer_to_index = {layer_id: idx for idx, layer_id in enumerate(self.layer_ids)}
        self.k = k
        self.v = v
        self.prefill_len = int(prefill_len)
        self.hf_cache = hf_cache

    @property
    def max_len(self) -> int:
        return int(self.k.shape[3])

    def append(self, layer_id: int, key_states: torch.Tensor, value_states: torch.Tensor, cache_position: Any):
        row = self.layer_to_index[int(layer_id)]
        if isinstance(cache_position, int):
            pos = cache_position
        else:
            pos = int(cache_position.reshape(-1)[0].item())
        if pos < 0 or pos >= self.max_len:
            raise ValueError(f"cache position {pos} exceeds native cache max_len {self.max_len}")
        self.k[row, :, :, pos : pos + 1, :].copy_(key_states)
        self.v[row, :, :, pos : pos + 1, :].copy_(value_states)
        self.sync_hf_view(layer_id, pos + 1)
        return self.k[row], self.v[row]

    def layer_cache(self, layer_id: int):
        row = self.layer_to_index[int(layer_id)]
        return self.k[row], self.v[row]

    def sync_hf_view(self, layer_id: int, length: int) -> None:
        if self.hf_cache is None:
            return
        layer = _cache_layer_object(self.hf_cache, int(layer_id))
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            row = self.layer_to_index[int(layer_id)]
            layer.keys = self.k[row, :, :, : int(length), :]
            layer.values = self.v[row, :, :, : int(length), :]

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "layer_ids": self.layer_ids,
            "shape": list(self.k.shape),
            "dtype": str(self.k.dtype).replace("torch.", ""),
            "device": str(self.k.device),
            "prefill_len": self.prefill_len,
            "max_len": self.max_len,
        }


def _cache_layer_object(cache: Any, layer_id: int):
    layers = getattr(cache, "layers", None)
    if layers is not None:
        try:
            return layers[int(layer_id)]
        except Exception as exc:
            raise ValueError(f"could not read HF cache layer {layer_id} from cache.layers") from exc
    try:
        return cache[int(layer_id)]
    except Exception as exc:
        raise ValueError(f"could not read HF cache layer {layer_id}") from exc


def _cache_layer_tensors(cache: Any, layer_id: int):
    layer = _cache_layer_object(cache, layer_id)
    if isinstance(layer, (tuple, list)) and len(layer) >= 2:
        return layer[0], layer[1]
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if keys is not None and values is not None:
        return keys, values
    raise ValueError(f"unsupported HF cache layer object for layer {layer_id}: {type(layer).__name__}")


def _native_cache_from_hf_cache(
    cache: Any,
    bundle,
    *,
    max_len: int,
    attach_hf_views: bool = True,
) -> StreamAttnNativeKVCache:
    if not bundle.policies:
        raise ValueError("route bundle is empty")
    first_k, first_v = _cache_layer_tensors(cache, int(bundle.policies[0].layer_id))
    if first_k.dim() != 4:
        raise ValueError("expected HF cache tensors shaped [batch,kv_heads,seq,dim]")
    layer_count = len(bundle.policies)
    batch, kv_heads, prefill_len, dim = first_k.shape
    if max_len < prefill_len:
        raise ValueError(f"native cache max_len {max_len} is smaller than prefill_len {prefill_len}")
    k_native = torch.empty(
        (layer_count, batch, kv_heads, max_len, dim),
        device=first_k.device,
        dtype=first_k.dtype,
    )
    v_native = torch.empty_like(k_native)
    for row, policy in enumerate(bundle.policies):
        k_layer, v_layer = _cache_layer_tensors(cache, int(policy.layer_id))
        if k_layer.shape != first_k.shape or v_layer.shape != first_v.shape:
            raise ValueError(f"HF cache shape mismatch for routed layer {policy.layer_id}")
        k_native[row, :, :, :prefill_len, :].copy_(k_layer)
        v_native[row, :, :, :prefill_len, :].copy_(v_layer)
    native = StreamAttnNativeKVCache(
        layer_ids=[int(policy.layer_id) for policy in bundle.policies],
        k=k_native,
        v=v_native,
        prefill_len=int(prefill_len),
        hf_cache=cache if attach_hf_views else None,
    )
    if attach_hf_views:
        for policy in bundle.policies:
            native.sync_hf_view(int(policy.layer_id), int(prefill_len))
    return native


@contextmanager
def _native_cache_mask_bookkeeping(cache: Any, *, enabled: bool):
    """Override HF mask-size bookkeeping while routed layers own native K/V.

    Qwen's causal-mask helper asks the cache for a global KV length before layer
    forwards run. If routed layer 0 bypasses HF DynamicCache.update, that length
    becomes stale. Some HF helpers also ask the cache/layer objects for the
    already-seen length directly, so patch both the top-level cache methods and
    per-layer methods from CPU-side decode-loop lengths instead of rebinding HF
    cache tensor views or paying a layer-0 HF append.
    """

    if not enabled or cache is None:
        yield
        return

    original_get_mask_sizes = getattr(cache, "get_mask_sizes", None)
    original_get_seq_length = getattr(cache, "get_seq_length", None)
    if original_get_mask_sizes is None and original_get_seq_length is None:
        yield
        return

    def get_mask_sizes(self, cache_position, layer_idx):
        kv_length = getattr(self, "_streamattn_mask_kv_length", None)
        if kv_length is not None:
            return int(kv_length), 0
        if original_get_mask_sizes is None:
            return 0, 0
        return original_get_mask_sizes(cache_position, layer_idx)

    def get_seq_length(self, layer_idx: int = 0, cache_position=None):
        past_length = getattr(self, "_streamattn_past_kv_length", None)
        if past_length is not None:
            return int(past_length)
        if original_get_seq_length is None:
            return 0
        return original_get_seq_length(layer_idx, cache_position)

    had_length = hasattr(cache, "_streamattn_mask_kv_length")
    previous_length = getattr(cache, "_streamattn_mask_kv_length", None)
    had_past_length = hasattr(cache, "_streamattn_past_kv_length")
    previous_past_length = getattr(cache, "_streamattn_past_kv_length", None)
    if original_get_mask_sizes is not None:
        cache.get_mask_sizes = types.MethodType(get_mask_sizes, cache)
    if original_get_seq_length is not None:
        cache.get_seq_length = types.MethodType(get_seq_length, cache)
    cache._streamattn_mask_kv_length = None
    cache._streamattn_past_kv_length = None
    layer_originals = []
    for layer in getattr(cache, "layers", []) or []:
        original_layer_get_mask_sizes = getattr(layer, "get_mask_sizes", None)
        original_layer_get_seq_length = getattr(layer, "get_seq_length", None)
        if original_layer_get_mask_sizes is None and original_layer_get_seq_length is None:
            continue

        def layer_get_mask_sizes(layer_self, cache_position, _original=original_layer_get_mask_sizes):
            kv_length = getattr(cache, "_streamattn_mask_kv_length", None)
            if kv_length is not None:
                return int(kv_length), 0
            if _original is None:
                return 0, 0
            return _original(cache_position)

        def layer_get_seq_length(layer_self, cache_position=None, _original=original_layer_get_seq_length):
            past_length = getattr(cache, "_streamattn_past_kv_length", None)
            if past_length is not None:
                return int(past_length)
            if _original is None:
                return 0
            return _original(cache_position)

        layer_originals.append((layer, original_layer_get_mask_sizes, original_layer_get_seq_length))
        if original_layer_get_mask_sizes is not None:
            layer.get_mask_sizes = types.MethodType(layer_get_mask_sizes, layer)
        if original_layer_get_seq_length is not None:
            layer.get_seq_length = types.MethodType(layer_get_seq_length, layer)
    try:
        yield
    finally:
        if original_get_mask_sizes is not None:
            cache.get_mask_sizes = original_get_mask_sizes
        if original_get_seq_length is not None:
            cache.get_seq_length = original_get_seq_length
        for layer, original_layer_get_mask_sizes, original_layer_get_seq_length in layer_originals:
            if original_layer_get_mask_sizes is not None:
                layer.get_mask_sizes = original_layer_get_mask_sizes
            if original_layer_get_seq_length is not None:
                layer.get_seq_length = original_layer_get_seq_length
        if had_length:
            cache._streamattn_mask_kv_length = previous_length
        else:
            try:
                delattr(cache, "_streamattn_mask_kv_length")
            except AttributeError:
                pass
        if had_past_length:
            cache._streamattn_past_kv_length = previous_past_length
        else:
            try:
                delattr(cache, "_streamattn_past_kv_length")
            except AttributeError:
                pass


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarize_patch_timing_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    if not rows:
        return {"call_count": 0, "stages": {}, "total_ms": {}}
    keys = [stage[0] for stage in _PATCH_TIMING_STAGES] + ["total_ms"]
    summary: Dict[str, Any] = {"call_count": len(rows), "stages": {}, "total_ms": {}}
    total_sum = sum(float(row.get("total_ms", 0.0)) for row in rows)
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        item = {
            "sum_ms": sum(values),
            "mean_ms": sum(values) / max(1, len(values)),
            "p50_ms": _percentile(values, 0.50),
            "p90_ms": _percentile(values, 0.90),
        }
        if key != "total_ms":
            item["share_of_patch_total"] = item["sum_ms"] / max(total_sum, 1.0e-12)
            summary["stages"][key] = item
        else:
            summary["total_ms"] = item
    return summary


def _import_qwen_rotary():
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except Exception as exc:  # pragma: no cover - optional dependency/version
        raise RuntimeError("Qwen route bundle decode requires transformers Qwen2 helpers") from exc
    return apply_rotary_pos_emb


def _batch_tokens(
    tokenizer,
    prompts: Sequence[Dict[str, str]],
    *,
    max_seq: int,
    device: torch.device,
    truncation_side: str = "",
):
    old_truncation_side = getattr(tokenizer, "truncation_side", None)
    if truncation_side:
        tokenizer.truncation_side = truncation_side
    try:
        encoded = tokenizer(
            [row["prompt"] for row in prompts],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq,
            padding="max_length",
        ).to(device)
    finally:
        if truncation_side and old_truncation_side is not None:
            tokenizer.truncation_side = old_truncation_side
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
        "kl_p95": _percentile(kl_values, 0.95),
        "kl_p99": _percentile(kl_values, 0.99),
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


def _summarize_logit_steps_by_field(
    step_rows: Sequence[Dict[str, Any]],
    *,
    field: str,
) -> Dict[str, Any]:
    values = sorted(
        {
            str(row.get(field, ""))
            for step in step_rows
            for row in step["rows"]
            if row.get(field, "")
        }
    )
    grouped: Dict[str, Any] = {}
    for value in values:
        filtered_steps = []
        row_ids = set()
        for step in step_rows:
            rows = [row for row in step["rows"] if str(row.get(field, "")) == value]
            if rows:
                filtered_steps.append({**step, "rows": rows})
                row_ids.update(int(row["row"]) for row in rows)
        grouped[value] = _summarize_logit_steps(filtered_steps, batch_size=len(row_ids))
    return grouped


class _SeedOnlyQwenDecodePatch:
    def __init__(
        self,
        *,
        policy,
        original_forward,
        profile_timing: bool = False,
        native_cache: Optional[StreamAttnNativeKVCache] = None,
        native_cache_hf_sync_layers: Optional[Set[int]] = None,
        fused_rope_append_seed: bool = False,
        packed_qkv_projection: bool = False,
        packed_qkv_fused_input: bool = False,
        direct_o_proj: bool = False,
        triton_o_proj: bool = False,
        dynamic_selector_layers: Optional[Set[int]] = None,
        dynamic_selector_profile: str = "",
    ):
        self.policy = policy
        self.original_forward = original_forward
        self.profile_timing = profile_timing
        self.native_cache = native_cache
        self.native_cache_hf_sync_layers = native_cache_hf_sync_layers or set()
        self.fused_rope_append_seed = fused_rope_append_seed
        self.packed_qkv_projection = packed_qkv_projection
        self.packed_qkv_fused_input = packed_qkv_fused_input
        self.direct_o_proj = direct_o_proj
        self.triton_o_proj = triton_o_proj
        self.dynamic_selector_layers = dynamic_selector_layers or set()
        self.dynamic_selector_profile = dynamic_selector_profile
        self.call_count = 0
        self.seed_call_count = 0
        self.dynamic_selector_call_count = 0
        self.native_cache_update_count = 0
        self.hf_sync_update_count = 0
        self._native_next_pos = native_cache.prefill_len if native_cache is not None else 0
        self.fallback_reasons: Counter[str] = Counter()
        self.fallback_samples: List[Dict[str, Any]] = []
        self._timing_event_rows: List[Dict[str, torch.cuda.Event]] = []
        self._out_buffer: Optional[torch.Tensor] = None
        self._packed_qkv_weight: Optional[torch.Tensor] = None
        self._packed_qkv_bias: Optional[torch.Tensor] = None
        self._packed_qkv_sizes: Optional[tuple[int, int, int]] = None
        self._packed_qkv_module_id: Optional[int] = None

    @staticmethod
    def _linear_bias(linear: torch.nn.Module) -> Optional[torch.Tensor]:
        bias = getattr(linear, "bias", None)
        return bias if isinstance(bias, torch.Tensor) else None

    def prepare_packed_qkv(self, module: torch.nn.Module) -> None:
        """Prepack q/k/v projection weights for the routed decode loop."""

        weights = [module.q_proj.weight, module.k_proj.weight, module.v_proj.weight]
        weight = torch.cat(weights, dim=0).contiguous()
        biases = [
            self._linear_bias(module.q_proj),
            self._linear_bias(module.k_proj),
            self._linear_bias(module.v_proj),
        ]
        if any(bias is not None for bias in biases):
            packed_biases = [
                bias
                if bias is not None
                else torch.zeros(linear.out_features, device=weight.device, dtype=weight.dtype)
                for bias, linear in zip(biases, (module.q_proj, module.k_proj, module.v_proj))
            ]
            packed_bias = torch.cat(packed_biases, dim=0).contiguous()
        else:
            packed_bias = None
        self._packed_qkv_weight = weight
        self._packed_qkv_bias = packed_bias
        self._packed_qkv_sizes = (
            int(module.q_proj.out_features),
            int(module.k_proj.out_features),
            int(module.v_proj.out_features),
        )
        self._packed_qkv_module_id = id(module)

    def _packed_qkv_linear(self, module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._packed_qkv_weight is None or self._packed_qkv_module_id != id(module):
            self.prepare_packed_qkv(module)
        assert self._packed_qkv_weight is not None
        return F.linear(hidden_states, self._packed_qkv_weight, self._packed_qkv_bias)

    def _o_projection(self, module: torch.nn.Module, attn_output: torch.Tensor) -> torch.Tensor:
        if self.triton_o_proj:
            return qwen_o_proj_triton_forward(
                attn_output,
                module.o_proj.weight,
                self._linear_bias(module.o_proj),
            )
        if self.direct_o_proj:
            return F.linear(attn_output, module.o_proj.weight, self._linear_bias(module.o_proj))
        return module.o_proj(attn_output)

    def _qkv_projection(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        hidden_shape: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.packed_qkv_projection:
            return (
                module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2),
                module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2),
                module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2),
            )
        packed = self._packed_qkv_linear(module, hidden_states)
        assert self._packed_qkv_sizes is not None
        q, k, v = torch.split(packed, self._packed_qkv_sizes, dim=-1)
        return (
            q.view(hidden_shape).transpose(1, 2),
            k.view(hidden_shape).transpose(1, 2),
            v.view(hidden_shape).transpose(1, 2),
        )

    def _mark(self, events: Optional[Dict[str, torch.cuda.Event]], name: str) -> None:
        if events is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events[name] = event

    def _output_buffer(
        self,
        *,
        batch: int,
        q_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = (int(batch), 1, int(q_heads), int(head_dim))
        if (
            self._out_buffer is None
            or tuple(self._out_buffer.shape) != shape
            or self._out_buffer.device != device
            or self._out_buffer.dtype != dtype
        ):
            self._out_buffer = torch.empty(shape, device=device, dtype=dtype)
        return self._out_buffer

    def timing_rows(self) -> List[Dict[str, float]]:
        rows = []
        for events in self._timing_event_rows:
            row = {}
            for key, start_name, end_name in _PATCH_TIMING_STAGES:
                row[key] = float(events[start_name].elapsed_time(events[end_name]))
            row["total_ms"] = float(events["start"].elapsed_time(events["end"]))
            rows.append(row)
        return rows

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
        layer_id = int(module.layer_idx)
        use_dynamic_selector = (
            bool(self.dynamic_selector_profile)
            and layer_id in self.dynamic_selector_layers
        )
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
        elif use_dynamic_selector and self.native_cache is None:
            fallback_reason = "dynamic_selector_requires_native_cache"
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

        timing_events: Optional[Dict[str, torch.cuda.Event]] = None
        if self.profile_timing and hidden_states.is_cuda and torch.cuda.is_available():
            timing_events = {}
            self._mark(timing_events, "start")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        cos, sin = position_embeddings
        use_fused_rope_append_seed = (
            self.fused_rope_append_seed
            and self.native_cache is not None
            and layer_id not in self.native_cache_hf_sync_layers
            and not use_dynamic_selector
        )
        use_packed_qkv_fused_input = (
            use_fused_rope_append_seed
            and self.packed_qkv_projection
            and self.packed_qkv_fused_input
        )
        if use_packed_qkv_fused_input:
            packed_qkv = self._packed_qkv_linear(module, hidden_states)
            self._mark(timing_events, "after_qkv")
            self._mark(timing_events, "after_rope")
            native_pos = self._native_next_pos
            key_cache, value_cache = self.native_cache.layer_cache(layer_id)
            assert self._packed_qkv_sizes is not None
            q_heads = int(self._packed_qkv_sizes[0] // module.head_dim)
            out = self._output_buffer(
                batch=hidden_states.shape[0],
                q_heads=q_heads,
                head_dim=module.head_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            self._mark(timing_events, "after_cache_update")
            self._mark(timing_events, "after_layout")
            gate0_seed_only_rope_append_packed_qkv_triton_forward_out_cachepos_bhnd(
                packed_qkv,
                cos,
                sin,
                key_cache,
                value_cache,
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
            self.native_cache.sync_hf_view(layer_id, native_pos + 1)
            self._native_next_pos += 1
            self.native_cache_update_count += 1
            self._mark(timing_events, "after_seed_kernel")
        else:
            apply_rotary_pos_emb = _import_qwen_rotary()
            query_states, key_states, value_states = self._qkv_projection(
                module,
                hidden_states,
                hidden_shape,
            )
            self._mark(timing_events, "after_qkv")
            if not use_fused_rope_append_seed:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            self._mark(timing_events, "after_rope")
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if use_fused_rope_append_seed:
                native_pos = self._native_next_pos
                key_cache, value_cache = self.native_cache.layer_cache(layer_id)
                q = query_states
                out = self._output_buffer(
                    batch=hidden_states.shape[0],
                    q_heads=query_states.shape[1],
                    head_dim=module.head_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                self._mark(timing_events, "after_cache_update")
                self._mark(timing_events, "after_layout")
                gate0_seed_only_rope_append_triton_forward_out_cachepos_bhnd(
                    q,
                    key_states,
                    value_states,
                    cos,
                    sin,
                    key_cache,
                    value_cache,
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
                self.native_cache.sync_hf_view(layer_id, native_pos + 1)
                self._native_next_pos += 1
                self.native_cache_update_count += 1
                self._mark(timing_events, "after_seed_kernel")
            elif self.native_cache is None:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    module.layer_idx,
                    cache_kwargs,
                )
            else:
                if layer_id in self.native_cache_hf_sync_layers:
                    past_key_value.update(
                        key_states,
                        value_states,
                        layer_id,
                        cache_kwargs,
                    )
                    self.hf_sync_update_count += 1
                native_pos = self._native_next_pos
                key_states, value_states = self.native_cache.append(
                    layer_id,
                    key_states,
                    value_states,
                    native_pos,
                )
                self._native_next_pos += 1
                self.native_cache_update_count += 1
                self._mark(timing_events, "after_cache_update")

                k = key_states
                v = value_states
                out = self._output_buffer(
                    batch=hidden_states.shape[0],
                    q_heads=query_states.shape[1],
                    head_dim=module.head_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                self._mark(timing_events, "after_layout")
                if use_dynamic_selector:
                    _dynamic_seed_attention_reference_bhnd(
                        query_states,
                        k,
                        v,
                        seq_len=native_pos + 1,
                        policy=self.policy,
                        selector_profile=self.dynamic_selector_profile,
                        out=out,
                    )
                    self.dynamic_selector_call_count += 1
                else:
                    q = query_states.transpose(1, 2).contiguous()
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
                self._mark(timing_events, "after_seed_kernel")
            if not use_fused_rope_append_seed and self.native_cache is None:
                self._mark(timing_events, "after_cache_update")

                q = query_states.transpose(1, 2).contiguous()
                k = key_states
                v = value_states
                out = self._output_buffer(
                    batch=hidden_states.shape[0],
                    q_heads=query_states.shape[1],
                    head_dim=module.head_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                self._mark(timing_events, "after_layout")
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
                self._mark(timing_events, "after_seed_kernel")
        attn_output = out.reshape(*input_shape, -1).contiguous()
        attn_output = self._o_projection(module, attn_output)
        self._mark(timing_events, "end")
        if timing_events is not None:
            self._timing_event_rows.append(timing_events)
        self.seed_call_count += 1
        return attn_output, None


class StreamAttnQwenAttentionModule(torch.nn.Module):
    """Qwen attention replacement that routes decode through StreamAttn."""

    def __init__(self, original_module: torch.nn.Module, patch: _SeedOnlyQwenDecodePatch):
        super().__init__()
        self.original_module = original_module
        self.patch = patch
        self.layer_idx = getattr(original_module, "layer_idx", None)
        self.attention_type = getattr(original_module, "attention_type", "full_attention")
        self.head_dim = getattr(original_module, "head_dim", None)
        for attr in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "num_heads",
            "num_key_value_heads",
            "num_key_value_groups",
            "scaling",
            "is_causal",
            "config",
        ):
            if hasattr(original_module, attr):
                setattr(self, attr, getattr(original_module, attr))
        if getattr(patch, "packed_qkv_projection", False) and all(
            hasattr(self, attr) for attr in ("q_proj", "k_proj", "v_proj")
        ):
            patch.prepare_packed_qkv(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        return self.patch.forward(
            self,
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )


def _parent_module_and_attr(root: torch.nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


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
    *,
    profile_timing: bool = False,
    native_cache: Optional[StreamAttnNativeKVCache] = None,
    native_cache_hf_sync_layers: Optional[Set[int]] = None,
    native_attention_module: bool = False,
    fused_rope_append_seed: bool = False,
    packed_qkv_projection: bool = False,
    packed_qkv_fused_input: bool = False,
    direct_o_proj: bool = False,
    triton_o_proj: bool = False,
    dynamic_selector_layers: Optional[Set[int]] = None,
    dynamic_selector_profile: str = "",
) -> Iterator[Dict[str, _SeedOnlyQwenDecodePatch]]:
    modules_by_layer = {
        int(layer_id): (name, module) for layer_id, name, module in _attention_modules(model)
    }
    patches: Dict[str, _SeedOnlyQwenDecodePatch] = {}
    originals = []
    effective_packed_qkv_projection = bool(packed_qkv_projection or native_attention_module)
    for policy in bundle.policies:
        item = modules_by_layer.get(int(policy.layer_id))
        if item is None:
            raise ValueError(f"model is missing layer {policy.layer_id}")
        name, module = item
        patch = _SeedOnlyQwenDecodePatch(
            policy=policy,
            original_forward=module.forward,
            profile_timing=profile_timing,
            native_cache=native_cache,
            native_cache_hf_sync_layers=native_cache_hf_sync_layers,
            fused_rope_append_seed=fused_rope_append_seed,
            packed_qkv_projection=effective_packed_qkv_projection,
            packed_qkv_fused_input=packed_qkv_fused_input,
            direct_o_proj=direct_o_proj,
            triton_o_proj=triton_o_proj,
            dynamic_selector_layers=dynamic_selector_layers,
            dynamic_selector_profile=dynamic_selector_profile,
        )
        if native_attention_module:
            parent, attr = _parent_module_and_attr(model, name)
            originals.append((parent, attr, module))
            setattr(parent, attr, StreamAttnQwenAttentionModule(module, patch))
        else:
            if effective_packed_qkv_projection:
                patch.prepare_packed_qkv(module)
            originals.append((module, "forward", module.forward))
            module.forward = types.MethodType(patch.forward, module)
        patches[str(policy.layer_id)] = patch
    try:
        yield patches
    finally:
        for owner, attr, original in originals:
            setattr(owner, attr, original)


def _prefill(model, tokens: Dict[str, torch.Tensor]):
    with torch.inference_mode():
        return model(**tokens, use_cache=True, logits_to_keep=1)


def _args_with_steps(args: argparse.Namespace, steps: int) -> argparse.Namespace:
    clone = argparse.Namespace(**vars(args))
    clone.native_cache_capacity_steps = max(int(args.steps), int(steps))
    clone.steps = int(steps)
    return clone


def _native_cache_max_len(tokens: Dict[str, torch.Tensor], args: argparse.Namespace) -> int:
    capacity_steps = int(getattr(args, "native_cache_capacity_steps", args.steps))
    return int(tokens["input_ids"].shape[1]) + max(int(args.steps), capacity_steps) + 1


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
            model_kwargs = {
                "input_ids": input_token,
                "attention_mask": _append_decode_attention_mask(mask, input_token),
                "past_key_values": past_key_values,
                "use_cache": True,
                "logits_to_keep": 1,
            }
            if args.explicit_cache_position or args.native_routed_cache:
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
    native_cache = None
    if args.native_routed_cache:
        native_cache = _native_cache_from_hf_cache(
            prefill.past_key_values,
            bundle,
            max_len=_native_cache_max_len(tokens, warmup_args),
            attach_hf_views=args.native_cache_attach_hf_views,
        )
    with _native_cache_mask_bookkeeping(prefill.past_key_values, enabled=args.native_routed_cache):
        with _patched_seed_only_decode_modules(
            model,
            bundle,
            native_cache=native_cache,
            native_cache_hf_sync_layers=parse_layer_id_set(args.native_cache_hf_sync_layers),
            native_attention_module=args.native_attention_module,
            fused_rope_append_seed=args.fused_rope_append_seed,
            packed_qkv_projection=args.packed_qkv_projection,
            packed_qkv_fused_input=args.packed_qkv_fused_input,
            direct_o_proj=args.direct_o_proj,
            triton_o_proj=args.triton_o_proj,
            dynamic_selector_layers=parse_layer_id_set(getattr(args, "dynamic_selector_layers", "")),
            dynamic_selector_profile=getattr(args, "dynamic_selector_profile", ""),
        ) as patches:
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
            "dynamic_selector_calls": patch.dynamic_selector_call_count,
            "dynamic_selector_profile": patch.dynamic_selector_profile,
            "native_cache_update_calls": patch.native_cache_update_count,
            "hf_sync_update_calls": patch.hf_sync_update_count,
            "packed_qkv_projection": bool(patch.packed_qkv_projection),
            "packed_qkv_fused_input": bool(patch.packed_qkv_fused_input),
            "direct_o_proj": bool(patch.direct_o_proj),
            "triton_o_proj": bool(patch.triton_o_proj),
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
                        "prompt_id": prompt_rows[int(row["row"])].get("id", ""),
                        "prompt_bucket": prompt_rows[int(row["row"])].get(
                            "bucket",
                            prompt_rows[int(row["row"])]["kind"],
                        ),
                        "sample_token_changed": bool(changed[int(row["row"])].item()),
                    }
                    for row in rows
                ],
            }
        )
    summary = _summarize_logit_steps(steps, batch_size=len(prompt_rows))
    summary["by_prompt_bucket"] = _summarize_logit_steps_by_field(steps, field="prompt_bucket")
    return {
        "summary": summary,
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


def _patch_timing_summary(
    patches: Dict[str, _SeedOnlyQwenDecodePatch],
    *,
    decode_steps: int,
) -> Optional[Dict[str, Any]]:
    if not patches or not any(patch._timing_event_rows for patch in patches.values()):
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    per_layer = {}
    bundle_stage_sums: Dict[str, float] = {stage[0]: 0.0 for stage in _PATCH_TIMING_STAGES}
    bundle_total_ms = 0.0
    for layer_id, patch in patches.items():
        rows = patch.timing_rows()
        summary = summarize_patch_timing_rows(rows)
        per_layer[layer_id] = summary
        bundle_total_ms += float(summary.get("total_ms", {}).get("sum_ms", 0.0))
        for stage_name in bundle_stage_sums:
            bundle_stage_sums[stage_name] += float(
                summary.get("stages", {}).get(stage_name, {}).get("sum_ms", 0.0)
            )
    return {
        "note": "CUDA-event diagnostic timing; enabling this changes the measured decode path.",
        "decode_steps": int(decode_steps),
        "routed_layer_count": len(per_layer),
        "bundle_total_patch_ms": bundle_total_ms,
        "bundle_patch_ms_per_decode_step": bundle_total_ms / max(1, int(decode_steps)),
        "bundle_stage_sums_ms": bundle_stage_sums,
        "bundle_stage_share_of_patch_total": {
            stage_name: value / max(bundle_total_ms, 1.0e-12)
            for stage_name, value in bundle_stage_sums.items()
        },
        "per_layer": per_layer,
    }


def _prompt_bucket(row: Dict[str, str]) -> str:
    return str(row.get("bucket") or row.get("kind") or "default")


def _bucket_route_policy_decision(
    prompt_rows: Sequence[Dict[str, str]],
    *,
    policy_name: str,
    product_strict: bool,
) -> Dict[str, Any]:
    if policy_name in {"", "none"}:
        return {"enabled": False, "batch_mode": "manual_route"}
    if policy_name != "qwen25_3b_b8":
        raise ValueError(f"unsupported bucket route policy {policy_name!r}")

    prompt_decisions = []
    policy_sets = set()
    exact_reasons = []
    for idx, row in enumerate(prompt_rows):
        bucket = _prompt_bucket(row)
        decision = qwen25_3b_bucket_route_decision(bucket, product_strict=product_strict)
        prompt_decisions.append(
            {
                "row": idx,
                "bucket": bucket,
                "mode": decision.mode,
                "seed_only_layers": list(decision.seed_only_layers),
                "policy_names": list(decision.policy_names),
                "status": decision.status,
                "reason": decision.reason,
            }
        )
        if decision.mode == "exact_native":
            exact_reasons.append(f"{bucket}:{decision.reason}")
        else:
            policy_sets.add(decision.policy_names)

    if exact_reasons:
        return {
            "enabled": True,
            "policy": policy_name,
            "product_strict": bool(product_strict),
            "batch_mode": "exact_native",
            "fallback_reason": "batch_contains_exact_bucket",
            "fallback_details": exact_reasons,
            "prompt_decisions": prompt_decisions,
            "policy_names": [],
            "seed_only_layers": [],
        }
    if len(policy_sets) != 1:
        return {
            "enabled": True,
            "policy": policy_name,
            "product_strict": bool(product_strict),
            "batch_mode": "exact_native",
            "fallback_reason": "mixed_bucket_route_decisions",
            "fallback_details": [sorted(list(item)) for item in policy_sets],
            "prompt_decisions": prompt_decisions,
            "policy_names": [],
            "seed_only_layers": [],
        }
    policy_names = tuple(next(iter(policy_sets)))
    seed_layers = []
    for decision in prompt_decisions:
        if decision["policy_names"] == list(policy_names):
            seed_layers = list(decision["seed_only_layers"])
            break
    return {
        "enabled": True,
        "policy": policy_name,
        "product_strict": bool(product_strict),
        "batch_mode": "seed_only_bundle",
        "fallback_reason": None,
        "fallback_details": [],
        "prompt_decisions": prompt_decisions,
        "policy_names": list(policy_names),
        "seed_only_layers": seed_layers,
    }


def _empty_route_bundle_summary(
    *,
    layer_seed_overrides: Sequence[Dict[str, Any]],
    native_cache_hf_sync_layers: Set[int],
    args: argparse.Namespace,
    bucket_policy: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "policy_names": [],
        "policy_ids": [],
        "layers": [],
        "seed_configs": [],
        "layer_seed_overrides": list(layer_seed_overrides),
        "native_cache_hf_sync_layers": sorted(native_cache_hf_sync_layers),
        "native_cache_attach_hf_views": bool(args.native_cache_attach_hf_views),
        "native_attention_module": bool(args.native_attention_module),
        "fused_rope_append_seed": False,
        "packed_qkv_projection": False,
        "packed_qkv_fused_input": False,
        "direct_o_proj": False,
        "triton_o_proj": False,
        "dynamic_selector_layers": sorted(parse_layer_id_set(args.dynamic_selector_layers)),
        "dynamic_selector_profile": args.dynamic_selector_profile,
        "dynamic_selector_reference_only": bool(args.dynamic_selector_profile),
        "allow_mixed_seed_configs": bool(args.allow_mixed_seed_configs),
        "native_routed_cache": {"enabled": False},
        "candidate_backend": "exact_native",
        "fallback_reason": bucket_policy.get("fallback_reason"),
        "bucket_route_policy": bucket_policy,
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dynamic_selector_layers = parse_layer_id_set(args.dynamic_selector_layers)
    if dynamic_selector_layers and not args.dynamic_selector_profile:
        raise ValueError("--dynamic-selector-layers requires --dynamic-selector-profile")
    if args.dynamic_selector_profile and not dynamic_selector_layers:
        raise ValueError("--dynamic-selector-profile requires --dynamic-selector-layers")
    if args.dynamic_selector_profile and args.dynamic_selector_profile not in SELECTOR_PROFILES:
        raise ValueError(
            f"unknown dynamic selector profile {args.dynamic_selector_profile!r}; "
            f"supported={sorted(SELECTOR_PROFILES)}"
        )
    if args.dynamic_selector_profile and not args.native_routed_cache:
        raise ValueError("--dynamic-selector-profile requires --native-routed-cache")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    native_cache_hf_sync_layers = parse_layer_id_set(args.native_cache_hf_sync_layers)
    prompt_rows = _prompts_from_args(args)
    bucket_policy = _bucket_route_policy_decision(
        prompt_rows,
        policy_name=args.bucket_route_policy,
        product_strict=args.product_strict,
    )
    exact_bucket_fallback = bucket_policy.get("batch_mode") == "exact_native"
    bundle = None
    layer_seed_overrides: Sequence[Dict[str, Any]] = []
    if not exact_bucket_fallback:
        if bucket_policy.get("batch_mode") == "seed_only_bundle":
            args.policy_names = ",".join(bucket_policy["policy_names"])
            args.layers = ""
            args.use_packaged_policies = True
        bundle = _route_bundle_from_args(args)
        bundle, layer_seed_overrides = apply_layer_seed_overrides(bundle, args.layer_seed_overrides)

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
    prompt_truncation_side = args.prompt_truncation_side if args.prompt_file else ""
    tokens = _batch_tokens(
        tokenizer,
        prompt_rows,
        max_seq=args.max_seq,
        device=device,
        truncation_side=prompt_truncation_side,
    )
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

        print("[route-bundle-decode] warmup candidate decode", flush=True)
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
    native_cache = None
    patches = {}
    patch_timing = None
    if args.native_routed_cache and bundle is not None:
        native_cache = _native_cache_from_hf_cache(
            seed_prefill.past_key_values,
            bundle,
            max_len=_native_cache_max_len(tokens, args),
            attach_hf_views=args.native_cache_attach_hf_views,
        )
    if exact_bucket_fallback:
        print("[route-bundle-decode] timing exact-native bucket fallback decode", flush=True)
        seed = _decode_loop(
            model=model,
            past_key_values=seed_prefill.past_key_values,
            attention_mask=tokens["attention_mask"],
            first_token=first_token,
            fixed_input_tokens=dense["input_tokens"],
            prompt_rows=prompt_rows,
            args=args,
        )
    else:
        print("[route-bundle-decode] timing StreamAttn seed-only bundle decode", flush=True)
        with _native_cache_mask_bookkeeping(seed_prefill.past_key_values, enabled=args.native_routed_cache):
            with _patched_seed_only_decode_modules(
                model,
                bundle,
                profile_timing=args.profile_patch_timing,
                native_cache=native_cache,
                native_cache_hf_sync_layers=native_cache_hf_sync_layers,
                native_attention_module=args.native_attention_module,
                fused_rope_append_seed=args.fused_rope_append_seed,
                packed_qkv_projection=args.packed_qkv_projection,
                packed_qkv_fused_input=args.packed_qkv_fused_input,
                direct_o_proj=args.direct_o_proj,
                triton_o_proj=args.triton_o_proj,
                dynamic_selector_layers=parse_layer_id_set(args.dynamic_selector_layers),
                dynamic_selector_profile=args.dynamic_selector_profile,
            ) as patches:
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
            "dynamic_selector_calls": patch.dynamic_selector_call_count,
            "dynamic_selector_profile": patch.dynamic_selector_profile,
            "native_cache_update_calls": patch.native_cache_update_count,
            "hf_sync_update_calls": patch.hf_sync_update_count,
            "packed_qkv_projection": bool(patch.packed_qkv_projection),
            "packed_qkv_fused_input": bool(patch.packed_qkv_fused_input),
            "direct_o_proj": bool(patch.direct_o_proj),
            "triton_o_proj": bool(patch.triton_o_proj),
            "fallback_reasons": dict(patch.fallback_reasons),
            "fallback_samples": patch.fallback_samples,
        }
        for layer_id, patch in patches.items()
    }
    if not exact_bucket_fallback:
        patch_timing = _patch_timing_summary(patches, decode_steps=args.steps)
    if bundle is None:
        route_bundle_summary = _empty_route_bundle_summary(
            layer_seed_overrides=layer_seed_overrides,
            native_cache_hf_sync_layers=native_cache_hf_sync_layers,
            args=args,
            bucket_policy=bucket_policy,
        )
    else:
        route_bundle_summary = {
            "policy_names": bundle.policy_names,
            "policy_ids": [policy.policy_id for policy in bundle.policies],
            "layers": bundle.layer_ids,
            "seed_configs": [_policy_seed_config(policy) for policy in bundle.policies],
            "layer_seed_overrides": layer_seed_overrides,
            "native_cache_hf_sync_layers": sorted(native_cache_hf_sync_layers),
            "native_cache_attach_hf_views": bool(args.native_cache_attach_hf_views),
            "native_attention_module": bool(args.native_attention_module),
            "fused_rope_append_seed": bool(args.fused_rope_append_seed),
            "packed_qkv_projection": bool(args.packed_qkv_projection or args.native_attention_module),
            "packed_qkv_fused_input": bool(args.packed_qkv_fused_input),
            "direct_o_proj": bool(args.direct_o_proj),
            "triton_o_proj": bool(args.triton_o_proj),
            "dynamic_selector_layers": sorted(parse_layer_id_set(args.dynamic_selector_layers)),
            "dynamic_selector_profile": args.dynamic_selector_profile,
            "dynamic_selector_reference_only": bool(args.dynamic_selector_profile),
            "allow_mixed_seed_configs": bool(args.allow_mixed_seed_configs),
            "native_routed_cache": native_cache.summary() if native_cache is not None else {"enabled": False},
            "candidate_backend": "seed_only_bundle",
            "fallback_reason": None,
            "bucket_route_policy": bucket_policy,
        }
    return {
        "schema": "streamattn.seed_only_route_bundle_decode.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model": {
            "model_id": args.model,
            "attn_implementation": args.attn_implementation or "default",
        },
        "route_bundle": route_bundle_summary,
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
        "patch_timing": patch_timing,
        "prompts": [
            {
                key: value
                for key, value in {
                    "row": idx,
                    "kind": row.get("kind", ""),
                    "id": row.get("id", ""),
                    "bucket": row.get("bucket", ""),
                    "language": row.get("language", ""),
                    "risk": row.get("risk") or row.get("expected_risk", ""),
                    "difficulty": row.get("difficulty", ""),
                }.items()
                if value != ""
            }
            for idx, row in enumerate(prompt_rows)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--policy-names", default="")
    parser.add_argument("--use-packaged-policies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--bucket-route-policy",
        choices=["none", "qwen25_3b_b8"],
        default="none",
        help="Batch-level bucket-conditioned route selector. Stress/unknown buckets fail closed in product mode.",
    )
    parser.add_argument(
        "--product-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --bucket-route-policy, fail closed on stress-risk or unknown buckets.",
    )
    parser.add_argument("--prompt-kinds", default="needle,code,long_doc,chat_doc")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument(
        "--prompt-file-kinds",
        default="",
        help="Optional comma-separated kind/bucket filter applied when --prompt-file is used.",
    )
    parser.add_argument(
        "--prompt-file-rows-per-kind",
        type=int,
        default=0,
        help="When filtering --prompt-file, keep at most this many rows per kind/bucket.",
    )
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument(
        "--prompt-truncation-side",
        choices=["left", "right"],
        default="right",
        help="Tokenizer truncation side for --prompt-file rows. Use left to keep final questions.",
    )
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
    parser.add_argument(
        "--layer-seed-overrides",
        default="",
        help=(
            "Semicolon-separated per-layer seed overrides, for example "
            "'2:sink=2,recent=4,middle=10,block=32;18:2,6,12'."
        ),
    )
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    parser.add_argument("--min-topk-overlap", type=int, default=4)
    parser.add_argument("--max-logprob-delta", type=float, default=2.0e-3)
    parser.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--explicit-cache-position",
        action="store_true",
        help="Pass cache_position explicitly during decode instead of relying on HF cache length.",
    )
    parser.add_argument(
        "--native-routed-cache",
        action="store_true",
        help="Use preallocated StreamAttn BHND cache for routed layers and bypass HF cache update there.",
    )
    parser.add_argument(
        "--native-cache-hf-sync-layers",
        default="",
        help=(
            "Comma-separated routed layer ids that still call HF cache update while native cache "
            "serves seed-only reads. Useful for keeping HF mask/cache-length bookkeeping alive."
        ),
    )
    parser.add_argument(
        "--native-cache-attach-hf-views",
        action="store_true",
        help=(
            "Experimental: point HF DynamicCache routed-layer keys/values at native-cache views "
            "instead of using HF update calls for cache metadata."
        ),
    )
    parser.add_argument(
        "--native-attention-module",
        action="store_true",
        help=(
            "Replace routed Qwen attention modules with StreamAttnQwenAttentionModule during decode. "
            "This native module path pre-packs QKV projection weights by default."
        ),
    )
    parser.add_argument(
        "--fused-rope-append-seed",
        action="store_true",
        help=(
            "Experimental: for native-cache routed layers not using HF sync, fuse Qwen RoPE, "
            "native K/V append, and seed-only attention into one Triton launch."
        ),
    )
    parser.add_argument(
        "--packed-qkv-projection",
        action="store_true",
        help=(
            "Prepack routed q/k/v projection weights and use one packed QKV F.linear during "
            "decode instead of three separate q_proj/k_proj/v_proj calls."
        ),
    )
    parser.add_argument(
        "--packed-qkv-fused-input",
        action="store_true",
        help=(
            "Experimental: when used with --packed-qkv-projection and --fused-rope-append-seed, "
            "pass the packed QKV projection output directly into the fused Triton decode kernel."
        ),
    )
    parser.add_argument(
        "--direct-o-proj",
        action="store_true",
        help="Apply routed attention output projection with direct F.linear instead of module o_proj dispatch.",
    )
    parser.add_argument(
        "--triton-o-proj",
        action="store_true",
        help=(
            "Experimental: apply routed decode output projection with a StreamAttn Triton "
            "kernel for the [batch,1,hidden] shape."
        ),
    )
    parser.add_argument(
        "--dynamic-selector-layers",
        default="",
        help=(
            "Comma-separated routed layer ids that use reference selected-block seed attention "
            "instead of the fixed-block Triton seed kernel."
        ),
    )
    parser.add_argument(
        "--dynamic-selector-profile",
        default="",
        help=(
            "Runtime selector profile for --dynamic-selector-layers, for example "
            "support_rand8_refine32. This path is reference-only and not a speed result."
        ),
    )
    parser.add_argument(
        "--allow-mixed-seed-configs",
        action="store_true",
        help=(
            "Allow explicit policy-name bundles to contain per-layer seed configs. "
            "Use for candidate routes such as Qwen3B L2 S640 plus S384 layers."
        ),
    )
    parser.add_argument(
        "--profile-patch-timing",
        action="store_true",
        help="Collect CUDA-event component timings inside each routed seed-only attention patch.",
    )
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
