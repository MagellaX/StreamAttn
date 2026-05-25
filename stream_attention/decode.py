"""Planned decode path for StreamAttn Gate-1 routing.

This module keeps decode routing separate from kernel execution. The first
version targets contiguous KV cache tensors and top-level ``causal=False``
decode mechanics, where the KV cache already contains only valid past/current
tokens.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch

from .certified import StreamAttnMetadataCache
from .gate0_fused_hybrid import (
    Gate0FusedHybridPolicy,
    Gate0ProjectionMetadata,
    build_gate0_projection_metadata,
    make_gate0_fused_hybrid_workspace,
    stream_attn_gate0_fused_hybrid,
)
from .gate1 import dense_attention_forward, make_route_request, stream_attn_gate1
from .router import StreamAttnRouter, normalize_device_class
from .telemetry import Prediction, seq_bucket


DenseFallbackFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
DEFAULT_GATE0_SEED_ONLY_BATCHED_POLICY = "qwen25_05b_l8_32k_seed_only_batched"
_PACKAGED_GATE0_SEED_ONLY_BATCHED_POLICIES = {
    DEFAULT_GATE0_SEED_ONLY_BATCHED_POLICY: "policies/qwen25_05b_l8_32k_seed_only_batched.json",
    "qwen25_05b_l8_32k_fp16_b8_seed_only_v1": "policies/qwen25_05b_l8_32k_seed_only_batched.json",
}


def _read_packaged_text(relative_path: str) -> str:
    """Read a text resource from the installed stream_attention package."""

    parts = tuple(part for part in relative_path.replace("\\", "/").split("/") if part)
    try:
        from importlib import resources as importlib_resources

        files = getattr(importlib_resources, "files", None)
        if files is not None:
            resource = importlib_resources.files(__package__ or "stream_attention")
            for part in parts:
                resource = resource.joinpath(part)
            return resource.read_text(encoding="utf-8")
    except Exception:
        pass
    return (Path(__file__).resolve().parent.joinpath(*parts)).read_text(encoding="utf-8")


@dataclass(frozen=True)
class StreamAttnDecodePolicy:
    """Conservative defaults for planned contiguous-KV decode."""

    max_router_regret_pct: float = 0.05
    safety_margin: float = 1.10
    allow_mass: bool = True
    allow_value_bound: bool = True
    prefer_value_bound_if_within: float = 1.10
    collect_telemetry_every: int = 16
    min_kv_len_for_gate1: int = 4096
    max_active_fraction_mass: float = 0.35
    max_active_fraction_value_bound: float = 0.30
    min_confidence: float = 0.70
    require_metadata_for_value_bound: bool = True
    allow_gate0_fused_hybrid: bool = True
    min_kv_len_for_gate0: int = 16384
    allow_gate0_seed_only_batched: bool = True
    min_kv_len_for_gate0_seed_only: int = 16384

    def __post_init__(self) -> None:
        if self.max_router_regret_pct < 0.0:
            raise ValueError("max_router_regret_pct must be non-negative")
        if self.safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1")
        if self.prefer_value_bound_if_within < 1.0:
            raise ValueError("prefer_value_bound_if_within must be >= 1")
        if self.collect_telemetry_every < 0:
            raise ValueError("collect_telemetry_every must be non-negative")
        if self.min_kv_len_for_gate1 <= 0:
            raise ValueError("min_kv_len_for_gate1 must be positive")
        if self.min_kv_len_for_gate0 <= 0:
            raise ValueError("min_kv_len_for_gate0 must be positive")
        if self.min_kv_len_for_gate0_seed_only <= 0:
            raise ValueError("min_kv_len_for_gate0_seed_only must be positive")
        for name in ("max_active_fraction_mass", "max_active_fraction_value_bound"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")


def _normalize_dtype_name(dtype: Union[str, torch.dtype, None]) -> Optional[str]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        if dtype is torch.float16:
            return "fp16"
        if dtype is torch.bfloat16:
            return "bf16"
        if dtype is torch.float32:
            return "fp32"
        return str(dtype).replace("torch.", "")
    name = str(dtype).lower().replace("torch.", "")
    return {
        "float16": "fp16",
        "half": "fp16",
        "fp16": "fp16",
        "bfloat16": "bf16",
        "bf16": "bf16",
        "float32": "fp32",
        "fp32": "fp32",
    }.get(name, name)


@dataclass(frozen=True)
class Gate0SeedOnlyBatchedPolicy:
    """Distribution-calibrated all-seed Gate-0 decode route.

    This policy targets the current product wedge: logit-safe layers where a
    batched true-GQA seed-only decode can replace exact attention and fall back
    to dense on any shape or runtime mismatch.
    """

    mode: str = "all_seed_only"
    model_id: str = "unknown"
    layer_id: int = -1
    policy_id: str = ""
    tensor_space: str = "post_rope"
    dtype: Optional[str] = None
    kv_len_bucket: Optional[int] = None
    min_batch: int = 4
    heads: Optional[int] = None
    kv_heads: Optional[int] = None
    dim: Optional[int] = None
    block_size: int = 32
    sink_blocks: int = 2
    recent_blocks: int = 2
    middle_seed_blocks: int = 8
    block_order: str = "recent_first"
    num_warps: int = 4
    num_stages: int = 2
    safety_gate: str = "distribution_aware"
    top1_must_match: bool = True
    min_topk_overlap: int = 4
    max_kl: Optional[float] = None
    max_logit_delta: Optional[float] = None
    max_logprob_delta: Optional[float] = None
    expected_seed_only_ms: Optional[float] = None
    expected_dense_ms: Optional[float] = None
    expected_speedup_vs_dense: Optional[float] = None
    expected_max_abs_error: Optional[float] = None
    fallback: str = "dense"

    def __post_init__(self) -> None:
        if self.mode != "all_seed_only":
            raise ValueError("Gate0SeedOnlyBatchedPolicy only supports all_seed_only mode")
        if self.tensor_space != "post_rope":
            raise ValueError("Gate0 seed-only policy currently requires post_rope tensors")
        if self.min_batch <= 0:
            raise ValueError("min_batch must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.sink_blocks < 0 or self.recent_blocks < 0 or self.middle_seed_blocks < 0:
            raise ValueError("seed block counts must be non-negative")
        if self.block_order not in {"sequential", "recent_first", "sink_recent_first"}:
            raise ValueError("block_order must be sequential, recent_first, or sink_recent_first")
        if self.num_warps <= 0 or self.num_stages <= 0:
            raise ValueError("num_warps and num_stages must be positive")
        if self.min_topk_overlap < 0:
            raise ValueError("min_topk_overlap must be non-negative")
        for name in ("max_kl", "max_logit_delta", "max_logprob_delta"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "expected_seed_only_ms",
            "expected_dense_ms",
            "expected_speedup_vs_dense",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.expected_max_abs_error is not None and self.expected_max_abs_error < 0.0:
            raise ValueError("expected_max_abs_error must be non-negative")

    @classmethod
    def from_entry(cls, entry: Dict[str, Any]) -> "Gate0SeedOnlyBatchedPolicy":
        """Create a policy from a seed-only route artifact entry."""

        runtime = dict(entry.get("runtime") or entry.get("seed_config") or {})
        safety = dict(entry.get("safety") or {})
        timing = dict(entry.get("timing") or entry.get("quality") or {})
        shape = dict(entry.get("shape") or {})
        capture = dict(entry.get("capture") or {})
        quality = dict(entry.get("quality") or {})
        seed_error = dict(quality.get("seed_vs_flashinfer_exact") or {})
        return cls(
            mode=str(entry.get("mode") or runtime.get("mode") or "all_seed_only"),
            model_id=str(entry.get("model_id") or entry.get("model") or "unknown"),
            layer_id=int(
                entry.get("layer_id")
                if entry.get("layer_id") is not None
                else capture.get("layer_id")
                if capture.get("layer_id") is not None
                else -1
            ),
            policy_id=str(entry.get("policy_id") or entry.get("id") or ""),
            tensor_space=str(entry.get("tensor_space") or runtime.get("tensor_space") or "post_rope"),
            dtype=_normalize_dtype_name(
                entry.get("dtype") or shape.get("dtype") or runtime.get("dtype")
            ),
            kv_len_bucket=(
                int(entry["kv_len_bucket"])
                if entry.get("kv_len_bucket") is not None
                else int(shape["kv_len_bucket"])
                if shape.get("kv_len_bucket") is not None
                else int(shape["kv_len"])
                if shape.get("kv_len") is not None
                else None
            ),
            min_batch=int(
                entry.get("min_batch")
                or runtime.get("min_batch")
                or shape.get("batch")
                or 4
            ),
            heads=(
                int(entry["heads"])
                if entry.get("heads") is not None
                else int(shape["heads"])
                if shape.get("heads") is not None
                else int(shape["q_heads"])
                if shape.get("q_heads") is not None
                else None
            ),
            kv_heads=(
                int(entry["kv_heads"])
                if entry.get("kv_heads") is not None
                else int(shape["kv_heads"])
                if shape.get("kv_heads") is not None
                else int(shape["true_kv_heads"])
                if shape.get("true_kv_heads") is not None
                else None
            ),
            dim=(
                int(entry["dim"])
                if entry.get("dim") is not None
                else int(shape["dim"])
                if shape.get("dim") is not None
                else None
            ),
            block_size=int(runtime.get("block_size") or 32),
            sink_blocks=int(runtime.get("sink_blocks") or 2),
            recent_blocks=int(runtime.get("recent_blocks") or 2),
            middle_seed_blocks=int(runtime.get("middle_seed_blocks") or 8),
            block_order=str(runtime.get("block_order") or "recent_first"),
            num_warps=int(runtime.get("num_warps") or 4),
            num_stages=int(runtime.get("num_stages") or 2),
            safety_gate=str(safety.get("gate") or safety.get("safety_gate") or "distribution_aware"),
            top1_must_match=bool(safety.get("top1_must_match", True)),
            min_topk_overlap=int(safety.get("min_top5_overlap") or safety.get("min_topk_overlap") or 4),
            max_kl=(float(safety["max_kl"]) if safety.get("max_kl") is not None else None),
            max_logit_delta=(
                float(safety["max_logit_delta"])
                if safety.get("max_logit_delta") is not None
                else None
            ),
            max_logprob_delta=(
                float(safety["max_logprob_delta"])
                if safety.get("max_logprob_delta") is not None
                else None
            ),
            expected_seed_only_ms=(
                float(timing["seed_only_ms"])
                if timing.get("seed_only_ms") is not None
                else float(timing["streamattn_seed_only_ms"])
                if timing.get("streamattn_seed_only_ms") is not None
                else float(timing["seed_direct_full_prealloc_ms"])
                if timing.get("seed_direct_full_prealloc_ms") is not None
                else None
            ),
            expected_dense_ms=(
                float(timing["dense_ms"])
                if timing.get("dense_ms") is not None
                else float(timing["flashinfer_tc_ms"])
                if timing.get("flashinfer_tc_ms") is not None
                else float(timing["flashinfer_batch_tc_exact_ms"])
                if timing.get("flashinfer_batch_tc_exact_ms") is not None
                else None
            ),
            expected_speedup_vs_dense=(
                float(timing["speedup_vs_dense"])
                if timing.get("speedup_vs_dense") is not None
                else float(timing["speedup"])
                if timing.get("speedup") is not None
                else float(timing["speedup_vs_flashinfer_batch"])
                if timing.get("speedup_vs_flashinfer_batch") is not None
                else None
            ),
            expected_max_abs_error=(
                float(timing["max_abs_error"])
                if timing.get("max_abs_error") is not None
                else float(seed_error["max_abs_error"])
                if seed_error.get("max_abs_error") is not None
                else None
            ),
            fallback=str(entry.get("fallback") or "dense"),
        )

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        *,
        entry_index: int = 0,
    ) -> "Gate0SeedOnlyBatchedPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = payload.get("stable_entries") or payload.get("entries")
        if entries:
            return cls.from_entry(entries[entry_index])
        return cls.from_entry(payload)

    @classmethod
    def from_packaged(
        cls,
        name: str = DEFAULT_GATE0_SEED_ONLY_BATCHED_POLICY,
        *,
        entry_index: int = 0,
    ) -> "Gate0SeedOnlyBatchedPolicy":
        """Load a packaged, validated seed-only policy by name."""

        try:
            relative_path = _PACKAGED_GATE0_SEED_ONLY_BATCHED_POLICIES[name]
        except KeyError as exc:
            known = ", ".join(sorted(_PACKAGED_GATE0_SEED_ONLY_BATCHED_POLICIES))
            raise ValueError(f"unknown packaged Gate-0 seed-only policy {name!r}; known: {known}") from exc
        payload = json.loads(_read_packaged_text(relative_path))
        entries = payload.get("stable_entries") or payload.get("entries")
        if entries:
            return cls.from_entry(entries[entry_index])
        return cls.from_entry(payload)

    def mismatch_reasons(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        min_kv_len: int = 1,
        model_id: Optional[str] = None,
        layer_id: Optional[int] = None,
        backend_available: bool = True,
    ) -> List[str]:
        """Return fail-closed reasons for rejecting this policy on tensors."""

        reasons: List[str] = []
        if model_id is not None and model_id != self.model_id:
            reasons.append("model_mismatch")
        if layer_id is not None and int(layer_id) != self.layer_id:
            reasons.append("layer_mismatch")
        if not backend_available:
            reasons.append("backend_unavailable")
        if query.dim() != 4 or key.dim() != 4:
            return reasons + ["shape_mismatch"]
        batch, query_len, heads, dim = query.shape
        key_batch, kv_len, kv_heads, key_dim = key.shape
        if batch != key_batch or dim != key_dim:
            reasons.append("shape_mismatch")
        if query_len != 1:
            reasons.append("query_len_mismatch")
        if batch < self.min_batch:
            reasons.append("batch_below_min")
        if kv_len < min_kv_len:
            reasons.append("kv_len_below_min")
        if self.kv_len_bucket is not None and seq_bucket(kv_len) != seq_bucket(int(self.kv_len_bucket)):
            reasons.append("kv_len_bucket_mismatch")
        expected_dtype = _normalize_dtype_name(self.dtype)
        actual_dtype = _normalize_dtype_name(query.dtype)
        if expected_dtype is not None and expected_dtype != actual_dtype:
            reasons.append("dtype_mismatch")
        if kv_heads <= 0 or heads % kv_heads != 0:
            reasons.append("gqa_shape_mismatch")
        if self.heads is not None and self.heads != heads:
            reasons.append("heads_mismatch")
        if self.kv_heads is not None and self.kv_heads != kv_heads:
            reasons.append("kv_heads_mismatch")
        if self.dim is not None and self.dim != dim:
            reasons.append("head_dim_mismatch")
        return reasons

    def matches_tensors(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        min_kv_len: int = 1,
        model_id: Optional[str] = None,
        layer_id: Optional[int] = None,
        backend_available: bool = True,
    ) -> bool:
        """Return whether this policy can be used without fail-closed fallback."""

        return not self.mismatch_reasons(
            query,
            key,
            min_kv_len=min_kv_len,
            model_id=model_id,
            layer_id=layer_id,
            backend_available=backend_available,
        )


def load_packaged_gate0_seed_only_batched_policy(
    name: str = DEFAULT_GATE0_SEED_ONLY_BATCHED_POLICY,
    *,
    entry_index: int = 0,
) -> Gate0SeedOnlyBatchedPolicy:
    """Load a packaged Gate-0 seed-only batched policy."""

    return Gate0SeedOnlyBatchedPolicy.from_packaged(name, entry_index=entry_index)


@dataclass(frozen=True)
class Gate0SeedOnlyBatchedStats:
    batch_size: int
    heads: int
    kv_heads: int
    kv_len: int
    block_size: int
    seed_blocks: int
    total_blocks: int

    @property
    def seed_fraction(self) -> float:
        return 0.0 if self.total_blocks <= 0 else self.seed_blocks / self.total_blocks


@dataclass(frozen=True)
class Gate0SeedOnlyBatchedRunInfo:
    policy: Gate0SeedOnlyBatchedPolicy
    stats: Optional[Gate0SeedOnlyBatchedStats]
    backend_used: str
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class StreamAttnServingInfo:
    """Structured telemetry returned by the serving-facing seed-only API."""

    backend_used: str
    policy_id: Optional[str]
    fallback_reason: Optional[str]
    batch_size: int
    kv_len: int
    layer_id: Optional[int]
    model_id: Optional[str]
    dtype: str
    device: str
    plan_backend: str
    plan_reason: str
    seed_only_enabled: bool
    safety_policy_matched: bool
    runtime_counters: Dict[str, Dict[str, int]]
    stats: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable telemetry snapshot."""

        return asdict(self)


@dataclass(frozen=True)
class DecodeCostKey:
    device_class: str
    dtype: str
    query_len: int
    kv_bucket: int
    heads: int
    kv_heads: int
    dim: int
    attention_type: str
    block_size: int
    tile_size_q: int
    num_warps: int
    num_stages: int
    causal: bool = False

    @classmethod
    def from_tensors(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        kv_heads: Optional[int] = None,
        attention_type: str = "mha",
        block_size: int,
        tile_size_q: int,
        num_warps: int,
        num_stages: int,
        causal: bool = False,
    ) -> "DecodeCostKey":
        request = make_route_request(
            query,
            key,
            causal=causal,
            block_size=block_size,
            tile_size_q=tile_size_q,
            phase="decode",
        )
        return cls(
            device_class=request.normalized_device_class,
            dtype=request.dtype,
            query_len=request.seq_q,
            kv_bucket=seq_bucket(request.seq_k),
            heads=request.heads,
            kv_heads=kv_heads if kv_heads is not None else key.shape[2],
            dim=request.dim,
            attention_type=attention_type,
            block_size=block_size,
            tile_size_q=tile_size_q,
            num_warps=num_warps,
            num_stages=num_stages,
            causal=causal,
        )


@dataclass(frozen=True)
class DecodeCostEntry:
    dense_ms: float
    qk_scan_ms: float
    gate1_mass_ms_at_calibration: float
    calibration_active_fraction: float
    gate1_dense_equiv_ms: Optional[float] = None
    gate1_value_bound_ms_at_calibration: Optional[float] = None
    predicate_overhead_ms: float = 0.0
    metadata_update_ms: float = 0.0
    metadata_build_ms: Optional[float] = None
    measured_router_regret_pct: Optional[float] = None
    sample_count: int = 1
    last_updated: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dense_ms <= 0.0:
            raise ValueError("dense_ms must be positive")
        if self.qk_scan_ms <= 0.0:
            raise ValueError("qk_scan_ms must be positive")
        if self.gate1_mass_ms_at_calibration <= 0.0:
            raise ValueError("gate1_mass_ms_at_calibration must be positive")
        if not 0.0 <= self.calibration_active_fraction <= 1.0:
            raise ValueError("calibration_active_fraction must be in [0, 1]")
        if self.gate1_dense_equiv_ms is not None and self.gate1_dense_equiv_ms <= 0.0:
            raise ValueError("gate1_dense_equiv_ms must be positive")
        if self.predicate_overhead_ms < 0.0:
            raise ValueError("predicate_overhead_ms must be non-negative")
        if self.metadata_update_ms < 0.0:
            raise ValueError("metadata_update_ms must be non-negative")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")

    @property
    def gate1_pv_ms(self) -> float:
        dense_equiv = self.gate1_dense_equiv_ms
        if dense_equiv is None:
            dense_equiv = self.dense_ms
        return max(0.0, dense_equiv - self.qk_scan_ms)

    @property
    def value_bound_extra_ms(self) -> float:
        if self.gate1_value_bound_ms_at_calibration is None:
            return 0.0
        return max(
            0.0,
            self.gate1_value_bound_ms_at_calibration
            - self.gate1_mass_ms_at_calibration,
        )

    def predict_mass_ms(
        self,
        active_fraction: float,
        *,
        include_metadata_update: bool = False,
    ) -> float:
        active = min(1.0, max(0.0, float(active_fraction)))
        metadata_cost = self.metadata_update_ms if include_metadata_update else 0.0
        return (
            self.qk_scan_ms
            + active * self.gate1_pv_ms
            + self.predicate_overhead_ms
            + metadata_cost
        )

    def predict_value_bound_ms(
        self,
        active_fraction: float,
        *,
        include_metadata_update: bool = False,
    ) -> float:
        return (
            self.predict_mass_ms(
                active_fraction,
                include_metadata_update=include_metadata_update,
            )
            + self.value_bound_extra_ms
        )

    def break_even_active_fraction(self, *, safety_margin: float) -> float:
        if safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1")
        if self.gate1_pv_ms <= 0.0:
            return 1.0 if self.qk_scan_ms + self.predicate_overhead_ms < self.dense_ms else 0.0
        threshold = (
            self.dense_ms / safety_margin
            - self.qk_scan_ms
            - self.predicate_overhead_ms
        ) / self.gate1_pv_ms
        return min(1.0, max(0.0, threshold))

    @classmethod
    def from_measurement(
        cls,
        *,
        dense_ms: float,
        qk_scan_ms: float,
        gate1_mass_ms: float,
        active_fraction: float,
        gate1_dense_equiv_ms: Optional[float] = None,
        gate1_value_bound_ms: Optional[float] = None,
        metadata_update_ms: float = 0.0,
        metadata_build_ms: Optional[float] = None,
        measured_router_regret_pct: Optional[float] = None,
        last_updated: Optional[str] = None,
    ) -> "DecodeCostEntry":
        dense_equiv = gate1_dense_equiv_ms if gate1_dense_equiv_ms is not None else dense_ms
        gate1_pv_ms = max(0.0, dense_equiv - qk_scan_ms)
        active = min(1.0, max(0.0, float(active_fraction)))
        predicted_without_overhead = qk_scan_ms + active * gate1_pv_ms
        overhead = max(0.0, gate1_mass_ms - predicted_without_overhead)
        return cls(
            dense_ms=dense_ms,
            qk_scan_ms=qk_scan_ms,
            gate1_mass_ms_at_calibration=gate1_mass_ms,
            calibration_active_fraction=active,
            gate1_dense_equiv_ms=gate1_dense_equiv_ms,
            gate1_value_bound_ms_at_calibration=gate1_value_bound_ms,
            predicate_overhead_ms=overhead,
            metadata_update_ms=metadata_update_ms,
            metadata_build_ms=metadata_build_ms,
            measured_router_regret_pct=measured_router_regret_pct,
            last_updated=last_updated,
        )


class DecodeCostModel:
    """Lookup table for planned decode routing."""

    def __init__(self) -> None:
        self._entries: Dict[DecodeCostKey, DecodeCostEntry] = {}

    def update(self, key: DecodeCostKey, entry: DecodeCostEntry) -> None:
        self._entries[key] = entry

    def lookup(self, key: DecodeCostKey) -> Optional[DecodeCostEntry]:
        return self._entries.get(key)

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "entries": [
                {"key": asdict(key), "entry": asdict(entry)}
                for key, entry in self._entries.items()
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "DecodeCostModel":
        model = cls()
        for item in payload.get("entries", []):
            model.update(
                DecodeCostKey(**dict(item["key"])),
                DecodeCostEntry(**dict(item["entry"])),
            )
        return model

    def to_json(self, path: Union[str, Path]) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DecodeCostModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class StreamAttnDecodeWorkspace:
    """Reusable state for the planned contiguous-KV decode runtime."""

    device: torch.device
    max_batch: int
    max_query_len: int
    max_kv_len: int
    max_heads: int
    head_dim: int
    block_size: int
    raw_stats: Optional[torch.Tensor] = None
    output: Optional[torch.Tensor] = None
    metadata_update_scratch: Optional[torch.Tensor] = None
    step_index: int = 0

    @classmethod
    def allocate(
        cls,
        *,
        device: Union[str, torch.device],
        max_batch: int,
        max_query_len: int,
        max_kv_len: int,
        max_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> "StreamAttnDecodeWorkspace":
        device = torch.device(device)
        if device.type == "cuda" and device.index is None and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        for name, value in {
            "max_batch": max_batch,
            "max_query_len": max_query_len,
            "max_kv_len": max_kv_len,
            "max_heads": max_heads,
            "head_dim": head_dim,
            "block_size": block_size,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

        max_q_blocks = max(1, math.ceil(max_query_len / block_size))
        max_kv_blocks = max(1, math.ceil(max_kv_len / block_size))
        return cls(
            device=device,
            max_batch=int(max_batch),
            max_query_len=int(max_query_len),
            max_kv_len=int(max_kv_len),
            max_heads=int(max_heads),
            head_dim=int(head_dim),
            block_size=int(block_size),
            raw_stats=torch.zeros(
                (max_batch, max_heads, max_q_blocks, 6),
                device=device,
                dtype=torch.int64,
            ),
            output=torch.empty(
                (max_batch, max_query_len, max_heads, head_dim),
                device=device,
                dtype=dtype,
            ),
            metadata_update_scratch=torch.empty(
                (max_batch, max_heads, max_kv_blocks),
                device=device,
                dtype=torch.float32,
            ),
        )

    def validate(
        self,
        query: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        batch: Optional[int] = None,
        query_len: Optional[int] = None,
        kv_len: Optional[int] = None,
        heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Validate that a decode call fits in this workspace."""

        if query is not None:
            if query.dim() != 4:
                raise ValueError("query must have shape [batch, seq, heads, dim]")
            batch = query.shape[0]
            query_len = query.shape[1]
            heads = query.shape[2]
            head_dim = query.shape[3]
            device = query.device
        if key is not None:
            if key.dim() != 4:
                raise ValueError("key must have shape [batch, seq, heads, dim]")
            if batch is not None and key.shape[0] != batch:
                raise ValueError("query/key batch must match")
            if head_dim is not None and key.shape[3] != head_dim:
                raise ValueError("query/key head dim must match")
            kv_len = key.shape[1]
            heads = max(heads or 0, key.shape[2])
            device = key.device if device is None else device
        if value is not None:
            if value.dim() != 4:
                raise ValueError("value must have shape [batch, seq, heads, dim]")
            if batch is not None and value.shape[0] != batch:
                raise ValueError("query/value batch must match")
            if kv_len is not None and value.shape[1] != kv_len:
                raise ValueError("key/value sequence length must match")
            if head_dim is not None and value.shape[3] != head_dim:
                raise ValueError("query/value head dim must match")
            heads = max(heads or 0, value.shape[2])
            device = value.device if device is None else device

        checks = {
            "batch": (batch, self.max_batch),
            "query_len": (query_len, self.max_query_len),
            "kv_len": (kv_len, self.max_kv_len),
            "heads": (heads, self.max_heads),
            "head_dim": (head_dim, self.head_dim),
        }
        for name, (actual, limit) in checks.items():
            if actual is None:
                continue
            if name == "head_dim":
                if int(actual) != limit:
                    raise ValueError(f"{name} {actual} does not match workspace {limit}")
            elif int(actual) > limit:
                raise ValueError(f"{name} {actual} exceeds workspace limit {limit}")
        if device is not None and torch.device(device) != self.device:
            raise ValueError(f"device {torch.device(device)} does not match workspace {self.device}")

    def reset_step(self) -> None:
        self.step_index = 0

    def advance_step(self) -> int:
        self.step_index += 1
        return self.step_index


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def decode_cost_model_from_profile_rows(rows: Iterable[Dict[str, object]]) -> DecodeCostModel:
    """Build a calibrated decode cost model from decode profiler rows."""

    grouped: Dict[DecodeCostKey, List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("error"):
            continue
        shape = row.get("shape")
        if not isinstance(shape, dict):
            continue
        required = [
            row.get("dense_decode_ms"),
            row.get("gate1_qk_scan_ms"),
            row.get("gate1_mass_ms"),
            row.get("active_pv_fraction_mass"),
        ]
        if any(value is None for value in required):
            continue
        key = DecodeCostKey(
            device_class=normalize_device_class(str(row.get("device", ""))),
            dtype=str(shape.get("dtype")),
            query_len=int(shape.get("query_len")),
            kv_bucket=seq_bucket(int(shape.get("kv_len"))),
            heads=int(shape.get("heads")),
            kv_heads=int(shape.get("kv_heads")),
            dim=int(shape.get("dim")),
            attention_type=str(shape.get("attention_type")),
            block_size=int(row.get("block_size")),
            tile_size_q=int(row.get("tile_size_q")),
            num_warps=int(row.get("num_warps")),
            num_stages=int(row.get("num_stages")),
            causal=row.get("causal_mode") not in {None, "none", False},
        )
        grouped.setdefault(key, []).append(row)

    model = DecodeCostModel()
    for key, group in grouped.items():
        dense_ms = _mean(row["dense_decode_ms"] for row in group)
        qk_ms = _mean(row["gate1_qk_scan_ms"] for row in group)
        dense_equiv_values = [
            row["gate1_dense_equiv_ms"]
            for row in group
            if row.get("gate1_dense_equiv_ms") is not None
        ]
        gate1_dense_equiv_ms = (
            _mean(dense_equiv_values) if dense_equiv_values else None
        )
        metadata_update_values = [
            row["metadata_update_wall_ms"]
            for row in group
            if row.get("metadata_update_wall_ms") is not None
        ]
        metadata_build_values = [
            row["metadata_full_build_ms"]
            for row in group
            if row.get("metadata_full_build_ms") is not None
        ]
        metadata_update_ms = (
            _mean(metadata_update_values) if metadata_update_values else 0.0
        )
        metadata_build_ms = (
            _mean(metadata_build_values) if metadata_build_values else None
        )
        gate1_pv_ms = max(0.0, (gate1_dense_equiv_ms or dense_ms) - qk_ms)
        residual_rows = []
        for row in group:
            active = float(row["active_pv_fraction_mass"])
            mass_ms = float(row["gate1_mass_ms"])
            residual = max(0.0, mass_ms - (qk_ms + active * gate1_pv_ms))
            residual_rows.append((residual, row))
        residual_rows.sort(key=lambda item: item[0])
        residual_idx = max(0, int(round(0.9 * (len(residual_rows) - 1))))
        predicate_overhead_ms, calibration_row = residual_rows[residual_idx]
        value_extra_values = [
            max(0.0, float(row["gate1_value_bound_ms"]) - float(row["gate1_mass_ms"]))
            for row in group
            if row.get("gate1_value_bound_ms") is not None
        ]
        value_bound_ms = None
        if value_extra_values:
            value_bound_ms = float(calibration_row["gate1_mass_ms"]) + _mean(
                value_extra_values
            )
        regret_values = [
            row["router_regret_pct"]
            for row in group
            if row.get("router_regret_pct") is not None
        ]
        model.update(
            key,
            DecodeCostEntry(
                dense_ms=dense_ms,
                qk_scan_ms=qk_ms,
                gate1_mass_ms_at_calibration=float(
                    calibration_row["gate1_mass_ms"]
                ),
                calibration_active_fraction=float(
                    calibration_row["active_pv_fraction_mass"]
                ),
                gate1_dense_equiv_ms=gate1_dense_equiv_ms,
                gate1_value_bound_ms_at_calibration=value_bound_ms,
                predicate_overhead_ms=predicate_overhead_ms,
                metadata_update_ms=metadata_update_ms,
                metadata_build_ms=metadata_build_ms,
                measured_router_regret_pct=(
                    _mean(regret_values) if regret_values else None
                ),
                sample_count=len(group),
            ),
        )
    return model


@dataclass(frozen=True)
class StreamAttnDecodePlan:
    backend: str
    reason: str
    query_len: int
    kv_len: int
    heads: int
    kv_heads: int
    dim: int
    attention_type: str
    block_size: int
    tile_size_q: int
    num_warps: int
    num_stages: int
    metadata_required: bool
    metadata_update_required: bool
    predicted_active_fraction: float
    predicted_ms: float
    dense_ms: float
    expected_regret_ms: float
    collect_telemetry: bool
    skip_predicate: Optional[str] = None
    active_threshold: Optional[float] = None
    projection_metadata_required: bool = False
    gate0_fused_hybrid_policy: Optional[Gate0FusedHybridPolicy] = None
    gate0_seed_only_batched_policy: Optional[Gate0SeedOnlyBatchedPolicy] = None
    fallback_reason: Optional[str] = None


def _predict_active_fraction(
    request,
    *,
    router: Optional[StreamAttnRouter],
    active_fraction_hint: Optional[float],
) -> Prediction:
    if active_fraction_hint is not None:
        return Prediction(
            active_frac_hat=min(1.0, max(0.0, float(active_fraction_hint))),
            confidence=1.0,
            source="hint",
        )
    if router is not None:
        return router.predict_with_aggregate_prior(request)
    return Prediction(active_frac_hat=1.0, confidence=0.0, source="missing")


def stream_attn_decode_plan(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    metadata: Optional[StreamAttnMetadataCache] = None,
    gate0_fused_hybrid_policy: Optional[Gate0FusedHybridPolicy] = None,
    gate0_seed_only_batched_policy: Optional[Gate0SeedOnlyBatchedPolicy] = None,
    gate0_projection_metadata: Optional[Gate0ProjectionMetadata] = None,
    router: Optional[StreamAttnRouter] = None,
    decode_cost_model: Optional[DecodeCostModel] = None,
    policy: Optional[StreamAttnDecodePolicy] = None,
    active_fraction_hint: Optional[float] = None,
    attention_type: str = "mha",
    kv_heads: Optional[int] = None,
    block_size: int = 64,
    tile_size_q: int = 16,
    num_warps: int = 4,
    num_stages: int = 3,
    error_budget: float = 1.0e-3,
    step_index: int = 0,
) -> StreamAttnDecodePlan:
    """Plan one contiguous-KV decode attention call without launching kernels."""

    if query.dim() != 4 or key.dim() != 4:
        raise ValueError("query and key must have shape [batch, seq, heads, dim]")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query/key batch and dim must match")
    if kv_heads is None:
        kv_heads = key.shape[2]
    policy = policy or StreamAttnDecodePolicy()
    request = make_route_request(
        query,
        key,
        causal=False,
        block_size=block_size,
        tile_size_q=tile_size_q,
        phase="decode",
        metadata_available=metadata is not None
        and metadata.value_norm_bounds is not None,
    )
    prediction = _predict_active_fraction(
        request,
        router=router,
        active_fraction_hint=active_fraction_hint,
    )
    active = prediction.active_frac_hat
    key_cost = DecodeCostKey.from_tensors(
        query,
        key,
        kv_heads=kv_heads,
        attention_type=attention_type,
        block_size=block_size,
        tile_size_q=tile_size_q,
        num_warps=num_warps,
        num_stages=num_stages,
        causal=False,
    )
    cost = decode_cost_model.lookup(key_cost) if decode_cost_model is not None else None
    collect_telemetry = (
        policy.collect_telemetry_every > 0
        and step_index % policy.collect_telemetry_every == 0
    )

    dense_ms = cost.dense_ms if cost is not None else 0.0
    if dense_ms <= 0.0 and gate0_fused_hybrid_policy is not None:
        dense_ms = float(gate0_fused_hybrid_policy.expected_dense_ms or 0.0)
    if dense_ms <= 0.0 and gate0_seed_only_batched_policy is not None:
        dense_ms = float(gate0_seed_only_batched_policy.expected_dense_ms or 0.0)
    candidates = {"dense": dense_ms}
    reasons = {"dense": "dense_fallback"}
    fallback_reason: Optional[str] = None
    metadata_available = metadata is not None and metadata.value_norm_bounds is not None
    active_threshold = (
        cost.break_even_active_fraction(safety_margin=policy.safety_margin)
        if cost is not None
        else None
    )

    if gate0_fused_hybrid_policy is not None and policy.allow_gate0_fused_hybrid:
        gate0 = gate0_fused_hybrid_policy
        gate0_bucket_ok = (
            gate0.kv_len_bucket is None
            or seq_bucket(key.shape[1]) == seq_bucket(int(gate0.kv_len_bucket))
        )
        gate0_shape_ok = (
            query.shape[1] == 1
            and gate0.heads == query.shape[2]
            and key.shape[2] > 0
            and query.shape[2] % key.shape[2] == 0
            and (gate0.kv_heads is None or gate0.kv_heads == key.shape[2])
        )
        gate0_ms = gate0.expected_fused_hybrid_ms
        if gate0_ms is None and gate0.expected_speedup_vs_dense and dense_ms > 0.0:
            gate0_ms = dense_ms / float(gate0.expected_speedup_vs_dense)
        gate0_speedup = gate0.expected_speedup_vs_dense
        if gate0_speedup is None and gate0_ms is not None and dense_ms > 0.0:
            gate0_speedup = dense_ms / gate0_ms
        gate0_error_ok = (
            gate0.expected_max_abs_error is None
            or gate0.expected_max_abs_error <= error_budget
        )
        if (
            gate0_shape_ok
            and gate0_bucket_ok
            and key.shape[1] >= policy.min_kv_len_for_gate0
            and gate0_ms is not None
            and gate0_ms > 0.0
            and dense_ms > 0.0
            and gate0_speedup is not None
            and gate0_speedup >= policy.safety_margin
            and gate0_error_ok
        ):
            candidates["gate0_fused_hybrid"] = float(gate0_ms)
            reasons["gate0_fused_hybrid"] = "calibrated_gate0_fused_hybrid_policy"

    if (
        gate0_seed_only_batched_policy is not None
        and policy.allow_gate0_seed_only_batched
    ):
        seed_policy = gate0_seed_only_batched_policy
        seed_mismatches = seed_policy.mismatch_reasons(
            query,
            key,
            min_kv_len=policy.min_kv_len_for_gate0_seed_only,
        )
        seed_ms = seed_policy.expected_seed_only_ms
        if seed_ms is None and seed_policy.expected_speedup_vs_dense and dense_ms > 0.0:
            seed_ms = dense_ms / float(seed_policy.expected_speedup_vs_dense)
        seed_speedup = None
        if seed_ms is not None and dense_ms > 0.0:
            seed_speedup = dense_ms / seed_ms
        elif seed_policy.expected_speedup_vs_dense is not None:
            seed_speedup = seed_policy.expected_speedup_vs_dense
        if seed_mismatches:
            fallback_reason = seed_mismatches[0]
        elif seed_ms is None or seed_ms <= 0.0:
            fallback_reason = "missing_seed_only_batched_timing"
        elif dense_ms <= 0.0:
            fallback_reason = "missing_dense_timing"
        elif seed_speedup is None or seed_speedup < policy.safety_margin:
            fallback_reason = "seed_only_batched_not_profitable"
        if (
            not seed_mismatches
            and seed_ms is not None
            and seed_ms > 0.0
            and dense_ms > 0.0
            and seed_speedup is not None
            and seed_speedup >= policy.safety_margin
        ):
            candidates["gate0_seed_only_batched"] = float(seed_ms)
            reasons["gate0_seed_only_batched"] = (
                "calibrated_gate0_seed_only_batched_policy"
            )

    if (
        cost is None
        and "gate0_fused_hybrid" not in candidates
        and "gate0_seed_only_batched" not in candidates
    ):
        return StreamAttnDecodePlan(
            backend="dense",
            reason="missing_decode_cost",
            query_len=query.shape[1],
            kv_len=key.shape[1],
            heads=query.shape[2],
            kv_heads=kv_heads,
            dim=query.shape[3],
            attention_type=attention_type,
            block_size=block_size,
            tile_size_q=tile_size_q,
            num_warps=num_warps,
            num_stages=num_stages,
            metadata_required=False,
            metadata_update_required=False,
            predicted_active_fraction=active,
            predicted_ms=0.0,
            dense_ms=0.0,
            expected_regret_ms=0.0,
            collect_telemetry=collect_telemetry,
            fallback_reason=fallback_reason,
        )

    if (
        cost is not None
        and active_threshold is not None
        and policy.allow_mass
        and key.shape[1] >= policy.min_kv_len_for_gate1
        and prediction.confidence >= policy.min_confidence
        and active <= min(policy.max_active_fraction_mass, active_threshold)
    ):
        mass_ms = cost.predict_mass_ms(active)
        if mass_ms * policy.safety_margin < dense_ms:
            candidates["gate1_mass"] = mass_ms
            reasons["gate1_mass"] = "calibrated_mass_profitable"

    if (
        cost is not None
        and active_threshold is not None
        and policy.allow_value_bound
        and key.shape[1] >= policy.min_kv_len_for_gate1
        and prediction.confidence >= policy.min_confidence
        and active <= min(policy.max_active_fraction_value_bound, active_threshold)
        and (metadata_available or not policy.require_metadata_for_value_bound)
    ):
        value_ms = cost.predict_value_bound_ms(active)
        if value_ms * policy.safety_margin < dense_ms:
            candidates["gate1_value_bound"] = value_ms
            reasons["gate1_value_bound"] = "calibrated_value_bound_profitable"

    backend = min(candidates.items(), key=lambda item: item[1])[0]
    if (
        backend == "gate1_mass"
        and "gate1_value_bound" in candidates
        and candidates["gate1_value_bound"]
        <= candidates["gate1_mass"] * policy.prefer_value_bound_if_within
    ):
        backend = "gate1_value_bound"
    predicted_ms = candidates[backend]
    oracle_pred = min(candidates.values())
    return StreamAttnDecodePlan(
        backend=backend,
        reason=reasons[backend],
        query_len=query.shape[1],
        kv_len=key.shape[1],
        heads=query.shape[2],
        kv_heads=kv_heads,
        dim=query.shape[3],
        attention_type=attention_type,
        block_size=block_size,
        tile_size_q=tile_size_q,
        num_warps=num_warps,
        num_stages=num_stages,
        metadata_required=backend == "gate1_value_bound",
        metadata_update_required=(
            backend == "gate0_fused_hybrid" and gate0_projection_metadata is None
        ),
        predicted_active_fraction=active,
        predicted_ms=predicted_ms,
        dense_ms=dense_ms,
        expected_regret_ms=max(0.0, predicted_ms - oracle_pred),
        collect_telemetry=collect_telemetry,
        skip_predicate=(
            "mass"
            if backend == "gate1_mass"
            else "value_bound"
            if backend == "gate1_value_bound"
            else None
        ),
        active_threshold=active_threshold,
        projection_metadata_required=backend == "gate0_fused_hybrid",
        gate0_fused_hybrid_policy=(
            gate0_fused_hybrid_policy if backend == "gate0_fused_hybrid" else None
        ),
        gate0_seed_only_batched_policy=(
            gate0_seed_only_batched_policy
            if backend == "gate0_seed_only_batched"
            else None
        ),
        fallback_reason=(fallback_reason if backend == "dense" else None),
    )


def stream_attn_decode_run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    plan: StreamAttnDecodePlan,
    metadata: Optional[StreamAttnMetadataCache] = None,
    gate0_projection_metadata: Optional[Gate0ProjectionMetadata] = None,
    gate0_workspace: Optional[Dict[str, torch.Tensor]] = None,
    output_workspace: Optional[torch.Tensor] = None,
    dense_fallback: Optional[DenseFallbackFn] = None,
    dense_fallback_backend: str = "dense",
    build_gate0_metadata_if_missing: bool = True,
    error_budget: float = 1.0e-3,
    return_info: bool = False,
):
    """Execute a decode plan produced by :func:`stream_attn_decode_plan`."""

    if plan.backend == "dense":
        output = (
            dense_fallback(query, key, value)
            if dense_fallback is not None
            else dense_attention_forward(query, key, value, causal=False)
        )
        return (output, None) if return_info else output
    if plan.backend == "gate0_fused_hybrid":
        if plan.gate0_fused_hybrid_policy is None:
            raise ValueError("gate0_fused_hybrid plan is missing its policy")
        return stream_attn_gate0_fused_hybrid(
            query,
            key,
            value,
            policy=plan.gate0_fused_hybrid_policy,
            metadata=gate0_projection_metadata,
            build_metadata_if_missing=build_gate0_metadata_if_missing,
            return_info=return_info,
            workspace=gate0_workspace,
            fallback="dense",
            num_warps=plan.num_warps,
            num_stages=plan.num_stages,
        )
    if plan.backend == "gate0_seed_only_batched":
        if plan.gate0_seed_only_batched_policy is None:
            raise ValueError("gate0_seed_only_batched plan is missing its policy")
        seed_policy = plan.gate0_seed_only_batched_policy

        def _dense_fallback(reason: str):
            output = (
                dense_fallback(query, key, value)
                if dense_fallback is not None
                else dense_attention_forward(query, key, value, causal=False)
            )
            info = Gate0SeedOnlyBatchedRunInfo(
                policy=seed_policy,
                stats=None,
                backend_used=dense_fallback_backend,
                fallback_reason=reason,
            )
            return (output, info) if return_info else output

        if not all(t.is_cuda for t in (query, key, value)):
            return _dense_fallback("seed_only_batched_requires_cuda")
        try:
            from .kernels.gate0_seed_only_triton import (
                gate0_seed_only_attention_triton_forward_out,
            )

            output = (
                output_workspace
                if output_workspace is not None
                and output_workspace.shape == query.shape
                and output_workspace.device == query.device
                and output_workspace.dtype == query.dtype
                and output_workspace.is_contiguous()
                else torch.empty_like(query)
            )
            gate0_seed_only_attention_triton_forward_out(
                query,
                key,
                value,
                output,
                block_size=seed_policy.block_size,
                sink_blocks=seed_policy.sink_blocks,
                recent_blocks=seed_policy.recent_blocks,
                middle_seed_blocks=seed_policy.middle_seed_blocks,
                block_order=seed_policy.block_order,
                num_warps=seed_policy.num_warps,
                num_stages=seed_policy.num_stages,
            )
        except Exception as exc:  # pragma: no cover - CUDA/Triton environment dependent
            if seed_policy.fallback != "dense":
                raise
            return _dense_fallback(f"seed_only_batched_backend_error:{type(exc).__name__}")
        if not return_info:
            return output
        total_blocks = (key.shape[1] + seed_policy.block_size - 1) // seed_policy.block_size
        seed_blocks = (
            seed_policy.sink_blocks
            + seed_policy.recent_blocks
            + seed_policy.middle_seed_blocks
        )
        info = Gate0SeedOnlyBatchedRunInfo(
            policy=seed_policy,
            stats=Gate0SeedOnlyBatchedStats(
                batch_size=query.shape[0],
                heads=query.shape[2],
                kv_heads=key.shape[2],
                kv_len=key.shape[1],
                block_size=seed_policy.block_size,
                seed_blocks=min(seed_blocks, total_blocks),
                total_blocks=total_blocks,
            ),
            backend_used="gate0_seed_only_batched",
            fallback_reason=None,
        )
        return output, info
    if plan.backend not in {"gate1_mass", "gate1_value_bound"}:
        raise ValueError(f"unknown decode backend: {plan.backend}")
    if plan.backend == "gate1_value_bound" and metadata is None:
        raise ValueError("value-bound decode plan requires metadata")
    return stream_attn_gate1(
        query,
        key,
        value,
        causal=False,
        mode="gate1",
        metadata=metadata,
        skip_predicate="mass" if plan.backend == "gate1_mass" else "value_bound",
        error_budget=error_budget,
        block_size=plan.block_size,
        tile_size_q=plan.tile_size_q,
        telemetry=return_info and plan.collect_telemetry,
        return_info=return_info,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )


def _shape_tuple(shape_or_tensor) -> tuple:
    if isinstance(shape_or_tensor, torch.Tensor):
        return tuple(shape_or_tensor.shape)
    return tuple(shape_or_tensor)


class StreamAttnDecodeWrapper:
    """Stateful planned decode wrapper for contiguous KV cache calls.

    The wrapper owns runtime state such as step index and previous active
    fraction. It delegates cost decisions and kernel launches to
    ``stream_attn_decode_plan`` and ``stream_attn_decode_run``.
    """

    def __init__(
        self,
        workspace: StreamAttnDecodeWorkspace,
        *,
        policy: Optional[StreamAttnDecodePolicy] = None,
        decode_cost_model: Optional[DecodeCostModel] = None,
        router: Optional[StreamAttnRouter] = None,
        gate0_fused_hybrid_policy: Optional[Gate0FusedHybridPolicy] = None,
        gate0_seed_only_batched_policy: Optional[Gate0SeedOnlyBatchedPolicy] = None,
        gate0_projection_metadata: Optional[Gate0ProjectionMetadata] = None,
        dense_fallback: Optional[DenseFallbackFn] = None,
        dense_fallback_backend: str = "dense",
    ) -> None:
        self.workspace = workspace
        self.policy = policy or StreamAttnDecodePolicy()
        self.decode_cost_model = decode_cost_model or DecodeCostModel()
        self.router = router
        self.gate0_fused_hybrid_policy = gate0_fused_hybrid_policy
        self.gate0_seed_only_batched_policy = gate0_seed_only_batched_policy
        self.gate0_projection_metadata = gate0_projection_metadata
        self.dense_fallback = dense_fallback
        self.dense_fallback_backend = dense_fallback_backend
        self.gate0_workspace: Optional[Dict[str, torch.Tensor]] = None
        self.last_active_fraction: Optional[float] = None
        self.last_plan: Optional[StreamAttnDecodePlan] = None
        self.last_info = None
        self._static_plan: Optional[Dict[str, object]] = None
        self._last_plan_metadata_available: Optional[bool] = None
        self.backend_counts: Dict[str, int] = {}
        self.fallback_reasons: Dict[str, int] = {}

    def plan(
        self,
        *,
        query_shape,
        kv_shape,
        attention_type: str = "mha",
        kv_heads: Optional[int] = None,
        block_size: Optional[int] = None,
        tile_size_q: int = 16,
        num_warps: int = 4,
        num_stages: int = 3,
        error_budget: float = 1.0e-3,
    ) -> "StreamAttnDecodeWrapper":
        """Store static shape/config state without launching kernels."""

        query_shape = _shape_tuple(query_shape)
        kv_shape = _shape_tuple(kv_shape)
        if len(query_shape) != 4 or len(kv_shape) != 4:
            raise ValueError("query_shape and kv_shape must be [batch, seq, heads, dim]")
        if query_shape[0] != kv_shape[0] or query_shape[3] != kv_shape[3]:
            raise ValueError("query/KV batch and dim must match")
        block_size = int(block_size or self.workspace.block_size)
        kv_heads = int(kv_heads if kv_heads is not None else kv_shape[2])
        self.workspace.validate(
            batch=int(query_shape[0]),
            query_len=int(query_shape[1]),
            kv_len=int(kv_shape[1]),
            heads=max(int(query_shape[2]), int(kv_shape[2])),
            head_dim=int(query_shape[3]),
            device=self.workspace.device,
        )
        self._static_plan = {
            "query_shape": query_shape,
            "kv_shape": kv_shape,
            "attention_type": attention_type,
            "kv_heads": kv_heads,
            "block_size": block_size,
            "tile_size_q": int(tile_size_q),
            "num_warps": int(num_warps),
            "num_stages": int(num_stages),
            "error_budget": float(error_budget),
        }
        self.last_plan = None
        self.last_info = None
        self.last_active_fraction = None
        self._last_plan_metadata_available = None
        self.reset_runtime_counters()
        self.workspace.reset_step()
        return self

    def observe_active_fraction(self, active_fraction: float) -> None:
        self.last_active_fraction = min(1.0, max(0.0, float(active_fraction)))

    def reset_runtime_counters(self) -> None:
        """Clear backend/fallback counters collected by :meth:`run`."""

        self.backend_counts.clear()
        self.fallback_reasons.clear()

    def runtime_counters(self) -> Dict[str, Dict[str, int]]:
        """Return a snapshot of backend and fallback routing telemetry."""

        return {
            "backend_counts": dict(self.backend_counts),
            "fallback_reasons": dict(self.fallback_reasons),
        }

    def _record_runtime_info(self, plan: StreamAttnDecodePlan, info) -> None:
        backend_used = getattr(info, "backend_used", None) if info is not None else None
        fallback_reason = getattr(info, "fallback_reason", None) if info is not None else None
        if fallback_reason is None and plan.backend == "dense":
            fallback_reason = plan.fallback_reason
        if backend_used is None:
            backend_used = (
                self.dense_fallback_backend
                if plan.backend == "dense" and self.dense_fallback is not None
                else plan.backend
            )
        self.backend_counts[str(backend_used)] = self.backend_counts.get(str(backend_used), 0) + 1
        if fallback_reason:
            reason = str(fallback_reason)
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def _require_plan(self) -> Dict[str, object]:
        if self._static_plan is None:
            raise RuntimeError("call plan(...) before run(...)")
        return self._static_plan

    def _check_static_tensors(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: Optional[torch.Tensor] = None,
    ) -> None:
        static = self._require_plan()
        if tuple(query.shape) != tuple(static["query_shape"]):
            raise ValueError("query shape does not match planned query_shape")
        if tuple(key_cache.shape) != tuple(static["kv_shape"]):
            raise ValueError("key_cache shape does not match planned kv_shape")
        if value_cache is not None and tuple(value_cache.shape) != tuple(static["kv_shape"]):
            raise ValueError("value_cache shape does not match planned kv_shape")
        if query.device != self.workspace.device or key_cache.device != self.workspace.device:
            raise ValueError("query/key device does not match workspace device")
        if value_cache is not None and value_cache.device != self.workspace.device:
            raise ValueError("value device does not match workspace device")

    def plan_step(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        *,
        metadata: Optional[StreamAttnMetadataCache] = None,
        active_fraction_hint: Optional[float] = None,
        _validated: bool = False,
    ) -> StreamAttnDecodePlan:
        """Build a per-step decode plan from stored static config."""

        static = self._require_plan()
        if not _validated:
            self._check_static_tensors(query, key_cache)
        hint = active_fraction_hint
        if hint is None:
            hint = self.last_active_fraction
        metadata_available = metadata is not None and metadata.value_norm_bounds is not None
        if (
            self.policy.collect_telemetry_every == 0
            and self.last_plan is not None
            and hint is not None
            and self._last_plan_metadata_available == metadata_available
            and abs(self.last_plan.predicted_active_fraction - float(hint)) <= 1.0e-12
        ):
            return self.last_plan
        plan = stream_attn_decode_plan(
            query,
            key_cache,
            metadata=metadata,
            gate0_fused_hybrid_policy=self.gate0_fused_hybrid_policy,
            gate0_seed_only_batched_policy=self.gate0_seed_only_batched_policy,
            gate0_projection_metadata=self.gate0_projection_metadata,
            router=self.router,
            decode_cost_model=self.decode_cost_model,
            policy=self.policy,
            active_fraction_hint=hint,
            attention_type=str(static["attention_type"]),
            kv_heads=int(static["kv_heads"]),
            block_size=int(static["block_size"]),
            tile_size_q=int(static["tile_size_q"]),
            num_warps=int(static["num_warps"]),
            num_stages=int(static["num_stages"]),
            error_budget=float(static["error_budget"]),
            step_index=self.workspace.step_index,
        )
        self.last_plan = plan
        self._last_plan_metadata_available = metadata_available
        return plan

    def run(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        metadata: Optional[StreamAttnMetadataCache] = None,
        active_fraction_hint: Optional[float] = None,
        return_info: bool = False,
    ):
        """Plan and execute one decode step."""

        static = self._require_plan()
        self._check_static_tensors(query, key_cache, value_cache)
        plan = self.plan_step(
            query,
            key_cache,
            metadata=metadata,
            active_fraction_hint=active_fraction_hint,
            _validated=True,
        )
        wants_info = return_info or (
            plan.backend != "dense"
            and (plan.collect_telemetry or self.last_active_fraction is None)
        )
        gate0_metadata = self.gate0_projection_metadata
        if plan.backend == "gate0_fused_hybrid" and gate0_metadata is None:
            if self.gate0_fused_hybrid_policy is None:
                raise ValueError("gate0_fused_hybrid plan requires a policy")
            gate0_metadata = build_gate0_projection_metadata(
                key_cache,
                self.gate0_fused_hybrid_policy,
            )
            self.gate0_projection_metadata = gate0_metadata
        if (
            plan.backend == "gate0_fused_hybrid"
            and self.gate0_workspace is None
            and self.gate0_fused_hybrid_policy is not None
            and query.is_cuda
        ):
            self.gate0_workspace = make_gate0_fused_hybrid_workspace(
                query,
                self.gate0_fused_hybrid_policy,
            )
        result = stream_attn_decode_run(
            query,
            key_cache,
            value_cache,
            plan=plan,
            metadata=metadata,
            gate0_projection_metadata=gate0_metadata,
            gate0_workspace=self.gate0_workspace,
            output_workspace=self.workspace.output,
            dense_fallback=self.dense_fallback,
            dense_fallback_backend=self.dense_fallback_backend,
            error_budget=float(static["error_budget"]),
            return_info=wants_info,
        )
        if wants_info:
            output, info = result
        else:
            output, info = result, None
        self.last_info = info
        self._record_runtime_info(plan, info)
        if info is not None and info.stats is not None:
            if hasattr(info.stats, "active_pv_fraction"):
                self.last_active_fraction = float(info.stats.active_pv_fraction)
            elif hasattr(info.stats, "pv_executed_fraction"):
                self.last_active_fraction = float(info.stats.pv_executed_fraction)
            elif hasattr(info.stats, "seed_fraction"):
                self.last_active_fraction = float(info.stats.seed_fraction)
        self.workspace.advance_step()
        return (output, info) if return_info else output


class StreamAttnSeedOnlyDecodeService:
    """Serving-facing wrapper for the packaged seed-only StreamAttn route.

    The service keeps the first deployable wedge narrow: it loads one validated
    seed-only policy, reuses decode workspace for stable request shapes, and
    fails closed to dense whenever policy/runtime invariants do not match.
    """

    def __init__(
        self,
        *,
        policy: Optional[Gate0SeedOnlyBatchedPolicy] = None,
        policy_name: str = DEFAULT_GATE0_SEED_ONLY_BATCHED_POLICY,
        decode_policy: Optional[StreamAttnDecodePolicy] = None,
        dense_fallback: Optional[DenseFallbackFn] = None,
        dense_fallback_backend: str = "dense",
        model_id: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        self.policy = policy or load_packaged_gate0_seed_only_batched_policy(policy_name)
        self.decode_policy = decode_policy or StreamAttnDecodePolicy(
            collect_telemetry_every=0,
        )
        self.dense_fallback = dense_fallback
        self.dense_fallback_backend = dense_fallback_backend
        self.model_id = model_id if model_id is not None else self.policy.model_id
        self.layer_id = layer_id if layer_id is not None else self.policy.layer_id
        self.workspace: Optional[StreamAttnDecodeWorkspace] = None
        self.wrapper: Optional[StreamAttnDecodeWrapper] = None
        self._planned_shape: Optional[tuple] = None

    @classmethod
    def from_packaged(
        cls,
        name: str = DEFAULT_GATE0_SEED_ONLY_BATCHED_POLICY,
        **kwargs,
    ) -> "StreamAttnSeedOnlyDecodeService":
        """Build a serving service from a packaged policy artifact."""

        return cls(policy=load_packaged_gate0_seed_only_batched_policy(name), **kwargs)

    def _preflight_reasons(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        *,
        model_id: Optional[str],
        layer_id: Optional[int],
        mode: str,
    ) -> List[str]:
        if mode not in {"auto", "dense"}:
            raise ValueError("mode must be 'auto' or 'dense'")
        if mode == "dense":
            return ["manual_dense"]
        safety_reasons = self.policy.mismatch_reasons(
            query,
            key_cache,
            min_kv_len=self.decode_policy.min_kv_len_for_gate0_seed_only,
            model_id=model_id,
            layer_id=layer_id,
        )
        if safety_reasons:
            return safety_reasons
        if not all(t.is_cuda for t in (query, key_cache)):
            return ["backend_unavailable"]
        return []

    def _allocate_wrapper(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> StreamAttnDecodeWrapper:
        shape_key = (tuple(query.shape), tuple(key_cache.shape), tuple(value_cache.shape))
        if self.wrapper is not None and self._planned_shape == shape_key:
            return self.wrapper
        self.workspace = StreamAttnDecodeWorkspace.allocate(
            device=query.device,
            max_batch=query.shape[0],
            max_query_len=query.shape[1],
            max_kv_len=key_cache.shape[1],
            max_heads=max(query.shape[2], key_cache.shape[2]),
            head_dim=query.shape[3],
            block_size=self.policy.block_size,
            dtype=query.dtype,
        )
        self.wrapper = StreamAttnDecodeWrapper(
            self.workspace,
            policy=self.decode_policy,
            gate0_seed_only_batched_policy=self.policy,
            dense_fallback=self.dense_fallback,
            dense_fallback_backend=self.dense_fallback_backend,
        )
        self.wrapper.plan(
            query_shape=query.shape,
            kv_shape=key_cache.shape,
            block_size=self.policy.block_size,
            tile_size_q=16,
            num_warps=self.policy.num_warps,
            num_stages=self.policy.num_stages,
        )
        self._planned_shape = shape_key
        return self.wrapper

    def _dense_output(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        if self.dense_fallback is not None:
            return self.dense_fallback(query, key_cache, value_cache)
        return dense_attention_forward(query, key_cache, value_cache, causal=False)

    def _serving_info(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        model_id: Optional[str],
        layer_id: Optional[int],
        backend_used: str,
        fallback_reason: Optional[str],
        plan_backend: str,
        plan_reason: str,
        safety_policy_matched: bool,
        runtime_counters: Optional[Dict[str, Dict[str, int]]] = None,
        stats: Optional[object] = None,
    ) -> StreamAttnServingInfo:
        return StreamAttnServingInfo(
            backend_used=backend_used,
            policy_id=self.policy.policy_id,
            fallback_reason=fallback_reason,
            batch_size=int(query.shape[0]),
            kv_len=int(key_cache.shape[1]),
            layer_id=layer_id,
            model_id=model_id,
            dtype=_normalize_dtype_name(query.dtype) or str(query.dtype),
            device=str(query.device),
            plan_backend=plan_backend,
            plan_reason=plan_reason,
            seed_only_enabled=backend_used == "gate0_seed_only_batched",
            safety_policy_matched=safety_policy_matched,
            runtime_counters=runtime_counters or {
                "backend_counts": {backend_used: 1},
                "fallback_reasons": (
                    {fallback_reason: 1} if fallback_reason is not None else {}
                ),
            },
            stats=asdict(stats) if stats is not None else None,
        )

    def run(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        model_id: Optional[str] = None,
        layer_id: Optional[int] = None,
        mode: str = "auto",
        active_fraction_hint: Optional[float] = 1.0,
        return_info: bool = True,
    ):
        """Run one decode call with policy-aware seed-only routing.

        ``mode='auto'`` uses the packaged policy if all invariants match and the
        backend is available.  ``mode='dense'`` forces dense fallback while still
        returning structured telemetry.
        """

        if query.dim() != 4 or key_cache.dim() != 4 or value_cache.dim() != 4:
            raise ValueError("query, key_cache, and value_cache must be [batch, seq, heads, dim]")
        if key_cache.shape != value_cache.shape:
            raise ValueError("key_cache and value_cache shapes must match")
        if query.shape[0] != key_cache.shape[0] or query.shape[3] != key_cache.shape[3]:
            raise ValueError("query/KV batch and head_dim must match")
        effective_model_id = model_id if model_id is not None else self.model_id
        effective_layer_id = layer_id if layer_id is not None else self.layer_id
        safety_reasons = self.policy.mismatch_reasons(
            query,
            key_cache,
            min_kv_len=self.decode_policy.min_kv_len_for_gate0_seed_only,
            model_id=effective_model_id,
            layer_id=effective_layer_id,
        )
        reasons = self._preflight_reasons(
            query,
            key_cache,
            model_id=effective_model_id,
            layer_id=effective_layer_id,
            mode=mode,
        )
        if reasons:
            reason = reasons[0]
            output = self._dense_output(query, key_cache, value_cache)
            backend = self.dense_fallback_backend if self.dense_fallback is not None else "dense"
            info = self._serving_info(
                query=query,
                key_cache=key_cache,
                model_id=effective_model_id,
                layer_id=effective_layer_id,
                backend_used=backend,
                fallback_reason=reason,
                plan_backend="dense",
                plan_reason=reason,
                safety_policy_matched=not safety_reasons,
            )
            return (output, info) if return_info else output

        wrapper = self._allocate_wrapper(query, key_cache, value_cache)
        output, run_info = wrapper.run(
            query,
            key_cache,
            value_cache,
            active_fraction_hint=active_fraction_hint,
            return_info=True,
        )
        plan = wrapper.last_plan
        assert plan is not None
        backend = getattr(run_info, "backend_used", None) or plan.backend
        fallback_reason = getattr(run_info, "fallback_reason", None) or plan.fallback_reason
        info = self._serving_info(
            query=query,
            key_cache=key_cache,
            model_id=effective_model_id,
            layer_id=effective_layer_id,
            backend_used=str(backend),
            fallback_reason=fallback_reason,
            plan_backend=plan.backend,
            plan_reason=plan.reason,
            safety_policy_matched=True,
            runtime_counters=wrapper.runtime_counters(),
            stats=getattr(run_info, "stats", None),
        )
        return (output, info) if return_info else output


def stream_attn_seed_only_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    policy: Optional[Gate0SeedOnlyBatchedPolicy] = None,
    policy_name: str = DEFAULT_GATE0_SEED_ONLY_BATCHED_POLICY,
    decode_policy: Optional[StreamAttnDecodePolicy] = None,
    dense_fallback: Optional[DenseFallbackFn] = None,
    dense_fallback_backend: str = "dense",
    model_id: Optional[str] = None,
    layer_id: Optional[int] = None,
    mode: str = "auto",
    active_fraction_hint: Optional[float] = 1.0,
    return_info: bool = True,
):
    """One-shot serving helper for policy-aware seed-only decode."""

    service = StreamAttnSeedOnlyDecodeService(
        policy=policy,
        policy_name=policy_name,
        decode_policy=decode_policy,
        dense_fallback=dense_fallback,
        dense_fallback_backend=dense_fallback_backend,
        model_id=model_id,
        layer_id=layer_id,
    )
    return service.run(
        query,
        key_cache,
        value_cache,
        model_id=model_id,
        layer_id=layer_id,
        mode=mode,
        active_fraction_hint=active_fraction_hint,
        return_info=return_info,
    )
