"""Planned decode path for StreamAttn Gate-1 routing.

This module keeps decode routing separate from kernel execution. The first
version targets contiguous KV cache tensors and top-level ``causal=False``
decode mechanics, where the KV cache already contains only valid past/current
tokens.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import torch

from .certified import StreamAttnMetadataCache
from .gate1 import dense_attention_forward, make_route_request, stream_attn_gate1
from .router import StreamAttnRouter, normalize_device_class
from .telemetry import Prediction, seq_bucket


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
        for name in ("max_active_fraction_mass", "max_active_fraction_value_bound"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")


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

    if cost is None:
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
        )

    dense_ms = cost.dense_ms
    candidates = {"dense": dense_ms}
    reasons = {"dense": "dense_fallback"}
    metadata_available = metadata is not None and metadata.value_norm_bounds is not None
    active_threshold = cost.break_even_active_fraction(
        safety_margin=policy.safety_margin,
    )

    if (
        policy.allow_mass
        and key.shape[1] >= policy.min_kv_len_for_gate1
        and prediction.confidence >= policy.min_confidence
        and active <= min(policy.max_active_fraction_mass, active_threshold)
    ):
        mass_ms = cost.predict_mass_ms(active)
        if mass_ms * policy.safety_margin < dense_ms:
            candidates["gate1_mass"] = mass_ms
            reasons["gate1_mass"] = "calibrated_mass_profitable"

    if (
        policy.allow_value_bound
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
        metadata_update_required=False,
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
    )


def stream_attn_decode_run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    plan: StreamAttnDecodePlan,
    metadata: Optional[StreamAttnMetadataCache] = None,
    error_budget: float = 1.0e-3,
    return_info: bool = False,
):
    """Execute a decode plan produced by :func:`stream_attn_decode_plan`."""

    if plan.backend == "dense":
        output = dense_attention_forward(query, key, value, causal=False)
        return (output, None) if return_info else output
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
