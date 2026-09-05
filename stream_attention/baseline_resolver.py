"""Semantic capability resolution for exact-attention comparison backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml

from .inference_workload import AttentionBatchV2, RequestPhase


BASELINE_DESCRIPTOR_SCHEMA_VERSION = 1
DEFAULT_BASELINE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "manifests"
    / "exact_baselines_v2.yaml"
)


@dataclass(frozen=True)
class ExactBaselineDescriptor:
    """Declared direct-call surface for one exact baseline implementation."""

    baseline_id: str
    implementation: str
    revision: str
    architectures: frozenset[str]
    phases: frozenset[str]
    attention_kinds: frozenset[str]
    q_dtypes: frozenset[str]
    kv_dtypes: frozenset[str]
    output_dtypes: frozenset[str]
    scale_formats: frozenset[str]
    d_qk: frozenset[int]
    d_v: frozenset[int]
    cache_kinds: frozenset[str]
    cache_layouts: frozenset[str]
    mask_kinds: frozenset[str]
    page_sizes: frozenset[int] = field(default_factory=frozenset)
    execution_modes: frozenset[str] = field(default_factory=frozenset)
    supported_features: frozenset[str] = field(default_factory=frozenset)
    supports_mixed_batches: bool = False
    supports_ragged_batches: bool = False
    supports_shared_prefixes: bool = False
    supports_speculative_trees: bool = False
    direct_layout: bool = True

    def __post_init__(self) -> None:
        if not self.baseline_id or not self.implementation or not self.revision:
            raise ValueError("baseline identifiers and revision must be non-empty")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExactBaselineDescriptor":
        def strings(name: str) -> frozenset[str]:
            return frozenset(str(value) for value in raw.get(name, ()))

        return cls(
            baseline_id=str(raw["baseline_id"]),
            implementation=str(raw["implementation"]),
            revision=str(raw["revision"]),
            architectures=strings("architectures"),
            phases=strings("phases"),
            attention_kinds=strings("attention_kinds"),
            q_dtypes=strings("q_dtypes"),
            kv_dtypes=strings("kv_dtypes"),
            output_dtypes=strings("output_dtypes"),
            scale_formats=strings("scale_formats"),
            d_qk=frozenset(int(value) for value in raw.get("d_qk", ())),
            d_v=frozenset(int(value) for value in raw.get("d_v", ())),
            cache_kinds=strings("cache_kinds"),
            cache_layouts=strings("cache_layouts"),
            mask_kinds=strings("mask_kinds"),
            page_sizes=frozenset(int(value) for value in raw.get("page_sizes", ())),
            execution_modes=strings("execution_modes"),
            supported_features=strings("supported_features"),
            supports_mixed_batches=bool(raw.get("supports_mixed_batches", False)),
            supports_ragged_batches=bool(raw.get("supports_ragged_batches", False)),
            supports_shared_prefixes=bool(raw.get("supports_shared_prefixes", False)),
            supports_speculative_trees=bool(raw.get("supports_speculative_trees", False)),
            direct_layout=bool(raw.get("direct_layout", True)),
        )


@dataclass(frozen=True)
class BaselineEligibility:
    baseline_id: str
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExactBaselineMeasurement:
    baseline_id: str
    backend_revision: str
    workload_sha256: str
    environment_sha256: str
    latency_us: float
    correctness_passed: bool
    graph_replay: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_us) or self.latency_us <= 0:
            raise ValueError("baseline latency must be finite and positive")
        if not self.baseline_id:
            raise ValueError("baseline_id must be non-empty")
        if not self.backend_revision:
            raise ValueError("backend_revision must be non-empty")
        _validate_sha256(self.workload_sha256, "workload_sha256")
        _validate_sha256(self.environment_sha256, "environment_sha256")


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")


def _validate_unique_descriptor_ids(
    descriptors: tuple[ExactBaselineDescriptor, ...],
) -> None:
    ids = [descriptor.baseline_id for descriptor in descriptors]
    if len(set(ids)) != len(ids):
        raise ValueError("exact-baseline descriptor IDs must be unique")


def _unsupported(value: object, supported: frozenset[object], reason: str) -> str | None:
    return None if not supported or value in supported else reason


def resolve_direct_exact_baseline(
    workload: AttentionBatchV2,
    descriptor: ExactBaselineDescriptor,
) -> BaselineEligibility:
    """Return explicit incompatibility reasons for one direct exact backend."""

    reasons = [
        reason
        for reason in (
            _unsupported(workload.architecture, descriptor.architectures, "architecture"),
            _unsupported(workload.phase.value, descriptor.phases, "phase"),
            _unsupported(
                workload.attention_kind.value,
                descriptor.attention_kinds,
                "attention_kind",
            ),
            _unsupported(workload.q_dtype, descriptor.q_dtypes, "q_dtype"),
            _unsupported(workload.kv_dtype, descriptor.kv_dtypes, "kv_dtype"),
            _unsupported(workload.output_dtype, descriptor.output_dtypes, "output_dtype"),
            _unsupported(workload.scale_format, descriptor.scale_formats, "scale_format"),
            _unsupported(workload.d_qk, descriptor.d_qk, "d_qk"),
            _unsupported(workload.d_v, descriptor.d_v, "d_v"),
            _unsupported(workload.cache_kind.value, descriptor.cache_kinds, "cache_kind"),
            _unsupported(
                workload.cache_layout.value, descriptor.cache_layouts, "cache_layout"
            ),
            _unsupported(workload.mask_kind.value, descriptor.mask_kinds, "mask_kind"),
            _unsupported(
                workload.execution_mode.value,
                descriptor.execution_modes,
                "execution_mode",
            ),
        )
        if reason is not None
    ]
    if workload.cache_kind.value == "paged":
        page_reason = _unsupported(workload.page_size, descriptor.page_sizes, "page_size")
        if page_reason:
            reasons.append(page_reason)
    if workload.phase is RequestPhase.MIXED and not descriptor.supports_mixed_batches:
        reasons.append("mixed_batch")
    if workload.is_ragged and not descriptor.supports_ragged_batches:
        reasons.append("ragged_batch")
    if workload.has_shared_prefixes and not descriptor.supports_shared_prefixes:
        reasons.append("shared_prefix")
    if any(request.speculative_tree_parents for request in workload.requests):
        if not descriptor.supports_speculative_trees:
            reasons.append("speculative_tree")
    unsupported = workload.semantic_features - descriptor.supported_features
    if unsupported:
        reasons.extend(f"unsupported_feature:{feature}" for feature in sorted(unsupported))
    if not descriptor.direct_layout:
        reasons.append("requires_layout_conversion")
    unique = tuple(sorted(set(reasons)))
    return BaselineEligibility(descriptor.baseline_id, not unique, unique)


def resolve_direct_exact_baselines(
    workload: AttentionBatchV2,
    descriptors: Iterable[ExactBaselineDescriptor],
) -> tuple[BaselineEligibility, ...]:
    descriptor_rows = tuple(descriptors)
    _validate_unique_descriptor_ids(descriptor_rows)
    return tuple(
        resolve_direct_exact_baseline(workload, descriptor)
        for descriptor in descriptor_rows
    )


def fastest_measured_exact_baseline(
    workload: AttentionBatchV2,
    descriptors: Iterable[ExactBaselineDescriptor],
    measurements: Iterable[ExactBaselineMeasurement],
    *,
    expected_environment_sha256: str | None = None,
) -> ExactBaselineMeasurement | None:
    """Select eligible, correct measurements from a single environment.

    An expected environment filters measurements to that digest. Without one,
    otherwise eligible candidates must agree on their environment. Descriptor
    IDs must be unique, including across revisions; repeated measurements of
    the declared revision are allowed.
    """

    if expected_environment_sha256 is not None:
        _validate_sha256(expected_environment_sha256, "expected_environment_sha256")
        expected_environment_sha256 = expected_environment_sha256.lower()
    descriptor_rows = tuple(descriptors)
    eligibility = {
        row.baseline_id: row.eligible
        for row in resolve_direct_exact_baselines(workload, descriptor_rows)
    }
    revisions = {descriptor.baseline_id: descriptor.revision for descriptor in descriptor_rows}
    workload_sha256 = workload.fingerprint
    candidates = [
        measurement
        for measurement in measurements
        if measurement.workload_sha256.lower() == workload_sha256
        and measurement.correctness_passed
        and eligibility.get(measurement.baseline_id, False)
        and measurement.backend_revision == revisions.get(measurement.baseline_id)
        and (
            workload.execution_mode.value != "cuda_graph" or measurement.graph_replay
        )
        and (
            expected_environment_sha256 is None
            or measurement.environment_sha256.lower() == expected_environment_sha256
        )
    ]
    if len({measurement.environment_sha256.lower() for measurement in candidates}) > 1:
        raise ValueError(
            "eligible baseline measurements span multiple environments; "
            "provide expected_environment_sha256"
        )
    return min(candidates, key=lambda measurement: measurement.latency_us, default=None)


def load_exact_baseline_descriptors(
    path: Path | None = None,
) -> tuple[ExactBaselineDescriptor, ...]:
    source = DEFAULT_BASELINE_MANIFEST if path is None else path
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != BASELINE_DESCRIPTOR_SCHEMA_VERSION:
        raise ValueError("unsupported exact-baseline descriptor schema")
    descriptors = tuple(
        ExactBaselineDescriptor.from_dict(row) for row in raw.get("baselines", ())
    )
    _validate_unique_descriptor_ids(descriptors)
    if not descriptors:
        raise ValueError("exact-baseline manifest must contain at least one descriptor")
    return descriptors


__all__ = [
    "BASELINE_DESCRIPTOR_SCHEMA_VERSION",
    "DEFAULT_BASELINE_MANIFEST",
    "BaselineEligibility",
    "ExactBaselineDescriptor",
    "ExactBaselineMeasurement",
    "fastest_measured_exact_baseline",
    "load_exact_baseline_descriptors",
    "resolve_direct_exact_baseline",
    "resolve_direct_exact_baselines",
]
