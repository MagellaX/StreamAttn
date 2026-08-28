"""Evidence resolution and phase database for the universal exact compiler.

The phase database is deliberately compiled from immutable measurement records.
It never benchmarks kernels itself.  This separation lets profilers evolve while
the evidence contract remains strict about correctness, backend resolution,
timed allocations, negative results, and environment identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .exact_compiler import (
    ExactWorkloadCell,
    KernelFamily,
    UniversalExactManifest,
    matching_kernel_families,
    registered_exact_kernel_families,
)


PHASE_DATABASE_SCHEMA_VERSION = 1
EVIDENCE_PROVIDERS = frozenset({"streamattn", "external"})


class MeasurementStatus(str, Enum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    INVALID = "invalid"


class PhaseEntryStatus(str, Enum):
    NATIVE = "native"
    EXTERNAL_FALLBACK = "external_fallback"
    INCOMPLETE_BASELINES = "incomplete_baselines"
    NO_CORRECT_BASELINE = "no_correct_baseline"
    NATIVE_UNMEASURED = "native_unmeasured"


def _require_finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class EnvironmentFingerprint:
    architecture: str
    device_name: str
    device_uuid: str
    driver_version: str
    cuda_version: str
    torch_version: str
    library_versions: Mapping[str, str]
    compiler_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.architecture not in {"sm80", "sm90", "sm100"}:
            raise ValueError(f"unsupported environment architecture: {self.architecture}")
        for name, value in (
            ("device_name", self.device_name),
            ("device_uuid", self.device_uuid),
            ("driver_version", self.driver_version),
            ("cuda_version", self.cuda_version),
            ("torch_version", self.torch_version),
        ):
            if not value:
                raise ValueError(f"environment {name} must be non-empty")
        if any(not key or not value for key, value in self.library_versions.items()):
            raise ValueError("library version names and values must be non-empty")
        if any(not key or not value for key, value in self.compiler_versions.items()):
            raise ValueError("compiler version names and values must be non-empty")

    @property
    def fingerprint_id(self) -> str:
        digest = hashlib.sha256(_canonical_json(self.as_dict()).encode()).hexdigest()
        return digest[:20]

    @property
    def compatibility_id(self) -> str:
        payload = self.as_dict()
        payload.pop("device_uuid")
        # Optional baseline libraries may be unqueried by phase-specific workers.
        # Pairwise compatibility below still rejects any conflicting recorded value.
        payload.pop("library_versions")
        digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        return digest[:20]

    def compatible_with(self, other: "EnvironmentFingerprint") -> bool:
        if (
            self.architecture,
            self.device_name,
            self.driver_version,
            self.cuda_version,
            self.torch_version,
        ) != (
            other.architecture,
            other.device_name,
            other.driver_version,
            other.cuda_version,
            other.torch_version,
        ):
            return False

        def mappings_agree(
            first: Mapping[str, str], second: Mapping[str, str]
        ) -> bool:
            return all(
                first[name] == second[name] for name in first.keys() & second.keys()
            )

        return mappings_agree(
            self.library_versions, other.library_versions
        ) and mappings_agree(self.compiler_versions, other.compiler_versions)

    def as_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "device_name": self.device_name,
            "device_uuid": self.device_uuid,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "library_versions": dict(sorted(self.library_versions.items())),
            "compiler_versions": dict(sorted(self.compiler_versions.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentFingerprint":
        return cls(
            architecture=str(data["architecture"]),
            device_name=str(data["device_name"]),
            device_uuid=str(data["device_uuid"]),
            driver_version=str(data["driver_version"]),
            cuda_version=str(data["cuda_version"]),
            torch_version=str(data["torch_version"]),
            library_versions={
                str(key): str(value)
                for key, value in dict(data.get("library_versions", {})).items()
            },
            compiler_versions={
                str(key): str(value)
                for key, value in dict(data.get("compiler_versions", {})).items()
            },
        )


@dataclass(frozen=True)
class TimingEvidence:
    cold_ms: float
    p10_ms: float
    p50_ms: float
    p90_ms: float
    variance_ms2: float
    process_count: int
    sample_count: int
    timed_allocation_count: int
    confidence: float

    def __post_init__(self) -> None:
        for name, value in (
            ("cold_ms", self.cold_ms),
            ("p10_ms", self.p10_ms),
            ("p50_ms", self.p50_ms),
            ("p90_ms", self.p90_ms),
            ("variance_ms2", self.variance_ms2),
        ):
            _require_finite_non_negative(float(value), name)
        if self.p10_ms <= 0.0 or self.p50_ms <= 0.0 or self.p90_ms <= 0.0:
            raise ValueError("timing quantiles must be positive")
        if not self.p10_ms <= self.p50_ms <= self.p90_ms:
            raise ValueError("timing quantiles must satisfy p10 <= p50 <= p90")
        if self.process_count <= 0 or self.sample_count <= 0:
            raise ValueError("timing process_count and sample_count must be positive")
        if self.timed_allocation_count < 0:
            raise ValueError("timed_allocation_count must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("timing confidence must be in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        return {
            "cold_ms": self.cold_ms,
            "p10_ms": self.p10_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "variance_ms2": self.variance_ms2,
            "process_count": self.process_count,
            "sample_count": self.sample_count,
            "timed_allocation_count": self.timed_allocation_count,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimingEvidence":
        return cls(
            cold_ms=float(data["cold_ms"]),
            p10_ms=float(data["p10_ms"]),
            p50_ms=float(data["p50_ms"]),
            p90_ms=float(data["p90_ms"]),
            variance_ms2=float(data["variance_ms2"]),
            process_count=int(data["process_count"]),
            sample_count=int(data["sample_count"]),
            timed_allocation_count=int(data["timed_allocation_count"]),
            confidence=float(data["confidence"]),
        )


@dataclass(frozen=True)
class CorrectnessEvidence:
    passed: bool
    reference: str
    checked_cases: int
    max_abs_error: float
    max_relative_error: float
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.reference:
            raise ValueError("correctness reference must be non-empty")
        if self.checked_cases <= 0:
            raise ValueError("correctness checked_cases must be positive")
        _require_finite_non_negative(self.max_abs_error, "max_abs_error")
        _require_finite_non_negative(self.max_relative_error, "max_relative_error")
        if not self.passed and not self.failure_reason:
            raise ValueError("failed correctness evidence requires failure_reason")

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reference": self.reference,
            "checked_cases": self.checked_cases,
            "max_abs_error": self.max_abs_error,
            "max_relative_error": self.max_relative_error,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorrectnessEvidence":
        return cls(
            passed=bool(data["passed"]),
            reference=str(data["reference"]),
            checked_cases=int(data["checked_cases"]),
            max_abs_error=float(data["max_abs_error"]),
            max_relative_error=float(data["max_relative_error"]),
            failure_reason=(
                None if data.get("failure_reason") is None else str(data["failure_reason"])
            ),
        )


@dataclass(frozen=True)
class BackendEvidence:
    evidence_id: str
    cell_id: str
    provider: str
    requested_backend: str
    resolved_backend: Optional[str]
    status: MeasurementStatus
    native: bool
    environment: EnvironmentFingerprint
    family_id: Optional[str] = None
    kernel_key: Optional[str] = None
    workspace_bytes: Optional[int] = None
    supported_range: Optional[Mapping[str, object]] = None
    timing: Optional[TimingEvidence] = None
    correctness: Optional[CorrectnessEvidence] = None
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.cell_id or not self.requested_backend:
            raise ValueError("evidence identifiers must be non-empty")
        if self.provider not in EVIDENCE_PROVIDERS:
            raise ValueError(f"unsupported evidence provider: {self.provider}")
        if self.status is MeasurementStatus.MEASURED:
            if not self.resolved_backend:
                raise ValueError("measured evidence requires resolved_backend")
            if self.timing is None or self.correctness is None:
                raise ValueError("measured evidence requires timing and correctness")
            if self.workspace_bytes is None or self.workspace_bytes < 0:
                raise ValueError("measured evidence requires non-negative workspace_bytes")
            if not self.supported_range:
                raise ValueError("measured evidence requires an explicit supported_range")
            if self.provider == "streamattn" and (not self.family_id or not self.kernel_key):
                raise ValueError("StreamAttn measurements require family_id and kernel_key")
        elif not self.detail:
            raise ValueError("non-measured evidence requires detail")
        if self.provider == "external" and self.native:
            raise ValueError("external evidence cannot be marked native")

    @property
    def is_routable(self) -> bool:
        return bool(
            self.status is MeasurementStatus.MEASURED
            and self.correctness is not None
            and self.correctness.passed
            and self.timing is not None
        )

    @property
    def is_usable(self) -> bool:
        return bool(
            self.is_routable
            and self.timing is not None
            and self.timing.timed_allocation_count == 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "cell_id": self.cell_id,
            "provider": self.provider,
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "status": self.status.value,
            "native": self.native,
            "environment": self.environment.as_dict(),
            "environment_id": self.environment.fingerprint_id,
            "environment_compatibility_id": self.environment.compatibility_id,
            "family_id": self.family_id,
            "kernel_key": self.kernel_key,
            "workspace_bytes": self.workspace_bytes,
            "supported_range": (
                None if self.supported_range is None else dict(self.supported_range)
            ),
            "timing": None if self.timing is None else self.timing.as_dict(),
            "correctness": (
                None if self.correctness is None else self.correctness.as_dict()
            ),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackendEvidence":
        timing = data.get("timing")
        correctness = data.get("correctness")
        return cls(
            evidence_id=str(data["evidence_id"]),
            cell_id=str(data["cell_id"]),
            provider=str(data["provider"]),
            requested_backend=str(data["requested_backend"]),
            resolved_backend=(
                None
                if data.get("resolved_backend") is None
                else str(data["resolved_backend"])
            ),
            status=MeasurementStatus(str(data["status"])),
            native=bool(data["native"]),
            environment=EnvironmentFingerprint.from_dict(dict(data["environment"])),
            family_id=None if data.get("family_id") is None else str(data["family_id"]),
            kernel_key=None if data.get("kernel_key") is None else str(data["kernel_key"]),
            workspace_bytes=(
                None if data.get("workspace_bytes") is None else int(data["workspace_bytes"])
            ),
            supported_range=(
                None
                if data.get("supported_range") is None
                else dict(data["supported_range"])
            ),
            timing=None if timing is None else TimingEvidence.from_dict(dict(timing)),
            correctness=(
                None
                if correctness is None
                else CorrectnessEvidence.from_dict(dict(correctness))
            ),
            detail=None if data.get("detail") is None else str(data["detail"]),
        )


def resolve_fastest_correct_baseline(
    cell: ExactWorkloadCell,
    evidence: Iterable[BackendEvidence],
) -> Optional[BackendEvidence]:
    """Return the fastest usable eligible external baseline for one cell."""

    eligible = set(cell.baseline_candidates)
    routable = [
        row
        for row in evidence
        if row.cell_id == cell.cell_id
        and row.provider == "external"
        and row.requested_backend in eligible
        and row.is_routable
    ]
    candidates = [row for row in routable if row.is_usable] or routable
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row.timing.p50_ms, row.evidence_id))  # type: ignore[union-attr]


def resolve_fastest_streamattn_candidate(
    cell: ExactWorkloadCell,
    evidence: Iterable[BackendEvidence],
) -> Optional[BackendEvidence]:
    routable = [
        row
        for row in evidence
        if row.cell_id == cell.cell_id
        and row.provider == "streamattn"
        and row.native
        and row.is_routable
    ]
    candidates = [row for row in routable if row.is_usable] or routable
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row.timing.p50_ms, row.evidence_id))  # type: ignore[union-attr]


@dataclass(frozen=True)
class PhaseDatabaseEntry:
    cell_id: str
    problem: Mapping[str, object]
    status: PhaseEntryStatus
    baseline_telemetry_complete: bool
    missing_baselines: tuple[str, ...]
    selected_evidence_id: Optional[str]
    selected_family_id: Optional[str]
    kernel_key: Optional[str]
    workspace_bytes: Optional[int]
    supported_range: Optional[Mapping[str, object]]
    selected_latency_ms: Optional[float]
    selected_confidence: Optional[float]
    baseline_evidence_id: Optional[str]
    baseline_requested_backend: Optional[str]
    baseline_resolved_backend: Optional[str]
    baseline_latency_ms: Optional[float]
    speedup_vs_baseline: Optional[float]
    routing_regret: Optional[float]
    correctness_passed: Optional[bool]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("phase entry cell_id must be non-empty")
        if self.baseline_telemetry_complete == bool(self.missing_baselines):
            raise ValueError("baseline telemetry flag disagrees with missing_baselines")
        if self.status is PhaseEntryStatus.NATIVE:
            if not self.selected_evidence_id or not self.selected_family_id or not self.kernel_key:
                raise ValueError("native phase entry requires selected native identifiers")
        if self.status is PhaseEntryStatus.EXTERNAL_FALLBACK:
            if not self.selected_evidence_id or self.kernel_key is not None:
                raise ValueError("external fallback requires evidence and no kernel key")
        if self.resolved and self.correctness_passed is not True:
            raise ValueError("resolved phase entry requires passing correctness")
        if self.speedup_vs_baseline is not None and self.speedup_vs_baseline <= 0.0:
            raise ValueError("speedup_vs_baseline must be positive")
        if self.routing_regret is not None and self.routing_regret < -1.0e-12:
            raise ValueError("routing_regret must be non-negative")

    @property
    def resolved(self) -> bool:
        return self.status in {PhaseEntryStatus.NATIVE, PhaseEntryStatus.EXTERNAL_FALLBACK}

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "problem": dict(self.problem),
            "status": self.status.value,
            "baseline_telemetry_complete": self.baseline_telemetry_complete,
            "missing_baselines": list(self.missing_baselines),
            "selected_evidence_id": self.selected_evidence_id,
            "selected_family_id": self.selected_family_id,
            "kernel_key": self.kernel_key,
            "workspace_bytes": self.workspace_bytes,
            "supported_range": (
                None if self.supported_range is None else dict(self.supported_range)
            ),
            "selected_latency_ms": self.selected_latency_ms,
            "selected_confidence": self.selected_confidence,
            "baseline_evidence_id": self.baseline_evidence_id,
            "baseline_requested_backend": self.baseline_requested_backend,
            "baseline_resolved_backend": self.baseline_resolved_backend,
            "baseline_latency_ms": self.baseline_latency_ms,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "routing_regret": self.routing_regret,
            "correctness_passed": self.correctness_passed,
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseDatabaseEntry":
        return cls(
            cell_id=str(data["cell_id"]),
            problem=dict(data["problem"]),
            status=PhaseEntryStatus(str(data["status"])),
            baseline_telemetry_complete=bool(data["baseline_telemetry_complete"]),
            missing_baselines=tuple(str(value) for value in data["missing_baselines"]),
            selected_evidence_id=(
                None
                if data.get("selected_evidence_id") is None
                else str(data["selected_evidence_id"])
            ),
            selected_family_id=(
                None
                if data.get("selected_family_id") is None
                else str(data["selected_family_id"])
            ),
            kernel_key=(
                None if data.get("kernel_key") is None else str(data["kernel_key"])
            ),
            workspace_bytes=(
                None
                if data.get("workspace_bytes") is None
                else int(data["workspace_bytes"])
            ),
            supported_range=(
                None
                if data.get("supported_range") is None
                else dict(data["supported_range"])
            ),
            selected_latency_ms=(
                None
                if data.get("selected_latency_ms") is None
                else float(data["selected_latency_ms"])
            ),
            selected_confidence=(
                None
                if data.get("selected_confidence") is None
                else float(data["selected_confidence"])
            ),
            baseline_evidence_id=(
                None
                if data.get("baseline_evidence_id") is None
                else str(data["baseline_evidence_id"])
            ),
            baseline_requested_backend=(
                None
                if data.get("baseline_requested_backend") is None
                else str(data["baseline_requested_backend"])
            ),
            baseline_resolved_backend=(
                None
                if data.get("baseline_resolved_backend") is None
                else str(data["baseline_resolved_backend"])
            ),
            baseline_latency_ms=(
                None
                if data.get("baseline_latency_ms") is None
                else float(data["baseline_latency_ms"])
            ),
            speedup_vs_baseline=(
                None
                if data.get("speedup_vs_baseline") is None
                else float(data["speedup_vs_baseline"])
            ),
            routing_regret=(
                None
                if data.get("routing_regret") is None
                else float(data["routing_regret"])
            ),
            correctness_passed=(
                None
                if data.get("correctness_passed") is None
                else bool(data["correctness_passed"])
            ),
            evidence_ids=tuple(str(value) for value in data["evidence_ids"]),
        )


@dataclass(frozen=True)
class PhaseDatabase:
    schema_version: int
    manifest_id: str
    architecture: str
    source_commit: str
    environment: EnvironmentFingerprint
    entries: tuple[PhaseDatabaseEntry, ...]
    evidence: tuple[BackendEvidence, ...]
    acceptance: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != PHASE_DATABASE_SCHEMA_VERSION:
            raise ValueError(f"unsupported phase database schema: {self.schema_version}")
        if self.architecture != self.environment.architecture:
            raise ValueError("phase database architecture does not match environment")
        if not self.manifest_id or not self.source_commit:
            raise ValueError("phase database manifest_id and source_commit must be non-empty")
        entry_ids = [entry.cell_id for entry in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("phase database entry cell IDs must be unique")
        evidence_ids = [row.evidence_id for row in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("phase database evidence IDs must be unique")
        entry_id_set = set(entry_ids)
        evidence_id_set = set(evidence_ids)
        if any(row.cell_id not in entry_id_set for row in self.evidence):
            raise ValueError("phase database evidence references an absent cell")
        if any(
            not row.environment.compatible_with(self.environment)
            for row in self.evidence
        ):
            raise ValueError("phase database evidence mixes incompatible environments")
        for entry in self.entries:
            if not set(entry.evidence_ids) <= evidence_id_set:
                raise ValueError("phase entry references missing evidence")
            if (
                entry.selected_evidence_id is not None
                and entry.selected_evidence_id not in evidence_id_set
            ):
                raise ValueError("phase entry selected evidence is missing")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "architecture": self.architecture,
            "source_commit": self.source_commit,
            "environment": self.environment.as_dict(),
            "environment_id": self.environment.fingerprint_id,
            "environment_compatibility_id": self.environment.compatibility_id,
            "acceptance": dict(self.acceptance),
            "entries": [entry.as_dict() for entry in self.entries],
            "evidence": [row.as_dict() for row in self.evidence],
        }

    def entry_for(self, cell_id: str) -> PhaseDatabaseEntry:
        for entry in self.entries:
            if entry.cell_id == cell_id:
                return entry
        raise KeyError(f"phase database has no cell {cell_id!r}")

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseDatabase":
        return cls(
            schema_version=int(data["schema_version"]),
            manifest_id=str(data["manifest_id"]),
            architecture=str(data["architecture"]),
            source_commit=str(data["source_commit"]),
            environment=EnvironmentFingerprint.from_dict(dict(data["environment"])),
            entries=tuple(
                PhaseDatabaseEntry.from_dict(dict(row)) for row in data["entries"]
            ),
            evidence=tuple(
                BackendEvidence.from_dict(dict(row)) for row in data["evidence"]
            ),
            acceptance=dict(data["acceptance"]),
        )


def _percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _entry_for_cell(
    cell: ExactWorkloadCell,
    evidence: Sequence[BackendEvidence],
    families: Sequence[KernelFamily],
    selected_evidence_ids: Mapping[str, str],
) -> PhaseDatabaseEntry:
    rows = tuple(sorted((row for row in evidence if row.cell_id == cell.cell_id), key=lambda r: r.evidence_id))
    attempted_baselines = {
        row.requested_backend for row in rows if row.provider == "external"
    }
    missing_baselines = tuple(
        baseline for baseline in cell.baseline_candidates if baseline not in attempted_baselines
    )
    telemetry_complete = not missing_baselines
    baseline = resolve_fastest_correct_baseline(cell, rows)
    native_oracle = resolve_fastest_streamattn_candidate(cell, rows)
    requested_selection = selected_evidence_ids.get(cell.cell_id)
    requested_row: Optional[BackendEvidence] = None
    if requested_selection is not None:
        requested_row = next(
            (
                row
                for row in rows
                if row.evidence_id == requested_selection
                and row.is_routable
            ),
            None,
        )
        if requested_row is None:
            raise ValueError(
                f"explicit selection for {cell.cell_id} is not routable: "
                f"{requested_selection}"
            )

    native_families = matching_kernel_families(cell, families, native_only=True)
    selected = requested_row if requested_row is not None else native_oracle
    if not telemetry_complete:
        status = PhaseEntryStatus.INCOMPLETE_BASELINES
    elif baseline is None:
        status = PhaseEntryStatus.NO_CORRECT_BASELINE
    elif requested_row is not None:
        status = (
            PhaseEntryStatus.NATIVE
            if requested_row.provider == "streamattn" and requested_row.native
            else PhaseEntryStatus.EXTERNAL_FALLBACK
        )
    elif (
        native_oracle is not None
        and native_oracle.timing is not None
        and baseline.timing is not None
        and native_oracle.timing.p50_ms <= baseline.timing.p50_ms
    ):
        status = PhaseEntryStatus.NATIVE
    elif native_oracle is not None:
        status = PhaseEntryStatus.EXTERNAL_FALLBACK
        selected = baseline
    elif not native_families:
        status = PhaseEntryStatus.EXTERNAL_FALLBACK
        selected = baseline
    else:
        status = PhaseEntryStatus.NATIVE_UNMEASURED

    selected_timing = None if selected is None else selected.timing
    baseline_timing = None if baseline is None else baseline.timing
    selected_latency = None if selected_timing is None else selected_timing.p50_ms
    baseline_latency = None if baseline_timing is None else baseline_timing.p50_ms
    speedup = (
        None
        if selected_latency is None or baseline_latency is None
        else baseline_latency / selected_latency
    )
    usable_oracles = [
        row
        for row in (baseline, native_oracle)
        if row is not None and row.timing is not None
    ]
    performance_oracle = min(
        usable_oracles,
        key=lambda row: (row.timing.p50_ms, row.evidence_id),  # type: ignore[union-attr]
        default=None,
    )
    regret = None
    if (
        selected_timing is not None
        and performance_oracle is not None
        and performance_oracle.timing is not None
    ):
        regret = selected_timing.p50_ms / performance_oracle.timing.p50_ms - 1.0
    return PhaseDatabaseEntry(
        cell_id=cell.cell_id,
        problem=cell.as_dict(),
        status=status,
        baseline_telemetry_complete=telemetry_complete,
        missing_baselines=missing_baselines,
        selected_evidence_id=None if selected is None else selected.evidence_id,
        selected_family_id=None if selected is None else selected.family_id,
        kernel_key=None if selected is None else selected.kernel_key,
        workspace_bytes=None if selected is None else selected.workspace_bytes,
        supported_range=None if selected is None else selected.supported_range,
        selected_latency_ms=selected_latency,
        selected_confidence=None if selected_timing is None else selected_timing.confidence,
        baseline_evidence_id=None if baseline is None else baseline.evidence_id,
        baseline_requested_backend=(
            None if baseline is None else baseline.requested_backend
        ),
        baseline_resolved_backend=None if baseline is None else baseline.resolved_backend,
        baseline_latency_ms=baseline_latency,
        speedup_vs_baseline=speedup,
        routing_regret=regret,
        correctness_passed=(
            None
            if selected is None or selected.correctness is None
            else selected.correctness.passed
        ),
        evidence_ids=tuple(row.evidence_id for row in rows),
    )


def compile_phase_database(
    manifest: UniversalExactManifest,
    evidence: Iterable[BackendEvidence],
    *,
    architecture: str,
    source_commit: str,
    families: Optional[Sequence[KernelFamily]] = None,
    selected_evidence_ids: Optional[Mapping[str, str]] = None,
) -> PhaseDatabase:
    """Compile one architecture phase database without dropping negative rows."""

    all_evidence = tuple(evidence)
    cells = tuple(cell for cell in manifest.cells if cell.architecture == architecture)
    if not cells:
        raise ValueError(f"manifest has no cells for architecture {architecture}")
    cell_ids = {cell.cell_id for cell in cells}
    rows = tuple(row for row in all_evidence if row.cell_id in cell_ids)
    unknown = sorted(
        {row.cell_id for row in all_evidence}
        - {cell.cell_id for cell in manifest.cells}
    )
    if unknown:
        raise ValueError(f"evidence references unknown manifest cells: {unknown}")
    if not rows:
        raise ValueError(f"no evidence supplied for architecture {architecture}")
    environment = rows[0].environment
    if any(not row.environment.compatible_with(environment) for row in rows[1:]):
        raise ValueError("one phase database cannot mix incompatible environments")
    if environment.architecture != architecture:
        raise ValueError("evidence environment architecture does not match database")
    evidence_ids = [row.evidence_id for row in rows]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence IDs must be globally unique per phase database")

    family_rows = tuple(families or registered_exact_kernel_families())
    selections = dict(selected_evidence_ids or {})
    entries = tuple(
        _entry_for_cell(cell, rows, family_rows, selections) for cell in cells
    )
    regrets = [entry.routing_regret for entry in entries if entry.routing_regret is not None]
    resolved = [entry for entry in entries if entry.resolved]
    telemetry_complete = [entry for entry in entries if entry.baseline_telemetry_complete]
    evidence_by_id = {row.evidence_id: row for row in rows}
    selected_rows = [
        evidence_by_id[entry.selected_evidence_id]
        for entry in entries
        if entry.selected_evidence_id is not None
    ]
    zero_allocation = all(
        row.timing is not None and row.timing.timed_allocation_count == 0
        for row in selected_rows
    )
    p90_regret = _percentile([float(value) for value in regrets], 0.90)
    negative_cells = {
        entry.cell_id
        for entry in entries
        if entry.speedup_vs_baseline is not None and entry.speedup_vs_baseline < 1.0
    }
    for cell in cells:
        baseline = resolve_fastest_correct_baseline(cell, rows)
        native = resolve_fastest_streamattn_candidate(cell, rows)
        if (
            baseline is not None
            and baseline.timing is not None
            and native is not None
            and native.timing is not None
            and native.timing.p50_ms > baseline.timing.p50_ms
        ):
            negative_cells.add(cell.cell_id)
    acceptance = {
        "semantic_coverage": len(entries) / len(cells),
        "telemetry_coverage": len(telemetry_complete) / len(cells),
        "resolved_coverage": len(resolved) / len(cells),
        "native_coverage": sum(entry.status is PhaseEntryStatus.NATIVE for entry in entries)
        / len(cells),
        "p90_routing_regret": p90_regret,
        "zero_timed_loop_allocations": zero_allocation,
        "negative_cells": sorted(negative_cells),
        "negative_cells_retained": True,
        "compiler_v1_pass": bool(
            len(entries) == len(cells)
            and len(telemetry_complete) == len(cells)
            and len(resolved) == len(cells)
            and zero_allocation
            and p90_regret is not None
            and p90_regret <= manifest.acceptance.p90_routing_regret
        ),
    }
    return PhaseDatabase(
        schema_version=PHASE_DATABASE_SCHEMA_VERSION,
        manifest_id=manifest.manifest_id,
        architecture=architecture,
        source_commit=source_commit,
        environment=environment,
        entries=entries,
        evidence=tuple(sorted(rows, key=lambda row: row.evidence_id)),
        acceptance=acceptance,
    )


def load_backend_evidence(path: str | Path) -> tuple[BackendEvidence, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != PHASE_DATABASE_SCHEMA_VERSION:
        raise ValueError("unsupported evidence schema_version")
    raw_rows = data.get("evidence")
    if not isinstance(raw_rows, list):
        raise ValueError("evidence artifact must contain an evidence list")
    return tuple(BackendEvidence.from_dict(dict(row)) for row in raw_rows)


def load_phase_database(path: str | Path) -> PhaseDatabase:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PhaseDatabase.from_dict(dict(data))


def write_backend_evidence(
    evidence: Iterable[BackendEvidence],
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PHASE_DATABASE_SCHEMA_VERSION,
        "evidence": [row.as_dict() for row in evidence],
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
