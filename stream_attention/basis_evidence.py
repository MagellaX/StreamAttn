"""Canonical basis-operation evidence for architecture-native attention work."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
from typing import Iterable

import yaml

from .inference_workload import INFERENCE_ARCHITECTURES


BASIS_EVIDENCE_SCHEMA_VERSION = 1
BASIS_SUITE_SCHEMA_VERSION = 1


class BasisOperation(str, Enum):
    LOAD = "load"
    ASYNC_LOAD = "async_load"
    QK = "qk"
    SOFTMAX = "softmax"
    PV = "pv"
    QK_SOFTMAX_SERIAL = "qk_softmax_serial"
    QK_SOFTMAX_OVERLAP = "qk_softmax_overlap"
    PV_RESCALE = "pv_rescale"
    PARTIAL_STATE_WRITE = "partial_state_write"
    PARTIAL_STATE_MERGE = "partial_state_merge"
    SCHEDULER = "scheduler"
    EPILOGUE = "epilogue"
    ATTENTION_EPOCH_SERIAL = "attention_epoch_serial"
    ATTENTION_EPOCH_OVERLAP = "attention_epoch_overlap"


CANONICAL_NCU_METRICS = (
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
    "launch__local_mem_per_thread",
    "sm__ctas_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
)


@dataclass(frozen=True)
class BasisAnchor:
    anchor_id: str
    architecture: str
    batch_size: int
    query_len: int
    kv_len: int
    q_heads: int
    kv_heads: int
    d_qk: int
    d_v: int
    q_dtype: str
    kv_dtype: str
    cache_kind: str
    cache_layout: str
    mask_kind: str
    ragged: bool = False
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.anchor_id:
            raise ValueError("basis anchor_id must be non-empty")
        if self.architecture not in INFERENCE_ARCHITECTURES:
            raise ValueError(f"unsupported basis architecture: {self.architecture}")
        if min(
            self.batch_size,
            self.query_len,
            self.kv_len,
            self.q_heads,
            self.kv_heads,
            self.d_qk,
            self.d_v,
        ) <= 0:
            raise ValueError("basis geometry must be positive")
        if self.q_heads % self.kv_heads:
            raise ValueError("basis q_heads must be divisible by kv_heads")
        if not all((self.q_dtype, self.kv_dtype, self.cache_kind, self.cache_layout)):
            raise ValueError("basis dtype and cache fields must be named")

    def as_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "architecture": self.architecture,
            "batch_size": self.batch_size,
            "query_len": self.query_len,
            "kv_len": self.kv_len,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "d_qk": self.d_qk,
            "d_v": self.d_v,
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "cache_kind": self.cache_kind,
            "cache_layout": self.cache_layout,
            "mask_kind": self.mask_kind,
            "ragged": self.ragged,
            "tags": sorted(self.tags),
        }


@dataclass(frozen=True)
class BasisCase:
    suite_id: str
    anchor: BasisAnchor
    operation: BasisOperation

    @property
    def case_id(self) -> str:
        return f"{self.anchor.anchor_id}.{self.operation.value}"

    def as_dict(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "case_id": self.case_id,
            "operation": self.operation.value,
            "anchor": self.anchor.as_dict(),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BasisSuite:
    suite_id: str
    architecture: str
    operations: tuple[BasisOperation, ...]
    anchors: tuple[BasisAnchor, ...]
    required_ncu_metrics: tuple[str, ...]
    graph_replay: bool
    warmup: int
    iterations: int
    repeats: int

    def __post_init__(self) -> None:
        if not self.suite_id or self.architecture not in INFERENCE_ARCHITECTURES:
            raise ValueError("invalid basis suite identity")
        if not self.operations or not self.anchors:
            raise ValueError("basis suite requires operations and anchors")
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("basis suite operations must be unique")
        ids = [anchor.anchor_id for anchor in self.anchors]
        if len(set(ids)) != len(ids):
            raise ValueError("basis anchor IDs must be unique")
        if any(anchor.architecture != self.architecture for anchor in self.anchors):
            raise ValueError("basis anchor architecture does not match its suite")
        if min(self.warmup, self.iterations, self.repeats) <= 0:
            raise ValueError("basis timing counts must be positive")
        if not self.required_ncu_metrics:
            raise ValueError("basis suite must declare required NCU metrics")

    @property
    def cases(self) -> tuple[BasisCase, ...]:
        return tuple(
            BasisCase(self.suite_id, anchor, operation)
            for anchor in self.anchors
            for operation in self.operations
        )

    def case(self, case_id: str) -> BasisCase:
        matches = [case for case in self.cases if case.case_id == case_id]
        if len(matches) != 1:
            raise KeyError(f"unknown basis case: {case_id}")
        return matches[0]


@dataclass(frozen=True)
class BasisEnvironment:
    source_commit: str
    provider: str
    device_name: str
    compute_capability: str
    gpu_count: int
    python_version: str
    torch_version: str
    cuda_version: str
    driver_version: str
    ncu_version: str | None

    def __post_init__(self) -> None:
        if not all(
            (
                self.source_commit,
                self.provider,
                self.device_name,
                self.compute_capability,
                self.python_version,
                self.torch_version,
                self.cuda_version,
                self.driver_version,
            )
        ):
            raise ValueError("basis environment fields must be non-empty")
        if self.gpu_count <= 0:
            raise ValueError("basis evidence requires at least one GPU")

    def as_dict(self) -> dict[str, object]:
        return {
            "source_commit": self.source_commit,
            "provider": self.provider,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "gpu_count": self.gpu_count,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "ncu_version": self.ncu_version,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BasisCounter:
    metric: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not self.metric or not self.unit:
            raise ValueError("basis counter metric and unit must be non-empty")


@dataclass(frozen=True)
class BasisEvidence:
    case: BasisCase
    environment: BasisEnvironment
    adapter_id: str
    adapter_revision: str
    timing_mode: str
    samples_ms: tuple[float, ...]
    correctness_passed: bool
    resources: tuple[tuple[str, float], ...]
    counters: tuple[BasisCounter, ...]
    missing_counters: tuple[str, ...]
    raw_artifact_sha256: str
    schema_version: int = BASIS_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BASIS_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported basis evidence schema")
        if not self.adapter_id or not self.adapter_revision or not self.timing_mode:
            raise ValueError("basis adapter and timing identity must be non-empty")
        if not self.samples_ms or min(self.samples_ms) <= 0:
            raise ValueError("basis timing samples must be positive")
        if len(self.raw_artifact_sha256) != 64:
            raise ValueError("raw_artifact_sha256 must be a 64-character digest")
        names = [counter.metric for counter in self.counters]
        if len(set(names)) != len(names):
            raise ValueError("basis counter metrics must be unique")

    @property
    def median_ms(self) -> float:
        ordered = sorted(self.samples_ms)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "case": self.case.as_dict(),
            "case_sha256": self.case.fingerprint,
            "environment": self.environment.as_dict(),
            "environment_sha256": self.environment.fingerprint,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "timing_mode": self.timing_mode,
            "samples_ms": list(self.samples_ms),
            "median_ms": self.median_ms,
            "correctness_passed": self.correctness_passed,
            "resources": dict(self.resources),
            "counters": [counter.__dict__ for counter in self.counters],
            "missing_counters": list(self.missing_counters),
            "raw_artifact_sha256": self.raw_artifact_sha256,
        }


def load_basis_suite(path: Path) -> BasisSuite:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != BASIS_SUITE_SCHEMA_VERSION:
        raise ValueError("unsupported basis suite schema")
    architecture = str(raw["architecture"])
    anchors = tuple(
        BasisAnchor(
            anchor_id=str(row["anchor_id"]),
            architecture=architecture,
            batch_size=int(row["batch_size"]),
            query_len=int(row["query_len"]),
            kv_len=int(row["kv_len"]),
            q_heads=int(row["q_heads"]),
            kv_heads=int(row["kv_heads"]),
            d_qk=int(row["d_qk"]),
            d_v=int(row["d_v"]),
            q_dtype=str(row["q_dtype"]),
            kv_dtype=str(row["kv_dtype"]),
            cache_kind=str(row["cache_kind"]),
            cache_layout=str(row["cache_layout"]),
            mask_kind=str(row["mask_kind"]),
            ragged=bool(row.get("ragged", False)),
            tags=frozenset(str(value) for value in row.get("tags", ())),
        )
        for row in raw["anchors"]
    )
    timing = raw["timing"]
    return BasisSuite(
        suite_id=str(raw["suite_id"]),
        architecture=architecture,
        operations=tuple(BasisOperation(value) for value in raw["operations"]),
        anchors=anchors,
        required_ncu_metrics=tuple(str(value) for value in raw["ncu_metrics"]),
        graph_replay=bool(timing["graph_replay"]),
        warmup=int(timing["warmup"]),
        iterations=int(timing["iterations"]),
        repeats=int(timing["repeats"]),
    )


def parse_ncu_csv(text: str) -> tuple[BasisCounter, ...]:
    """Parse NCU CSV output while tolerating its preamble and locale commas."""

    lines = [line for line in text.splitlines() if line.lstrip().startswith('"')]
    if not lines:
        return ()
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    result: dict[str, BasisCounter] = {}
    for row in rows:
        metric = row.get("Metric Name") or row.get("Metric Name ")
        raw_value = row.get("Metric Value") or row.get("Metric Value ")
        unit = row.get("Metric Unit") or row.get("Metric Unit ") or "count"
        if not metric or raw_value is None:
            continue
        try:
            value = float(raw_value.replace(",", ""))
        except ValueError:
            continue
        result[metric] = BasisCounter(metric=metric, value=value, unit=unit)
    return tuple(result[key] for key in sorted(result))


def capture_local_basis_environment(
    *, source_commit: str, provider: str, ncu_version: str | None = None
) -> BasisEnvironment:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("basis evidence requires CUDA")
    capability = torch.cuda.get_device_capability(0)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = "unavailable"
    return BasisEnvironment(
        source_commit=source_commit,
        provider=provider,
        device_name=torch.cuda.get_device_name(0),
        compute_capability=f"{capability[0]}.{capability[1]}",
        gpu_count=torch.cuda.device_count(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda or "unavailable"),
        driver_version=driver.strip(),
        ncu_version=ncu_version,
    )


def missing_required_counters(
    required: Iterable[str], counters: Iterable[BasisCounter]
) -> tuple[str, ...]:
    present = {counter.metric for counter in counters}
    return tuple(sorted(set(required) - present))


__all__ = [
    "BASIS_EVIDENCE_SCHEMA_VERSION",
    "BASIS_SUITE_SCHEMA_VERSION",
    "CANONICAL_NCU_METRICS",
    "BasisAnchor",
    "BasisCase",
    "BasisCounter",
    "BasisEnvironment",
    "BasisEvidence",
    "BasisOperation",
    "BasisSuite",
    "capture_local_basis_environment",
    "load_basis_suite",
    "missing_required_counters",
    "parse_ncu_csv",
]
