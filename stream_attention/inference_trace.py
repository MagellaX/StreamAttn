"""Immutable trace import and frozen calibration/holdout partitioning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .inference_workload import AttentionBatchV2, DatasetSplit


TRACE_SCHEMA_VERSION = 1
DEFAULT_SPLIT_SALT = "streamattn-universal-inference-v2"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def frozen_split_for(
    record_id: str,
    *,
    calibration_fraction: float = 0.8,
    salt: str = DEFAULT_SPLIT_SALT,
) -> DatasetSplit:
    """Assign a stable split without depending on trace order or corpus size."""

    if not record_id:
        raise ValueError("record_id must be non-empty")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between zero and one")
    digest = hashlib.sha256(f"{salt}\0{record_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return (
        DatasetSplit.CALIBRATION
        if bucket < calibration_fraction
        else DatasetSplit.HOLDOUT
    )


@dataclass(frozen=True)
class InferenceTraceRecord:
    record_id: str
    source: str
    source_trace_id: str
    split: DatasetSplit
    workload: AttentionBatchV2
    workload_sha256: str
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema version: {self.schema_version}")
        if not self.record_id or not self.source or not self.source_trace_id:
            raise ValueError("trace identifiers must be non-empty")
        if len(self.workload_sha256) != 64:
            raise ValueError("workload_sha256 must be a 64-character digest")
        if self.workload_sha256 != self.workload.fingerprint:
            raise ValueError("workload payload does not match workload_sha256")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InferenceTraceRecord":
        workload = AttentionBatchV2.from_dict(raw["workload"])
        return cls(
            schema_version=int(raw.get("schema_version", TRACE_SCHEMA_VERSION)),
            record_id=str(raw["record_id"]),
            source=str(raw["source"]),
            source_trace_id=str(raw["source_trace_id"]),
            split=DatasetSplit(raw["split"]),
            workload=workload,
            workload_sha256=str(raw["workload_sha256"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "source": self.source,
            "source_trace_id": self.source_trace_id,
            "split": self.split.value,
            "workload_sha256": self.workload_sha256,
            "workload": self.workload.as_dict(),
        }


def freeze_trace_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    existing: Iterable[InferenceTraceRecord] = (),
    calibration_fraction: float = 0.8,
    salt: str = DEFAULT_SPLIT_SALT,
) -> tuple[InferenceTraceRecord, ...]:
    """Validate raw records and freeze their stable dataset partition.

    Existing record IDs are immutable: re-importing the same payload is
    idempotent, while changing its source or workload raises an error.
    """

    frozen = {record.record_id: record for record in existing}
    seen: set[str] = set()
    result: list[InferenceTraceRecord] = []
    for raw in rows:
        record_id = str(raw["record_id"])
        if record_id in seen:
            raise ValueError(f"duplicate trace record_id: {record_id}")
        seen.add(record_id)
        source = str(raw["source"])
        source_trace_id = str(raw.get("source_trace_id", record_id))
        workload_raw = raw.get("workload", raw.get("batch"))
        if not isinstance(workload_raw, Mapping):
            raise ValueError(f"trace record {record_id} is missing a workload object")
        workload = AttentionBatchV2.from_dict(workload_raw)
        prior = frozen.get(record_id)
        if prior is not None:
            if (
                prior.source != source
                or prior.source_trace_id != source_trace_id
                or prior.workload_sha256 != workload.fingerprint
            ):
                raise ValueError(f"frozen trace record changed: {record_id}")
            result.append(prior)
            continue
        result.append(
            InferenceTraceRecord(
                record_id=record_id,
                source=source,
                source_trace_id=source_trace_id,
                split=frozen_split_for(
                    record_id,
                    calibration_fraction=calibration_fraction,
                    salt=salt,
                ),
                workload=workload,
                workload_sha256=workload.fingerprint,
            )
        )
    return tuple(sorted(result, key=lambda record: record.record_id))


def read_trace_jsonl(path: Path) -> tuple[InferenceTraceRecord, ...]:
    records: list[InferenceTraceRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = InferenceTraceRecord.from_dict(json.loads(line))
            if record.record_id in seen:
                raise ValueError(f"duplicate record_id at {path}:{line_number}")
            seen.add(record.record_id)
            records.append(record)
    return tuple(records)


def read_raw_trace_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"trace row at {path}:{line_number} must be an object")
            rows.append(raw)
    return tuple(rows)


def write_trace_jsonl(path: Path, records: Iterable[InferenceTraceRecord]) -> None:
    materialized = tuple(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "\n".join(_canonical_json(record.as_dict()) for record in materialized)
    path.write_text(encoded + ("\n" if encoded else ""), encoding="utf-8")


def summarize_trace(records: Iterable[InferenceTraceRecord]) -> dict[str, object]:
    materialized = tuple(records)

    def counts(values: Iterable[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "record_count": len(materialized),
        "request_count": sum(record.workload.batch_size for record in materialized),
        "splits": counts(record.split.value for record in materialized),
        "sources": counts(record.source for record in materialized),
        "architectures": counts(record.workload.architecture for record in materialized),
        "phases": counts(record.workload.phase.value for record in materialized),
        "cache_kinds": counts(record.workload.cache_kind.value for record in materialized),
        "ragged_batches": sum(record.workload.is_ragged for record in materialized),
    }


__all__ = [
    "DEFAULT_SPLIT_SALT",
    "TRACE_SCHEMA_VERSION",
    "InferenceTraceRecord",
    "freeze_trace_records",
    "frozen_split_for",
    "read_raw_trace_jsonl",
    "read_trace_jsonl",
    "summarize_trace",
    "write_trace_jsonl",
]
