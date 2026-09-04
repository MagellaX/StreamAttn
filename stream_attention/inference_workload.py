"""Versioned serving-workload contract for Universal Inference v2.

Unlike the rectangular v1 calibration cells, a v2 workload describes an entire
serving batch.  The contract is deliberately independent of torch so traces can
be validated, split, and inspected on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


UNIVERSAL_INFERENCE_SCHEMA_VERSION = 2
INFERENCE_ARCHITECTURES = frozenset({"sm80", "sm90", "sm100", "sm103", "sm120"})
INFERENCE_DTYPES = frozenset({"fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "nvfp4"})
ACCUMULATOR_DTYPES = frozenset({"fp32", "fp16"})
DEFAULT_INFERENCE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "manifests"
    / "universal_inference_v2.yaml"
)


class RequestPhase(str, Enum):
    DECODE = "decode"
    VERIFY = "verify"
    APPEND = "append"
    MICRO_PREFILL = "micro_prefill"
    CHUNKED_PREFILL = "chunked_prefill"
    FULL_PREFILL = "full_prefill"
    MIXED = "mixed"


class AttentionKind(str, Enum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"
    MLA = "mla"


class CacheKind(str, Enum):
    CONTIGUOUS = "contiguous"
    PAGED = "paged"
    RAGGED = "ragged"


class CacheLayout(str, Enum):
    BSHD = "bshd"
    HND = "hnd"
    NHD = "nhd"
    PAGE_MAJOR = "page_major"


class MaskKind(str, Enum):
    CAUSAL = "causal"
    NONCAUSAL = "noncausal"
    SLIDING = "sliding"
    ADDITIVE = "additive"


class ExecutionMode(str, Enum):
    EAGER = "eager"
    CUDA_GRAPH = "cuda_graph"


class OptimizationObjective(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"


class DatasetSplit(str, Enum):
    CALIBRATION = "calibration"
    HOLDOUT = "holdout"


class WorkloadSourceKind(str, Enum):
    TRACE = "trace"
    STRATIFIED = "stratified"
    BOUNDARY = "boundary"


def _enum_value(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"unsupported {field_name} {value!r}; expected one of {choices}") from exc


def _tuple_of_ints(value: Any, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of integers")
    result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        raise ValueError(f"{field_name} values must be non-negative")
    return result


def _tuple_of_signed_ints(value: Any, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of integers")
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class AttentionRequestV2:
    """One logical request inside a possibly heterogeneous serving batch."""

    request_id: str
    phase: RequestPhase
    query_len: int
    kv_len: int
    prefix_group: str | None = None
    cache_page_ids: tuple[int, ...] = ()
    last_page_len: int | None = None
    shared_prefix_len: int = 0
    speculative_tree_parents: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.phase is RequestPhase.MIXED:
            raise ValueError("an individual request cannot have phase=mixed")
        if self.query_len <= 0 or self.kv_len < 0:
            raise ValueError("query_len must be positive and kv_len non-negative")
        if not 0 <= self.shared_prefix_len <= self.kv_len:
            raise ValueError("shared_prefix_len must lie inside the KV extent")
        if self.last_page_len is not None and self.last_page_len <= 0:
            raise ValueError("last_page_len must be positive when present")
        if self.speculative_tree_parents:
            if len(self.speculative_tree_parents) != self.query_len:
                raise ValueError("speculative_tree_parents must have query_len entries")
            for node, parent in enumerate(self.speculative_tree_parents):
                if parent >= node:
                    raise ValueError("speculative tree parents must precede their child")
            if self.phase not in {RequestPhase.VERIFY, RequestPhase.APPEND}:
                raise ValueError("speculative trees are valid only for verify/append requests")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AttentionRequestV2":
        return cls(
            request_id=str(raw["request_id"]),
            phase=_enum_value(RequestPhase, raw["phase"], "request phase"),
            query_len=int(raw["query_len"]),
            kv_len=int(raw["kv_len"]),
            prefix_group=(
                None if raw.get("prefix_group") is None else str(raw["prefix_group"])
            ),
            cache_page_ids=_tuple_of_ints(raw.get("cache_page_ids"), "cache_page_ids"),
            last_page_len=(
                None if raw.get("last_page_len") is None else int(raw["last_page_len"])
            ),
            shared_prefix_len=int(raw.get("shared_prefix_len", 0)),
            speculative_tree_parents=_tuple_of_signed_ints(
                raw.get("speculative_tree_parents"), "speculative_tree_parents"
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "phase": self.phase.value,
            "query_len": self.query_len,
            "kv_len": self.kv_len,
            "prefix_group": self.prefix_group,
            "cache_page_ids": list(self.cache_page_ids),
            "last_page_len": self.last_page_len,
            "shared_prefix_len": self.shared_prefix_len,
            "speculative_tree_parents": list(self.speculative_tree_parents),
        }


@dataclass(frozen=True)
class AttentionBatchV2:
    """Canonical exact-inference workload for one complete serving batch."""

    batch_id: str
    architecture: str
    phase: RequestPhase
    requests: tuple[AttentionRequestV2, ...]
    attention_kind: AttentionKind
    q_heads: int
    kv_heads: int
    d_qk: int
    d_v: int
    q_dtype: str
    kv_dtype: str
    accumulator_dtype: str
    output_dtype: str
    scale_format: str
    cache_kind: CacheKind
    cache_layout: CacheLayout
    mask_kind: MaskKind
    execution_mode: ExecutionMode
    objective: OptimizationObjective
    page_size: int | None = None
    sliding_window: int | None = None
    sink_tokens: int = 0
    softcap: float | None = None
    alibi: bool = False
    append_kv: bool = False
    rope_mode: str = "none"
    fixed_workspace_bytes: int = 0
    maximum_captured_batch: int | None = None
    cache_residency: str = "device"
    features: frozenset[str] = field(default_factory=frozenset)
    trace_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty")
        if self.architecture not in INFERENCE_ARCHITECTURES:
            raise ValueError(f"unsupported inference architecture: {self.architecture}")
        if not self.requests:
            raise ValueError("a serving batch must contain at least one request")
        request_ids = [request.request_id for request in self.requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_id values must be unique inside a batch")
        request_phases = {request.phase for request in self.requests}
        if self.phase is RequestPhase.MIXED:
            if len(request_phases) < 2:
                raise ValueError("phase=mixed requires at least two request phases")
        elif request_phases != {self.phase}:
            raise ValueError("homogeneous batch phase must match every request")
        if min(self.q_heads, self.kv_heads, self.d_qk, self.d_v) <= 0:
            raise ValueError("head counts and dimensions must be positive")
        if self.q_heads % self.kv_heads:
            raise ValueError("q_heads must be divisible by kv_heads")
        expected_kind = (
            AttentionKind.MHA
            if self.q_heads == self.kv_heads
            else AttentionKind.MQA
            if self.kv_heads == 1
            else AttentionKind.GQA
        )
        if self.attention_kind is not AttentionKind.MLA and self.attention_kind is not expected_kind:
            raise ValueError(
                f"attention_kind={self.attention_kind.value} conflicts with Hq/Hkv; "
                f"expected {expected_kind.value}"
            )
        if self.q_dtype not in INFERENCE_DTYPES or self.kv_dtype not in INFERENCE_DTYPES:
            raise ValueError("unsupported query or KV dtype")
        if self.output_dtype not in INFERENCE_DTYPES:
            raise ValueError("unsupported output dtype")
        if self.accumulator_dtype not in ACCUMULATOR_DTYPES:
            raise ValueError("unsupported accumulator dtype")
        if not self.scale_format:
            raise ValueError("scale_format must be named")
        if self.cache_kind is CacheKind.PAGED:
            if self.page_size is None or self.page_size <= 0:
                raise ValueError("paged workloads require a positive page_size")
            for request in self.requests:
                if not request.cache_page_ids:
                    raise ValueError("every paged request requires cache_page_ids")
                if request.last_page_len is None or request.last_page_len > self.page_size:
                    raise ValueError("paged request last_page_len must lie in [1, page_size]")
                capacity = len(request.cache_page_ids) * self.page_size
                minimum = capacity - self.page_size + request.last_page_len
                if request.kv_len != minimum:
                    raise ValueError("paged request metadata must exactly describe kv_len")
        else:
            if self.page_size is not None:
                raise ValueError("page_size is valid only for paged workloads")
            if any(request.cache_page_ids for request in self.requests):
                raise ValueError("cache_page_ids are valid only for paged workloads")
        if self.mask_kind is MaskKind.SLIDING:
            if self.sliding_window is None or self.sliding_window <= 0:
                raise ValueError("sliding masks require a positive sliding_window")
        elif self.sliding_window is not None:
            raise ValueError("sliding_window is valid only for sliding masks")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be non-negative")
        if self.softcap is not None and self.softcap <= 0:
            raise ValueError("softcap must be positive when present")
        if self.fixed_workspace_bytes < 0:
            raise ValueError("fixed_workspace_bytes must be non-negative")
        if self.execution_mode is ExecutionMode.CUDA_GRAPH:
            if self.maximum_captured_batch is None:
                raise ValueError("CUDA-graph workloads require maximum_captured_batch")
            if self.maximum_captured_batch < len(self.requests):
                raise ValueError("maximum_captured_batch is smaller than the live batch")
        elif self.maximum_captured_batch is not None:
            raise ValueError("maximum_captured_batch is valid only for CUDA graphs")
        if self.trace_weight <= 0:
            raise ValueError("trace_weight must be positive")

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def query_lengths(self) -> tuple[int, ...]:
        return tuple(request.query_len for request in self.requests)

    @property
    def kv_lengths(self) -> tuple[int, ...]:
        return tuple(request.kv_len for request in self.requests)

    @property
    def group_size(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def is_ragged(self) -> bool:
        return len(set(zip(self.query_lengths, self.kv_lengths))) > 1

    @property
    def has_shared_prefixes(self) -> bool:
        groups = [request.prefix_group for request in self.requests if request.prefix_group]
        return len(groups) != len(set(groups)) or any(
            request.shared_prefix_len for request in self.requests
        )

    @property
    def semantic_features(self) -> frozenset[str]:
        derived = set(self.features)
        if self.alibi:
            derived.add("alibi")
        if self.softcap is not None:
            derived.add("softcap")
        if self.sink_tokens:
            derived.add("sink_tokens")
        if self.append_kv:
            derived.add("append_kv")
        if self.rope_mode != "none":
            derived.add(f"rope:{self.rope_mode}")
        return frozenset(derived)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AttentionBatchV2":
        return cls(
            batch_id=str(raw["batch_id"]),
            architecture=str(raw["architecture"]),
            phase=_enum_value(RequestPhase, raw["phase"], "batch phase"),
            requests=tuple(
                AttentionRequestV2.from_dict(request) for request in raw["requests"]
            ),
            attention_kind=_enum_value(
                AttentionKind, raw["attention_kind"], "attention kind"
            ),
            q_heads=int(raw["q_heads"]),
            kv_heads=int(raw["kv_heads"]),
            d_qk=int(raw["d_qk"]),
            d_v=int(raw["d_v"]),
            q_dtype=str(raw["q_dtype"]),
            kv_dtype=str(raw["kv_dtype"]),
            accumulator_dtype=str(raw.get("accumulator_dtype", "fp32")),
            output_dtype=str(raw["output_dtype"]),
            scale_format=str(raw.get("scale_format", "scalar_fp32")),
            cache_kind=_enum_value(CacheKind, raw["cache_kind"], "cache kind"),
            cache_layout=_enum_value(CacheLayout, raw["cache_layout"], "cache layout"),
            mask_kind=_enum_value(MaskKind, raw["mask_kind"], "mask kind"),
            execution_mode=_enum_value(
                ExecutionMode, raw["execution_mode"], "execution mode"
            ),
            objective=_enum_value(
                OptimizationObjective, raw["objective"], "optimization objective"
            ),
            page_size=None if raw.get("page_size") is None else int(raw["page_size"]),
            sliding_window=(
                None if raw.get("sliding_window") is None else int(raw["sliding_window"])
            ),
            sink_tokens=int(raw.get("sink_tokens", 0)),
            softcap=None if raw.get("softcap") is None else float(raw["softcap"]),
            alibi=bool(raw.get("alibi", False)),
            append_kv=bool(raw.get("append_kv", False)),
            rope_mode=str(raw.get("rope_mode", "none")),
            fixed_workspace_bytes=int(raw.get("fixed_workspace_bytes", 0)),
            maximum_captured_batch=(
                None
                if raw.get("maximum_captured_batch") is None
                else int(raw["maximum_captured_batch"])
            ),
            cache_residency=str(raw.get("cache_residency", "device")),
            features=frozenset(str(value) for value in raw.get("features", ())),
            trace_weight=float(raw.get("trace_weight", 1.0)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": UNIVERSAL_INFERENCE_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "architecture": self.architecture,
            "phase": self.phase.value,
            "requests": [request.as_dict() for request in self.requests],
            "attention_kind": self.attention_kind.value,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "d_qk": self.d_qk,
            "d_v": self.d_v,
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "output_dtype": self.output_dtype,
            "scale_format": self.scale_format,
            "cache_kind": self.cache_kind.value,
            "cache_layout": self.cache_layout.value,
            "page_size": self.page_size,
            "mask_kind": self.mask_kind.value,
            "sliding_window": self.sliding_window,
            "sink_tokens": self.sink_tokens,
            "softcap": self.softcap,
            "alibi": self.alibi,
            "append_kv": self.append_kv,
            "rope_mode": self.rope_mode,
            "execution_mode": self.execution_mode.value,
            "fixed_workspace_bytes": self.fixed_workspace_bytes,
            "maximum_captured_batch": self.maximum_captured_batch,
            "objective": self.objective.value,
            "cache_residency": self.cache_residency,
            "features": sorted(self.features),
            "trace_weight": self.trace_weight,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InferenceWorkloadSource:
    source_id: str
    kind: WorkloadSourceKind
    weight: float
    path: str | None = None
    generator: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id or self.weight <= 0:
            raise ValueError("workload source requires a name and positive weight")
        if self.kind is WorkloadSourceKind.TRACE:
            if not self.path or self.generator is not None:
                raise ValueError("trace sources require path and prohibit generator")
        elif not self.generator or self.path is not None:
            raise ValueError("generated sources require generator and prohibit path")


@dataclass(frozen=True)
class InferenceRegime:
    name: str
    q_len_min: int
    q_len_max: int | None

    def __post_init__(self) -> None:
        if not self.name or self.q_len_min <= 0:
            raise ValueError("regime requires a name and positive lower bound")
        if self.q_len_max is not None and self.q_len_max < self.q_len_min:
            raise ValueError("regime upper bound precedes lower bound")

    def contains(self, query_len: int) -> bool:
        return query_len >= self.q_len_min and (
            self.q_len_max is None or query_len <= self.q_len_max
        )


@dataclass(frozen=True)
class InferenceAcceptance:
    semantic_coverage: float
    holdout_p90_routing_regret: float
    zero_timed_loop_allocations: bool
    retain_negative_cells: bool
    require_fastest_correct_baseline: bool

    def __post_init__(self) -> None:
        if not 0 < self.semantic_coverage <= 1:
            raise ValueError("semantic_coverage must lie in (0, 1]")
        if not 0 <= self.holdout_p90_routing_regret <= 1:
            raise ValueError("holdout routing regret must lie in [0, 1]")


@dataclass(frozen=True)
class UniversalInferenceManifestV2:
    manifest_id: str
    schema_version: int
    architectures: frozenset[str]
    sources: tuple[InferenceWorkloadSource, ...]
    regimes: tuple[InferenceRegime, ...]
    split_salt: str
    calibration_fraction: float
    acceptance: InferenceAcceptance

    def __post_init__(self) -> None:
        if self.schema_version != UNIVERSAL_INFERENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported inference manifest schema: {self.schema_version}")
        if not self.manifest_id or not self.split_salt:
            raise ValueError("manifest_id and split_salt must be non-empty")
        if not self.architectures or not self.architectures <= INFERENCE_ARCHITECTURES:
            raise ValueError("manifest contains unsupported or empty architecture set")
        if not self.sources or not self.regimes:
            raise ValueError("manifest requires workload sources and query regimes")
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("manifest workload source IDs must be unique")
        if {source.kind for source in self.sources} != set(WorkloadSourceKind):
            raise ValueError("manifest requires trace, stratified, and boundary sources")
        if not 0 < self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must lie strictly between zero and one")
        ordered = sorted(self.regimes, key=lambda regime: regime.q_len_min)
        if ordered[0].q_len_min != 1:
            raise ValueError("inference query regimes must begin at query length one")
        for previous, current in zip(ordered, ordered[1:]):
            if previous.q_len_max is None or current.q_len_min != previous.q_len_max + 1:
                raise ValueError("inference query regimes must be contiguous and non-overlapping")
        if sum(regime.q_len_max is None for regime in self.regimes) != 1:
            raise ValueError("manifest requires exactly one open-ended query regime")

    def regime_for(self, query_len: int) -> InferenceRegime:
        matches = [regime for regime in self.regimes if regime.contains(query_len)]
        if len(matches) != 1:
            raise ValueError(f"query length {query_len} maps to {len(matches)} regimes")
        return matches[0]


def load_universal_inference_manifest(
    path: Path | None = None,
) -> UniversalInferenceManifestV2:
    import yaml

    source = DEFAULT_INFERENCE_MANIFEST if path is None else path
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    sources = tuple(
        InferenceWorkloadSource(
            source_id=str(row["source_id"]),
            kind=WorkloadSourceKind(row["kind"]),
            weight=float(row["weight"]),
            path=None if row.get("path") is None else str(row["path"]),
            generator=None if row.get("generator") is None else str(row["generator"]),
        )
        for row in raw["workload_sources"]
    )
    regimes = tuple(
        InferenceRegime(
            name=str(row["name"]),
            q_len_min=int(row["q_len_min"]),
            q_len_max=None if row.get("q_len_max") is None else int(row["q_len_max"]),
        )
        for row in raw["query_regimes"]
    )
    acceptance_raw = raw["acceptance"]
    acceptance = InferenceAcceptance(
        semantic_coverage=float(acceptance_raw["semantic_coverage"]),
        holdout_p90_routing_regret=float(
            acceptance_raw["holdout_p90_routing_regret"]
        ),
        zero_timed_loop_allocations=bool(
            acceptance_raw["zero_timed_loop_allocations"]
        ),
        retain_negative_cells=bool(acceptance_raw["retain_negative_cells"]),
        require_fastest_correct_baseline=bool(
            acceptance_raw["require_fastest_correct_baseline"]
        ),
    )
    split_raw = raw["dataset_split"]
    return UniversalInferenceManifestV2(
        manifest_id=str(raw["manifest_id"]),
        schema_version=int(raw["schema_version"]),
        architectures=frozenset(str(value) for value in raw["architectures"]),
        sources=sources,
        regimes=regimes,
        split_salt=str(split_raw["salt"]),
        calibration_fraction=float(split_raw["calibration_fraction"]),
        acceptance=acceptance,
    )


__all__ = [
    "ACCUMULATOR_DTYPES",
    "DEFAULT_INFERENCE_MANIFEST",
    "INFERENCE_ARCHITECTURES",
    "INFERENCE_DTYPES",
    "UNIVERSAL_INFERENCE_SCHEMA_VERSION",
    "AttentionBatchV2",
    "AttentionKind",
    "AttentionRequestV2",
    "CacheKind",
    "CacheLayout",
    "DatasetSplit",
    "ExecutionMode",
    "MaskKind",
    "OptimizationObjective",
    "RequestPhase",
    "InferenceAcceptance",
    "InferenceRegime",
    "InferenceWorkloadSource",
    "UniversalInferenceManifestV2",
    "WorkloadSourceKind",
    "load_universal_inference_manifest",
]
