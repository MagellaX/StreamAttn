"""Hierarchical macro and physical schedule IR for exact inference v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from .exact_compiler import GuaranteeClass
from .inference_workload import AttentionBatchV2


SCHEDULE_IR_SCHEMA_VERSION = 2


class MacroPlanKind(str, Enum):
    UNIFIED_PERSISTENT = "unified_persistent"
    SPLIT_PHASE = "split_phase"
    QUERY_LENGTH_COHORT = "query_length_cohort"
    STATIC_RECTANGULAR = "static_rectangular"
    PREFIX_SUM_RAGGED = "prefix_sum_ragged"


class OperandSource(str, Enum):
    SS = "ss"
    RS = "rs"


class ProducerTopology(str, Enum):
    NONE = "none"
    PRODUCER_WARP = "producer_warp"
    PRODUCER_WARPGROUP = "producer_warpgroup"
    PRODUCER_CTA = "producer_cta"


class BarrierTopology(str, Enum):
    CTA_WIDE = "cta_wide"
    NAMED = "named"
    STAGE_LOCAL_MBARRIER = "stage_local_mbarrier"


class ConsumerOverlap(str, Enum):
    NONE = "none"
    INTER_WARPGROUP_PING_PONG = "inter_warpgroup_ping_pong"
    INTRA_WARPGROUP_TWO_STAGE = "intra_warpgroup_two_stage"


class TaskGranularity(str, Enum):
    REQUEST = "request"
    ROW = "row"
    Q_TILE = "q_tile"
    KV_GROUP_TILE = "kv_group_tile"
    SEQUENCE_SPLIT = "sequence_split"


class OutputMode(str, Enum):
    DIRECT = "direct"
    PARTIAL_STATE = "partial_state"
    FUSED_CACHE_APPEND = "fused_cache_append"


@dataclass(frozen=True)
class WorkCohort:
    cohort_id: str
    request_indices: tuple[int, ...]
    phase_family: str
    q_len_min: int
    q_len_max: int

    def __post_init__(self) -> None:
        if not self.cohort_id or not self.phase_family:
            raise ValueError("cohort identifiers must be non-empty")
        if not self.request_indices or min(self.request_indices) < 0:
            raise ValueError("cohort request_indices must be non-empty and non-negative")
        if len(set(self.request_indices)) != len(self.request_indices):
            raise ValueError("cohort request_indices must be unique")
        if self.q_len_min <= 0 or self.q_len_max < self.q_len_min:
            raise ValueError("invalid cohort query-length range")

    def as_dict(self) -> dict[str, object]:
        return {
            "cohort_id": self.cohort_id,
            "request_indices": list(self.request_indices),
            "phase_family": self.phase_family,
            "q_len_min": self.q_len_min,
            "q_len_max": self.q_len_max,
        }


@dataclass(frozen=True)
class MacroExecutionPlan:
    """Batch decomposition chosen before architecture-specific lowering."""

    workload_sha256: str
    plan_kind: MacroPlanKind
    cohorts: tuple[WorkCohort, ...]
    task_granularity: TaskGranularity
    launch_count: int
    persistent_queue: bool
    split_kv: bool
    graph_safe: bool
    fixed_workspace_bytes: int
    planner_version: str
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if len(self.workload_sha256) != 64:
            raise ValueError("workload_sha256 must be a 64-character digest")
        if not self.cohorts or self.launch_count <= 0:
            raise ValueError("macro plans require cohorts and at least one launch")
        if self.fixed_workspace_bytes < 0 or not self.planner_version:
            raise ValueError("invalid macro-plan workspace or planner version")

    def validate_for(self, workload: AttentionBatchV2) -> None:
        if self.workload_sha256 != workload.fingerprint:
            raise ValueError("macro plan was compiled for a different workload")
        covered = [index for cohort in self.cohorts for index in cohort.request_indices]
        if sorted(covered) != list(range(workload.batch_size)):
            raise ValueError("macro-plan cohorts must partition every request exactly once")
        for cohort in self.cohorts:
            lengths = [workload.requests[index].query_len for index in cohort.request_indices]
            if min(lengths) < cohort.q_len_min or max(lengths) > cohort.q_len_max:
                raise ValueError(f"cohort {cohort.cohort_id} query bounds are incorrect")
        if self.graph_safe and self.fixed_workspace_bytes > workload.fixed_workspace_bytes:
            raise ValueError("macro plan exceeds the workload's fixed workspace")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEDULE_IR_SCHEMA_VERSION,
            "workload_sha256": self.workload_sha256,
            "plan_kind": self.plan_kind.value,
            "cohorts": [cohort.as_dict() for cohort in self.cohorts],
            "task_granularity": self.task_granularity.value,
            "launch_count": self.launch_count,
            "persistent_queue": self.persistent_queue,
            "split_kv": self.split_kv,
            "graph_safe": self.graph_safe,
            "fixed_workspace_bytes": self.fixed_workspace_bytes,
            "planner_version": self.planner_version,
            "metadata": dict(self.metadata),
        }

    @property
    def plan_key(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return f"macro-v2:{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


@dataclass(frozen=True)
class PhysicalScheduleV2:
    """Architecture-native lowering for one homogeneous macro-plan cohort."""

    family_id: str
    architecture: str
    cohort_id: str
    guarantee: GuaranteeClass
    qk_operand_source: OperandSource
    pv_operand_source: OperandSource
    producer_topology: ProducerTopology
    barrier_topology: BarrierTopology
    consumer_overlap: ConsumerOverlap
    task_granularity: TaskGranularity
    output_mode: OutputMode
    q_heads_per_cta: int
    q_positions_per_cta: int
    kv_tokens_per_tile: int
    split_q: int
    split_kv: int
    pipeline_stages: int
    producer_warps: int
    consumer_warpgroups: int
    cluster_shape: tuple[int, int, int]
    accumulator_space: str
    load_engine: str
    mma_engine: str
    softmax_variant: str
    merge_variant: str
    epilogue_variant: str
    expected_registers_per_thread: int | None = None
    expected_shared_memory_bytes: int | None = None
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.family_id or not self.architecture or not self.cohort_id:
            raise ValueError("physical schedule identifiers must be non-empty")
        positive = (
            self.q_heads_per_cta,
            self.q_positions_per_cta,
            self.kv_tokens_per_tile,
            self.split_q,
            self.split_kv,
            self.pipeline_stages,
        )
        if min(positive) <= 0:
            raise ValueError("physical tile, split, and stage values must be positive")
        if self.producer_warps < 0 or self.consumer_warpgroups < 0:
            raise ValueError("warp counts must be non-negative")
        if len(self.cluster_shape) != 3 or min(self.cluster_shape) <= 0:
            raise ValueError("cluster_shape must contain three positive dimensions")
        if self.expected_registers_per_thread is not None:
            if self.expected_registers_per_thread <= 0:
                raise ValueError("expected register count must be positive")
        if self.expected_shared_memory_bytes is not None:
            if self.expected_shared_memory_bytes < 0:
                raise ValueError("expected shared-memory bytes must be non-negative")
        if not all(
            (
                self.accumulator_space,
                self.load_engine,
                self.mma_engine,
                self.softmax_variant,
                self.merge_variant,
                self.epilogue_variant,
            )
        ):
            raise ValueError("physical engine and variant names must be non-empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEDULE_IR_SCHEMA_VERSION,
            "family_id": self.family_id,
            "architecture": self.architecture,
            "cohort_id": self.cohort_id,
            "guarantee": self.guarantee.value,
            "qk_operand_source": self.qk_operand_source.value,
            "pv_operand_source": self.pv_operand_source.value,
            "producer_topology": self.producer_topology.value,
            "barrier_topology": self.barrier_topology.value,
            "consumer_overlap": self.consumer_overlap.value,
            "task_granularity": self.task_granularity.value,
            "output_mode": self.output_mode.value,
            "q_heads_per_cta": self.q_heads_per_cta,
            "q_positions_per_cta": self.q_positions_per_cta,
            "kv_tokens_per_tile": self.kv_tokens_per_tile,
            "split_q": self.split_q,
            "split_kv": self.split_kv,
            "pipeline_stages": self.pipeline_stages,
            "producer_warps": self.producer_warps,
            "consumer_warpgroups": self.consumer_warpgroups,
            "cluster_shape": list(self.cluster_shape),
            "accumulator_space": self.accumulator_space,
            "load_engine": self.load_engine,
            "mma_engine": self.mma_engine,
            "softmax_variant": self.softmax_variant,
            "merge_variant": self.merge_variant,
            "epilogue_variant": self.epilogue_variant,
            "expected_registers_per_thread": self.expected_registers_per_thread,
            "expected_shared_memory_bytes": self.expected_shared_memory_bytes,
            "metadata": dict(self.metadata),
        }

    @property
    def kernel_key(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
        return f"{self.architecture}:{self.family_id}:v2:{digest}"


@dataclass(frozen=True)
class HierarchicalExecutionPlan:
    macro: MacroExecutionPlan
    physical_schedules: tuple[PhysicalScheduleV2, ...]

    def validate_for(self, workload: AttentionBatchV2) -> None:
        self.macro.validate_for(workload)
        cohort_ids = {cohort.cohort_id for cohort in self.macro.cohorts}
        scheduled = {schedule.cohort_id for schedule in self.physical_schedules}
        if cohort_ids != scheduled:
            raise ValueError("every macro cohort requires exactly one physical schedule")
        if len(scheduled) != len(self.physical_schedules):
            raise ValueError("a cohort cannot have multiple selected physical schedules")
        if any(schedule.architecture != workload.architecture for schedule in self.physical_schedules):
            raise ValueError("physical schedule architecture does not match the workload")


__all__ = [
    "SCHEDULE_IR_SCHEMA_VERSION",
    "BarrierTopology",
    "ConsumerOverlap",
    "HierarchicalExecutionPlan",
    "MacroExecutionPlan",
    "MacroPlanKind",
    "OperandSource",
    "OutputMode",
    "PhysicalScheduleV2",
    "ProducerTopology",
    "TaskGranularity",
    "WorkCohort",
]
