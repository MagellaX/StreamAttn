"""Universal exact-attention phase compiler contracts.

This module does not execute attention kernels.  It defines the durable inputs
to the offline compiler: a valid workload manifest, a small physical schedule
grammar, architecture resource constraints, and explicit records for compiled
artifacts.  Runtime problem sizes remain in :class:`AttentionProblem`; only
instruction-shaping choices belong in :class:`ScheduleCandidate`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from .planning import (
    ATTENTION_CACHE_CONTIGUOUS,
    ATTENTION_CACHE_PAGED,
    ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    ATTENTION_GUARANTEE_EXACT,
    ATTENTION_GUARANTEE_SCHEDULE_EXACT,
    ATTENTION_PHASE_DECODE,
    ATTENTION_PHASE_PREFILL,
    ATTENTION_PHASE_TRAIN,
    AttentionProblem,
)


UNIVERSAL_EXACT_MANIFEST_SCHEMA_VERSION = 1
WORKLOAD_SURFACES = frozenset({"real", "boundary", "feature"})
TARGET_ARCHITECTURES = frozenset({"sm80", "sm90", "sm100"})
TARGET_DTYPES = frozenset({"float16", "bfloat16"})
WEIGHT_SCHEMES = ("trace", "stratified", "boundary")


class GuaranteeClass(str, Enum):
    """Semantics promised by a compiled kernel family."""

    FULL_CONTEXT_EXACT = ATTENTION_GUARANTEE_EXACT
    SCHEDULE_EXACT = ATTENTION_GUARANTEE_SCHEDULE_EXACT
    DISTRIBUTION_VERIFIED = ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED


class AlgebraOrientation(str, Enum):
    QK = "qk_t"
    TRANSPOSED_KQ = "kq_t"


class Ownership(str, Enum):
    Q_HEAD = "q_head"
    KV_GROUP = "kv_group"
    Q_TILE = "q_tile"
    KV_TILE = "kv_tile"


class LoadEngine(str, Enum):
    LDST = "ldst"
    CP_ASYNC = "cp_async"
    TMA = "tma"


class MmaEngine(str, Enum):
    MMA_SYNC = "mma_sync"
    WGMMA = "wgmma"
    TCGEN05 = "tcgen05"
    TRITON_DOT = "triton_dot"
    EXTERNAL = "external"


class AccumulatorSpace(str, Enum):
    REGISTERS = "registers"
    SHARED = "shared"
    TMEM = "tmem"


class SchedulerKind(str, Enum):
    STATIC = "static"
    SOFTWARE_PERSISTENT = "software_persistent"
    CLUSTER_PERSISTENT = "cluster_persistent"
    CLC = "clc"


class BackwardRole(str, Enum):
    FORWARD = "forward"
    DQ = "dq"
    DKV = "dkv"


@dataclass(frozen=True)
class WorkloadTolerance:
    rtol: float
    atol: float

    def __post_init__(self) -> None:
        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("correctness tolerances must be non-negative")


@dataclass(frozen=True)
class ExactWorkloadCell:
    """One explicit semantic cell in the universal workload manifest."""

    cell_id: str
    surface: str
    architecture: str
    phase: str
    guarantee: GuaranteeClass
    batch_size: int
    query_len: int
    kv_lengths: tuple[int, ...]
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    cache_kind: str
    cache_layout: str
    mask_kind: str
    page_size: Optional[int]
    features: frozenset[str]
    weights: Mapping[str, float]
    baseline_candidates: tuple[str, ...]
    tolerance: WorkloadTolerance

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("workload cell_id must be non-empty")
        if self.surface not in WORKLOAD_SURFACES:
            raise ValueError(f"unsupported workload surface: {self.surface}")
        if self.architecture not in TARGET_ARCHITECTURES:
            raise ValueError(f"unsupported target architecture: {self.architecture}")
        if self.guarantee is not GuaranteeClass.FULL_CONTEXT_EXACT:
            raise ValueError("universal exact manifest requires full-context exact cells")
        if self.phase not in {
            ATTENTION_PHASE_DECODE,
            ATTENTION_PHASE_PREFILL,
            ATTENTION_PHASE_TRAIN,
        }:
            raise ValueError(f"unsupported phase: {self.phase}")
        if self.batch_size <= 0 or self.query_len <= 0:
            raise ValueError("batch_size and query_len must be positive")
        if len(self.kv_lengths) != self.batch_size:
            raise ValueError("kv_lengths must contain one length per batch row")
        if any(length <= 0 for length in self.kv_lengths):
            raise ValueError("KV lengths must be positive")
        if self.phase == ATTENTION_PHASE_DECODE and self.query_len != 1:
            raise ValueError("decode manifest cells require query_len == 1")
        if self.q_heads <= 0 or self.kv_heads <= 0 or self.q_heads % self.kv_heads:
            raise ValueError("q_heads must be a positive multiple of kv_heads")
        if self.head_dim not in {64, 128, 256}:
            raise ValueError("v1 head_dim must be 64, 128, or 256")
        if self.dtype not in TARGET_DTYPES:
            raise ValueError(f"unsupported v1 dtype: {self.dtype}")
        if self.cache_kind not in {ATTENTION_CACHE_CONTIGUOUS, ATTENTION_CACHE_PAGED}:
            raise ValueError(f"unsupported cache kind: {self.cache_kind}")
        if self.cache_layout not in {"NHD", "HND"}:
            raise ValueError("cache_layout must be NHD or HND")
        if self.cache_kind == ATTENTION_CACHE_PAGED:
            if self.page_size not in {16, 32, 64}:
                raise ValueError("paged cells require page_size 16, 32, or 64")
            if self.phase == ATTENTION_PHASE_TRAIN:
                raise ValueError("paged KV training is outside the v1 valid surface")
        elif self.page_size is not None:
            raise ValueError("contiguous cells must not define page_size")
        if "dropout" in self.features and self.phase != ATTENTION_PHASE_TRAIN:
            raise ValueError("dropout is valid only for training cells")
        if not self.baseline_candidates:
            raise ValueError("every cell requires at least one baseline candidate")
        unknown_weights = set(self.weights) - set(WEIGHT_SCHEMES)
        if unknown_weights:
            raise ValueError(f"unknown weight schemes: {sorted(unknown_weights)}")
        if any(float(value) < 0.0 for value in self.weights.values()):
            raise ValueError("workload weights must be non-negative")

    @property
    def group_size(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def is_ragged(self) -> bool:
        return min(self.kv_lengths) != max(self.kv_lengths)

    def to_attention_problem(self, *, device: str = "cuda") -> AttentionProblem:
        """Lower manifest semantics into the existing runtime problem contract."""

        return AttentionProblem(
            phase=self.phase,
            guarantee=self.guarantee.value,
            mask=self.mask_kind,
            batch_size=self.batch_size,
            query_len=self.query_len,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=device,
            kv_lengths=self.kv_lengths,
            cache_kind=self.cache_kind,
            cache_layout=self.cache_layout,
            page_size=self.page_size,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "surface": self.surface,
            "architecture": self.architecture,
            "phase": self.phase,
            "guarantee": self.guarantee.value,
            "batch_size": self.batch_size,
            "query_len": self.query_len,
            "kv_lengths": list(self.kv_lengths),
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "group_size": self.group_size,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "cache_kind": self.cache_kind,
            "cache_layout": self.cache_layout,
            "mask_kind": self.mask_kind,
            "page_size": self.page_size,
            "features": sorted(self.features),
            "weights": dict(self.weights),
            "baseline_candidates": list(self.baseline_candidates),
            "tolerance": {
                "rtol": self.tolerance.rtol,
                "atol": self.tolerance.atol,
            },
        }


@dataclass(frozen=True)
class CompilerAcceptance:
    semantic_coverage: float
    telemetry_coverage: float
    p90_routing_regret: float
    zero_timed_loop_allocations: bool
    retain_negative_cells: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.semantic_coverage <= 1.0:
            raise ValueError("semantic_coverage must be in [0, 1]")
        if not 0.0 <= self.telemetry_coverage <= 1.0:
            raise ValueError("telemetry_coverage must be in [0, 1]")
        if self.p90_routing_regret < 0.0:
            raise ValueError("p90_routing_regret must be non-negative")


@dataclass(frozen=True)
class UniversalExactManifest:
    schema_version: int
    manifest_id: str
    cells: tuple[ExactWorkloadCell, ...]
    acceptance: CompilerAcceptance
    source_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.schema_version != UNIVERSAL_EXACT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema_version: {self.schema_version}"
            )
        if not self.manifest_id:
            raise ValueError("manifest_id must be non-empty")
        ids = [cell.cell_id for cell in self.cells]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest cell IDs must be unique")
        if {cell.surface for cell in self.cells} != WORKLOAD_SURFACES:
            raise ValueError("manifest must contain real, boundary, and feature surfaces")
        if {cell.architecture for cell in self.cells} != TARGET_ARCHITECTURES:
            raise ValueError("manifest must cover SM80, SM90, and SM100")
        phases = {cell.phase for cell in self.cells}
        if phases != {
            ATTENTION_PHASE_DECODE,
            ATTENTION_PHASE_PREFILL,
            ATTENTION_PHASE_TRAIN,
        }:
            raise ValueError("manifest must cover decode, prefill, and train")
        for scheme in WEIGHT_SCHEMES:
            if sum(float(cell.weights.get(scheme, 0.0)) for cell in self.cells) <= 0.0:
                raise ValueError(f"manifest has no positive {scheme} weights")

    def normalized_weights(self, scheme: str) -> dict[str, float]:
        if scheme not in WEIGHT_SCHEMES:
            raise ValueError(f"unknown weight scheme: {scheme}")
        values = {
            cell.cell_id: float(cell.weights.get(scheme, 0.0)) for cell in self.cells
        }
        total = sum(values.values())
        return {cell_id: value / total for cell_id, value in values.items()}

    def summary(self) -> dict[str, object]:
        def counts(values: Iterable[str]) -> dict[str, int]:
            result: dict[str, int] = {}
            for value in values:
                result[value] = result.get(value, 0) + 1
            return dict(sorted(result.items()))

        return {
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "cell_count": len(self.cells),
            "surfaces": counts(cell.surface for cell in self.cells),
            "architectures": counts(cell.architecture for cell in self.cells),
            "phases": counts(cell.phase for cell in self.cells),
            "dtypes": counts(cell.dtype for cell in self.cells),
            "head_dims": counts(str(cell.head_dim) for cell in self.cells),
        }


@dataclass(frozen=True)
class ScheduleCandidate:
    """Compile-time physical schedule; it deliberately excludes B, M, and N."""

    family_id: str
    architecture: str
    phase_family: str
    guarantee: GuaranteeClass
    algebra_orientation: AlgebraOrientation
    ownership: Ownership
    q_heads_per_cta: int
    q_positions_per_cta: int
    kv_tokens_per_tile: int
    head_dim_stage: int
    split_q: int
    split_kv: int
    producer_ctas: int
    load_engine: LoadEngine
    mma_engine: MmaEngine
    accumulator_space: AccumulatorSpace
    num_load_warps: int
    num_consumer_warpgroups: int
    pipeline_stages: int
    scheduler: SchedulerKind
    cluster_shape: tuple[int, int, int]
    softmax_variant: str
    merge_variant: str
    epilogue_variant: str
    backward_role: BackwardRole = BackwardRole.FORWARD

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("candidate family_id must be non-empty")
        if self.architecture not in TARGET_ARCHITECTURES:
            raise ValueError(f"unsupported candidate architecture: {self.architecture}")
        if min(
            self.q_heads_per_cta,
            self.q_positions_per_cta,
            self.kv_tokens_per_tile,
            self.head_dim_stage,
            self.split_q,
            self.split_kv,
            self.producer_ctas,
            self.pipeline_stages,
        ) <= 0:
            raise ValueError("candidate tile, split, and stage counts must be positive")
        if self.num_load_warps < 0 or self.num_consumer_warpgroups < 0:
            raise ValueError("candidate warp counts must be non-negative")
        if len(self.cluster_shape) != 3 or any(value <= 0 for value in self.cluster_shape):
            raise ValueError("cluster_shape must contain three positive dimensions")
        if not self.softmax_variant or not self.merge_variant or not self.epilogue_variant:
            raise ValueError("candidate variants must be named")

    @property
    def kernel_key(self) -> str:
        payload = self.as_dict()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return f"{self.architecture}:{self.family_id}:{digest}"

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "architecture": self.architecture,
            "phase_family": self.phase_family,
            "guarantee": self.guarantee.value,
            "algebra_orientation": self.algebra_orientation.value,
            "ownership": self.ownership.value,
            "q_heads_per_cta": self.q_heads_per_cta,
            "q_positions_per_cta": self.q_positions_per_cta,
            "kv_tokens_per_tile": self.kv_tokens_per_tile,
            "head_dim_stage": self.head_dim_stage,
            "split_q": self.split_q,
            "split_kv": self.split_kv,
            "producer_ctas": self.producer_ctas,
            "load_engine": self.load_engine.value,
            "mma_engine": self.mma_engine.value,
            "accumulator_space": self.accumulator_space.value,
            "num_load_warps": self.num_load_warps,
            "num_consumer_warpgroups": self.num_consumer_warpgroups,
            "pipeline_stages": self.pipeline_stages,
            "scheduler": self.scheduler.value,
            "cluster_shape": list(self.cluster_shape),
            "softmax_variant": self.softmax_variant,
            "merge_variant": self.merge_variant,
            "epilogue_variant": self.epilogue_variant,
            "backward_role": self.backward_role.value,
        }


@dataclass(frozen=True)
class CandidateResourceEstimate:
    threads_per_cta: int
    dynamic_shared_memory_bytes: int
    registers_per_thread: int
    tmem_columns: int = 0
    spill_bytes: int = 0

    def __post_init__(self) -> None:
        if self.threads_per_cta <= 0:
            raise ValueError("threads_per_cta must be positive")
        if min(
            self.dynamic_shared_memory_bytes,
            self.registers_per_thread,
            self.tmem_columns,
            self.spill_bytes,
        ) < 0:
            raise ValueError("resource estimates must be non-negative")


@dataclass(frozen=True)
class ResourceLegality:
    legal: bool
    reasons: tuple[str, ...]
    estimated_active_ctas_per_sm: int


@dataclass(frozen=True)
class ArchitectureResourceModel:
    architecture: str
    max_threads_per_cta: int
    max_threads_per_sm: int
    max_ctas_per_sm: int
    registers_per_sm: int
    max_registers_per_thread: int
    shared_memory_per_sm: int
    max_shared_memory_per_cta: int
    max_cluster_ctas: int
    max_tmem_columns: Optional[int] = None

    def __post_init__(self) -> None:
        if self.architecture not in TARGET_ARCHITECTURES:
            raise ValueError(f"unsupported architecture model: {self.architecture}")

    def check(
        self,
        candidate: ScheduleCandidate,
        resources: CandidateResourceEstimate,
    ) -> ResourceLegality:
        if candidate.architecture != self.architecture:
            return ResourceLegality(False, ("architecture_mismatch",), 0)
        reasons: list[str] = []
        if resources.threads_per_cta > self.max_threads_per_cta:
            reasons.append("threads_per_cta")
        if resources.dynamic_shared_memory_bytes > self.max_shared_memory_per_cta:
            reasons.append("shared_memory_per_cta")
        if resources.registers_per_thread > self.max_registers_per_thread:
            reasons.append("registers_per_thread")
        registers_per_cta = resources.registers_per_thread * resources.threads_per_cta
        if registers_per_cta > self.registers_per_sm:
            reasons.append("register_file")
        cluster_ctas = math.prod(candidate.cluster_shape)
        if cluster_ctas > self.max_cluster_ctas:
            reasons.append("cluster_shape")
        if resources.tmem_columns:
            if self.max_tmem_columns is None:
                reasons.append("tmem_unsupported")
            elif resources.tmem_columns > self.max_tmem_columns:
                reasons.append("tmem_columns")
        if resources.spill_bytes:
            reasons.append("register_spill")

        active_by_threads = self.max_threads_per_sm // resources.threads_per_cta
        active_by_registers = (
            self.max_ctas_per_sm
            if registers_per_cta == 0
            else self.registers_per_sm // registers_per_cta
        )
        active_by_shared = (
            self.max_ctas_per_sm
            if resources.dynamic_shared_memory_bytes == 0
            else self.shared_memory_per_sm // resources.dynamic_shared_memory_bytes
        )
        active = min(
            self.max_ctas_per_sm,
            active_by_threads,
            active_by_registers,
            active_by_shared,
        )
        if active <= 0:
            reasons.append("zero_occupancy")
        return ResourceLegality(not reasons, tuple(sorted(set(reasons))), max(0, active))


@dataclass(frozen=True)
class CompiledKernelRecord:
    kernel_key: str
    family_id: str
    architecture: str
    compiler: str
    compiler_version: str
    binary_path: str
    resources: CandidateResourceEstimate
    correctness_passed: bool
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.kernel_key or not self.family_id or not self.binary_path:
            raise ValueError("compiled record identifiers must be non-empty")
        if len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a 64-character digest")


@dataclass(frozen=True)
class KernelFamily:
    family_id: str
    implementation: str
    guarantees: frozenset[GuaranteeClass]
    architectures: frozenset[str]
    phases: frozenset[str]
    cache_kinds: frozenset[str]
    layouts: frozenset[str]
    dtypes: frozenset[str]
    head_dims: frozenset[int]
    mask_kinds: frozenset[str]
    min_group_size: int
    max_group_size: int
    min_query_len: int
    max_query_len: int
    native: bool
    maturity: str
    required_features: frozenset[str] = field(default_factory=frozenset)
    excluded_features: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.family_id or not self.implementation:
            raise ValueError("kernel family identifiers must be non-empty")
        if not self.architectures <= TARGET_ARCHITECTURES:
            raise ValueError("kernel family contains an unsupported architecture")
        if self.min_group_size <= 0 or self.max_group_size < self.min_group_size:
            raise ValueError("invalid kernel-family GQA range")
        if self.min_query_len <= 0 or self.max_query_len < self.min_query_len:
            raise ValueError("invalid kernel-family query range")
        if self.maturity not in {"promoted", "calibration", "generic", "fallback"}:
            raise ValueError(f"unsupported kernel-family maturity: {self.maturity}")

    def supports(self, cell: ExactWorkloadCell) -> bool:
        return bool(
            cell.guarantee in self.guarantees
            and cell.architecture in self.architectures
            and cell.phase in self.phases
            and cell.cache_kind in self.cache_kinds
            and cell.cache_layout in self.layouts
            and cell.dtype in self.dtypes
            and cell.head_dim in self.head_dims
            and cell.mask_kind in self.mask_kinds
            and self.required_features <= cell.features
            and not (self.excluded_features & cell.features)
            and self.min_group_size <= cell.group_size <= self.max_group_size
            and self.min_query_len <= cell.query_len <= self.max_query_len
        )


def default_architecture_resource_models() -> dict[str, ArchitectureResourceModel]:
    """Conservative v1 pruning limits; compiled reports replace estimates."""

    return {
        "sm80": ArchitectureResourceModel(
            architecture="sm80",
            max_threads_per_cta=1024,
            max_threads_per_sm=2048,
            max_ctas_per_sm=32,
            registers_per_sm=65536,
            max_registers_per_thread=255,
            shared_memory_per_sm=167936,
            max_shared_memory_per_cta=163840,
            max_cluster_ctas=1,
        ),
        "sm90": ArchitectureResourceModel(
            architecture="sm90",
            max_threads_per_cta=1024,
            max_threads_per_sm=2048,
            max_ctas_per_sm=32,
            registers_per_sm=65536,
            max_registers_per_thread=255,
            shared_memory_per_sm=233472,
            max_shared_memory_per_cta=227328,
            max_cluster_ctas=8,
        ),
        "sm100": ArchitectureResourceModel(
            architecture="sm100",
            max_threads_per_cta=1024,
            max_threads_per_sm=2048,
            max_ctas_per_sm=32,
            registers_per_sm=65536,
            max_registers_per_thread=255,
            shared_memory_per_sm=233472,
            max_shared_memory_per_cta=227328,
            max_cluster_ctas=16,
            max_tmem_columns=128,
        ),
    }


def registered_exact_kernel_families() -> tuple[KernelFamily, ...]:
    """Register current assets as compiler families without rewriting them."""

    exact = frozenset({GuaranteeClass.FULL_CONTEXT_EXACT})
    contiguous_paged = frozenset({ATTENTION_CACHE_CONTIGUOUS, ATTENTION_CACHE_PAGED})
    both_layouts = frozenset({"NHD", "HND"})
    both_dtypes = frozenset(TARGET_DTYPES)
    common_masks = frozenset({"none", "causal", "noncausal", "sliding_window"})
    return (
        KernelFamily(
            family_id="sm80_paged_gqa_exact_decode",
            implementation="stream_attention.paged:PAGED_EXACT_SM80_CP_ASYNC_BACKEND",
            guarantees=exact,
            architectures=frozenset({"sm80"}),
            phases=frozenset({ATTENTION_PHASE_DECODE}),
            cache_kinds=frozenset({ATTENTION_CACHE_PAGED}),
            layouts=both_layouts,
            dtypes=both_dtypes,
            head_dims=frozenset({64, 128}),
            mask_kinds=frozenset({"none", "causal"}),
            min_group_size=1,
            max_group_size=16,
            min_query_len=1,
            max_query_len=1,
            native=True,
            maturity="calibration",
        ),
        KernelFamily(
            family_id="sm90_transposed_gqa_exact_decode",
            implementation="stream_attention.backends.sm90:ExactDecodePlan",
            guarantees=exact,
            architectures=frozenset({"sm90"}),
            phases=frozenset({ATTENTION_PHASE_DECODE}),
            cache_kinds=contiguous_paged,
            layouts=both_layouts,
            dtypes=both_dtypes,
            head_dims=frozenset({64, 128}),
            mask_kinds=frozenset({"none", "causal"}),
            min_group_size=4,
            max_group_size=8,
            min_query_len=1,
            max_query_len=1,
            native=True,
            maturity="promoted",
        ),
        KernelFamily(
            family_id="sm100_tgv_paged_gqa_exact_decode",
            implementation="stream_attention.paged:PAGED_EXACT_SM100_TGV_BACKEND",
            guarantees=exact,
            architectures=frozenset({"sm100"}),
            phases=frozenset({ATTENTION_PHASE_DECODE}),
            cache_kinds=frozenset({ATTENTION_CACHE_PAGED}),
            layouts=both_layouts,
            dtypes=frozenset({"bfloat16"}),
            head_dims=frozenset({128}),
            mask_kinds=frozenset({"none", "causal"}),
            min_group_size=8,
            max_group_size=8,
            min_query_len=1,
            max_query_len=1,
            native=True,
            maturity="promoted",
        ),
        KernelFamily(
            family_id="sm100_tgv_gqa_causal_prefill",
            implementation="stream_attention.backends.sm100:Sm100GqaPrefillPlan",
            guarantees=exact,
            architectures=frozenset({"sm100"}),
            phases=frozenset({ATTENTION_PHASE_PREFILL}),
            cache_kinds=frozenset({ATTENTION_CACHE_CONTIGUOUS}),
            layouts=frozenset({"NHD"}),
            dtypes=frozenset({"bfloat16"}),
            head_dims=frozenset({128}),
            mask_kinds=frozenset({"causal"}),
            min_group_size=8,
            max_group_size=8,
            min_query_len=1,
            max_query_len=8192,
            native=True,
            maturity="calibration",
        ),
        KernelFamily(
            family_id="triton_grouped_gqa_prefill",
            implementation="stream_attention.kernels.grouped_gqa_prefill_triton",
            guarantees=exact,
            architectures=TARGET_ARCHITECTURES,
            phases=frozenset({ATTENTION_PHASE_PREFILL}),
            cache_kinds=frozenset({ATTENTION_CACHE_CONTIGUOUS}),
            layouts=frozenset({"NHD"}),
            dtypes=both_dtypes,
            head_dims=frozenset({64, 128}),
            mask_kinds=frozenset({"none", "causal", "noncausal"}),
            min_group_size=2,
            max_group_size=16,
            min_query_len=1,
            max_query_len=131072,
            native=True,
            maturity="generic",
        ),
        KernelFamily(
            family_id="triton_online_softmax_exact",
            implementation="stream_attention.core.fused_online_attention",
            guarantees=exact,
            architectures=TARGET_ARCHITECTURES,
            phases=frozenset({ATTENTION_PHASE_PREFILL}),
            cache_kinds=frozenset({ATTENTION_CACHE_CONTIGUOUS}),
            layouts=both_layouts,
            dtypes=both_dtypes,
            head_dims=frozenset({64, 128, 256}),
            mask_kinds=frozenset({*common_masks, "arbitrary", "additive"}),
            min_group_size=1,
            max_group_size=16,
            min_query_len=1,
            max_query_len=131072,
            native=True,
            maturity="generic",
        ),
        KernelFamily(
            family_id="triton_online_softmax_exact_train",
            implementation="stream_attention.core.fused_online_attention",
            guarantees=exact,
            architectures=TARGET_ARCHITECTURES,
            phases=frozenset({ATTENTION_PHASE_TRAIN}),
            cache_kinds=frozenset({ATTENTION_CACHE_CONTIGUOUS}),
            layouts=both_layouts,
            dtypes=both_dtypes,
            head_dims=frozenset({64, 128, 256}),
            mask_kinds=frozenset({*common_masks, "arbitrary", "additive"}),
            min_group_size=1,
            max_group_size=16,
            min_query_len=1,
            max_query_len=131072,
            native=True,
            maturity="generic",
            excluded_features=frozenset({"dropout"}),
        ),
        KernelFamily(
            family_id="explicit_external_exact_fallback",
            implementation="torch.nn.functional.scaled_dot_product_attention",
            guarantees=exact,
            architectures=TARGET_ARCHITECTURES,
            phases=frozenset(
                {ATTENTION_PHASE_DECODE, ATTENTION_PHASE_PREFILL, ATTENTION_PHASE_TRAIN}
            ),
            cache_kinds=contiguous_paged,
            layouts=both_layouts,
            dtypes=both_dtypes,
            head_dims=frozenset({64, 128, 256}),
            mask_kinds=frozenset({*common_masks, "arbitrary", "additive"}),
            min_group_size=1,
            max_group_size=16,
            min_query_len=1,
            max_query_len=131072,
            native=False,
            maturity="fallback",
        ),
    )


def matching_kernel_families(
    cell: ExactWorkloadCell,
    families: Iterable[KernelFamily],
    *,
    native_only: bool = False,
) -> tuple[KernelFamily, ...]:
    return tuple(
        family
        for family in families
        if (family.native or not native_only) and family.supports(cell)
    )


def default_universal_exact_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "manifests"
        / "universal_exact_v1.yaml"
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_universal_exact_manifest(
    path: str | Path | None = None,
) -> UniversalExactManifest:
    """Load and fully validate the committed universal exact v1 manifest."""

    resolved = Path(path) if path is not None else default_universal_exact_manifest_path()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    data = _mapping(raw, name="manifest")
    defaults = _mapping(data.get("defaults", {}), name="defaults")
    baselines = _mapping(data.get("baselines", {}), name="baselines")
    tolerances = _mapping(data.get("tolerances", {}), name="tolerances")
    surfaces = _mapping(data.get("surfaces", {}), name="surfaces")

    cells: list[ExactWorkloadCell] = []
    for surface_name, raw_cells in surfaces.items():
        if not isinstance(raw_cells, list):
            raise ValueError(f"surface {surface_name!r} must be a list")
        for raw_cell in raw_cells:
            cell = _mapping(raw_cell, name=f"surface {surface_name} cell")
            phase = str(cell["phase"])
            dtype = str(cell.get("dtype", defaults.get("dtype", "bfloat16")))
            batch = int(cell["batch_size"])
            if "kv_lengths" in cell:
                kv_lengths = tuple(int(value) for value in cell["kv_lengths"])
            else:
                kv_lengths = (int(cell["kv_len"]),) * batch
            tolerance_data = _mapping(
                tolerances.get(dtype, {}), name=f"tolerance {dtype}"
            )
            cell_baselines = cell.get("baseline_candidates", baselines.get(phase, ()))
            cells.append(
                ExactWorkloadCell(
                    cell_id=str(cell["id"]),
                    surface=str(surface_name),
                    architecture=str(cell["architecture"]),
                    phase=phase,
                    guarantee=GuaranteeClass(
                        str(cell.get("guarantee", defaults.get("guarantee", "exact")))
                    ),
                    batch_size=batch,
                    query_len=int(cell["query_len"]),
                    kv_lengths=kv_lengths,
                    q_heads=int(cell["q_heads"]),
                    kv_heads=int(cell["kv_heads"]),
                    head_dim=int(cell["head_dim"]),
                    dtype=dtype,
                    cache_kind=str(cell["cache_kind"]),
                    cache_layout=str(cell["cache_layout"]).upper(),
                    mask_kind=str(cell.get("mask_kind", defaults.get("mask_kind", "none"))),
                    page_size=(
                        None if cell.get("page_size") is None else int(cell["page_size"])
                    ),
                    features=frozenset(str(value) for value in cell.get("features", ())),
                    weights={
                        str(key): float(value)
                        for key, value in _mapping(
                            cell.get("weights", {}), name=f"weights {cell['id']}"
                        ).items()
                    },
                    baseline_candidates=tuple(str(value) for value in cell_baselines),
                    tolerance=WorkloadTolerance(
                        rtol=float(tolerance_data["rtol"]),
                        atol=float(tolerance_data["atol"]),
                    ),
                )
            )

    acceptance_data = _mapping(data.get("acceptance", {}), name="acceptance")
    return UniversalExactManifest(
        schema_version=int(data["schema_version"]),
        manifest_id=str(data["manifest_id"]),
        cells=tuple(cells),
        acceptance=CompilerAcceptance(
            semantic_coverage=float(acceptance_data["semantic_coverage"]),
            telemetry_coverage=float(acceptance_data["telemetry_coverage"]),
            p90_routing_regret=float(acceptance_data["p90_routing_regret"]),
            zero_timed_loop_allocations=bool(
                acceptance_data["zero_timed_loop_allocations"]
            ),
            retain_negative_cells=bool(acceptance_data["retain_negative_cells"]),
        ),
        source_path=resolved.resolve(),
    )


__all__ = [
    "AccumulatorSpace",
    "AlgebraOrientation",
    "ArchitectureResourceModel",
    "BackwardRole",
    "CandidateResourceEstimate",
    "CompiledKernelRecord",
    "CompilerAcceptance",
    "ExactWorkloadCell",
    "GuaranteeClass",
    "KernelFamily",
    "LoadEngine",
    "MmaEngine",
    "Ownership",
    "ResourceLegality",
    "ScheduleCandidate",
    "SchedulerKind",
    "UniversalExactManifest",
    "default_architecture_resource_models",
    "default_universal_exact_manifest_path",
    "load_universal_exact_manifest",
    "matching_kernel_families",
    "registered_exact_kernel_families",
]
