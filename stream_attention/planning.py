"""Model-independent semantic planning for StreamAttn attention work.

The planner separates three questions that used to be coupled in individual
decode paths:

* ``AttentionProblem`` describes the attention semantics and cache geometry.
* ``AttentionTilePlan`` describes which logical KV tiles are legal to execute.
* ``AttentionBackendPlan`` records how a device backend will execute that work.

Exact, fixed-block, sliding-window, and query-selected attention therefore use
the same plan contract. They differ in schedule, not in attention algebra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


ATTENTION_GUARANTEE_EXACT = "exact"
ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED = "distribution_verified"
ATTENTION_GUARANTEES = frozenset(
    {
        ATTENTION_GUARANTEE_EXACT,
        ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    }
)

ATTENTION_CACHE_CONTIGUOUS = "contiguous"
ATTENTION_CACHE_PAGED = "paged"
ATTENTION_CACHE_KINDS = frozenset({ATTENTION_CACHE_CONTIGUOUS, ATTENTION_CACHE_PAGED})

ATTENTION_SCHEDULE_ALL = "all"
ATTENTION_SCHEDULE_SELECTED = "selected"
ATTENTION_SCHEDULE_KINDS = frozenset({ATTENTION_SCHEDULE_ALL, ATTENTION_SCHEDULE_SELECTED})


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def fixed_block_tile_ids(
    *,
    kv_len: int,
    tile_size: int,
    sink_tiles: int,
    recent_tiles: int,
    middle_tiles: int,
    tile_order: str,
) -> tuple[int, ...]:
    """Compile the legacy fixed-block policy into logical tile IDs.

    This is a schedule builder, not a backend decision. The returned order
    matches the existing fixed-block kernel contract; ``AttentionTilePlan``
    canonicalizes it when only set membership matters.
    """

    if kv_len <= 0 or tile_size <= 0:
        raise ValueError("kv_len and tile_size must be positive")
    if sink_tiles < 0 or recent_tiles < 0 or middle_tiles < 0:
        raise ValueError("fixed-block tile counts must be non-negative")
    if tile_order not in {"sequential", "recent_first", "sink_recent_first"}:
        raise ValueError(
            "tile_order must be sequential, recent_first, or sink_recent_first"
        )
    tile_count = math.ceil(kv_len / tile_size)
    recent_start = tile_count - recent_tiles
    tiles: list[int] = []
    tiles.extend(range(0, sink_tiles))
    tiles.extend(range(recent_start, tile_count))
    if tile_order == "sequential":
        tiles.extend(range(sink_tiles, sink_tiles + middle_tiles))
    else:
        tiles.extend(range(recent_start - 1, recent_start - 1 - middle_tiles, -1))
    seen: set[int] = set()
    valid: list[int] = []
    for tile in tiles:
        if tile < 0 or tile >= tile_count or tile in seen:
            continue
        seen.add(tile)
        valid.append(int(tile))
    return tuple(valid)


@dataclass(frozen=True)
class AttentionProblem:
    """Semantic contract for one planned attention call."""

    phase: str
    guarantee: str
    mask: str
    batch_size: int
    query_len: int
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    device: str
    kv_lengths: tuple[int, ...]
    cache_kind: str
    cache_layout: str
    page_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.phase != "decode":
            raise ValueError("AttentionProblem currently supports decode phase only")
        if self.guarantee not in ATTENTION_GUARANTEES:
            raise ValueError(f"unsupported attention guarantee: {self.guarantee}")
        if self.cache_kind not in ATTENTION_CACHE_KINDS:
            raise ValueError(f"unsupported cache kind: {self.cache_kind}")
        if self.batch_size <= 0 or self.query_len <= 0:
            raise ValueError("batch_size and query_len must be positive")
        if self.q_heads <= 0 or self.kv_heads <= 0 or self.q_heads % self.kv_heads:
            raise ValueError("q_heads must be a positive multiple of kv_heads")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if len(self.kv_lengths) != self.batch_size or any(length <= 0 for length in self.kv_lengths):
            raise ValueError("kv_lengths must contain one positive length per batch row")
        if self.cache_kind == ATTENTION_CACHE_PAGED:
            if self.page_size is None or self.page_size <= 0:
                raise ValueError("paged attention requires a positive page_size")
        elif self.page_size is not None:
            raise ValueError("contiguous attention must not define page_size")

    @property
    def group_size(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def max_kv_len(self) -> int:
        return max(self.kv_lengths)

    @property
    def min_kv_len(self) -> int:
        return min(self.kv_lengths)

    @property
    def is_ragged(self) -> bool:
        return self.min_kv_len != self.max_kv_len

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "guarantee": self.guarantee,
            "mask": self.mask,
            "batch_size": self.batch_size,
            "query_len": self.query_len,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "group_size": self.group_size,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "device": self.device,
            "kv_lengths": list(self.kv_lengths),
            "cache_kind": self.cache_kind,
            "cache_layout": self.cache_layout,
            "page_size": self.page_size,
            "ragged": self.is_ragged,
        }

    @classmethod
    def from_contiguous(
        cls,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        guarantee: str,
        mask: str = "none",
        cache_layout: str = "auto",
    ) -> "AttentionProblem":
        if query.dim() != 4 or key_cache.dim() != 4 or value_cache.dim() != 4:
            raise ValueError("contiguous Q/K/V must be rank-4 B[N]HD tensors")
        if query.shape[1] != 1:
            raise ValueError("decode query must have query_len == 1")
        if key_cache.shape != value_cache.shape:
            raise ValueError("contiguous key/value cache shapes must match")
        if query.shape[0] != key_cache.shape[0] or query.shape[3] != key_cache.shape[3]:
            raise ValueError("query and KV cache batch/head dimensions must match")
        if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
            raise ValueError("query and KV cache dtypes must match")
        if query.device != key_cache.device or query.device != value_cache.device:
            raise ValueError("query and KV cache devices must match")
        layout = cache_layout.upper()
        if layout == "AUTO":
            axis1_is_heads = int(query.shape[2]) % int(key_cache.shape[1]) == 0
            axis2_is_heads = int(query.shape[2]) % int(key_cache.shape[2]) == 0
            if axis2_is_heads and (not axis1_is_heads or key_cache.shape[1] >= key_cache.shape[2]):
                layout = "NHD"
            elif axis1_is_heads:
                layout = "HND"
            else:
                raise ValueError("cannot infer contiguous KV layout from query-head geometry")
        if layout not in {"NHD", "HND"}:
            raise ValueError("contiguous cache_layout must be auto, NHD, or HND")
        kv_axis = 2 if layout == "NHD" else 1
        sequence_axis = 1 if layout == "NHD" else 2
        if int(query.shape[2]) % int(key_cache.shape[kv_axis]):
            raise ValueError("query heads must be a multiple of KV heads")
        batch = int(query.shape[0])
        kv_len = int(key_cache.shape[sequence_axis])
        return cls(
            phase="decode",
            guarantee=guarantee,
            mask=mask,
            batch_size=batch,
            query_len=int(query.shape[1]),
            q_heads=int(query.shape[2]),
            kv_heads=int(key_cache.shape[kv_axis]),
            head_dim=int(query.shape[3]),
            dtype=_dtype_name(query.dtype),
            device=str(query.device),
            kv_lengths=(kv_len,) * batch,
            cache_kind=ATTENTION_CACHE_CONTIGUOUS,
            cache_layout=layout,
        )

    @classmethod
    def from_paged(
        cls,
        query: torch.Tensor,
        cache,
        *,
        guarantee: str,
        mask: str = "none",
    ) -> "AttentionProblem":
        if query.dim() != 4 or query.shape[1] != 1:
            raise ValueError("paged decode query must be [batch, 1, heads, dim]")
        lengths = tuple(int(length) for length in cache.sequence_lengths.detach().cpu().tolist())
        return cls(
            phase="decode",
            guarantee=guarantee,
            mask=mask,
            batch_size=int(query.shape[0]),
            query_len=int(query.shape[1]),
            q_heads=int(query.shape[2]),
            kv_heads=int(cache.kv_heads),
            head_dim=int(query.shape[3]),
            dtype=_dtype_name(query.dtype),
            device=str(query.device),
            kv_lengths=lengths,
            cache_kind=ATTENTION_CACHE_PAGED,
            cache_layout=str(cache.normalized_layout),
            page_size=int(cache.page_size),
        )


@dataclass(frozen=True)
class AttentionTileSource:
    """Logical KV tile geometry independent of its physical cache mapping."""

    cache_kind: str
    cache_layout: str
    logical_tile_size: int
    logical_tile_counts: tuple[int, ...]
    page_size: Optional[int]
    fragments_per_tile: int

    def as_dict(self) -> dict[str, object]:
        return {
            "cache_kind": self.cache_kind,
            "cache_layout": self.cache_layout,
            "logical_tile_size": self.logical_tile_size,
            "logical_tile_counts": list(self.logical_tile_counts),
            "page_size": self.page_size,
            "fragments_per_tile": self.fragments_per_tile,
        }

    @classmethod
    def from_problem(
        cls,
        problem: AttentionProblem,
        *,
        logical_tile_size: int,
    ) -> "AttentionTileSource":
        if logical_tile_size <= 0:
            raise ValueError("logical_tile_size must be positive")
        if problem.cache_kind == ATTENTION_CACHE_PAGED:
            assert problem.page_size is not None
            if logical_tile_size < problem.page_size or logical_tile_size % problem.page_size:
                raise ValueError(
                    "paged logical_tile_size must be a multiple of the physical page size"
                )
            fragments = logical_tile_size // problem.page_size
        else:
            fragments = 1
        return cls(
            cache_kind=problem.cache_kind,
            cache_layout=problem.cache_layout,
            logical_tile_size=int(logical_tile_size),
            logical_tile_counts=tuple(
                math.ceil(length / logical_tile_size) for length in problem.kv_lengths
            ),
            page_size=problem.page_size,
            fragments_per_tile=fragments,
        )


@dataclass(frozen=True)
class AttentionTileSchedule:
    """Logical tile IDs required for each batch row."""

    kind: str
    selected_tile_ids: Optional[tuple[tuple[int, ...], ...]] = None

    def __post_init__(self) -> None:
        if self.kind not in ATTENTION_SCHEDULE_KINDS:
            raise ValueError(f"unsupported tile schedule kind: {self.kind}")
        if self.kind == ATTENTION_SCHEDULE_ALL and self.selected_tile_ids is not None:
            raise ValueError("all-tile schedule must not materialize selected IDs")
        if self.kind == ATTENTION_SCHEDULE_SELECTED and self.selected_tile_ids is None:
            raise ValueError("selected schedule requires tile IDs")


@dataclass(frozen=True)
class AttentionTilePlan:
    """Semantic tile schedule consumed by architecture-specific backends."""

    problem: AttentionProblem
    source: AttentionTileSource
    schedule: AttentionTileSchedule
    policy_id: Optional[str] = None
    selection_reason: str = ""

    def __post_init__(self) -> None:
        selected = self.schedule.selected_tile_ids
        if selected is None:
            return
        if len(selected) != self.problem.batch_size:
            raise ValueError("selected tile schedule must contain one row per batch item")
        for row_ids, tile_count in zip(selected, self.source.logical_tile_counts):
            if tuple(sorted(set(row_ids))) != row_ids:
                raise ValueError("selected tile IDs must be sorted and unique")
            if any(tile < 0 or tile >= tile_count for tile in row_ids):
                raise ValueError("selected tile ID is outside the logical cache extent")
        if self.problem.guarantee == ATTENTION_GUARANTEE_EXACT:
            raise ValueError("exact attention requires an all-tile schedule")

    @classmethod
    def exact(
        cls,
        problem: AttentionProblem,
        *,
        logical_tile_size: int,
        reason: str = "exact_all_tiles",
    ) -> "AttentionTilePlan":
        if problem.guarantee != ATTENTION_GUARANTEE_EXACT:
            raise ValueError("exact tile plan requires exact problem guarantee")
        source = AttentionTileSource.from_problem(
            problem,
            logical_tile_size=logical_tile_size,
        )
        return cls(
            problem=problem,
            source=source,
            schedule=AttentionTileSchedule(kind=ATTENTION_SCHEDULE_ALL),
            selection_reason=reason,
        )

    @classmethod
    def selected(
        cls,
        problem: AttentionProblem,
        *,
        logical_tile_size: int,
        tile_ids_per_row: Sequence[Sequence[int]],
        policy_id: Optional[str],
        reason: str,
    ) -> "AttentionTilePlan":
        if problem.guarantee != ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED:
            raise ValueError("selected tile plan requires distribution-verified guarantee")
        source = AttentionTileSource.from_problem(
            problem,
            logical_tile_size=logical_tile_size,
        )
        selected = tuple(tuple(sorted(set(int(tile) for tile in row))) for row in tile_ids_per_row)
        return cls(
            problem=problem,
            source=source,
            schedule=AttentionTileSchedule(
                kind=ATTENTION_SCHEDULE_SELECTED,
                selected_tile_ids=selected,
            ),
            policy_id=policy_id,
            selection_reason=reason,
        )

    @property
    def scheduled_tile_counts(self) -> tuple[int, ...]:
        if self.schedule.selected_tile_ids is None:
            return self.source.logical_tile_counts
        return tuple(len(row) for row in self.schedule.selected_tile_ids)

    @property
    def tile_coverage(self) -> float:
        total = sum(self.source.logical_tile_counts)
        return 0.0 if total == 0 else sum(self.scheduled_tile_counts) / total

    def as_dict(self) -> dict[str, object]:
        selected = self.schedule.selected_tile_ids
        return {
            "problem": self.problem.as_dict(),
            "source": self.source.as_dict(),
            "schedule": {
                "kind": self.schedule.kind,
                "selected_tile_ids": (
                    None if selected is None else [list(row) for row in selected]
                ),
                "scheduled_tile_counts": list(self.scheduled_tile_counts),
                "tile_coverage": self.tile_coverage,
            },
            "policy_id": self.policy_id,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class AttentionBackendPlan:
    """Device execution choice for a semantic tile plan."""

    backend: str
    reason: str
    architecture: str
    splits: int = 1
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.splits <= 0 or self.workspace_bytes < 0:
            raise ValueError("splits must be positive and workspace_bytes non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "reason": self.reason,
            "architecture": self.architecture,
            "splits": self.splits,
            "workspace_bytes": self.workspace_bytes,
        }


def device_architecture(device: torch.device) -> str:
    """Return a stable architecture label without requiring CUDA."""

    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return resolved.type
    major, minor = torch.cuda.get_device_capability(resolved)
    return f"sm{major}{minor}"
