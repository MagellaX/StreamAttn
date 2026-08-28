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
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch


ATTENTION_GUARANTEE_EXACT = "exact"
ATTENTION_GUARANTEE_FULL_CONTEXT_EXACT = ATTENTION_GUARANTEE_EXACT
ATTENTION_GUARANTEE_SCHEDULE_EXACT = "schedule_exact"
ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED = "distribution_verified"
ATTENTION_GUARANTEES = frozenset(
    {
        ATTENTION_GUARANTEE_EXACT,
        ATTENTION_GUARANTEE_SCHEDULE_EXACT,
        ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    }
)

ATTENTION_CACHE_CONTIGUOUS = "contiguous"
ATTENTION_CACHE_PAGED = "paged"
ATTENTION_CACHE_KINDS = frozenset({ATTENTION_CACHE_CONTIGUOUS, ATTENTION_CACHE_PAGED})

ATTENTION_PHASE_DECODE = "decode"
ATTENTION_PHASE_PREFILL = "prefill"
ATTENTION_PHASE_TRAIN = "train"
ATTENTION_PHASES = frozenset(
    {
        ATTENTION_PHASE_DECODE,
        ATTENTION_PHASE_PREFILL,
        ATTENTION_PHASE_TRAIN,
    }
)

ATTENTION_SCHEDULE_ALL = "all"
ATTENTION_SCHEDULE_SELECTED = "selected"
ATTENTION_SCHEDULE_KINDS = frozenset({ATTENTION_SCHEDULE_ALL, ATTENTION_SCHEDULE_SELECTED})

ATTENTION_ROUTE_GRANULARITY_BATCH = "batch"
ATTENTION_ROUTE_GRANULARITY_KV_GROUP = "kv_group"
ATTENTION_ROUTE_GRANULARITY_Q_HEAD = "q_head"
ATTENTION_ROUTE_GRANULARITIES = frozenset(
    {
        ATTENTION_ROUTE_GRANULARITY_BATCH,
        ATTENTION_ROUTE_GRANULARITY_KV_GROUP,
        ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    }
)
ATTENTION_ROUTE_ABI_VERSION = 1


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
        if self.phase not in ATTENTION_PHASES:
            raise ValueError(f"unsupported attention phase: {self.phase}")
        if self.phase == ATTENTION_PHASE_DECODE and self.query_len != 1:
            raise ValueError("decode attention requires query_len == 1")
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
            phase=ATTENTION_PHASE_DECODE,
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
    def from_qkv(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        phase: str,
        guarantee: str = ATTENTION_GUARANTEE_EXACT,
        mask: str = "causal",
    ) -> "AttentionProblem":
        """Describe contiguous BHSD forward or training attention.

        This constructor records semantics only. Backend-specific limits, such
        as whether grouped-query prefill is implemented, are checked while
        lowering the problem to an execution plan.
        """

        if phase not in {ATTENTION_PHASE_PREFILL, ATTENTION_PHASE_TRAIN}:
            raise ValueError("from_qkv phase must be prefill or train")
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("forward Q/K/V must use rank-4 [B, S, H, D] tensors")
        if key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
            raise ValueError("query and key/value batch sizes and head dimensions must match")
        if query.shape[2] % key.shape[2]:
            raise ValueError("query heads must be a multiple of KV heads")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("query, key, and value dtypes must match")
        if query.device != key.device or query.device != value.device:
            raise ValueError("query, key, and value devices must match")
        batch = int(query.shape[0])
        kv_len = int(key.shape[1])
        return cls(
            phase=phase,
            guarantee=guarantee,
            mask=mask,
            batch_size=batch,
            query_len=int(query.shape[1]),
            q_heads=int(query.shape[2]),
            kv_heads=int(key.shape[2]),
            head_dim=int(query.shape[3]),
            dtype=_dtype_name(query.dtype),
            device=str(query.device),
            kv_lengths=(kv_len,) * batch,
            cache_kind=ATTENTION_CACHE_CONTIGUOUS,
            cache_layout="NHD",
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
            phase=ATTENTION_PHASE_DECODE,
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
class AttentionRouteCSR:
    """Immutable device ABI for an irregular selected-atom schedule.

    ``row_ptr`` and ``atom_ids`` describe logical cache atoms. They never
    contain K/V values or physical cache addresses. The backend lowering owns
    translation from logical atoms to physical pages.

    Route rows are explicitly scoped to a batch item, a ``(batch, KV-head)``
    group, or a ``(batch, Q-head)`` pair. Optional per-atom head masks are
    useful when a KV-group selector already emits a subset of the group's Q
    heads; Q-head schedules encode that ownership in the row index instead.
    """

    row_ptr: torch.Tensor
    atom_ids: torch.Tensor
    granularity: str
    atom_size: int
    schedule_epoch: int = 0
    abi_version: int = ATTENTION_ROUTE_ABI_VERSION
    active_head_masks: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.granularity not in ATTENTION_ROUTE_GRANULARITIES:
            raise ValueError(f"unsupported route granularity: {self.granularity}")
        if self.atom_size <= 0:
            raise ValueError("route atom_size must be positive")
        if self.schedule_epoch < 0:
            raise ValueError("schedule_epoch must be non-negative")
        if self.abi_version != ATTENTION_ROUTE_ABI_VERSION:
            raise ValueError(
                f"unsupported route ABI version: {self.abi_version}"
            )
        for name, tensor in (("row_ptr", self.row_ptr), ("atom_ids", self.atom_ids)):
            if tensor.dtype != torch.int32 or tensor.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional int32 tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if self.row_ptr.device != self.atom_ids.device:
            raise ValueError("route CSR tensors must share a device")
        if self.row_ptr.numel() == 0:
            raise ValueError("row_ptr must contain at least one offset")
        offsets = self.row_ptr.detach().to(device="cpu", dtype=torch.int64)
        if int(offsets[0]) != 0:
            raise ValueError("row_ptr must start at zero")
        if bool(torch.any(offsets[1:] < offsets[:-1])):
            raise ValueError("row_ptr must be monotonic non-decreasing")
        if int(offsets[-1]) != int(self.atom_ids.numel()):
            raise ValueError("row_ptr[-1] must equal atom_ids length")
        if self.atom_ids.numel() and bool(torch.any(self.atom_ids < 0).item()):
            raise ValueError("atom_ids must be non-negative")
        if self.active_head_masks is not None:
            masks = self.active_head_masks
            if masks.dtype != torch.int32 or masks.dim() != 1:
                raise ValueError(
                    "active_head_masks must be a one-dimensional int32 tensor"
                )
            if masks.device != self.atom_ids.device or not masks.is_contiguous():
                raise ValueError(
                    "active_head_masks must be contiguous and share the CSR device"
                )
            if masks.shape != self.atom_ids.shape:
                raise ValueError("active_head_masks must align with atom_ids")
            if masks.numel() and bool(torch.any(masks == 0).item()):
                raise ValueError("active_head_masks must select at least one head")
            if self.granularity == ATTENTION_ROUTE_GRANULARITY_Q_HEAD:
                raise ValueError(
                    "q_head route rows encode head ownership and must not carry masks"
                )

    @property
    def device(self) -> torch.device:
        return self.atom_ids.device

    @property
    def row_count(self) -> int:
        return int(self.row_ptr.numel()) - 1

    @property
    def nnz(self) -> int:
        return int(self.atom_ids.numel())

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence[int]],
        *,
        granularity: str,
        atom_size: int,
        device: torch.device | str = "cpu",
        schedule_epoch: int = 0,
        active_head_masks_per_row: Optional[Sequence[Sequence[int]]] = None,
    ) -> "AttentionRouteCSR":
        """Compile canonical host rows into the device CSR contract."""

        flat_atoms: list[int] = []
        flat_masks: list[int] = []
        offsets = [0]
        if (
            active_head_masks_per_row is not None
            and len(active_head_masks_per_row) != len(rows)
        ):
            raise ValueError("active head-mask rows must align with route rows")
        for row_idx, row in enumerate(rows):
            canonical = tuple(sorted(set(int(atom) for atom in row)))
            if any(atom < 0 for atom in canonical):
                raise ValueError("atom IDs must be non-negative")
            flat_atoms.extend(canonical)
            if active_head_masks_per_row is not None:
                source_masks = tuple(
                    int(mask) for mask in active_head_masks_per_row[row_idx]
                )
                if len(source_masks) != len(row):
                    raise ValueError(
                        "each active head-mask row must align with its atom row"
                    )
                mask_by_atom: dict[int, int] = {}
                for atom, mask in zip(row, source_masks, strict=True):
                    atom_id = int(atom)
                    mask_by_atom[atom_id] = mask_by_atom.get(atom_id, 0) | int(mask)
                for atom in canonical:
                    mask = mask_by_atom[atom]
                    if mask <= 0 or mask > 0xFFFFFFFF:
                        raise ValueError("active head masks must be unsigned 32-bit values")
                    flat_masks.append(mask if mask < (1 << 31) else mask - (1 << 32))
            offsets.append(len(flat_atoms))
        resolved_device = torch.device(device)
        masks_tensor = None
        if active_head_masks_per_row is not None:
            masks_tensor = torch.tensor(
                flat_masks,
                dtype=torch.int32,
                device=resolved_device,
            )
        return cls(
            row_ptr=torch.tensor(offsets, dtype=torch.int32, device=resolved_device),
            atom_ids=torch.tensor(flat_atoms, dtype=torch.int32, device=resolved_device),
            granularity=granularity,
            atom_size=int(atom_size),
            schedule_epoch=int(schedule_epoch),
            active_head_masks=masks_tensor,
        )

    def expected_row_count(self, problem: AttentionProblem) -> int:
        if self.granularity == ATTENTION_ROUTE_GRANULARITY_BATCH:
            return problem.batch_size
        if self.granularity == ATTENTION_ROUTE_GRANULARITY_KV_GROUP:
            return problem.batch_size * problem.kv_heads
        return problem.batch_size * problem.q_heads

    def validate_for(self, problem: AttentionProblem) -> None:
        """Validate row ownership and ragged logical bounds for a problem."""

        if self.row_count != self.expected_row_count(problem):
            raise ValueError(
                "route CSR row count does not match its declared granularity"
            )
        if str(self.device) != problem.device:
            raise ValueError("route CSR and attention problem must share a device")
        offsets = self.row_ptr.detach().to(device="cpu", dtype=torch.int64).tolist()
        atoms = self.atom_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        row_divisor = {
            ATTENTION_ROUTE_GRANULARITY_BATCH: 1,
            ATTENTION_ROUTE_GRANULARITY_KV_GROUP: problem.kv_heads,
            ATTENTION_ROUTE_GRANULARITY_Q_HEAD: problem.q_heads,
        }[self.granularity]
        for row_idx, (begin, end) in enumerate(
            zip(offsets[:-1], offsets[1:], strict=True)
        ):
            row_atoms = atoms[begin:end]
            if row_atoms != sorted(set(row_atoms)):
                raise ValueError("route CSR rows must contain sorted unique atom IDs")
            batch_idx = row_idx // row_divisor
            atom_count = math.ceil(problem.kv_lengths[batch_idx] / self.atom_size)
            if any(atom >= atom_count for atom in row_atoms):
                raise ValueError("route atom ID is outside the logical cache extent")

    def rows(self) -> tuple[tuple[int, ...], ...]:
        """Materialize host rows for diagnostics and plan-time compilation."""

        offsets = self.row_ptr.detach().to(device="cpu", dtype=torch.int64).tolist()
        atoms = self.atom_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        return tuple(
            tuple(atoms[begin:end])
            for begin, end in zip(offsets[:-1], offsets[1:], strict=True)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "abi_version": self.abi_version,
            "granularity": self.granularity,
            "atom_size": self.atom_size,
            "schedule_epoch": self.schedule_epoch,
            "row_count": self.row_count,
            "nnz": self.nnz,
            "device": str(self.device),
            "has_active_head_masks": self.active_head_masks is not None,
        }


@dataclass(frozen=True)
class AttentionTileSchedule:
    """Logical tile IDs required for each batch row."""

    kind: str
    selected_tile_ids: Optional[tuple[tuple[int, ...], ...]] = None
    route_granularity: str = ATTENTION_ROUTE_GRANULARITY_BATCH
    schedule_epoch: int = 0
    abi_version: int = ATTENTION_ROUTE_ABI_VERSION
    device_routes: Optional[AttentionRouteCSR] = None

    def __post_init__(self) -> None:
        if self.kind not in ATTENTION_SCHEDULE_KINDS:
            raise ValueError(f"unsupported tile schedule kind: {self.kind}")
        if self.route_granularity not in ATTENTION_ROUTE_GRANULARITIES:
            raise ValueError(f"unsupported route granularity: {self.route_granularity}")
        if self.schedule_epoch < 0:
            raise ValueError("schedule_epoch must be non-negative")
        if self.abi_version != ATTENTION_ROUTE_ABI_VERSION:
            raise ValueError(f"unsupported route ABI version: {self.abi_version}")
        if self.kind == ATTENTION_SCHEDULE_ALL:
            if self.selected_tile_ids is not None or self.device_routes is not None:
                raise ValueError("all-tile schedule must not materialize selected routes")
        elif self.selected_tile_ids is None and self.device_routes is None:
            raise ValueError("selected schedule requires tile IDs or device routes")
        if self.device_routes is not None:
            if self.device_routes.granularity != self.route_granularity:
                raise ValueError("device route granularity must match the schedule")
            if self.device_routes.schedule_epoch != self.schedule_epoch:
                raise ValueError("device route epoch must match the schedule")


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
        if (
            self.schedule.kind == ATTENTION_SCHEDULE_SELECTED
            and self.problem.guarantee == ATTENTION_GUARANTEE_EXACT
        ):
            raise ValueError("exact attention requires an all-tile schedule")
        if selected is not None:
            expected_rows = self._expected_schedule_rows()
            if len(selected) != expected_rows:
                raise ValueError(
                    "selected tile schedule row count does not match its granularity"
                )
            for row_idx, row_ids in enumerate(selected):
                batch_idx = self._schedule_row_batch(row_idx)
                tile_count = self.source.logical_tile_counts[batch_idx]
                if tuple(sorted(set(row_ids))) != row_ids:
                    raise ValueError("selected tile IDs must be sorted and unique")
                if any(tile < 0 or tile >= tile_count for tile in row_ids):
                    raise ValueError("selected tile ID is outside the logical cache extent")
        if self.schedule.device_routes is not None:
            routes = self.schedule.device_routes
            if routes.atom_size != self.source.logical_tile_size:
                raise ValueError("device route atom size must match the logical tile size")
            routes.validate_for(self.problem)
            if selected is not None and routes.rows() != selected:
                raise ValueError("device route CSR does not match selected host rows")

    def _expected_schedule_rows(self) -> int:
        granularity = self.schedule.route_granularity
        if granularity == ATTENTION_ROUTE_GRANULARITY_BATCH:
            return self.problem.batch_size
        if granularity == ATTENTION_ROUTE_GRANULARITY_KV_GROUP:
            return self.problem.batch_size * self.problem.kv_heads
        return self.problem.batch_size * self.problem.q_heads

    def _schedule_row_batch(self, row_idx: int) -> int:
        divisor = {
            ATTENTION_ROUTE_GRANULARITY_BATCH: 1,
            ATTENTION_ROUTE_GRANULARITY_KV_GROUP: self.problem.kv_heads,
            ATTENTION_ROUTE_GRANULARITY_Q_HEAD: self.problem.q_heads,
        }[self.schedule.route_granularity]
        return row_idx // divisor

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
        route_granularity: str = ATTENTION_ROUTE_GRANULARITY_BATCH,
        schedule_epoch: int = 0,
        device_routes: Optional[AttentionRouteCSR] = None,
    ) -> "AttentionTilePlan":
        if problem.guarantee not in {
            ATTENTION_GUARANTEE_SCHEDULE_EXACT,
            ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
        }:
            raise ValueError(
                "selected tile plan requires schedule-exact or "
                "distribution-verified guarantee"
            )
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
                route_granularity=route_granularity,
                schedule_epoch=int(schedule_epoch),
                device_routes=device_routes,
            ),
            policy_id=policy_id,
            selection_reason=reason,
        )

    @classmethod
    def selected_device(
        cls,
        problem: AttentionProblem,
        *,
        logical_tile_size: int,
        device_routes: AttentionRouteCSR,
        policy_id: Optional[str],
        reason: str,
    ) -> "AttentionTilePlan":
        """Build a selected plan from a device-produced CSR schedule only."""

        if problem.guarantee not in {
            ATTENTION_GUARANTEE_SCHEDULE_EXACT,
            ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
        }:
            raise ValueError(
                "selected tile plan requires schedule-exact or "
                "distribution-verified guarantee"
            )
        source = AttentionTileSource.from_problem(
            problem,
            logical_tile_size=logical_tile_size,
        )
        return cls(
            problem=problem,
            source=source,
            schedule=AttentionTileSchedule(
                kind=ATTENTION_SCHEDULE_SELECTED,
                route_granularity=device_routes.granularity,
                schedule_epoch=device_routes.schedule_epoch,
                device_routes=device_routes,
            ),
            policy_id=policy_id,
            selection_reason=reason,
        )

    def with_device_routes(
        self,
        *,
        device: torch.device | str,
    ) -> "AttentionTilePlan":
        """Compile selected host rows into the immutable device CSR ABI."""

        if self.schedule.kind != ATTENTION_SCHEDULE_SELECTED:
            raise ValueError("only selected schedules can compile device routes")
        if self.schedule.selected_tile_ids is None:
            raise ValueError("selected host rows are unavailable for CSR compilation")
        routes = AttentionRouteCSR.from_rows(
            self.schedule.selected_tile_ids,
            granularity=self.schedule.route_granularity,
            atom_size=self.source.logical_tile_size,
            device=device,
            schedule_epoch=self.schedule.schedule_epoch,
        )
        routes.validate_for(self.problem)
        return replace(
            self,
            schedule=replace(self.schedule, device_routes=routes),
        )

    @property
    def scheduled_tile_counts(self) -> tuple[int, ...]:
        if self.schedule.kind == ATTENTION_SCHEDULE_ALL:
            return self.source.logical_tile_counts
        if self.schedule.selected_tile_ids is not None:
            return tuple(len(row) for row in self.schedule.selected_tile_ids)
        assert self.schedule.device_routes is not None
        offsets = self.schedule.device_routes.row_ptr.detach().to(
            device="cpu", dtype=torch.int64
        )
        return tuple(int(value) for value in (offsets[1:] - offsets[:-1]).tolist())

    @property
    def tile_coverage(self) -> float:
        if self.schedule.kind == ATTENTION_SCHEDULE_ALL:
            return 1.0
        total = sum(
            self.source.logical_tile_counts[self._schedule_row_batch(row_idx)]
            for row_idx in range(self._expected_schedule_rows())
        )
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
                "route_granularity": self.schedule.route_granularity,
                "schedule_epoch": self.schedule.schedule_epoch,
                "abi_version": self.schedule.abi_version,
                "device_routes": (
                    None
                    if self.schedule.device_routes is None
                    else self.schedule.device_routes.as_dict()
                ),
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
