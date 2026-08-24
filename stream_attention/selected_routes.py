"""Physical route preparation for selected paged decode attention.

This module lowers a logical selected-atom CSR schedule into compact 64-token
page routes. It never reads or copies K/V values. The output is metadata for a
native backend that loads four physical page-16 fragments directly into one
compute tile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .planning import (
    ATTENTION_CACHE_PAGED,
    ATTENTION_ROUTE_ABI_VERSION,
    ATTENTION_ROUTE_GRANULARITY_BATCH,
    ATTENTION_ROUTE_GRANULARITY_KV_GROUP,
    ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    ATTENTION_SCHEDULE_SELECTED,
    AttentionRouteCSR,
    AttentionTilePlan,
)

if TYPE_CHECKING:
    from .paged import PagedKVCache


PACKED_ROUTE64_ABI_VERSION = 1
PACKED_ROUTE64_TOKENS = 64
PACKED_ROUTE64_PAGE_SIZE = 16
PACKED_ROUTE64_ATOMS = PACKED_ROUTE64_TOKENS // PACKED_ROUTE64_PAGE_SIZE

PACKED_ROUTE_FLAG_STRUCTURALLY_FULL = 1 << 0
PACKED_ROUTE_FLAG_ALL_HEADS = 1 << 1
PACKED_ROUTE_FLAG_TOKEN_FULL = 1 << 2

SELECTED_SCHEDULER_STATIC_UNIFORM = "static_uniform"
SELECTED_SCHEDULER_COMPACT_STATIC = "compact_static"


def _signed_i32_bits(values: torch.Tensor) -> torch.Tensor:
    """Store unsigned 32-bit masks in an int32 tensor without losing bits."""

    values_i64 = values.to(torch.int64) & 0xFFFFFFFF
    signed = torch.where(values_i64 >= (1 << 31), values_i64 - (1 << 32), values_i64)
    return signed.to(torch.int32)


def _full_head_mask(group_size: int) -> int:
    if group_size <= 0 or group_size > 32:
        raise ValueError("PackedRoute64 supports GQA groups in [1, 32]")
    return (1 << group_size) - 1


def _tensor_version(tensor: torch.Tensor) -> int:
    # PyTorch increments this counter for in-place writes, including live page
    # table remaps that retain the same storage address.
    return int(tensor._version)


@dataclass(frozen=True)
class PackedPagedRoute64:
    """Prepared page-16 metadata for 64-token selected-attention routes.

    Rows are always ``(batch, KV head)`` groups, regardless of the semantic
    selector granularity. Each atom carries its own Q-head mask because four
    atoms packed into one route may have different per-head selections.
    """

    row_ptr: torch.Tensor
    logical_atom_origins: torch.Tensor
    physical_page_ids: torch.Tensor
    atom_valid_masks: torch.Tensor
    active_head_masks: torch.Tensor
    token_valid_masks: torch.Tensor
    route_flags: torch.Tensor
    schedule_epoch: int
    source_route_abi_version: int
    abi_version: int
    group_route_efficiency: float
    scheduler_hint: str
    page_table_data_ptr: int
    page_table_version: int
    sequence_lengths_data_ptr: int
    sequence_lengths_version: int

    def __post_init__(self) -> None:
        if self.abi_version != PACKED_ROUTE64_ABI_VERSION:
            raise ValueError("unsupported PackedRoute64 ABI version")
        if self.source_route_abi_version != ATTENTION_ROUTE_ABI_VERSION:
            raise ValueError("unsupported source route ABI version")
        if self.schedule_epoch < 0:
            raise ValueError("schedule_epoch must be non-negative")
        if not 0.0 <= self.group_route_efficiency <= 1.0:
            raise ValueError("group_route_efficiency must be in [0, 1]")
        if self.scheduler_hint not in {
            SELECTED_SCHEDULER_STATIC_UNIFORM,
            SELECTED_SCHEDULER_COMPACT_STATIC,
        }:
            raise ValueError("unsupported selected-route scheduler hint")
        tensors = (
            self.row_ptr,
            self.logical_atom_origins,
            self.physical_page_ids,
            self.atom_valid_masks,
            self.active_head_masks,
            self.token_valid_masks,
            self.route_flags,
        )
        if len({str(tensor.device) for tensor in tensors}) != 1:
            raise ValueError("PackedRoute64 tensors must share a device")
        if not all(tensor.dtype == torch.int32 for tensor in tensors):
            raise ValueError("PackedRoute64 tensors must use int32 storage")
        if not all(tensor.is_contiguous() for tensor in tensors):
            raise ValueError("PackedRoute64 tensors must be contiguous")
        route_count = int(self.logical_atom_origins.shape[0])
        route_shape = (route_count, PACKED_ROUTE64_ATOMS)
        if self.row_ptr.dim() != 1 or self.row_ptr.numel() == 0:
            raise ValueError("PackedRoute64 row_ptr must be one-dimensional")
        offsets = self.row_ptr.detach().to(device="cpu", dtype=torch.int64)
        if int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
            raise ValueError("PackedRoute64 row_ptr must be canonical CSR offsets")
        if tuple(self.logical_atom_origins.shape) != route_shape:
            raise ValueError("logical_atom_origins must be [routes, 4]")
        if tuple(self.physical_page_ids.shape) != route_shape:
            raise ValueError("physical_page_ids must be [routes, 4]")
        if tuple(self.active_head_masks.shape) != route_shape:
            raise ValueError("active_head_masks must be [routes, 4]")
        if tuple(self.token_valid_masks.shape) != route_shape:
            raise ValueError("token_valid_masks must be [routes, 4]")
        if tuple(self.atom_valid_masks.shape) != (route_count,):
            raise ValueError("atom_valid_masks must be [routes]")
        if tuple(self.route_flags.shape) != (route_count,):
            raise ValueError("route_flags must be [routes]")
        if int(self.row_ptr[-1].item()) != route_count:
            raise ValueError("PackedRoute64 row_ptr[-1] must equal route count")

    @property
    def device(self) -> torch.device:
        return self.row_ptr.device

    @property
    def row_count(self) -> int:
        return int(self.row_ptr.numel()) - 1

    @property
    def route_count(self) -> int:
        return int(self.logical_atom_origins.shape[0])

    @property
    def metadata_bytes(self) -> int:
        tensors = (
            self.row_ptr,
            self.logical_atom_origins,
            self.physical_page_ids,
            self.atom_valid_masks,
            self.active_head_masks,
            self.token_valid_masks,
            self.route_flags,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def validate_current(
        self,
        cache: "PagedKVCache",
        *,
        schedule_epoch: int,
    ) -> None:
        """Reject stale prepared routes after schedule or page metadata mutation."""

        if int(schedule_epoch) != self.schedule_epoch:
            raise RuntimeError("prepared route schedule epoch is stale")
        if cache.page_table.data_ptr() != self.page_table_data_ptr:
            raise RuntimeError("prepared route page-table storage changed")
        if _tensor_version(cache.page_table) != self.page_table_version:
            raise RuntimeError("prepared route page-table contents changed")
        if cache.sequence_lengths.data_ptr() != self.sequence_lengths_data_ptr:
            raise RuntimeError("prepared route sequence-length storage changed")
        if _tensor_version(cache.sequence_lengths) != self.sequence_lengths_version:
            raise RuntimeError("prepared route sequence lengths changed")

    def as_dict(self) -> dict[str, object]:
        row_counts = self.row_ptr[1:] - self.row_ptr[:-1]
        return {
            "abi_version": self.abi_version,
            "source_route_abi_version": self.source_route_abi_version,
            "schedule_epoch": self.schedule_epoch,
            "row_count": self.row_count,
            "route_count": self.route_count,
            "metadata_bytes": self.metadata_bytes,
            "group_route_efficiency": self.group_route_efficiency,
            "scheduler_hint": self.scheduler_hint,
            "min_routes_per_row": int(row_counts.min().item()) if row_counts.numel() else 0,
            "max_routes_per_row": int(row_counts.max().item()) if row_counts.numel() else 0,
            "device": str(self.device),
        }


def _route_rows_on_device(plan: AttentionTilePlan, device: torch.device) -> AttentionRouteCSR:
    routes = plan.schedule.device_routes
    if routes is not None:
        if routes.device != device:
            raise ValueError("device routes and paged cache must share a device")
        return routes
    if plan.schedule.selected_tile_ids is None:
        raise ValueError("selected plan has no route rows to lower")
    return AttentionRouteCSR.from_rows(
        plan.schedule.selected_tile_ids,
        granularity=plan.schedule.route_granularity,
        atom_size=plan.source.logical_tile_size,
        device=device,
        schedule_epoch=plan.schedule.schedule_epoch,
    )


def _validate_paged_lowering_inputs(
    plan: AttentionTilePlan,
    cache: "PagedKVCache",
) -> None:
    if plan.problem.cache_kind != ATTENTION_CACHE_PAGED:
        raise ValueError("PackedRoute64 lowering requires a paged attention problem")
    if plan.schedule.kind != ATTENTION_SCHEDULE_SELECTED:
        raise ValueError("PackedRoute64 lowering requires a selected schedule")
    if cache.page_size != PACKED_ROUTE64_PAGE_SIZE:
        raise ValueError("PackedRoute64 currently requires page size 16")
    if plan.source.logical_tile_size % cache.page_size:
        raise ValueError("logical route atoms must be divisible into physical pages")
    if plan.problem.page_size != cache.page_size:
        raise ValueError("plan and cache page sizes do not match")
    if plan.problem.batch_size != cache.batch_size:
        raise ValueError("plan and cache batch sizes do not match")
    if plan.problem.kv_heads != cache.kv_heads:
        raise ValueError("plan and cache KV-head counts do not match")
    if plan.problem.cache_layout != cache.normalized_layout:
        raise ValueError("plan and cache layouts do not match")
    if str(cache.page_table.device) != plan.problem.device:
        raise ValueError("plan and cache must share a device")
    live_lengths = tuple(
        int(value)
        for value in cache.sequence_lengths.detach().to(device="cpu", dtype=torch.int64).tolist()
    )
    if live_lengths != plan.problem.kv_lengths:
        raise ValueError("plan KV lengths are stale relative to the paged cache")


def _expand_to_page_atoms(
    routes: AttentionRouteCSR,
    plan: AttentionTilePlan,
    cache: "PagedKVCache",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return semantic row, page-atom ID, and unsigned active-head mask."""

    device = routes.device
    row_counts = (routes.row_ptr[1:] - routes.row_ptr[:-1]).to(torch.int64)
    source_rows = torch.repeat_interleave(
        torch.arange(routes.row_count, device=device, dtype=torch.int64),
        row_counts,
    )
    fragments = routes.atom_size // cache.page_size
    expanded_rows = source_rows.repeat_interleave(fragments)
    expanded_atoms = routes.atom_ids.to(torch.int64).repeat_interleave(fragments)
    fragment_offsets = torch.arange(fragments, device=device, dtype=torch.int64).repeat(
        routes.nnz
    )
    page_atoms = expanded_atoms * fragments + fragment_offsets

    if routes.active_head_masks is None:
        unsigned_masks = torch.full(
            (routes.nnz,),
            _full_head_mask(plan.problem.group_size),
            device=device,
            dtype=torch.int64,
        )
    else:
        unsigned_masks = routes.active_head_masks.to(torch.int64) & 0xFFFFFFFF
    unsigned_masks = unsigned_masks.repeat_interleave(fragments)

    divisor = {
        ATTENTION_ROUTE_GRANULARITY_BATCH: 1,
        ATTENTION_ROUTE_GRANULARITY_KV_GROUP: plan.problem.kv_heads,
        ATTENTION_ROUTE_GRANULARITY_Q_HEAD: plan.problem.q_heads,
    }[routes.granularity]
    batch_ids = expanded_rows // divisor
    lengths = cache.sequence_lengths.to(torch.int64)
    live = page_atoms * cache.page_size < lengths[batch_ids]
    return expanded_rows[live], page_atoms[live], unsigned_masks[live]


def _coalesce_group_atoms(
    source_rows: torch.Tensor,
    page_atoms: torch.Tensor,
    source_masks: torch.Tensor,
    *,
    routes: AttentionRouteCSR,
    plan: AttentionTilePlan,
    cache: "PagedKVCache",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Map semantic rows to KV groups and union duplicate per-head atoms."""

    device = page_atoms.device
    group_size = plan.problem.group_size
    full_mask = _full_head_mask(group_size)
    if routes.granularity == ATTENTION_ROUTE_GRANULARITY_BATCH:
        kv_offsets = torch.arange(
            plan.problem.kv_heads,
            device=device,
            dtype=torch.int64,
        )
        group_rows = (
            source_rows[:, None] * plan.problem.kv_heads + kv_offsets[None, :]
        ).reshape(-1)
        page_atoms = page_atoms.repeat_interleave(plan.problem.kv_heads)
        source_masks = source_masks.repeat_interleave(plan.problem.kv_heads)
    elif routes.granularity == ATTENTION_ROUTE_GRANULARITY_KV_GROUP:
        group_rows = source_rows
    else:
        q_head = source_rows % plan.problem.q_heads
        batch_ids = source_rows // plan.problem.q_heads
        kv_head = q_head // group_size
        local_head = q_head % group_size
        group_rows = batch_ids * plan.problem.kv_heads + kv_head
        source_masks = torch.ones_like(local_head) << local_head

    max_page_atoms = math.ceil(plan.problem.max_kv_len / cache.page_size)
    composite = group_rows * max_page_atoms + page_atoms
    order = torch.argsort(composite, stable=True)
    composite = composite[order]
    source_masks = source_masks[order]
    unique, inverse = torch.unique_consecutive(composite, return_inverse=True)
    union_masks = torch.zeros(unique.shape, device=device, dtype=torch.int64)
    union_masks.scatter_add_(0, inverse, source_masks)
    union_masks &= 0xFFFFFFFF
    union_group_rows = unique // max_page_atoms
    union_page_atoms = unique % max_page_atoms

    mask_values = union_masks.detach().to(device="cpu", dtype=torch.int64).tolist()
    selected_head_atoms = sum((int(mask) & full_mask).bit_count() for mask in mask_values)
    denominator = group_size * len(mask_values)
    efficiency = 1.0 if denominator == 0 else selected_head_atoms / denominator
    return union_group_rows, union_page_atoms, union_masks, float(efficiency)


def prepare_paged_routes64(
    plan: AttentionTilePlan,
    cache: "PagedKVCache",
) -> PackedPagedRoute64:
    """Lower selected logical routes to physical page-16 metadata.

    All tensor transforms stay on the route/cache device. The operation may
    synchronize for plan-time validation and diagnostics; a future dynamic
    selector path will replace this adapter with a dedicated preparation
    kernel while preserving the same output ABI.
    """

    _validate_paged_lowering_inputs(plan, cache)
    routes = _route_rows_on_device(plan, cache.page_table.device)
    routes.validate_for(plan.problem)
    source_rows, page_atoms, source_masks = _expand_to_page_atoms(
        routes,
        plan,
        cache,
    )
    group_rows, page_atoms, head_masks, efficiency = _coalesce_group_atoms(
        source_rows,
        page_atoms,
        source_masks,
        routes=routes,
        plan=plan,
        cache=cache,
    )

    device = cache.page_table.device
    num_group_rows = plan.problem.batch_size * plan.problem.kv_heads
    atom_counts = torch.bincount(group_rows, minlength=num_group_rows).to(torch.int64)
    atom_row_ptr = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.int64),
            torch.cumsum(atom_counts, dim=0),
        )
    )
    route_counts = torch.div(
        atom_counts + PACKED_ROUTE64_ATOMS - 1,
        PACKED_ROUTE64_ATOMS,
        rounding_mode="floor",
    )
    route_row_ptr_i64 = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.int64),
            torch.cumsum(route_counts, dim=0),
        )
    )
    total_routes = int(route_row_ptr_i64[-1].item())
    packed_group_rows = torch.repeat_interleave(
        torch.arange(num_group_rows, device=device, dtype=torch.int64),
        route_counts,
    )
    route_row_starts = torch.repeat_interleave(route_row_ptr_i64[:-1], route_counts)
    route_in_row = torch.arange(total_routes, device=device, dtype=torch.int64) - route_row_starts
    source_offsets = (
        atom_row_ptr[packed_group_rows, None]
        + route_in_row[:, None] * PACKED_ROUTE64_ATOMS
        + torch.arange(PACKED_ROUTE64_ATOMS, device=device, dtype=torch.int64)[None, :]
    )
    source_ends = atom_row_ptr[packed_group_rows + 1, None]
    valid = source_offsets < source_ends
    safe_offsets = source_offsets.clamp(max=max(0, int(page_atoms.numel()) - 1))

    if page_atoms.numel() == 0:
        selected_page_atoms = torch.empty(
            (0, PACKED_ROUTE64_ATOMS), device=device, dtype=torch.int64
        )
        selected_head_masks = torch.empty_like(selected_page_atoms)
    else:
        selected_page_atoms = page_atoms[safe_offsets]
        selected_head_masks = head_masks[safe_offsets]
        selected_page_atoms = torch.where(valid, selected_page_atoms, -1)
        selected_head_masks = torch.where(valid, selected_head_masks, 0)

    batch_ids = packed_group_rows // plan.problem.kv_heads
    safe_page_atoms = selected_page_atoms.clamp(min=0)
    physical_page_ids = cache.page_table[batch_ids[:, None], safe_page_atoms]
    physical_valid = (physical_page_ids >= 0) & (physical_page_ids < cache.num_pages)
    if bool(torch.any(valid & ~physical_valid).item()):
        raise ValueError("selected logical atom resolves to an invalid physical page")
    valid = valid & physical_valid
    physical_page_ids = torch.where(valid, physical_page_ids, -1)
    logical_origins = selected_page_atoms * cache.page_size
    logical_origins = torch.where(valid, logical_origins, -1)
    selected_head_masks = torch.where(valid, selected_head_masks, 0)

    remaining = cache.sequence_lengths.to(torch.int64)[batch_ids, None] - logical_origins
    valid_tokens = remaining.clamp(min=0, max=cache.page_size)
    token_masks = (torch.ones_like(valid_tokens) << valid_tokens) - 1
    token_masks = torch.where(valid, token_masks, 0)

    slots = torch.arange(PACKED_ROUTE64_ATOMS, device=device, dtype=torch.int64)
    atom_valid_masks = torch.sum(valid.to(torch.int64) << slots[None, :], dim=1)
    full_head_mask = _full_head_mask(plan.problem.group_size)
    structurally_full = atom_valid_masks == (1 << PACKED_ROUTE64_ATOMS) - 1
    all_heads = torch.all(
        torch.where(valid, selected_head_masks == full_head_mask, True),
        dim=1,
    )
    token_full = torch.all(
        torch.where(valid, token_masks == (1 << cache.page_size) - 1, True),
        dim=1,
    )
    route_flags = (
        structurally_full.to(torch.int64) * PACKED_ROUTE_FLAG_STRUCTURALLY_FULL
        + all_heads.to(torch.int64) * PACKED_ROUTE_FLAG_ALL_HEADS
        + token_full.to(torch.int64) * PACKED_ROUTE_FLAG_TOKEN_FULL
    )
    scheduler_hint = (
        SELECTED_SCHEDULER_STATIC_UNIFORM
        if route_counts.numel() == 0 or bool(torch.all(route_counts == route_counts[:1]).item())
        else SELECTED_SCHEDULER_COMPACT_STATIC
    )

    return PackedPagedRoute64(
        row_ptr=route_row_ptr_i64.to(torch.int32).contiguous(),
        logical_atom_origins=logical_origins.to(torch.int32).contiguous(),
        physical_page_ids=physical_page_ids.to(torch.int32).contiguous(),
        atom_valid_masks=atom_valid_masks.to(torch.int32).contiguous(),
        active_head_masks=_signed_i32_bits(selected_head_masks).contiguous(),
        token_valid_masks=token_masks.to(torch.int32).contiguous(),
        route_flags=route_flags.to(torch.int32).contiguous(),
        schedule_epoch=routes.schedule_epoch,
        source_route_abi_version=routes.abi_version,
        abi_version=PACKED_ROUTE64_ABI_VERSION,
        group_route_efficiency=efficiency,
        scheduler_hint=scheduler_hint,
        page_table_data_ptr=cache.page_table.data_ptr(),
        page_table_version=_tensor_version(cache.page_table),
        sequence_lengths_data_ptr=cache.sequence_lengths.data_ptr(),
        sequence_lengths_version=_tensor_version(cache.sequence_lengths),
    )


__all__ = [
    "PACKED_ROUTE64_ABI_VERSION",
    "PACKED_ROUTE64_ATOMS",
    "PACKED_ROUTE64_PAGE_SIZE",
    "PACKED_ROUTE64_TOKENS",
    "PACKED_ROUTE_FLAG_ALL_HEADS",
    "PACKED_ROUTE_FLAG_STRUCTURALLY_FULL",
    "PACKED_ROUTE_FLAG_TOKEN_FULL",
    "SELECTED_SCHEDULER_COMPACT_STATIC",
    "SELECTED_SCHEDULER_STATIC_UNIFORM",
    "PackedPagedRoute64",
    "prepare_paged_routes64",
]
