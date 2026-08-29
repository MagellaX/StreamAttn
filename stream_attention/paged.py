"""Paged-KV cache contracts and exact decode planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .planning import ATTENTION_ROUTE_GRANULARITY_Q_HEAD, AttentionRouteCSR
from .selected_routes import PackedPagedRoute64


PAGED_EXACT_NATIVE_BACKEND = "streamattn_paged_exact_native"
PAGED_EXACT_SM80_CP_ASYNC_BACKEND = "streamattn_paged_sm80_cp_async_exact"
PAGED_EXACT_SM80_GROUPED_BACKEND = "streamattn_paged_sm80_grouped_exact"
PAGED_EXACT_SM100_GROUPED_BACKEND = "streamattn_paged_sm100_grouped_exact"
PAGED_EXACT_SM100_TGV_BACKEND = "streamattn_paged_sm100_tgv_exact"

# Direct-NHD page-16 D128/G8 cells that won both a B200 architecture phase and
# an independent 15-trial paired confirmation against FlashInfer 0.6.17.
PROMOTED_PAGED_EXACT_SM100_TGV_SPLITS = {
    (1, 32768): 16,
    (2, 32768): 16,
    (2, 65536): 16,
    (4, 32768): 8,
    (4, 65536): 8,
    (8, 32768): 4,
}
PAGED_EXACT_SM90_BACKEND = "streamattn_paged_sm90_wgmma_exact"
PAGED_EXACT_SM90_FRAGMENTED_BACKEND = "streamattn_paged_sm90_wgmma_fragmented_exact"
PAGED_EXACT_SM90_FRAGMENTED_RAGGED_BACKEND = (
    "streamattn_paged_sm90_wgmma_fragmented_ragged_exact"
)
PAGED_EXACT_SM90_NHD_FRAGMENTED_BACKEND = (
    "streamattn_paged_sm90_wgmma_nhd_fragmented_exact"
)
PAGED_EXACT_SM90_NHD_FRAGMENTED_RAGGED_BACKEND = (
    "streamattn_paged_sm90_wgmma_nhd_fragmented_ragged_exact"
)
PAGED_SELECTED_SM90_STATIC_BACKEND = "streamattn_paged_sm90_wgmma_selected_static"
PAGED_SELECTED_SM90_DYNAMIC_QHEAD_BACKEND = (
    "streamattn_paged_sm90_wgmma_selected_dynamic_qhead"
)
PAGED_QUERY_SELECTED_SM90_BACKEND = (
    "streamattn_paged_sm90_wgmma_query_selected_qhead"
)
PAGED_QUERY_REFINED_SM90_BACKEND = (
    "streamattn_paged_sm90_wgmma_query_refined_qhead"
)

PROMOTED_PAGED_EXACT_SPLITS = {
    (1, 16384): 64,
    (1, 32768): 64,
    (1, 65536): 64,
    (2, 16384): 64,
    (2, 32768): 64,
    (2, 65536): 64,
    (4, 16384): 32,
    (4, 32768): 32,
    (4, 65536): 64,
    (8, 16384): 32,
    (8, 32768): 32,
    (8, 65536): 32,
}

PROMOTED_PAGED_EXACT_PAGE16_SPLITS = {
    (1, 16384): 64,
    (1, 32768): 64,
    (1, 65536): 64,
    (2, 16384): 64,
    (2, 32768): 64,
    (2, 65536): 64,
    (4, 16384): 64,
    (4, 32768): 64,
    (4, 65536): 64,
    (8, 16384): 32,
    (8, 32768): 32,
    (8, 65536): 32,
}

# Ragged rows use the same launch geometry as full rows. H100 evidence covers
# every batch/bucket cell below from a one-token endpoint through full length.
PROMOTED_PAGED_EXACT_PAGE16_RAGGED_SPLITS = dict(PROMOTED_PAGED_EXACT_PAGE16_SPLITS)

# D128 uses a two-phase K/V shared-memory pipeline. The split tables below are
# measured H100 optima against FlashInfer FA2, not extrapolated shape support.
PROMOTED_PAGED_EXACT_PAGE16_D128_G8_SPLITS = {
    (1, 16384): 128,
    (1, 32768): 128,
    (1, 65536): 128,
    (2, 16384): 64,
    (2, 32768): 64,
    (2, 65536): 64,
    (4, 16384): 32,
    (4, 32768): 32,
    (4, 65536): 32,
    (8, 16384): 16,
    (8, 32768): 16,
    (8, 65536): 16,
}

PROMOTED_PAGED_EXACT_PAGE16_D128_G8_RAGGED_SPLITS = dict(
    PROMOTED_PAGED_EXACT_PAGE16_D128_G8_SPLITS
)
# Variable-length producers need a balanced third wave at this boundary.
PROMOTED_PAGED_EXACT_PAGE16_D128_G8_RAGGED_SPLITS[(8, 65536)] = 24

PROMOTED_PAGED_EXACT_PAGE16_D128_G4_SPLITS = {
    (1, 16384): 32,
    (1, 32768): 32,
    (1, 65536): 32,
    (2, 16384): 16,
    (2, 32768): 16,
    (2, 65536): 16,
    (4, 16384): 8,
    (4, 32768): 8,
    (4, 65536): 8,
    (8, 16384): 8,
    (8, 32768): 8,
    (8, 65536): 8,
}

PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS = dict(
    PROMOTED_PAGED_EXACT_PAGE16_D128_G4_SPLITS
)
PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS[(4, 32768)] = 12
PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS[(4, 65536)] = 16
PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS[(8, 32768)] = 12
PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS[(8, 65536)] = 16

PROMOTED_PAGED_EXACT_PAGE16_SHAPES = {
    (16, 2, 8, 64): PROMOTED_PAGED_EXACT_PAGE16_SPLITS,
    (16, 2, 8, 128): PROMOTED_PAGED_EXACT_PAGE16_D128_G8_SPLITS,
    (32, 8, 4, 128): PROMOTED_PAGED_EXACT_PAGE16_D128_G4_SPLITS,
}

PROMOTED_PAGED_EXACT_PAGE16_RAGGED_SHAPES = {
    (16, 2, 8, 64): PROMOTED_PAGED_EXACT_PAGE16_RAGGED_SPLITS,
    (16, 2, 8, 128): PROMOTED_PAGED_EXACT_PAGE16_D128_G8_RAGGED_SPLITS,
    (32, 8, 4, 128): PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS,
}

# Direct NHD D128/G8 measured optima on H100. These copy strided token rows
# directly into the WGMMA shared tile; no cache transpose or repack is allowed.
PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_SPLITS = {
    (1, 16384): 64,
    (1, 32768): 128,
    (1, 65536): 64,
    (2, 16384): 64,
    (2, 32768): 64,
    (2, 65536): 128,
    (4, 16384): 32,
    (4, 32768): 32,
    (4, 65536): 64,
    (8, 16384): 32,
    (8, 32768): 32,
    (8, 65536): 32,
}

PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_RAGGED_SPLITS = {
    (1, 16384): 64,
    (1, 32768): 64,
    (1, 65536): 128,
    (2, 16384): 64,
    (2, 32768): 64,
    (2, 65536): 128,
    (4, 16384): 64,
    (4, 32768): 32,
    (4, 65536): 64,
    (8, 16384): 16,
    (8, 32768): 32,
    (8, 65536): 32,
}

PROMOTED_PAGED_EXACT_PAGE16_NHD_SHAPES = {
    (16, 2, 8, 128): PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_SPLITS,
}

PROMOTED_PAGED_EXACT_PAGE16_NHD_RAGGED_SHAPES = {
    (16, 2, 8, 128): PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_RAGGED_SPLITS,
}


@dataclass(frozen=True)
class PagedKVCache:
    """Physical KV pages plus a logical request-to-page mapping.

    NHD pages are [num_pages, page_size, kv_heads, head_dim].
    HND pages are [num_pages, kv_heads, page_size, head_dim].
    page_table is [batch, max_pages_per_request] and may use -1 for inactive
    trailing slots. sequence_lengths is [batch].
    """

    key: torch.Tensor
    value: torch.Tensor
    page_table: torch.Tensor
    sequence_lengths: torch.Tensor
    layout: str = "NHD"

    @property
    def normalized_layout(self) -> str:
        return str(self.layout).upper()

    @property
    def num_pages(self) -> int:
        return int(self.key.shape[0]) if self.key.dim() == 4 else 0

    @property
    def page_size(self) -> int:
        if self.key.dim() != 4:
            return 0
        return int(
            self.key.shape[1] if self.normalized_layout == "NHD" else self.key.shape[2]
        )

    @property
    def kv_heads(self) -> int:
        if self.key.dim() != 4:
            return 0
        return int(
            self.key.shape[2] if self.normalized_layout == "NHD" else self.key.shape[1]
        )

    @property
    def head_dim(self) -> int:
        return int(self.key.shape[3]) if self.key.dim() == 4 else 0

    @property
    def batch_size(self) -> int:
        return int(self.page_table.shape[0]) if self.page_table.dim() == 2 else 0

    @property
    def max_pages_per_request(self) -> int:
        return int(self.page_table.shape[1]) if self.page_table.dim() == 2 else 0

    @property
    def max_sequence_length(self) -> int:
        return self.max_pages_per_request * self.page_size

    def validate(
        self,
        query: torch.Tensor,
        *,
        validate_metadata: bool = True,
    ) -> None:
        """Validate shape, device, layout, and active page-table entries."""

        if query.dim() != 4 or query.shape[1] != 1:
            raise ValueError("paged exact decode query must be [batch, 1, heads, dim]")
        if self.key.dim() != 4 or self.value.dim() != 4:
            raise ValueError("paged key/value caches must be rank 4")
        if self.key.shape != self.value.shape:
            raise ValueError("paged key/value cache shapes must match")
        if self.normalized_layout not in {"NHD", "HND"}:
            raise ValueError("paged KV layout must be NHD or HND")
        if self.page_table.dim() != 2:
            raise ValueError("page_table must be [batch, max_pages_per_request]")
        if self.sequence_lengths.dim() != 1:
            raise ValueError("sequence_lengths must be [batch]")
        if self.batch_size != int(query.shape[0]):
            raise ValueError("query and page_table batch sizes must match")
        if self.sequence_lengths.numel() != self.batch_size:
            raise ValueError("sequence_lengths size must match batch")
        if (
            self.num_pages <= 0
            or self.page_size <= 0
            or self.max_pages_per_request <= 0
        ):
            raise ValueError("paged cache dimensions must be positive")
        if self.page_size > 256:
            raise ValueError("paged exact decode currently supports page_size <= 256")
        if self.head_dim != int(query.shape[3]):
            raise ValueError("query and paged cache head dimensions must match")
        if self.kv_heads <= 0 or int(query.shape[2]) % self.kv_heads:
            raise ValueError("query heads must be a multiple of KV heads")
        if self.head_dim > 256 or self.head_dim & (self.head_dim - 1):
            raise ValueError("head_dim must be a power of two no larger than 256")
        if query.dtype != self.key.dtype or query.dtype != self.value.dtype:
            raise ValueError("query and paged key/value dtypes must match")
        if not torch.is_floating_point(query):
            raise ValueError("query and paged key/value tensors must be floating point")
        if self.page_table.dtype not in {torch.int32, torch.int64}:
            raise ValueError("page_table must use int32 or int64")
        if self.sequence_lengths.dtype not in {torch.int32, torch.int64}:
            raise ValueError("sequence_lengths must use int32 or int64")
        tensors = (
            query,
            self.key,
            self.value,
            self.page_table,
            self.sequence_lengths,
        )
        if len({str(t.device) for t in tensors}) != 1:
            raise ValueError("query and paged cache metadata must share a device")
        if not all(t.is_contiguous() for t in tensors):
            raise ValueError("query and paged cache tensors must be contiguous")

        if not validate_metadata:
            return
        lengths = self.sequence_lengths.detach().to(device="cpu", dtype=torch.int64)
        if bool(torch.any(lengths <= 0)):
            raise ValueError("sequence_lengths must be positive")
        if bool(torch.any(lengths > self.max_sequence_length)):
            raise ValueError("sequence_lengths exceed page_table capacity")
        table = self.page_table.detach().to(device="cpu", dtype=torch.int64)
        for batch_idx, length in enumerate(lengths.tolist()):
            active_pages = (int(length) + self.page_size - 1) // self.page_size
            active = table[batch_idx, :active_pages]
            if bool(torch.any(active < 0)) or bool(torch.any(active >= self.num_pages)):
                raise ValueError(
                    "active page_table entries must reference physical pages"
                )


def build_paged_support_keys(
    cache: PagedKVCache,
    *,
    support_width: int = 2,
    method: str = "centroid_extremes",
) -> torch.Tensor:
    """Build compact per-64-token key metadata in logical page order.

    The first support vector is the valid-token centroid. Additional vectors
    are real keys selected by distance from the centroid (or key norm). This
    metadata is built during planning/prefill; decode-time selection scans only
    ``support_width`` vectors per logical atom instead of all 64 keys.
    """

    if cache.key.dim() != 4 or cache.page_table.dim() != 2:
        raise ValueError("paged support metadata requires rank-4 K and rank-2 table")
    if cache.page_size != 16 or cache.max_pages_per_request % 4:
        raise ValueError("paged support metadata requires page-16 and 64-token atoms")
    if cache.normalized_layout not in {"NHD", "HND"}:
        raise ValueError("paged support metadata requires NHD or HND cache layout")
    if cache.page_table.device != cache.key.device:
        raise ValueError("page table and K pages must share a device")
    if support_width not in {1, 2, 4, 8}:
        raise ValueError("support_width must be 1, 2, 4, or 8")
    if method not in {"centroid_extremes", "centroid_top_norm"}:
        raise ValueError("unsupported support-key construction method")

    safe_pages = cache.page_table.clamp_min(0).to(dtype=torch.long)
    gathered = cache.key[safe_pages]
    if cache.normalized_layout == "NHD":
        # [B,logical_page,token,Hkv,D] -> [B,Hkv,logical_page,token,D]
        gathered = gathered.permute(0, 3, 1, 2, 4)
    else:
        # [B,logical_page,Hkv,token,D] -> [B,Hkv,logical_page,token,D]
        gathered = gathered.permute(0, 2, 1, 3, 4)

    batch = cache.batch_size
    num_atoms = cache.max_sequence_length // 64
    keys = gathered.reshape(batch, cache.kv_heads, num_atoms, 64, cache.head_dim)
    logical_tokens = torch.arange(
        cache.max_sequence_length,
        device=cache.key.device,
        dtype=torch.int64,
    ).view(1, num_atoms, 64)
    valid = logical_tokens < cache.sequence_lengths.to(torch.int64).view(batch, 1, 1)
    keys_f32 = keys.float() * valid[:, None, :, :, None]
    counts = valid.sum(dim=-1).clamp_min(1).to(torch.float32)
    centroid = keys_f32.sum(dim=3) / counts[:, None, :, None]
    if support_width == 1:
        return centroid[:, :, :, None, :].to(cache.key.dtype).contiguous()

    if method == "centroid_top_norm":
        priority = keys_f32.square().sum(dim=-1)
    else:
        priority = (keys_f32 - centroid[:, :, :, None, :]).square().sum(dim=-1)
    priority = priority.masked_fill(~valid[:, None, :, :], -float("inf"))
    indices = priority.topk(support_width - 1, dim=-1).indices
    extreme = torch.gather(
        keys_f32,
        dim=3,
        index=indices[..., None].expand(-1, -1, -1, -1, cache.head_dim),
    )
    return torch.cat((centroid[:, :, :, None, :], extreme), dim=3).to(
        cache.key.dtype
    ).contiguous()


def choose_paged_exact_splits(
    *,
    batch: int,
    query_heads: int,
    max_pages_per_request: int,
    target_ctas: int = 512,
    max_splits: int = 32,
) -> int:
    """Choose the minimum split count needed to expose enough producer CTAs."""

    if min(batch, query_heads, max_pages_per_request, target_ctas, max_splits) <= 0:
        raise ValueError("paged split inputs must be positive")
    needed = (target_ctas + batch * query_heads - 1) // (batch * query_heads)
    return max(1, min(max_pages_per_request, max_splits, needed))


def choose_sm80_merge_segments(
    *, batch: int, query_heads: int, target_ctas: int = 128
) -> int:
    """Choose exact output-dimension segments for the SM80 split-state merge."""

    if min(batch, query_heads, target_ctas) <= 0:
        raise ValueError("SM80 merge schedule inputs must be positive")
    required = (target_ctas + batch * query_heads - 1) // (batch * query_heads)
    return next((segments for segments in (1, 2, 4, 8) if segments >= required), 8)


def paged_exact_reference(
    query: torch.Tensor,
    cache: PagedKVCache,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Page-streaming exact reference that does not build a contiguous cache."""

    cache.validate(query)
    if output is None:
        output = torch.empty_like(query)
    if output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("output must match query shape and dtype")
    if output.device != query.device or not output.is_contiguous():
        raise ValueError("output must be contiguous and share the query device")

    batch, _query_len, heads, dim = map(int, query.shape)
    group_size = heads // cache.kv_heads
    scale = 1.0 / math.sqrt(float(dim))
    for batch_idx in range(batch):
        sequence_length = int(cache.sequence_lengths[batch_idx].item())
        active_pages = (sequence_length + cache.page_size - 1) // cache.page_size
        for head_idx in range(heads):
            kv_head = head_idx // group_size
            q = query[batch_idx, 0, head_idx].float()
            running_max = torch.tensor(
                -float("inf"), device=query.device, dtype=torch.float32
            )
            denominator = torch.zeros((), device=query.device, dtype=torch.float32)
            numerator = torch.zeros(dim, device=query.device, dtype=torch.float32)
            for logical_page in range(active_pages):
                physical_page = int(cache.page_table[batch_idx, logical_page].item())
                valid_tokens = min(
                    cache.page_size,
                    sequence_length - logical_page * cache.page_size,
                )
                if cache.normalized_layout == "NHD":
                    key_page = cache.key[physical_page, :valid_tokens, kv_head]
                    value_page = cache.value[physical_page, :valid_tokens, kv_head]
                else:
                    key_page = cache.key[physical_page, kv_head, :valid_tokens]
                    value_page = cache.value[physical_page, kv_head, :valid_tokens]
                scores = torch.matmul(key_page.float(), q) * scale
                page_max = torch.max(scores)
                new_max = torch.maximum(running_max, page_max)
                correction = torch.exp(running_max - new_max)
                probabilities = torch.exp(scores - new_max)
                numerator = numerator * correction + torch.sum(
                    probabilities[:, None] * value_page.float(), dim=0
                )
                denominator = denominator * correction + probabilities.sum()
                running_max = new_max
            output[batch_idx, 0, head_idx].copy_(
                (numerator / denominator).to(output.dtype)
            )
    return output


def paged_selected_reference(
    query: torch.Tensor,
    cache: PagedKVCache,
    routes: PackedPagedRoute64,
    *,
    schedule_epoch: int,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact reference over the per-head token sets encoded by PackedRoute64."""

    cache.validate(query)
    routes.validate_current(cache, schedule_epoch=schedule_epoch)
    if routes.row_count != cache.batch_size * cache.kv_heads:
        raise ValueError("selected route rows must match batch * KV heads")
    if output is None:
        output = torch.empty_like(query)
    if output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("output must match query shape and dtype")
    if output.device != query.device or not output.is_contiguous():
        raise ValueError("output must be contiguous and share the query device")

    row_ptr = routes.row_ptr.detach().to(device="cpu", dtype=torch.int64).tolist()
    physical_pages = routes.physical_page_ids.detach().to(
        device="cpu", dtype=torch.int64
    ).tolist()
    head_masks = routes.active_head_masks.detach().to(
        device="cpu", dtype=torch.int64
    ).tolist()
    token_masks = routes.token_valid_masks.detach().to(
        device="cpu", dtype=torch.int64
    ).tolist()
    group_size = int(query.shape[2]) // cache.kv_heads
    scale = 1.0 / math.sqrt(float(query.shape[3]))

    for group in range(routes.row_count):
        batch_idx = group // cache.kv_heads
        kv_head = group % cache.kv_heads
        for local_head in range(group_size):
            key_fragments: list[torch.Tensor] = []
            value_fragments: list[torch.Tensor] = []
            for route_idx in range(row_ptr[group], row_ptr[group + 1]):
                for atom in range(4):
                    head_mask = int(head_masks[route_idx][atom]) & 0xFFFFFFFF
                    token_mask = int(token_masks[route_idx][atom]) & 0xFFFF
                    if not (head_mask & (1 << local_head)) or token_mask == 0:
                        continue
                    physical_page = int(physical_pages[route_idx][atom])
                    token_ids = [
                        token for token in range(16) if token_mask & (1 << token)
                    ]
                    if cache.normalized_layout == "NHD":
                        key_page = cache.key[physical_page, token_ids, kv_head]
                        value_page = cache.value[physical_page, token_ids, kv_head]
                    else:
                        key_page = cache.key[physical_page, kv_head, token_ids]
                        value_page = cache.value[physical_page, kv_head, token_ids]
                    key_fragments.append(key_page)
                    value_fragments.append(value_page)
            if not key_fragments:
                raise ValueError(
                    f"selected route row {group} has no tokens for head {local_head}"
                )
            keys = torch.cat(key_fragments, dim=0).float()
            values = torch.cat(value_fragments, dim=0).float()
            q_head = kv_head * group_size + local_head
            scores = (keys @ query[batch_idx, 0, q_head].float()) * scale
            output[batch_idx, 0, q_head].copy_(
                (scores.softmax(dim=0)[:, None] * values).sum(dim=0).to(output.dtype)
            )
    return output


@dataclass
class PagedExactDecodePlan:
    """Allocation-free exact decode plan bound to paged KV buffers."""

    query: torch.Tensor
    cache: PagedKVCache
    output: torch.Tensor
    splits: int
    workspace: Optional[dict[str, torch.Tensor]]
    backend: str
    launch: Optional[Any] = None
    tokens_per_tile: int = 512
    partial_num_warps: int = 4
    query_group: Optional[torch.Tensor] = None
    output_group: Optional[torch.Tensor] = None
    merge_segments: int = 1

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        cache: PagedKVCache,
        *,
        output: Optional[torch.Tensor] = None,
        splits: Optional[int] = None,
        tokens_per_tile: int = 512,
        partial_num_warps: int = 4,
        sm80_merge_segments: Optional[int] = None,
        validate_metadata: bool = True,
        sm80_cp_async_experimental: bool = False,
        sm80_grouped_experimental: bool = False,
        sm100_grouped_experimental: bool = False,
        sm100_tgv_experimental: bool = False,
        sm90_fragmented_experimental: bool = False,
        sm90_fragmented_ragged_experimental: bool = False,
    ) -> "PagedExactDecodePlan":
        cache.validate(query, validate_metadata=validate_metadata)
        if output is None:
            output = torch.empty_like(query)
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("output must match query shape and dtype")
        if output.device != query.device or not output.is_contiguous():
            raise ValueError("output must be contiguous and share the query device")

        selected_splits = (
            choose_paged_exact_splits(
                batch=int(query.shape[0]),
                query_heads=int(query.shape[2]),
                max_pages_per_request=cache.max_pages_per_request,
            )
            if splits is None
            else int(splits)
        )
        if selected_splits <= 0 or selected_splits > cache.max_pages_per_request:
            raise ValueError("splits must be in [1, max_pages_per_request]")
        if tokens_per_tile < cache.page_size or tokens_per_tile & (tokens_per_tile - 1):
            raise ValueError("tokens_per_tile must be a power of two >= page_size")
        if tokens_per_tile % cache.page_size:
            raise ValueError("tokens_per_tile must be divisible by page_size")
        if partial_num_warps not in {1, 2, 4, 8}:
            raise ValueError("partial_num_warps must be one of 1, 2, 4, or 8")
        if sm80_merge_segments is not None and sm80_merge_segments not in {
            1,
            2,
            4,
            8,
        }:
            raise ValueError("sm80_merge_segments must be one of 1, 2, 4, or 8")

        if query.is_cuda:
            full_lengths = bool(
                torch.all(cache.sequence_lengths == cache.max_sequence_length).item()
            )
            q_heads = int(query.shape[2])
            head_dim = int(query.shape[3])
            group_size = q_heads // cache.kv_heads
            page16_shape = (q_heads, cache.kv_heads, group_size, head_dim)
            promoted_page16_shapes = (
                PROMOTED_PAGED_EXACT_PAGE16_NHD_SHAPES
                if cache.normalized_layout == "NHD"
                else PROMOTED_PAGED_EXACT_PAGE16_SHAPES
            )
            promoted_page16_ragged_shapes = (
                PROMOTED_PAGED_EXACT_PAGE16_NHD_RAGGED_SHAPES
                if cache.normalized_layout == "NHD"
                else PROMOTED_PAGED_EXACT_PAGE16_RAGGED_SHAPES
            )
            common_sm90_shape = (
                torch.cuda.get_device_capability(query.device) == (9, 0)
                and query.dtype == torch.bfloat16
                and group_size in {4, 8}
                and head_dim in {64, 128}
                and cache.page_table.dtype == torch.int32
            )
            use_sm90_page64 = (
                common_sm90_shape
                and cache.normalized_layout == "HND"
                and page16_shape == (16, 2, 8, 64)
                and full_lengths
                and cache.page_size == 64
                and (int(query.shape[0]), cache.max_sequence_length)
                in PROMOTED_PAGED_EXACT_SPLITS
            )
            use_sm90_page16 = (
                common_sm90_shape
                and full_lengths
                and cache.page_size == 16
                and (
                    sm90_fragmented_experimental
                    or (
                        (int(query.shape[0]), cache.max_sequence_length)
                        in promoted_page16_shapes.get(page16_shape, {})
                    )
                )
            )
            use_sm90_page16_ragged = (
                common_sm90_shape
                and not full_lengths
                and cache.page_size == 16
                and cache.sequence_lengths.dtype == torch.int32
                and (
                    sm90_fragmented_ragged_experimental
                    or (
                        (int(query.shape[0]), cache.max_sequence_length)
                        in promoted_page16_ragged_shapes.get(page16_shape, {})
                    )
                )
            )
            use_sm90_paged = (
                use_sm90_page64 or use_sm90_page16 or use_sm90_page16_ragged
            )
            if use_sm90_paged:
                from .backends.sm90.transposed_gqa_exact import (
                    choose_num_splits,
                    compile_transposed_gqa_exact_extension,
                    resolve_cutlass_root,
                )

                try:
                    resolve_cutlass_root()
                except FileNotFoundError:
                    use_sm90_paged = False

            if use_sm90_paged:
                if splits is None and use_sm90_page64:
                    selected_splits = PROMOTED_PAGED_EXACT_SPLITS.get(
                        (int(query.shape[0]), cache.max_sequence_length),
                        choose_num_splits(
                            batch=int(query.shape[0]),
                            kv_heads=cache.kv_heads,
                            kv_len=cache.max_sequence_length,
                        ),
                    )
                elif splits is None and (use_sm90_page16 or use_sm90_page16_ragged):
                    shape_tables = (
                        promoted_page16_ragged_shapes
                        if use_sm90_page16_ragged
                        else promoted_page16_shapes
                    )
                    split_table = shape_tables.get(page16_shape, {})
                    selected_splits = split_table.get(
                        (int(query.shape[0]), cache.max_sequence_length),
                        choose_num_splits(
                            batch=int(query.shape[0]),
                            kv_heads=cache.kv_heads,
                            kv_len=cache.max_sequence_length,
                        ),
                    )
                elif splits is None:
                    selected_splits = choose_num_splits(
                        batch=int(query.shape[0]),
                        kv_heads=cache.kv_heads,
                        kv_len=cache.max_sequence_length,
                    )
                groups = int(query.shape[0]) * cache.kv_heads
                partial_o = torch.empty(
                    groups,
                    selected_splits,
                    8,
                    head_dim,
                    device=query.device,
                    dtype=torch.float32,
                )
                partial_lse = torch.empty(
                    groups,
                    selected_splits,
                    8,
                    device=query.device,
                    dtype=torch.float32,
                )
                extension = compile_transposed_gqa_exact_extension(head_dim=head_dim)
                return cls(
                    query=query,
                    cache=cache,
                    output=output,
                    splits=selected_splits,
                    workspace={"partial_o": partial_o, "partial_lse": partial_lse},
                    backend=(
                        PAGED_EXACT_SM90_NHD_FRAGMENTED_RAGGED_BACKEND
                        if use_sm90_page16_ragged and cache.normalized_layout == "NHD"
                        else (
                            PAGED_EXACT_SM90_FRAGMENTED_RAGGED_BACKEND
                            if use_sm90_page16_ragged
                            else (
                                PAGED_EXACT_SM90_NHD_FRAGMENTED_BACKEND
                                if use_sm90_page16 and cache.normalized_layout == "NHD"
                                else (
                                    PAGED_EXACT_SM90_FRAGMENTED_BACKEND
                                    if use_sm90_page16
                                    else PAGED_EXACT_SM90_BACKEND
                                )
                            )
                        )
                    ),
                    launch=(
                        extension.paged_fragmented_nhd_ragged_exact_decode_out
                        if use_sm90_page16_ragged and cache.normalized_layout == "NHD"
                        else (
                            extension.paged_fragmented_ragged_exact_decode_out
                            if use_sm90_page16_ragged
                            else (
                                extension.paged_fragmented_nhd_exact_decode_out
                                if use_sm90_page16 and cache.normalized_layout == "NHD"
                                else (
                                    extension.paged_fragmented_exact_decode_out
                                    if use_sm90_page16
                                    else extension.paged_exact_decode_out
                                )
                            )
                        )
                    ),
                    # Every promoted SM90 path consumes one 64-token logical
                    # context tile, including four-fragment page-16 sources.
                    tokens_per_tile=64,
                    partial_num_warps=int(partial_num_warps),
                    query_group=query.view(
                        int(query.shape[0]), cache.kv_heads, group_size, head_dim
                    ),
                    output_group=output.view(groups, group_size, head_dim),
                )

            use_sm80_cp_async = (
                sm80_cp_async_experimental
                and torch.cuda.get_device_capability(query.device) == (8, 0)
                and query.dtype == torch.bfloat16
                and cache.normalized_layout in {"HND", "NHD"}
                and cache.page_size == 16
                and page16_shape == (16, 2, 8, 128)
                and full_lengths
                and cache.max_pages_per_request % 4 == 0
                and cache.page_table.dtype == torch.int32
                and cache.sequence_lengths.dtype == torch.int32
            )
            if use_sm80_cp_async:
                from .backends.sm80.paged_gqa_exact import (
                    compile_sm80_paged_gqa_extension,
                    resolve_cutlass_root,
                )

                try:
                    resolve_cutlass_root()
                except FileNotFoundError:
                    use_sm80_cp_async = False
            if use_sm80_cp_async:
                producer_tile = 128 if int(tokens_per_tile) == 128 else 64
                pages_per_tile = producer_tile // cache.page_size
                logical_tiles = (
                    cache.max_pages_per_request + pages_per_tile - 1
                ) // pages_per_tile
                if splits is None:
                    target_ctas = 256
                    group_count = int(query.shape[0]) * cache.kv_heads
                    selected_splits = max(
                        1,
                        min(
                            logical_tiles,
                            (target_ctas + group_count - 1) // group_count,
                        ),
                    )
                if selected_splits > min(logical_tiles, 512):
                    raise ValueError(
                        "SM80 cp.async splits must be <= "
                        "min(ceil(max_pages/pages_per_tile), 512)"
                    )
                groups = int(query.shape[0]) * cache.kv_heads
                selected_merge_segments = sm80_merge_segments
                if selected_merge_segments is None:
                    selected_merge_segments = choose_sm80_merge_segments(
                        batch=int(query.shape[0]),
                        query_heads=int(query.shape[2]),
                    )
                partial_o = torch.empty(
                    groups,
                    selected_splits,
                    8,
                    head_dim,
                    device=query.device,
                    dtype=torch.float32,
                )
                partial_lse = torch.empty(
                    groups,
                    selected_splits,
                    8,
                    device=query.device,
                    dtype=torch.float32,
                )
                extension = compile_sm80_paged_gqa_extension()
                return cls(
                    query=query,
                    cache=cache,
                    output=output,
                    splits=selected_splits,
                    workspace={"partial_o": partial_o, "partial_lse": partial_lse},
                    backend=PAGED_EXACT_SM80_CP_ASYNC_BACKEND,
                    launch=extension.paged_exact_decode_out,
                    tokens_per_tile=producer_tile,
                    partial_num_warps=4,
                    query_group=query.view(
                        int(query.shape[0]), cache.kv_heads, group_size, head_dim
                    ),
                    output_group=output.view(groups, group_size, head_dim),
                    merge_segments=int(selected_merge_segments),
                )
            sm100_tgv_cell = (
                int(query.shape[0]),
                cache.max_sequence_length,
            )
            promoted_sm100_tgv = (
                splits is None
                and sm100_tgv_cell in PROMOTED_PAGED_EXACT_SM100_TGV_SPLITS
            )
            use_sm100_tgv = (
                (sm100_tgv_experimental or promoted_sm100_tgv)
                and torch.cuda.get_device_capability(query.device) == (10, 0)
                and query.dtype == torch.bfloat16
                and cache.normalized_layout == "NHD"
                and cache.page_size == 16
                and page16_shape == (16, 2, 8, 128)
                and full_lengths
                and cache.page_table.dtype == torch.int32
                and cache.sequence_lengths.dtype == torch.int32
                and cache.key.is_contiguous()
                and cache.value.is_contiguous()
            )
            if use_sm100_tgv:
                from .backends.sm100.paged_gqa_exact import (
                    compile_sm100_paged_gqa_extension,
                    resolve_sm100_cutlass_root,
                )

                try:
                    resolve_sm100_cutlass_root()
                except FileNotFoundError:
                    use_sm100_tgv = False
            if use_sm100_tgv:
                selected_splits = (
                    PROMOTED_PAGED_EXACT_SM100_TGV_SPLITS[sm100_tgv_cell]
                    if promoted_sm100_tgv
                    else (8 if splits is None else int(splits))
                )
                if selected_splits not in {2, 4, 8, 16}:
                    raise ValueError("SM100 TGV splits must be 2, 4, 8, or 16")
                metadata_padding = 64
                padded_page_table = torch.zeros(
                    int(query.shape[0]),
                    cache.max_pages_per_request + metadata_padding,
                    dtype=torch.int32,
                    device=query.device,
                )
                padded_page_table[:, : cache.max_pages_per_request].copy_(
                    cache.page_table
                )
                extension = compile_sm100_paged_gqa_extension()
                return cls(
                    query=query,
                    cache=cache,
                    output=output,
                    splits=selected_splits,
                    workspace={"padded_page_table": padded_page_table},
                    backend=PAGED_EXACT_SM100_TGV_BACKEND,
                    launch=extension.paged_exact_decode_out,
                    tokens_per_tile=128,
                    partial_num_warps=4,
                    query_group=query.view(
                        int(query.shape[0]), cache.kv_heads, group_size, head_dim
                    ),
                    output_group=output.view(
                        int(query.shape[0]) * cache.kv_heads,
                        group_size,
                        head_dim,
                    ),
                )
            from .kernels.paged_exact_triton import (
                TRITON_AVAILABLE,
                make_paged_exact_workspace,
                paged_exact_decode_grouped_forward_out,
                paged_exact_decode_triton_forward_out,
            )

            if not TRITON_AVAILABLE:
                raise RuntimeError("Triton is required for CUDA paged exact decode")
            workspace = make_paged_exact_workspace(
                batch=int(query.shape[0]),
                heads=int(query.shape[2]),
                splits=selected_splits,
                dim=int(query.shape[3]),
                device=query.device,
            )
            use_sm80_grouped = (
                sm80_grouped_experimental
                and torch.cuda.get_device_capability(query.device) == (8, 0)
                and query.dtype == torch.bfloat16
                and cache.page_size == 16
                and group_size == 8
                and head_dim == 128
                and cache.page_table.dtype == torch.int32
                and cache.sequence_lengths.dtype == torch.int32
                and tokens_per_tile in {64, 128}
            )
            use_sm100_grouped = (
                sm100_grouped_experimental
                and torch.cuda.get_device_capability(query.device) == (10, 0)
                and query.dtype == torch.bfloat16
                and cache.page_size == 16
                and group_size == 8
                and head_dim == 128
                and cache.page_table.dtype == torch.int32
                and cache.sequence_lengths.dtype == torch.int32
                and tokens_per_tile in {64, 128}
            )
            return cls(
                query=query,
                cache=cache,
                output=output,
                splits=selected_splits,
                workspace=workspace,
                backend=(
                    PAGED_EXACT_SM100_GROUPED_BACKEND
                    if use_sm100_grouped
                    else (
                        PAGED_EXACT_SM80_GROUPED_BACKEND
                        if use_sm80_grouped
                        else PAGED_EXACT_NATIVE_BACKEND
                    )
                ),
                launch=(
                    paged_exact_decode_grouped_forward_out
                    if use_sm80_grouped or use_sm100_grouped
                    else paged_exact_decode_triton_forward_out
                ),
                tokens_per_tile=int(tokens_per_tile),
                partial_num_warps=int(partial_num_warps),
            )
        return cls(
            query=query,
            cache=cache,
            output=output,
            splits=1,
            workspace=None,
            backend="torch_paged_exact_reference",
            tokens_per_tile=int(tokens_per_tile),
            partial_num_warps=int(partial_num_warps),
        )

    @property
    def workspace_bytes(self) -> int:
        if self.workspace is None:
            return 0
        return sum(t.numel() * t.element_size() for t in self.workspace.values())

    def run(self) -> torch.Tensor:
        """Execute using the bound buffers and preallocated workspace."""

        if self.launch is None:
            return paged_exact_reference(self.query, self.cache, output=self.output)
        assert self.workspace is not None
        if self.backend in {
            PAGED_EXACT_SM90_BACKEND,
            PAGED_EXACT_SM90_FRAGMENTED_BACKEND,
            PAGED_EXACT_SM90_FRAGMENTED_RAGGED_BACKEND,
            PAGED_EXACT_SM90_NHD_FRAGMENTED_BACKEND,
            PAGED_EXACT_SM90_NHD_FRAGMENTED_RAGGED_BACKEND,
            PAGED_EXACT_SM80_CP_ASYNC_BACKEND,
            PAGED_EXACT_SM100_TGV_BACKEND,
        }:
            assert self.query_group is not None and self.output_group is not None
            if self.backend == PAGED_EXACT_SM100_TGV_BACKEND:
                self.launch(
                    self.query_group,
                    self.cache.key,
                    self.cache.value,
                    self.workspace["padded_page_table"],
                    self.cache.sequence_lengths,
                    self.output_group,
                    self.cache.max_pages_per_request,
                    self.splits,
                )
            elif self.backend == PAGED_EXACT_SM80_CP_ASYNC_BACKEND:
                self.launch(
                    self.query_group,
                    self.cache.key,
                    self.cache.value,
                    self.cache.page_table,
                    self.cache.sequence_lengths,
                    self.workspace["partial_o"],
                    self.workspace["partial_lse"],
                    self.output_group,
                    self.splits,
                    self.cache.normalized_layout == "HND",
                    self.merge_segments,
                    self.tokens_per_tile,
                )
            elif self.backend in {
                PAGED_EXACT_SM90_FRAGMENTED_RAGGED_BACKEND,
                PAGED_EXACT_SM90_NHD_FRAGMENTED_RAGGED_BACKEND,
            }:
                self.launch(
                    self.query_group,
                    self.cache.key,
                    self.cache.value,
                    self.cache.page_table,
                    self.cache.sequence_lengths,
                    self.workspace["partial_o"],
                    self.workspace["partial_lse"],
                    self.output_group,
                    self.splits,
                )
            else:
                self.launch(
                    self.query_group,
                    self.cache.key,
                    self.cache.value,
                    self.cache.page_table,
                    self.workspace["partial_o"],
                    self.workspace["partial_lse"],
                    self.output_group,
                    self.splits,
                )
            return self.output
        return self.launch(
            self.query,
            self.cache.key,
            self.cache.value,
            self.cache.page_table,
            self.cache.sequence_lengths,
            self.output,
            layout=self.cache.normalized_layout,
            splits=self.splits,
            workspace=self.workspace,
            tokens_per_tile=self.tokens_per_tile,
            partial_num_warps=self.partial_num_warps,
        )


@dataclass
class PagedSelectedDecodePlan:
    """Allocation-free H100 decode over a prepared selected paged route."""

    query: torch.Tensor
    cache: PagedKVCache
    routes: PackedPagedRoute64
    schedule_epoch: int
    output: torch.Tensor
    max_routes_per_row: int
    workspace: dict[str, torch.Tensor]
    query_group: torch.Tensor
    output_group: torch.Tensor
    launch: Any
    backend: str = PAGED_SELECTED_SM90_STATIC_BACKEND

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        cache: PagedKVCache,
        routes: PackedPagedRoute64,
        *,
        schedule_epoch: int,
        output: Optional[torch.Tensor] = None,
        validate_metadata: bool = True,
    ) -> "PagedSelectedDecodePlan":
        cache.validate(query, validate_metadata=validate_metadata)
        routes.validate_current(cache, schedule_epoch=schedule_epoch)
        if not query.is_cuda or torch.cuda.get_device_capability(query.device) != (9, 0):
            raise ValueError("selected paged WGMMA decode requires an H100/SM90 GPU")
        if query.dtype != torch.bfloat16:
            raise ValueError("selected paged WGMMA decode currently requires BF16")
        if cache.page_size != 16 or cache.normalized_layout not in {"NHD", "HND"}:
            raise ValueError("selected paged WGMMA decode requires NHD/HND page-16 KV")
        if cache.page_table.dtype != torch.int32 or cache.sequence_lengths.dtype != torch.int32:
            raise ValueError("selected paged WGMMA metadata must use int32")
        if not cache.key.is_contiguous() or not cache.value.is_contiguous():
            raise ValueError("selected paged WGMMA K/V pages must be contiguous")

        batch = int(query.shape[0])
        q_heads = int(query.shape[2])
        head_dim = int(query.shape[3])
        group_size = q_heads // cache.kv_heads
        if group_size not in {4, 8} or head_dim not in {64, 128}:
            raise ValueError("selected paged WGMMA supports G4/G8 and D64/D128")
        groups = batch * cache.kv_heads
        if routes.row_count != groups:
            raise ValueError("selected route rows must match batch * KV heads")
        if routes.device != query.device:
            raise ValueError("selected routes and attention buffers must share a device")

        row_ptr = routes.row_ptr.detach().to(device="cpu", dtype=torch.int64)
        row_counts = row_ptr[1:] - row_ptr[:-1]
        max_routes = int(row_counts.max().item()) if row_counts.numel() else 0
        if max_routes <= 0:
            raise ValueError("every selected execution plan needs at least one route")
        masks = routes.active_head_masks.detach().to(device="cpu", dtype=torch.int64)
        token_masks = routes.token_valid_masks.detach().to(
            device="cpu", dtype=torch.int64
        )
        full_head_mask = (1 << group_size) - 1
        for group in range(groups):
            coverage = 0
            for route_idx in range(int(row_ptr[group]), int(row_ptr[group + 1])):
                for atom in range(4):
                    if int(token_masks[route_idx, atom]) & 0xFFFF:
                        coverage |= int(masks[route_idx, atom]) & 0xFFFFFFFF
            if coverage & full_head_mask != full_head_mask:
                raise ValueError(
                    f"selected route row {group} leaves one or more Q heads empty"
                )

        if output is None:
            output = torch.empty_like(query)
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("output must match query shape and dtype")
        if output.device != query.device or not output.is_contiguous():
            raise ValueError("output must be contiguous and share the query device")

        partial_o = torch.empty(
            groups,
            max_routes,
            8,
            head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            groups,
            max_routes,
            8,
            device=query.device,
            dtype=torch.float32,
        )
        from .backends.sm90.transposed_gqa_exact import (
            compile_transposed_gqa_exact_extension,
        )

        extension = compile_transposed_gqa_exact_extension(head_dim=head_dim)
        launch = (
            extension.paged_selected_fragmented_nhd_exact_decode_out
            if cache.normalized_layout == "NHD"
            else extension.paged_selected_fragmented_exact_decode_out
        )
        return cls(
            query=query,
            cache=cache,
            routes=routes,
            schedule_epoch=int(schedule_epoch),
            output=output,
            max_routes_per_row=max_routes,
            workspace={"partial_o": partial_o, "partial_lse": partial_lse},
            query_group=query.view(batch, cache.kv_heads, group_size, head_dim),
            output_group=output.view(groups, group_size, head_dim),
            launch=launch,
        )

    @property
    def workspace_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.workspace.values())

    @property
    def producer_ctas(self) -> int:
        return self.routes.row_count * self.max_routes_per_row

    def run(self) -> torch.Tensor:
        """Run the static selected route and reject stale cache metadata."""

        self.routes.validate_current(
            self.cache,
            schedule_epoch=self.schedule_epoch,
        )
        self.launch(
            self.query_group,
            self.cache.key,
            self.cache.value,
            self.routes.row_ptr,
            self.routes.physical_page_ids,
            self.routes.active_head_masks,
            self.routes.token_valid_masks,
            self.workspace["partial_o"],
            self.workspace["partial_lse"],
            self.output_group,
            self.max_routes_per_row,
        )
        return self.output


@dataclass
class PagedDynamicSelectedDecodePlan:
    """No-sync H100 path from mutable Q-head CSR atoms to selected decode.

    CSR row offsets are fixed at plan time. ``atom_ids`` may be overwritten by
    a GPU selector before every run. A bounded shared-memory membership map and
    warp-prefix compaction form each GQA-group union; physical-page resolution,
    attention, and merge then execute on the current CUDA stream without a host
    route-count readback.
    """

    query: torch.Tensor
    cache: PagedKVCache
    routes: AttentionRouteCSR
    output: torch.Tensor
    max_routes_per_group: int
    metadata: dict[str, torch.Tensor]
    workspace: dict[str, torch.Tensor]
    query_group: torch.Tensor
    output_group: torch.Tensor
    prepare_launch: Any
    launch: Any
    row_ptr_version: int
    backend: str = PAGED_SELECTED_SM90_DYNAMIC_QHEAD_BACKEND

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        cache: PagedKVCache,
        routes: AttentionRouteCSR,
        *,
        output: Optional[torch.Tensor] = None,
        validate_metadata: bool = True,
    ) -> "PagedDynamicSelectedDecodePlan":
        cache.validate(query, validate_metadata=validate_metadata)
        if not query.is_cuda or torch.cuda.get_device_capability(query.device) != (9, 0):
            raise ValueError("dynamic selected paged decode requires H100/SM90")
        if query.dtype != torch.bfloat16:
            raise ValueError("dynamic selected paged decode currently requires BF16")
        if cache.page_size != 16 or cache.normalized_layout not in {"NHD", "HND"}:
            raise ValueError("dynamic selected paged decode requires NHD/HND page-16 KV")
        if not cache.key.is_contiguous() or not cache.value.is_contiguous():
            raise ValueError("dynamic selected paged K/V pages must be contiguous")
        if routes.granularity != ATTENTION_ROUTE_GRANULARITY_Q_HEAD:
            raise ValueError("dynamic selected lowering requires Q-head route rows")
        if routes.atom_size != 64:
            raise ValueError("dynamic selected lowering requires 64-token route atoms")
        logical_atoms = int(cache.page_table.size(1)) * cache.page_size // 64
        if logical_atoms > 12_288:
            raise ValueError(
                "dynamic selected lowering currently supports at most 12,288 "
                "logical route atoms per row"
            )
        if routes.device != query.device:
            raise ValueError("dynamic routes and attention buffers must share a device")

        batch = int(query.shape[0])
        q_heads = int(query.shape[2])
        head_dim = int(query.shape[3])
        group_size = q_heads // cache.kv_heads
        if group_size not in {4, 8} or head_dim not in {64, 128}:
            raise ValueError("dynamic selected WGMMA supports G4/G8 and D64/D128")
        if routes.row_count != batch * q_heads:
            raise ValueError("dynamic Q-head CSR must contain B*Hq rows")
        if routes.active_head_masks is not None:
            raise ValueError("Q-head CSR encodes ownership and cannot carry head masks")

        row_ptr = routes.row_ptr.detach().to(device="cpu", dtype=torch.int64)
        row_counts = row_ptr[1:] - row_ptr[:-1]
        if row_counts.numel() != batch * q_heads or bool(torch.any(row_counts <= 0)):
            raise ValueError("every dynamic Q-head route row must remain non-empty")
        host_atoms = routes.atom_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        host_offsets = row_ptr.tolist()
        live_lengths = cache.sequence_lengths.detach().to(
            device="cpu", dtype=torch.int64
        ).tolist()
        for row, (begin, end) in enumerate(
            zip(host_offsets[:-1], host_offsets[1:], strict=True)
        ):
            atoms = host_atoms[begin:end]
            if len(atoms) != len(set(atoms)):
                raise ValueError("dynamic route rows must initially contain unique atoms")
            max_atom = math.ceil(int(live_lengths[row // q_heads]) / 64)
            if any(atom < 0 or atom >= max_atom for atom in atoms):
                raise ValueError("dynamic route atom is outside the live KV extent")
        group_capacity = row_counts.view(batch, cache.kv_heads, group_size).sum(dim=2)
        max_routes = int(group_capacity.max().item())
        if max_routes <= 0 or max_routes > 512:
            raise ValueError("dynamic route capacity must be in [1,512]")

        if output is None:
            output = torch.empty_like(query)
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("output must match query shape and dtype")
        if output.device != query.device or not output.is_contiguous():
            raise ValueError("output must be contiguous and share the query device")

        groups = batch * cache.kv_heads
        atom_shape = (groups, max_routes, 4)
        route_shape = (groups, max_routes)
        metadata = {
            "route_counts": torch.empty(groups, device=query.device, dtype=torch.int32),
            "logical_atom_origins": torch.empty(
                atom_shape, device=query.device, dtype=torch.int32
            ),
            "physical_page_ids": torch.empty(
                atom_shape, device=query.device, dtype=torch.int32
            ),
            "atom_valid_masks": torch.empty(
                route_shape, device=query.device, dtype=torch.int32
            ),
            "active_head_masks": torch.empty(
                atom_shape, device=query.device, dtype=torch.int32
            ),
            "token_valid_masks": torch.empty(
                atom_shape, device=query.device, dtype=torch.int32
            ),
            "route_flags": torch.empty(
                route_shape, device=query.device, dtype=torch.int32
            ),
            "route_errors": torch.empty(groups, device=query.device, dtype=torch.int32),
        }
        workspace = {
            "partial_o": torch.empty(
                groups,
                max_routes,
                8,
                head_dim,
                device=query.device,
                dtype=torch.float32,
            ),
            "partial_lse": torch.empty(
                groups,
                max_routes,
                8,
                device=query.device,
                dtype=torch.float32,
            ),
        }
        from .backends.sm90.transposed_gqa_exact import (
            compile_transposed_gqa_exact_extension,
        )

        extension = compile_transposed_gqa_exact_extension(head_dim=head_dim)
        launch = (
            extension.paged_dynamic_qhead_fragmented_nhd_exact_decode_out
            if cache.normalized_layout == "NHD"
            else extension.paged_dynamic_qhead_fragmented_exact_decode_out
        )
        return cls(
            query=query,
            cache=cache,
            routes=routes,
            output=output,
            max_routes_per_group=max_routes,
            metadata=metadata,
            workspace=workspace,
            query_group=query.view(batch, cache.kv_heads, group_size, head_dim),
            output_group=output.view(groups, group_size, head_dim),
            prepare_launch=extension.prepare_qhead_paged_routes64_out,
            launch=launch,
            row_ptr_version=int(routes.row_ptr._version),
        )

    @property
    def metadata_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.metadata.values())

    @property
    def workspace_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.workspace.values())

    @property
    def producer_ctas(self) -> int:
        return int(self.metadata["route_counts"].numel()) * self.max_routes_per_group

    def _validate_fixed_structure(self) -> None:
        if int(self.routes.row_ptr._version) != self.row_ptr_version:
            raise RuntimeError("dynamic route row offsets changed after planning")

    def prepare(self) -> None:
        """Prepare row-local route metadata without synchronizing to the host."""

        self._validate_fixed_structure()
        self.prepare_launch(
            self.routes.row_ptr,
            self.routes.atom_ids,
            self.cache.page_table,
            self.cache.sequence_lengths,
            self.metadata["route_counts"],
            self.metadata["logical_atom_origins"],
            self.metadata["physical_page_ids"],
            self.metadata["atom_valid_masks"],
            self.metadata["active_head_masks"],
            self.metadata["token_valid_masks"],
            self.metadata["route_flags"],
            self.metadata["route_errors"],
            int(self.query.shape[2]),
            self.cache.kv_heads,
            self.cache.num_pages,
            self.max_routes_per_group,
        )

    def check_route_errors(self) -> None:
        """Synchronize diagnostics and reject malformed live selector output."""

        errors = self.metadata["route_errors"].detach().to(device="cpu")
        if bool(torch.any(errors != 0)):
            raise RuntimeError(f"dynamic route preparation failed: {errors.tolist()}")

    def run(self) -> torch.Tensor:
        """Prepare and execute the current Q-head selections in one host call."""

        self._validate_fixed_structure()
        self.launch(
            self.query_group,
            self.cache.key,
            self.cache.value,
            self.cache.page_table,
            self.cache.sequence_lengths,
            self.routes.row_ptr,
            self.routes.atom_ids,
            self.metadata["route_counts"],
            self.metadata["logical_atom_origins"],
            self.metadata["physical_page_ids"],
            self.metadata["atom_valid_masks"],
            self.metadata["active_head_masks"],
            self.metadata["token_valid_masks"],
            self.metadata["route_flags"],
            self.metadata["route_errors"],
            self.workspace["partial_o"],
            self.workspace["partial_lse"],
            self.output_group,
            self.max_routes_per_group,
        )
        return self.output


@dataclass
class PagedQuerySelectedDecodePlan:
    """Query-aware selector composed with the no-sync selected WGMMA path.

    Per-atom support keys are persistent prefill metadata. Every ``run`` scores
    those summaries with the live query, writes fixed-width Q-head CSR rows,
    lowers their union, and executes exact online-softmax attention over the
    selected atoms on the current CUDA stream.
    """

    query: torch.Tensor
    cache: PagedKVCache
    support_keys: torch.Tensor
    routes: AttentionRouteCSR
    selected_plan: PagedDynamicSelectedDecodePlan
    scores: torch.Tensor
    sink_atoms: int
    recent_atoms: int
    middle_atoms: int
    candidate_ids: Optional[torch.Tensor] = None
    candidate_scores: Optional[torch.Tensor] = None
    refine_candidates: int = 0
    backend: str = PAGED_QUERY_SELECTED_SM90_BACKEND

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        cache: PagedKVCache,
        *,
        selected_atoms: int = 6,
        sink_atoms: int = 1,
        recent_atoms: int = 1,
        support_width: int = 2,
        support_method: str = "centroid_extremes",
        refine_candidates: int = 0,
        support_keys: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        validate_metadata: bool = True,
    ) -> "PagedQuerySelectedDecodePlan":
        cache.validate(query, validate_metadata=validate_metadata)
        middle_atoms = int(selected_atoms) - int(sink_atoms) - int(recent_atoms)
        if min(selected_atoms, sink_atoms, recent_atoms, middle_atoms) <= 0:
            raise ValueError(
                "selected atoms must leave positive sink, recent, and middle sets"
            )
        live_atoms = (
            (cache.sequence_lengths.detach().to(device="cpu", dtype=torch.int64) + 63)
            // 64
        )
        if int(live_atoms.min().item()) < selected_atoms:
            raise ValueError("selected atom count exceeds the shortest live sequence")
        if refine_candidates:
            if refine_candidates not in {8, 16, 32, 64}:
                raise ValueError(
                    "refine_candidates must be 0, 8, 16, 32, or 64"
                )
            if refine_candidates < middle_atoms:
                raise ValueError(
                    "refine_candidates must cover the final middle atom count"
                )
            if int(live_atoms.min().item()) < (
                sink_atoms + recent_atoms + refine_candidates
            ):
                raise ValueError(
                    "refine candidate count exceeds the shortest middle region"
                )
        if support_keys is None:
            support_keys = build_paged_support_keys(
                cache,
                support_width=support_width,
                method=support_method,
            )
        expected_atoms = cache.max_sequence_length // 64
        expected_shape = (
            cache.batch_size,
            cache.kv_heads,
            expected_atoms,
            support_width,
            cache.head_dim,
        )
        if tuple(support_keys.shape) != expected_shape:
            raise ValueError(f"support_keys must have shape {expected_shape}")
        if support_keys.device != query.device or support_keys.dtype != query.dtype:
            raise ValueError("support keys must share query device and dtype")
        if not support_keys.is_contiguous():
            raise ValueError("support keys must be contiguous")

        rows = tuple(
            tuple(range(selected_atoms))
            for _ in range(cache.batch_size * int(query.shape[2]))
        )
        routes = AttentionRouteCSR.from_rows(
            rows,
            granularity=ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
            atom_size=64,
            device=query.device,
        )
        selected_plan = PagedDynamicSelectedDecodePlan.build(
            query,
            cache,
            routes,
            output=output,
            validate_metadata=False,
        )
        scores = torch.empty(
            cache.batch_size * int(query.shape[2]),
            expected_atoms,
            device=query.device,
            dtype=torch.float32,
        )
        candidate_ids = None
        candidate_scores = None
        backend = PAGED_QUERY_SELECTED_SM90_BACKEND
        if refine_candidates:
            candidate_ids = torch.empty(
                cache.batch_size * int(query.shape[2]),
                int(refine_candidates),
                device=query.device,
                dtype=torch.int32,
            )
            candidate_scores = torch.empty(
                candidate_ids.shape,
                device=query.device,
                dtype=torch.float32,
            )
            backend = PAGED_QUERY_REFINED_SM90_BACKEND
        return cls(
            query=query,
            cache=cache,
            support_keys=support_keys,
            routes=routes,
            selected_plan=selected_plan,
            scores=scores,
            sink_atoms=int(sink_atoms),
            recent_atoms=int(recent_atoms),
            middle_atoms=middle_atoms,
            candidate_ids=candidate_ids,
            candidate_scores=candidate_scores,
            refine_candidates=int(refine_candidates),
            backend=backend,
        )

    @property
    def output(self) -> torch.Tensor:
        return self.selected_plan.output

    @property
    def selector_workspace_bytes(self) -> int:
        total = self.scores.numel() * self.scores.element_size()
        for tensor in (self.candidate_ids, self.candidate_scores):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    @property
    def support_metadata_bytes(self) -> int:
        return self.support_keys.numel() * self.support_keys.element_size()

    def select(self) -> None:
        """Write score-ranked unique route atoms without a host synchronization."""

        from .kernels.paged_support_selector_triton import (
            paged_support_refined_select_triton,
            paged_support_select_triton,
        )

        if self.refine_candidates:
            if self.candidate_ids is None or self.candidate_scores is None:
                raise RuntimeError("refined selector workspaces were not allocated")
            paged_support_refined_select_triton(
                self.query,
                self.support_keys,
                self.cache.key,
                self.cache.page_table,
                self.cache.sequence_lengths,
                self.routes.atom_ids,
                layout=self.cache.normalized_layout,
                sink_atoms=self.sink_atoms,
                recent_atoms=self.recent_atoms,
                middle_atoms=self.middle_atoms,
                candidate_atoms=self.refine_candidates,
                scores=self.scores,
                candidate_ids=self.candidate_ids,
                candidate_scores=self.candidate_scores,
            )
            return
        paged_support_select_triton(
            self.query,
            self.support_keys,
            self.cache.sequence_lengths,
            self.routes.atom_ids,
            sink_atoms=self.sink_atoms,
            recent_atoms=self.recent_atoms,
            middle_atoms=self.middle_atoms,
            scores=self.scores,
        )

    def check_route_errors(self) -> None:
        self.selected_plan.check_route_errors()

    def run(self) -> torch.Tensor:
        """Select, lower, and execute attention in one no-sync decode path."""

        self.select()
        return self.selected_plan.run()


def stream_attn_paged_selected_decode(
    query: torch.Tensor,
    cache: PagedKVCache,
    routes: PackedPagedRoute64,
    *,
    schedule_epoch: int,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Plan and execute one H100 selected paged-KV decode call."""

    return PagedSelectedDecodePlan.build(
        query,
        cache,
        routes,
        schedule_epoch=schedule_epoch,
        output=output,
    ).run()


def stream_attn_paged_dynamic_selected_decode(
    query: torch.Tensor,
    cache: PagedKVCache,
    routes: AttentionRouteCSR,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Plan and run the H100 no-sync dynamic Q-head selected path."""

    return PagedDynamicSelectedDecodePlan.build(
        query,
        cache,
        routes,
        output=output,
    ).run()


def stream_attn_paged_query_selected_decode(
    query: torch.Tensor,
    cache: PagedKVCache,
    *,
    selected_atoms: int = 6,
    sink_atoms: int = 1,
    recent_atoms: int = 1,
    support_width: int = 2,
    support_method: str = "centroid_extremes",
    refine_candidates: int = 0,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Plan and execute one query-aware selected H100 paged decode call."""

    return PagedQuerySelectedDecodePlan.build(
        query,
        cache,
        selected_atoms=selected_atoms,
        sink_atoms=sink_atoms,
        recent_atoms=recent_atoms,
        support_width=support_width,
        support_method=support_method,
        refine_candidates=refine_candidates,
        output=output,
    ).run()


@dataclass
class PagedSelectedDecodeRunner:
    """Engine-compatible selected paged runner with serving telemetry."""

    plan: PagedSelectedDecodePlan
    info: Any

    def run(self) -> torch.Tensor:
        return self.plan.run()

    def run_with_info(self):
        return self.run(), self.info


@dataclass
class PagedDynamicSelectedDecodeRunner:
    """Engine-compatible runner for mutable GPU Q-head route atoms."""

    plan: PagedDynamicSelectedDecodePlan
    info: Any

    def run(self) -> torch.Tensor:
        return self.plan.run()

    def run_with_info(self):
        return self.run(), self.info


@dataclass
class PagedQuerySelectedDecodeRunner:
    """Engine-compatible query-selected paged runner with telemetry."""

    plan: PagedQuerySelectedDecodePlan
    info: Any

    def run(self) -> torch.Tensor:
        return self.plan.run()

    def run_with_info(self):
        return self.run(), self.info


@dataclass
class PagedExactDecodeRunner:
    """Engine-compatible paged exact runner with serving telemetry."""

    plan: PagedExactDecodePlan
    info: Any

    def run(self) -> torch.Tensor:
        return self.plan.run()

    def run_with_info(self):
        return self.run(), self.info


def stream_attn_paged_exact_decode(
    query: torch.Tensor,
    cache: PagedKVCache,
    *,
    output: Optional[torch.Tensor] = None,
    splits: Optional[int] = None,
) -> torch.Tensor:
    """Plan and execute one exact paged-KV decode call."""

    return PagedExactDecodePlan.build(
        query,
        cache,
        output=output,
        splits=splits,
    ).run()
