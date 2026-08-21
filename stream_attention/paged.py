"""Paged-KV cache contracts and exact decode planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch


PAGED_EXACT_NATIVE_BACKEND = "streamattn_paged_exact_native"
PAGED_EXACT_SM90_BACKEND = "streamattn_paged_sm90_wgmma_exact"

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
        return int(self.key.shape[1] if self.normalized_layout == "NHD" else self.key.shape[2])

    @property
    def kv_heads(self) -> int:
        if self.key.dim() != 4:
            return 0
        return int(self.key.shape[2] if self.normalized_layout == "NHD" else self.key.shape[1])

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
        if self.num_pages <= 0 or self.page_size <= 0 or self.max_pages_per_request <= 0:
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
                raise ValueError("active page_table entries must reference physical pages")


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
                numerator = (
                    numerator * correction
                    + torch.sum(probabilities[:, None] * value_page.float(), dim=0)
                )
                denominator = denominator * correction + probabilities.sum()
                running_max = new_max
            output[batch_idx, 0, head_idx].copy_(
                (numerator / denominator).to(output.dtype)
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
        validate_metadata: bool = True,
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

        if query.is_cuda:
            full_lengths = bool(
                torch.all(cache.sequence_lengths == cache.max_sequence_length).item()
            )
            use_sm90_paged = (
                torch.cuda.get_device_capability(query.device) == (9, 0)
                and query.dtype == torch.bfloat16
                and cache.normalized_layout == "HND"
                and cache.page_size == 64
                and int(query.shape[2]) == 16
                and cache.kv_heads == 2
                and int(query.shape[3]) == 64
                and cache.page_table.dtype == torch.int32
                and full_lengths
                and (
                    int(query.shape[0]), cache.max_sequence_length
                ) in PROMOTED_PAGED_EXACT_SPLITS
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
                if splits is None:
                    selected_splits = PROMOTED_PAGED_EXACT_SPLITS.get(
                        (int(query.shape[0]), cache.max_sequence_length),
                        choose_num_splits(
                            batch=int(query.shape[0]),
                            kv_heads=cache.kv_heads,
                            kv_len=cache.max_sequence_length,
                        ),
                    )
                groups = int(query.shape[0]) * cache.kv_heads
                partial_o = torch.empty(
                    groups,
                    selected_splits,
                    8,
                    64,
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
                extension = compile_transposed_gqa_exact_extension(head_dim=64)
                return cls(
                    query=query,
                    cache=cache,
                    output=output,
                    splits=selected_splits,
                    workspace={"partial_o": partial_o, "partial_lse": partial_lse},
                    backend=PAGED_EXACT_SM90_BACKEND,
                    launch=extension.paged_exact_decode_out,
                    tokens_per_tile=int(tokens_per_tile),
                    partial_num_warps=int(partial_num_warps),
                    query_group=query.view(int(query.shape[0]), 2, 8, 64),
                    output_group=output.view(groups, 8, 64),
                )
            from .kernels.paged_exact_triton import (
                TRITON_AVAILABLE,
                make_paged_exact_workspace,
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
            return cls(
                query=query,
                cache=cache,
                output=output,
                splits=selected_splits,
                workspace=workspace,
                backend=PAGED_EXACT_NATIVE_BACKEND,
                launch=paged_exact_decode_triton_forward_out,
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
        if self.backend == PAGED_EXACT_SM90_BACKEND:
            assert self.query_group is not None and self.output_group is not None
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
