"""Opt-in append ownership for the validated H100 paged decode shape."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .paged import (
    PAGED_EXACT_SM90_NHD_FRAGMENTED_BACKEND,
    PAGED_EXACT_SM90_NHD_FRAGMENTED_RAGGED_BACKEND,
    PagedExactDecodePlan,
    PagedKVCache,
)


def _is_sm90(query: torch.Tensor) -> bool:
    return query.is_cuda and torch.cuda.get_device_capability(query.device) == (9, 0)


def _validate_buffers(query, cache, new_key, new_value, output=None):
    if tuple(query.shape) != (1, 1, 16, 128) or query.dtype != torch.bfloat16:
        raise ValueError("append requires BF16 query [1,1,16,128]")
    cache.validate(query)
    if (cache.normalized_layout != "NHD" or cache.page_size != 16
            or cache.kv_heads != 2 or cache.max_sequence_length != 32768):
        raise ValueError("append requires NHD/page16/Hkv2/32K capacity")
    if cache.page_table.dtype != torch.int32 or cache.sequence_lengths.dtype != torch.int32:
        raise ValueError("append metadata must be int32")
    tensors = dict(query=query, key=cache.key, value=cache.value,
                   page_table=cache.page_table, lengths=cache.sequence_lengths,
                   new_key=new_key, new_value=new_value)
    for incoming in (new_key, new_value):
        if (tuple(incoming.shape) != (1, 2, 128) or incoming.dtype != query.dtype
                or incoming.device != query.device or not incoming.is_contiguous()):
            raise ValueError("incoming K/V must be contiguous BF16 [1,2,128] on the query device")
    if output is not None:
        if (output.shape != query.shape or output.dtype != query.dtype
                or output.device != query.device or not output.is_contiguous()):
            raise ValueError("output must match query shape, dtype and device and be contiguous")
        tensors['output'] = output
    ranges = []
    for name, tensor in tensors.items():
        low = tensor.data_ptr()
        high = low + tensor.numel() * tensor.element_size()
        if any(low < end and start < high for _, start, end in ranges):
            raise ValueError(f"append buffers must not overlap: {name}")
        ranges.append((name, low, high))
    length = int(cache.sequence_lengths[0].item())
    active = cache.page_table[0, :(length + 15) // 16].detach().cpu()
    tail_page = active[-1]
    if int((active == tail_page).sum().item()) != 1:
        raise ValueError("writable tail page aliases another active logical page")


@dataclass(frozen=True)
class PagedAppendDecodePlan:
    """Full attention with the unique tail reader also writing incoming K/V.

    Setup validates the current metadata snapshot and synchronizes. The caller
    must maintain exclusive writable-tail ownership, valid metadata and stable
    buffer allocations, and order all readers/writers on the execution stream.
    No other CTA, request or concurrent stream may read the writable tail.
    run() does not revalidate device metadata; graph replay cannot run Python
    preflight. Revalidate changed ownership unless the allocator maintains it.
    """

    plan: PagedExactDecodePlan
    new_key: torch.Tensor
    new_value: torch.Tensor
    launch: Any

    @classmethod
    def build(
        cls, query: torch.Tensor, cache: PagedKVCache,
        new_key: torch.Tensor, new_value: torch.Tensor, *,
        exclusive_tail_ownership: bool = False,
        output: Optional[torch.Tensor] = None, splits: int = 128,
    ) -> "PagedAppendDecodePlan":
        if exclusive_tail_ownership is not True:
            raise ValueError("caller must maintain exclusive_tail_ownership=True")
        if type(splits) is not int or splits not in (64, 128):
            raise ValueError("append splits must be 64 or 128")
        _validate_buffers(query, cache, new_key, new_value, output)
        if not _is_sm90(query):
            raise ValueError("owner-only append decode requires H100/SM90")
        plan = PagedExactDecodePlan.build(query, cache, output=output, splits=splits,
                                          validate_metadata=False)
        if plan.backend not in (PAGED_EXACT_SM90_NHD_FRAGMENTED_BACKEND,
                                PAGED_EXACT_SM90_NHD_FRAGMENTED_RAGGED_BACKEND):
            raise RuntimeError("native SM90 NHD backend unavailable; append was not performed")
        from .backends.sm90.transposed_gqa_exact import compile_transposed_gqa_exact_extension
        extension = compile_transposed_gqa_exact_extension(head_dim=128)
        return cls(plan, new_key, new_value, extension.paged_append_decode_out)

    @property
    def output(self) -> torch.Tensor:
        return self.plan.output

    @property
    def workspace_bytes(self) -> int:
        return self.plan.workspace_bytes

    def validate(self) -> None:
        """Paid, synchronizing preflight; cannot detect external concurrent users."""
        _validate_buffers(self.plan.query, self.plan.cache, self.new_key, self.new_value,
                          self.plan.output)

    def run(self) -> torch.Tensor:
        """Enqueue append, full-context producer and normalized-state merge."""
        p = self.plan
        self.launch(p.query_group, p.cache.key, p.cache.value, p.cache.page_table,
                    p.cache.sequence_lengths, self.new_key, self.new_value,
                    p.workspace['partial_o'], p.workspace['partial_lse'],
                    p.output_group, p.splits)
        return p.output
