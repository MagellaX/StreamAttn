"""Experimental allocation-free, direct paged/ragged SM90 micro-prefill."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .micro_prefill import natural_micro_prefill_query_tiles
from .micro_prefill_semantics import compile_semantic_extension, validate_positions
from stream_attention.paged import PagedKVCache


def validate_paged_micro_prefill(
    query: torch.Tensor, cache: PagedKVCache, query_lengths: torch.Tensor,
    *, causal: bool = False, query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None, validate_metadata: bool = True,
) -> None:
    """Validate values at planning time, never copy metadata during replay."""
    cache.validate_prefill(query, validate_metadata=validate_metadata)
    batch, capacity, heads, dim = query.shape
    if not 2 <= capacity <= 64 or dim not in (64, 128):
        raise ValueError("paged micro-prefill requires M in [2,64], D64/D128")
    if heads // cache.kv_heads not in (4, 8):
        raise ValueError("paged micro-prefill requires G4/G8")
    if query.dtype not in (torch.bfloat16, torch.float16) or cache.page_size != 16:
        raise ValueError("paged micro-prefill requires FP16/BF16 and page-16")
    for name, tensor in (
        ("page_table", cache.page_table),
        ("sequence_lengths", cache.sequence_lengths),
        ("query_lengths", query_lengths),
    ):
        if tensor.dtype != torch.int32 or tensor.device != query.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous int32 on the query device")
    if query_lengths.shape != (batch,):
        raise ValueError("query_lengths must have shape [batch]")
    if validate_metadata:
        lengths = query_lengths.detach().cpu()
        if bool(torch.any(lengths < 0)) or bool(torch.any(lengths > capacity)):
            raise ValueError("query_lengths must be in [0,query_capacity]")
    validate_positions(
        query, cache.key, causal=causal, query_positions=query_positions,
        key_positions=key_positions, key_capacity=cache.max_sequence_length,
    )


@dataclass
class PagedMicroPrefillPlan:
    """Two retained families sharing native page addressing and fixed buffers.

    Q/output are [B,Mmax,Hq,D]; query_lengths and cache.sequence_lengths define
    valid prefixes. Empty/padded query rows produce zero output and -inf LSE.
    Page IDs and lengths may be updated in place between graph replays, within
    the validated buffer capacities. Callers must keep active IDs in bounds
    and synchronize metadata updates. No replay-time host readback is used.
    """

    query: torch.Tensor
    cache: PagedKVCache
    query_lengths: torch.Tensor
    output: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    query_positions: torch.Tensor
    key_positions: torch.Tensor
    num_splits: int
    natural: bool
    extension: Any

    @classmethod
    def build(
        cls, query: torch.Tensor, cache: PagedKVCache,
        query_lengths: torch.Tensor, *, natural: bool = False,
        output: torch.Tensor | None = None, num_splits: int | None = None,
        target_producer_ctas: int = 256, causal: bool = False,
        query_positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
        cutlass_root: Path | None = None, build_dir: Path | None = None,
        compile_verbose: bool = False,
    ) -> "PagedMicroPrefillPlan":
        validate_paged_micro_prefill(
            query, cache, query_lengths, causal=causal,
            query_positions=query_positions, key_positions=key_positions,
        )
        if not query.is_cuda or torch.cuda.get_device_capability(query.device) != (9, 0):
            raise ValueError("paged micro-prefill requires an SM90 CUDA device")
        if not isinstance(natural, bool) or target_producer_ctas <= 0:
            raise ValueError("natural must be bool and target_producer_ctas positive")
        batch, capacity, heads, dim = map(int, query.shape)
        tiles = natural_micro_prefill_query_tiles(
            query_len=capacity, group_size=heads // cache.kv_heads,
        ) if natural else capacity
        groups, rows = batch * cache.kv_heads * tiles, 64 if natural else 8
        max_splits = min((cache.max_sequence_length + 63) // 64, 512)
        splits = num_splits if num_splits is not None else max(
            1, min(max_splits, (target_producer_ctas + groups - 1) // groups)
        )
        if not isinstance(splits, int) or not 1 <= splits <= max_splits:
            raise ValueError("num_splits exceeds the cache tile capacity or 512")
        if output is None:
            output = torch.empty_like(query)
        if (output.shape != query.shape or output.dtype != query.dtype
                or output.device != query.device or not output.is_contiguous()):
            raise ValueError("output must match contiguous query shape, dtype, device")
        if not causal:
            query_positions = torch.empty(0, dtype=torch.int64, device=query.device)
            key_positions = torch.empty(0, dtype=torch.int64, device=query.device)
        extension = compile_semantic_extension(
            head_dim=dim, dtype=query.dtype, causal=causal, paged=True,
            cutlass_root=cutlass_root, build_dir=build_dir, verbose=compile_verbose,
        )
        return cls(
            query, cache, query_lengths, output,
            torch.empty((groups, splits, rows, dim), device=query.device, dtype=torch.float32),
            torch.empty((groups, splits, rows), device=query.device, dtype=torch.float32),
            query_positions, key_positions, splits, natural, extension,
        )

    @property
    def workspace_bytes(self) -> int:
        return 4 * (self.partial_output.numel() + self.partial_lse.numel())

    @property
    def backend(self) -> str:
        family = "natural" if self.natural else "transposed"
        return f"sm90_{family}_wgmma_paged_micro_prefill"

    def run(self) -> torch.Tensor:
        self.extension.out(
            self.query, self.cache.key, self.cache.value,
            self.partial_output, self.partial_lse, self.output,
            self.query_positions, self.key_positions,
            self.cache.page_table, self.cache.sequence_lengths, self.query_lengths,
            self.num_splits, self.natural, self.cache.normalized_layout == "NHD",
        )
        return self.output
