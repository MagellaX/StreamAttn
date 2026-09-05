"""Experimental exact SM90 micro-prefill over contiguous head-major KV.

The first family reuses the proven transposed GQA WGMMA state machine across
all query positions. It is intentionally a candidate family, not a promoted
dispatcher route: measured compiler evidence decides where it wins against a
natural small-M family and the fastest eligible exact baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .micro_prefill_semantics import compile_semantic_extension, validate_positions
from .transposed_gqa_exact import (
    compile_transposed_gqa_exact_extension,
    resolve_cutlass_root,
)


MICRO_PREFILL_MIN_QUERY_LEN = 2
MICRO_PREFILL_MAX_QUERY_LEN = 64
MICRO_PREFILL_TILE_N = 64
MICRO_PREFILL_QUERY_TILE_ROWS = 64
MICRO_PREFILL_SUPPORTED_GROUPS = (4, 8)
MICRO_PREFILL_SUPPORTED_HEAD_DIMS = (64, 128)


def choose_micro_prefill_splits(
    *,
    batch: int,
    query_len: int,
    kv_heads: int,
    kv_len: int,
    target_producer_ctas: int = 256,
) -> int:
    """Choose exact split-KV parallelism for flattened query/head groups."""

    if batch <= 0 or kv_heads <= 0:
        raise ValueError("batch and kv_heads must be positive")
    if not MICRO_PREFILL_MIN_QUERY_LEN <= query_len <= MICRO_PREFILL_MAX_QUERY_LEN:
        raise ValueError("query_len must be in [2,64]")
    if kv_len <= 0 or kv_len % MICRO_PREFILL_TILE_N:
        raise ValueError("kv_len must be positive and divisible by 64")
    if target_producer_ctas <= 0:
        raise ValueError("target_producer_ctas must be positive")

    producer_groups = batch * query_len * kv_heads
    needed = (target_producer_ctas + producer_groups - 1) // producer_groups
    return max(1, min(kv_len // MICRO_PREFILL_TILE_N, needed))


def natural_micro_prefill_query_tiles(*, query_len: int, group_size: int) -> int:
    """Return 64-row WGMMA query tiles after packing positions and GQA heads."""

    if not MICRO_PREFILL_MIN_QUERY_LEN <= query_len <= MICRO_PREFILL_MAX_QUERY_LEN:
        raise ValueError("query_len must be in [2,64]")
    if group_size not in MICRO_PREFILL_SUPPORTED_GROUPS:
        raise ValueError("group_size must be 4 or 8")
    positions_per_tile = MICRO_PREFILL_QUERY_TILE_ROWS // group_size
    return (query_len + positions_per_tile - 1) // positions_per_tile


def choose_natural_micro_prefill_splits(
    *,
    batch: int,
    query_len: int,
    kv_heads: int,
    group_size: int,
    kv_len: int,
    target_producer_ctas: int = 256,
) -> int:
    """Choose split-KV parallelism after 64-row query/GQA packing."""

    if batch <= 0 or kv_heads <= 0:
        raise ValueError("batch and kv_heads must be positive")
    if kv_len <= 0 or kv_len % MICRO_PREFILL_TILE_N:
        raise ValueError("kv_len must be positive and divisible by 64")
    if target_producer_ctas <= 0:
        raise ValueError("target_producer_ctas must be positive")
    query_tiles = natural_micro_prefill_query_tiles(
        query_len=query_len, group_size=group_size
    )
    work_groups = batch * kv_heads * query_tiles
    needed = (target_producer_ctas + work_groups - 1) // work_groups
    return max(1, min(kv_len // MICRO_PREFILL_TILE_N, needed))


def micro_prefill_shape_reasons(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> list[str]:
    """Return stable reasons why buffers miss the initial family contract."""

    reasons: list[str] = []
    if query.dim() != 4 or key_cache.dim() != 4 or value_cache.dim() != 4:
        return ["rank"]
    if key_cache.shape != value_cache.shape:
        return ["kv_shape"]

    batch, query_len, q_heads, head_dim = map(int, query.shape)
    kv_batch, kv_heads, kv_len, kv_dim = map(int, key_cache.shape)
    if batch <= 0:
        reasons.append("batch")
    if not query.device == key_cache.device == value_cache.device:
        reasons.append("device")
    if batch != kv_batch or head_dim != kv_dim:
        reasons.append("shape")
    if not MICRO_PREFILL_MIN_QUERY_LEN <= query_len <= MICRO_PREFILL_MAX_QUERY_LEN:
        reasons.append("query_len")
    group_size = (
        q_heads // kv_heads
        if kv_heads > 0 and q_heads % kv_heads == 0
        else 0
    )
    if group_size not in MICRO_PREFILL_SUPPORTED_GROUPS:
        reasons.append("gqa")
    if head_dim not in MICRO_PREFILL_SUPPORTED_HEAD_DIMS:
        reasons.append("head_dim")
    if kv_len <= 0 or kv_len % MICRO_PREFILL_TILE_N:
        reasons.append("kv_len")
    if query.dtype not in (torch.bfloat16, torch.float16) or not all(
        tensor.dtype == query.dtype
        for tensor in (query, key_cache, value_cache)
    ):
        reasons.append("dtype")
    if not all(
        tensor.is_contiguous()
        for tensor in (query, key_cache, value_cache)
    ):
        reasons.append("layout")
    return reasons


def supports_sm90_micro_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    require_cutlass: bool = True,
) -> bool:
    """Return whether buffers can enter the experimental exact SM90 family."""

    if micro_prefill_shape_reasons(query, key_cache, value_cache):
        return False
    if not all(
        tensor.is_cuda for tensor in (query, key_cache, value_cache)
    ):
        return False
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        return False
    if require_cutlass:
        try:
            resolve_cutlass_root()
        except FileNotFoundError:
            return False
    return True


def _compile_plan(
    query, key_cache, *, causal, query_positions, key_positions,
    cutlass_root, build_dir, verbose,
):
    validate_positions(
        query, key_cache, causal=causal,
        query_positions=query_positions, key_positions=key_positions,
    )
    if causal or query.dtype == torch.float16:
        extension = compile_semantic_extension(
            head_dim=int(query.shape[-1]), dtype=query.dtype, causal=causal,
            cutlass_root=cutlass_root, build_dir=build_dir, verbose=verbose,
        )
        if not causal:
            query_positions = torch.empty(0, dtype=torch.int64, device=query.device)
            key_positions = torch.empty(0, dtype=torch.int64, device=query.device)
        return extension, (query_positions, key_positions)
    return compile_transposed_gqa_exact_extension(
        head_dim=int(query.shape[-1]), cutlass_root=cutlass_root,
        build_dir=build_dir, verbose=verbose,
    ), None


@dataclass
class MicroPrefillPlan:
    """Allocation-free replay plan for exact transposed GQA micro-prefill."""

    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    output: torch.Tensor
    query_group: torch.Tensor
    output_group: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    num_splits: int
    extension: Any
    launch: Any
    backend: str = "sm90_transposed_gqa_wgmma_micro_prefill"
    positions: tuple[torch.Tensor, torch.Tensor] | None = None

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        output: Optional[torch.Tensor] = None,
        num_splits: Optional[int] = None,
        target_producer_ctas: int = 256,
        cutlass_root: Optional[Path] = None,
        build_dir: Optional[Path] = None,
        compile_verbose: bool = False,
        causal: bool = False,
        query_positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
    ) -> "MicroPrefillPlan":
        if not supports_sm90_micro_prefill(
            query,
            key_cache,
            value_cache,
            require_cutlass=False,
        ):
            reasons = micro_prefill_shape_reasons(
                query, key_cache, value_cache
            )
            raise ValueError(
                "unsupported SM90 exact micro-prefill buffers: "
                + ",".join(reasons or ["device"])
            )

        batch, query_len, q_heads, head_dim = map(int, query.shape)
        kv_heads = int(key_cache.shape[1])
        kv_len = int(key_cache.shape[2])
        group_size = q_heads // kv_heads
        splits = num_splits
        if splits is None:
            splits = choose_micro_prefill_splits(
                batch=batch,
                query_len=query_len,
                kv_heads=kv_heads,
                kv_len=kv_len,
                target_producer_ctas=target_producer_ctas,
            )
        if splits <= 0 or splits > min(kv_len // MICRO_PREFILL_TILE_N, 512):
            raise ValueError("num_splits must be in [1,min(kv_len/64,512)]")

        if output is None:
            output = torch.empty_like(query)
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("output must match query shape and dtype")
        if output.device != query.device or not output.is_contiguous():
            raise ValueError("output must be contiguous on the query CUDA device")

        groups = batch * query_len * kv_heads
        partial_output = torch.empty(
            groups,
            splits,
            8,
            head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            groups,
            splits,
            8,
            device=query.device,
            dtype=torch.float32,
        )
        extension, positions = _compile_plan(
            query, key_cache, causal=causal,
            query_positions=query_positions, key_positions=key_positions,
            cutlass_root=cutlass_root,
            build_dir=build_dir,
            verbose=compile_verbose,
        )
        return cls(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_group=query.view(
                batch, query_len, kv_heads, group_size, head_dim
            ),
            output_group=output.view(groups, group_size, head_dim),
            partial_output=partial_output,
            partial_lse=partial_lse,
            num_splits=splits,
            extension=extension,
            launch=extension.out if positions is not None else extension.micro_prefill_out,
            positions=positions,
        )

    @property
    def workspace_bytes(self) -> int:
        return self.partial_output.numel() * self.partial_output.element_size() + (
            self.partial_lse.numel() * self.partial_lse.element_size()
        )

    def run(self) -> torch.Tensor:
        """Replay the preplanned exact producer and associative merge."""

        if self.positions is not None:
            self.launch(
                self.query, self.key_cache, self.value_cache,
                self.partial_output, self.partial_lse, self.output,
                *self.positions, self.num_splits, False,
            )
            return self.output
        self.launch(
            self.query_group,
            self.key_cache,
            self.value_cache,
            self.partial_output,
            self.partial_lse,
            self.output_group,
            self.num_splits,
        )
        return self.output


@dataclass
class NaturalMicroPrefillPlan:
    """Exact 64-row small-M plan that reuses each KV tile across query rows."""

    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    output: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    num_splits: int
    query_tiles: int
    extension: Any
    launch: Any
    backend: str = "sm90_natural_wgmma_micro_prefill"
    positions: tuple[torch.Tensor, torch.Tensor] | None = None

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        output: Optional[torch.Tensor] = None,
        num_splits: Optional[int] = None,
        target_producer_ctas: int = 256,
        cutlass_root: Optional[Path] = None,
        build_dir: Optional[Path] = None,
        compile_verbose: bool = False,
        causal: bool = False,
        query_positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
    ) -> "NaturalMicroPrefillPlan":
        if not supports_sm90_micro_prefill(
            query,
            key_cache,
            value_cache,
            require_cutlass=False,
        ):
            reasons = micro_prefill_shape_reasons(
                query, key_cache, value_cache
            )
            raise ValueError(
                "unsupported SM90 natural micro-prefill buffers: "
                + ",".join(reasons or ["device"])
            )

        batch, query_len, q_heads, head_dim = map(int, query.shape)
        kv_heads = int(key_cache.shape[1])
        kv_len = int(key_cache.shape[2])
        group_size = q_heads // kv_heads
        query_tiles = natural_micro_prefill_query_tiles(
            query_len=query_len, group_size=group_size
        )
        splits = num_splits
        if splits is None:
            splits = choose_natural_micro_prefill_splits(
                batch=batch,
                query_len=query_len,
                kv_heads=kv_heads,
                group_size=group_size,
                kv_len=kv_len,
                target_producer_ctas=target_producer_ctas,
            )
        if splits <= 0 or splits > min(kv_len // MICRO_PREFILL_TILE_N, 512):
            raise ValueError("num_splits must be in [1,min(kv_len/64,512)]")

        if output is None:
            output = torch.empty_like(query)
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("output must match query shape and dtype")
        if output.device != query.device or not output.is_contiguous():
            raise ValueError("output must be contiguous on the query CUDA device")

        work_groups = batch * kv_heads * query_tiles
        partial_output = torch.empty(
            work_groups,
            splits,
            MICRO_PREFILL_QUERY_TILE_ROWS,
            head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            work_groups,
            splits,
            MICRO_PREFILL_QUERY_TILE_ROWS,
            device=query.device,
            dtype=torch.float32,
        )
        extension, positions = _compile_plan(
            query, key_cache, causal=causal,
            query_positions=query_positions, key_positions=key_positions,
            cutlass_root=cutlass_root,
            build_dir=build_dir,
            verbose=compile_verbose,
        )
        return cls(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            partial_output=partial_output,
            partial_lse=partial_lse,
            num_splits=splits,
            query_tiles=query_tiles,
            extension=extension,
            launch=extension.out if positions is not None else extension.natural_micro_prefill_out,
            positions=positions,
        )

    @property
    def workspace_bytes(self) -> int:
        return self.partial_output.numel() * self.partial_output.element_size() + (
            self.partial_lse.numel() * self.partial_lse.element_size()
        )

    def run(self) -> torch.Tensor:
        """Replay the preplanned exact natural small-M producer and merge."""

        if self.positions is not None:
            self.launch(
                self.query, self.key_cache, self.value_cache,
                self.partial_output, self.partial_lse, self.output,
                *self.positions, self.num_splits, True,
            )
            return self.output
        self.launch(
            self.query,
            self.key_cache,
            self.value_cache,
            self.partial_output,
            self.partial_lse,
            self.output,
            self.num_splits,
        )
        return self.output


__all__ = [
    "MICRO_PREFILL_MAX_QUERY_LEN",
    "MICRO_PREFILL_MIN_QUERY_LEN",
    "MICRO_PREFILL_QUERY_TILE_ROWS",
    "MICRO_PREFILL_SUPPORTED_GROUPS",
    "MicroPrefillPlan",
    "NaturalMicroPrefillPlan",
    "choose_micro_prefill_splits",
    "choose_natural_micro_prefill_splits",
    "micro_prefill_shape_reasons",
    "natural_micro_prefill_query_tiles",
    "supports_sm90_micro_prefill",
]
