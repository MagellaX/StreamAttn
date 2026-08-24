"""Native paged-KV exact decode kernels.

The kernel consumes a block table directly. It never gathers or repacks the
physical pages into a contiguous per-request cache.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _paged_exact_grouped_partial_kernel(
        Q,
        K,
        V,
        PageTable,
        SequenceLengths,
        PartialM,
        PartialL,
        PartialNum,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_H: tl.constexpr,
        K_STRIDE_PAGE: tl.constexpr,
        K_STRIDE_TOKEN: tl.constexpr,
        K_STRIDE_HEAD: tl.constexpr,
        V_STRIDE_PAGE: tl.constexpr,
        V_STRIDE_TOKEN: tl.constexpr,
        V_STRIDE_HEAD: tl.constexpr,
        PAGE_TABLE_STRIDE_B: tl.constexpr,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        NUM_PHYSICAL_PAGES: tl.constexpr,
        MAX_PAGES: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        TOKENS_PER_TILE: tl.constexpr,
        SPLITS: tl.constexpr,
        PAGES_PER_SPLIT: tl.constexpr,
        TILES_PER_SPLIT: tl.constexpr,
        SCALE: tl.constexpr,
        HEAD_TILE: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_kv_h = tl.program_id(1)
        off_split = tl.program_id(2)
        offs_h_local = tl.arange(0, HEAD_TILE)
        offs_h = off_kv_h * GROUP_SIZE + offs_h_local
        head_mask = offs_h_local < GROUP_SIZE
        offs_d = tl.arange(0, D)
        offs_token = tl.arange(0, TOKENS_PER_TILE)

        q = tl.load(
            Q + off_b * Q_STRIDE_B + offs_h[:, None] * Q_STRIDE_H + offs_d[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
        sequence_length = tl.load(SequenceLengths + off_b).to(tl.int32)
        running_max = tl.full([HEAD_TILE], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([HEAD_TILE], dtype=tl.float32)
        acc_num = tl.zeros([HEAD_TILE, D], dtype=tl.float32)

        for local_tile in tl.range(0, TILES_PER_SPLIT):
            local_token = local_tile * TOKENS_PER_TILE + offs_token
            logical_page = off_split * PAGES_PER_SPLIT + local_token // PAGE_SIZE
            page_token = local_token % PAGE_SIZE
            page_slot_valid = logical_page < MAX_PAGES
            physical_page = tl.load(
                PageTable + off_b * PAGE_TABLE_STRIDE_B + logical_page,
                mask=page_slot_valid,
                other=-1,
            ).to(tl.int64)
            logical_token = logical_page * PAGE_SIZE + page_token
            token_mask = (
                page_slot_valid
                & (local_token < PAGES_PER_SPLIT * PAGE_SIZE)
                & (logical_token < sequence_length)
                & (physical_page >= 0)
                & (physical_page < NUM_PHYSICAL_PAGES)
            )

            k_tile = tl.load(
                K
                + physical_page[:, None] * K_STRIDE_PAGE
                + page_token[:, None] * K_STRIDE_TOKEN
                + off_kv_h * K_STRIDE_HEAD
                + offs_d[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
            qk = tl.where(
                head_mask[:, None] & token_mask[None, :],
                qk,
                -float("inf"),
            )
            tile_max = tl.max(qk, axis=1)
            previous_valid = acc_den > 0.0
            tile_valid = tile_max > -float("inf")
            new_valid = previous_valid | tile_valid
            new_max = tl.maximum(running_max, tile_max)
            safe_new_max = tl.where(new_valid, new_max, 0.0)
            correction = tl.where(
                previous_valid, tl.exp(running_max - safe_new_max), 0.0
            )
            probabilities = tl.exp(qk - safe_new_max[:, None])
            probabilities = tl.where(qk > -float("inf"), probabilities, 0.0)

            v_tile = tl.load(
                V
                + physical_page[:, None] * V_STRIDE_PAGE
                + page_token[:, None] * V_STRIDE_TOKEN
                + off_kv_h * V_STRIDE_HEAD
                + offs_d[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            acc_num = acc_num * correction[:, None] + tl.dot(
                probabilities.to(tl.bfloat16), v_tile, out_dtype=tl.float32
            )
            acc_den = acc_den * correction + tl.sum(probabilities, axis=1)
            running_max = tl.where(new_valid, new_max, running_max)

        state = ((off_b * H + offs_h) * SPLITS + off_split).to(tl.int64)
        tl.store(PartialM + state, running_max, mask=head_mask)
        tl.store(PartialL + state, acc_den, mask=head_mask)
        tl.store(
            PartialNum + state[:, None] * D + offs_d[None, :],
            acc_num,
            mask=head_mask[:, None],
        )

    @triton.jit
    def _paged_exact_splitk_partial_kernel(
        Q,
        K,
        V,
        PageTable,
        SequenceLengths,
        PartialM,
        PartialL,
        PartialNum,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_H: tl.constexpr,
        K_STRIDE_PAGE: tl.constexpr,
        K_STRIDE_TOKEN: tl.constexpr,
        K_STRIDE_HEAD: tl.constexpr,
        V_STRIDE_PAGE: tl.constexpr,
        V_STRIDE_TOKEN: tl.constexpr,
        V_STRIDE_HEAD: tl.constexpr,
        PAGE_TABLE_STRIDE_B: tl.constexpr,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        NUM_PHYSICAL_PAGES: tl.constexpr,
        MAX_PAGES: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        TOKENS_PER_TILE: tl.constexpr,
        SPLITS: tl.constexpr,
        PAGES_PER_SPLIT: tl.constexpr,
        TILES_PER_SPLIT: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_split = tl.program_id(2)
        off_kv_h = off_h // GROUP_SIZE
        offs_d = tl.arange(0, D)
        offs_token = tl.arange(0, TOKENS_PER_TILE)

        q = tl.load(Q + off_b * Q_STRIDE_B + off_h * Q_STRIDE_H + offs_d).to(tl.float32)
        sequence_length = tl.load(SequenceLengths + off_b).to(tl.int32)

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)

        for local_tile in tl.range(0, TILES_PER_SPLIT):
            local_token = local_tile * TOKENS_PER_TILE + offs_token
            logical_page = off_split * PAGES_PER_SPLIT + local_token // PAGE_SIZE
            page_token = local_token % PAGE_SIZE
            page_slot_valid = logical_page < MAX_PAGES
            physical_page = tl.load(
                PageTable + off_b * PAGE_TABLE_STRIDE_B + logical_page,
                mask=page_slot_valid,
                other=-1,
            ).to(tl.int64)
            page_valid = (
                page_slot_valid
                & (logical_page * PAGE_SIZE < sequence_length)
                & (physical_page >= 0)
                & (physical_page < NUM_PHYSICAL_PAGES)
            )
            logical_token = logical_page * PAGE_SIZE + page_token
            token_mask = (
                page_valid
                & (local_token < PAGES_PER_SPLIT * PAGE_SIZE)
                & (logical_token < sequence_length)
            )

            k_tile = tl.load(
                K
                + physical_page[:, None] * K_STRIDE_PAGE
                + page_token[:, None] * K_STRIDE_TOKEN
                + off_kv_h * K_STRIDE_HEAD
                + offs_d[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q[None, :] * k_tile, axis=1) * SCALE
            scores = tl.where(token_mask, scores, -float("inf"))

            tile_max = tl.max(scores, axis=0)
            tile_valid = tile_max > -float("inf")
            previous_valid = acc_den > 0.0
            new_valid = previous_valid | tile_valid
            new_max = tl.maximum(running_max, tile_max)
            safe_new_max = tl.where(new_valid, new_max, 0.0)
            correction = tl.where(
                previous_valid, tl.exp(running_max - safe_new_max), 0.0
            )
            probabilities = tl.exp(scores - safe_new_max)
            probabilities = tl.where(token_mask, probabilities, 0.0)

            v_tile = tl.load(
                V
                + physical_page[:, None] * V_STRIDE_PAGE
                + page_token[:, None] * V_STRIDE_TOKEN
                + off_kv_h * V_STRIDE_HEAD
                + offs_d[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc_num = acc_num * correction + tl.sum(
                probabilities[:, None] * v_tile, axis=0
            )
            acc_den = acc_den * correction + tl.sum(probabilities, axis=0)
            running_max = tl.where(new_valid, new_max, running_max)

        state = (off_b * H + off_h) * SPLITS + off_split
        tl.store(PartialM + state, running_max)
        tl.store(PartialL + state, acc_den)
        tl.store(PartialNum + state * D + offs_d, acc_num)

    @triton.jit
    def _paged_exact_splitk_merge_kernel(
        PartialM,
        PartialL,
        PartialNum,
        Out,
        H: tl.constexpr,
        D: tl.constexpr,
        SPLITS: tl.constexpr,
        SPLITS_TILE: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        offs_split = tl.arange(0, SPLITS_TILE)
        offs_d = tl.arange(0, D)
        split_mask = offs_split < SPLITS
        state = (off_b * H + off_h) * SPLITS

        partial_m = tl.load(
            PartialM + state + offs_split,
            mask=split_mask,
            other=-float("inf"),
        ).to(tl.float32)
        partial_l = tl.load(
            PartialL + state + offs_split,
            mask=split_mask,
            other=0.0,
        ).to(tl.float32)
        merged_m = tl.max(partial_m, axis=0)
        valid = merged_m > -float("inf")
        safe_m = tl.where(valid, merged_m, 0.0)
        factors = tl.exp(partial_m - safe_m)
        factors = tl.where(split_mask & (partial_l > 0.0), factors, 0.0)
        denominator = tl.sum(partial_l * factors, axis=0)

        partial_num = tl.load(
            PartialNum + (state + offs_split[:, None]) * D + offs_d[None, :],
            mask=split_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        numerator = tl.sum(partial_num * factors[:, None], axis=0)
        output = tl.where(denominator > 0.0, numerator / denominator, 0.0)
        tl.store(Out + off_b * H * D + off_h * D + offs_d, output)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def paged_exact_decode_triton_forward_out(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    output: torch.Tensor,
    *,
    layout: str,
    splits: int,
    workspace: dict[str, torch.Tensor],
    tokens_per_tile: int = 512,
    partial_num_warps: int = 4,
    partial_num_stages: int = 2,
    merge_num_warps: int = 1,
    merge_num_stages: int = 2,
) -> torch.Tensor:
    """Run exact M=1 decode directly over a paged KV cache."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    tensors = (
        query,
        key_cache,
        value_cache,
        page_table,
        sequence_lengths,
        output,
    )
    if not all(t.is_cuda for t in tensors):
        raise RuntimeError("paged exact Triton decode requires CUDA tensors")
    if splits <= 0:
        raise ValueError("splits must be positive")

    batch, _query_len, heads, dim = map(int, query.shape)
    normalized_layout = str(layout).upper()
    if normalized_layout == "NHD":
        num_pages, page_size, kv_heads, cache_dim = map(int, key_cache.shape)
        token_stride = int(key_cache.stride(1))
        head_stride = int(key_cache.stride(2))
        value_token_stride = int(value_cache.stride(1))
        value_head_stride = int(value_cache.stride(2))
    elif normalized_layout == "HND":
        num_pages, kv_heads, page_size, cache_dim = map(int, key_cache.shape)
        head_stride = int(key_cache.stride(1))
        token_stride = int(key_cache.stride(2))
        value_head_stride = int(value_cache.stride(1))
        value_token_stride = int(value_cache.stride(2))
    else:
        raise ValueError("layout must be NHD or HND")
    if cache_dim != dim:
        raise ValueError("query and paged cache head dimensions must match")

    max_pages = int(page_table.shape[1])
    pages_per_split = (max_pages + int(splits) - 1) // int(splits)
    tokens_per_tile = int(tokens_per_tile)
    if tokens_per_tile < page_size or tokens_per_tile & (tokens_per_tile - 1):
        raise ValueError("tokens_per_tile must be a power of two >= page_size")
    if tokens_per_tile % page_size:
        raise ValueError("tokens_per_tile must be divisible by page_size")
    tiles_per_split = (
        pages_per_split * page_size + tokens_per_tile - 1
    ) // tokens_per_tile
    score_scale = 1.0 / math.sqrt(float(dim))
    partial_m = workspace["partial_m"]
    partial_l = workspace["partial_l"]
    partial_num = workspace["partial_num"]

    _paged_exact_splitk_partial_kernel[(batch, heads, int(splits))](
        query,
        key_cache,
        value_cache,
        page_table,
        sequence_lengths,
        partial_m,
        partial_l,
        partial_num,
        query.stride(0),
        query.stride(2),
        key_cache.stride(0),
        token_stride,
        head_stride,
        value_cache.stride(0),
        value_token_stride,
        value_head_stride,
        page_table.stride(0),
        H=heads,
        H_KV=kv_heads,
        GROUP_SIZE=heads // kv_heads,
        D=dim,
        NUM_PHYSICAL_PAGES=num_pages,
        MAX_PAGES=max_pages,
        PAGE_SIZE=page_size,
        TOKENS_PER_TILE=tokens_per_tile,
        SPLITS=int(splits),
        PAGES_PER_SPLIT=pages_per_split,
        TILES_PER_SPLIT=tiles_per_split,
        SCALE=score_scale,
        num_warps=int(partial_num_warps),
        num_stages=int(partial_num_stages),
    )
    _paged_exact_splitk_merge_kernel[(batch, heads)](
        partial_m,
        partial_l,
        partial_num,
        output,
        H=heads,
        D=dim,
        SPLITS=int(splits),
        SPLITS_TILE=_next_power_of_2(int(splits)),
        num_warps=int(merge_num_warps),
        num_stages=int(merge_num_stages),
    )
    return output


def paged_exact_decode_grouped_forward_out(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    output: torch.Tensor,
    *,
    layout: str,
    splits: int,
    workspace: dict[str, torch.Tensor],
    tokens_per_tile: int = 128,
    partial_num_warps: int = 8,
    partial_num_stages: int = 2,
    merge_num_warps: int = 1,
    merge_num_stages: int = 2,
) -> torch.Tensor:
    """Run true-GQA paged exact decode with one tensor-core CTA per KV head."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    batch, _query_len, heads, dim = map(int, query.shape)
    normalized_layout = str(layout).upper()
    if normalized_layout == "NHD":
        num_pages, page_size, kv_heads, cache_dim = map(int, key_cache.shape)
        token_stride = int(key_cache.stride(1))
        head_stride = int(key_cache.stride(2))
        value_token_stride = int(value_cache.stride(1))
        value_head_stride = int(value_cache.stride(2))
    elif normalized_layout == "HND":
        num_pages, kv_heads, page_size, cache_dim = map(int, key_cache.shape)
        head_stride = int(key_cache.stride(1))
        token_stride = int(key_cache.stride(2))
        value_head_stride = int(value_cache.stride(1))
        value_token_stride = int(value_cache.stride(2))
    else:
        raise ValueError("layout must be NHD or HND")
    if cache_dim != dim or heads // kv_heads != 8 or dim != 128:
        raise ValueError("grouped exact decode requires G8 and D128")
    if query.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16:
        raise ValueError("grouped exact decode requires bf16 Q/K/V")
    if splits <= 0:
        raise ValueError("splits must be positive")
    if tokens_per_tile not in {64, 128}:
        raise ValueError("grouped tokens_per_tile must be 64 or 128")

    max_pages = int(page_table.shape[1])
    pages_per_split = (max_pages + int(splits) - 1) // int(splits)
    tiles_per_split = (
        pages_per_split * page_size + tokens_per_tile - 1
    ) // tokens_per_tile
    partial_m = workspace["partial_m"]
    partial_l = workspace["partial_l"]
    partial_num = workspace["partial_num"]
    _paged_exact_grouped_partial_kernel[(batch, kv_heads, int(splits))](
        query,
        key_cache,
        value_cache,
        page_table,
        sequence_lengths,
        partial_m,
        partial_l,
        partial_num,
        query.stride(0),
        query.stride(2),
        key_cache.stride(0),
        token_stride,
        head_stride,
        value_cache.stride(0),
        value_token_stride,
        value_head_stride,
        page_table.stride(0),
        H=heads,
        H_KV=kv_heads,
        GROUP_SIZE=8,
        D=128,
        NUM_PHYSICAL_PAGES=num_pages,
        MAX_PAGES=max_pages,
        PAGE_SIZE=page_size,
        TOKENS_PER_TILE=int(tokens_per_tile),
        SPLITS=int(splits),
        PAGES_PER_SPLIT=pages_per_split,
        TILES_PER_SPLIT=tiles_per_split,
        SCALE=1.0 / math.sqrt(128.0),
        HEAD_TILE=16,
        num_warps=int(partial_num_warps),
        num_stages=int(partial_num_stages),
    )
    _paged_exact_splitk_merge_kernel[(batch, heads)](
        partial_m,
        partial_l,
        partial_num,
        output,
        H=heads,
        D=dim,
        SPLITS=int(splits),
        SPLITS_TILE=_next_power_of_2(int(splits)),
        num_warps=int(merge_num_warps),
        num_stages=int(merge_num_stages),
    )
    return output


def make_paged_exact_workspace(
    *,
    batch: int,
    heads: int,
    splits: int,
    dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate split-K online-softmax state once during planning."""

    if min(batch, heads, splits, dim) <= 0:
        raise ValueError("workspace dimensions must be positive")
    return {
        "partial_m": torch.empty(
            batch, heads, splits, device=device, dtype=torch.float32
        ),
        "partial_l": torch.empty(
            batch, heads, splits, device=device, dtype=torch.float32
        ),
        "partial_num": torch.empty(
            batch, heads, splits, dim, device=device, dtype=torch.float32
        ),
    }
