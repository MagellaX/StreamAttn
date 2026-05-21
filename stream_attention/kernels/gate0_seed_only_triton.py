"""Calibrated seed-only Gate-0 decode kernel.

This is a deliberately aggressive science kernel. It computes only the
production seed set (sink blocks, recent blocks, and optional middle seed
blocks) and skips the remaining middle region without projection scanning.

The goal is to test whether some calibrated heads are effectively seed-only.
If true, this avoids the split-K projection/merge overhead that dominates the
current true-GQA path.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _seed_only_kernel(
        Q,
        K,
        V,
        Out,
        RawStats,
        N: tl.constexpr,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        RECENT_START: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
        HAS_STATS: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_kv_h = off_h // GROUP_SIZE
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)
        seed_blocks = tl.full([], SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS, dtype=tl.int32)

        for iter_idx in range(0, SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS):
            if iter_idx < SINK_BLOCKS:
                block_idx = iter_idx
            elif iter_idx < SINK_BLOCKS + RECENT_BLOCKS:
                recent_pos = iter_idx - SINK_BLOCKS
                block_idx = RECENT_START + recent_pos
            else:
                middle_pos = iter_idx - SINK_BLOCKS - RECENT_BLOCKS
                if BLOCK_ORDER == 0:  # sequential
                    block_idx = SINK_BLOCKS + middle_pos
                else:  # recent_first / sink_recent_first
                    block_idx = RECENT_START - 1 - middle_pos

            start_n = block_idx * TILE_N
            block_len_i = min(TILE_N, N - start_n)
            offs_n = start_n + tl.arange(0, TILE_N)
            col_mask = tl.arange(0, TILE_N) < block_len_i
            k_tile = tl.load(
                K
                + off_b * N * H_KV * D
                + offs_n[:, None] * H_KV * D
                + off_kv_h * D
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            qk = tl.sum(q[None, :].to(tl.float32) * k_tile.to(tl.float32), axis=1) * SCALE
            qk = tl.where(col_mask, qk, -float("inf"))
            tile_max = tl.max(qk, axis=0)
            tile_valid = tile_max > -float("inf")
            prev_valid = acc_den > 0.0
            new_valid = prev_valid | tile_valid
            new_max = tl.maximum(running_max, tile_max)
            safe_new_max = tl.where(new_valid, new_max, 0.0)
            correction = tl.where(prev_valid, tl.exp(running_max - safe_new_max), 0.0)
            p = tl.exp(qk - safe_new_max)
            p = tl.where(qk > -float("inf"), p, 0.0)
            v_tile = tl.load(
                V
                + off_b * N * H_KV * D
                + offs_n[:, None] * H_KV * D
                + off_kv_h * D
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            if PV_USE_BF16:
                weighted = p[:, None].to(tl.bfloat16) * v_tile.to(tl.bfloat16)
            else:
                weighted = p[:, None].to(tl.float16) * v_tile.to(tl.float16)
            acc_num = acc_num * correction + tl.sum(weighted.to(tl.float32), axis=0)
            acc_den = acc_den * correction + tl.sum(p, axis=0)
            running_max = tl.where(new_valid, new_max, running_max)

        out = acc_num / acc_den
        out = tl.where(acc_den > 0.0, out, 0.0)
        tl.store(Out + off_b * H * D + off_h * D + offs_d, out)

        if HAS_STATS:
            stats_base = (off_b * H + off_h) * 4
            tl.store(RawStats + stats_base + 0, seed_blocks)
            tl.store(RawStats + stats_base + 1, tl.full([], NUM_BLOCKS, dtype=tl.int32))
            tl.store(RawStats + stats_base + 2, tl.full([], SINK_BLOCKS + RECENT_BLOCKS, dtype=tl.int32))
            tl.store(RawStats + stats_base + 3, tl.full([], MIDDLE_SEED_BLOCKS, dtype=tl.int32))

    @triton.jit
    def _seed_only_selected_kernel(
        Q,
        K,
        V,
        SelectedHeads,
        Out,
        RawStats,
        N: tl.constexpr,
        H_Q: tl.constexpr,
        H_KV: tl.constexpr,
        SELECTED: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        RECENT_START: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
        HAS_STATS: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_s = tl.program_id(1)
        q_head = tl.load(SelectedHeads + off_s)
        off_kv_h = q_head // GROUP_SIZE
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H_Q * D + q_head * D + offs_d)

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)
        seed_blocks = tl.full([], SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS, dtype=tl.int32)

        for iter_idx in range(0, SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS):
            if iter_idx < SINK_BLOCKS:
                block_idx = iter_idx
            elif iter_idx < SINK_BLOCKS + RECENT_BLOCKS:
                recent_pos = iter_idx - SINK_BLOCKS
                block_idx = RECENT_START + recent_pos
            else:
                middle_pos = iter_idx - SINK_BLOCKS - RECENT_BLOCKS
                if BLOCK_ORDER == 0:  # sequential
                    block_idx = SINK_BLOCKS + middle_pos
                else:  # recent_first / sink_recent_first
                    block_idx = RECENT_START - 1 - middle_pos

            start_n = block_idx * TILE_N
            block_len_i = min(TILE_N, N - start_n)
            offs_n = start_n + tl.arange(0, TILE_N)
            col_mask = tl.arange(0, TILE_N) < block_len_i
            k_tile = tl.load(
                K
                + off_b * N * H_KV * D
                + offs_n[:, None] * H_KV * D
                + off_kv_h * D
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            qk = tl.sum(q[None, :].to(tl.float32) * k_tile.to(tl.float32), axis=1) * SCALE
            qk = tl.where(col_mask, qk, -float("inf"))
            tile_max = tl.max(qk, axis=0)
            tile_valid = tile_max > -float("inf")
            prev_valid = acc_den > 0.0
            new_valid = prev_valid | tile_valid
            new_max = tl.maximum(running_max, tile_max)
            safe_new_max = tl.where(new_valid, new_max, 0.0)
            correction = tl.where(prev_valid, tl.exp(running_max - safe_new_max), 0.0)
            p = tl.exp(qk - safe_new_max)
            p = tl.where(qk > -float("inf"), p, 0.0)
            v_tile = tl.load(
                V
                + off_b * N * H_KV * D
                + offs_n[:, None] * H_KV * D
                + off_kv_h * D
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            if PV_USE_BF16:
                weighted = p[:, None].to(tl.bfloat16) * v_tile.to(tl.bfloat16)
            else:
                weighted = p[:, None].to(tl.float16) * v_tile.to(tl.float16)
            acc_num = acc_num * correction + tl.sum(weighted.to(tl.float32), axis=0)
            acc_den = acc_den * correction + tl.sum(p, axis=0)
            running_max = tl.where(new_valid, new_max, running_max)

        out = acc_num / acc_den
        out = tl.where(acc_den > 0.0, out, 0.0)
        tl.store(Out + off_b * SELECTED * D + off_s * D + offs_d, out)

        if HAS_STATS:
            stats_base = (off_b * SELECTED + off_s) * 5
            tl.store(RawStats + stats_base + 0, q_head)
            tl.store(RawStats + stats_base + 1, seed_blocks)
            tl.store(RawStats + stats_base + 2, tl.full([], NUM_BLOCKS, dtype=tl.int32))
            tl.store(RawStats + stats_base + 3, tl.full([], SINK_BLOCKS + RECENT_BLOCKS, dtype=tl.int32))
            tl.store(RawStats + stats_base + 4, tl.full([], MIDDLE_SEED_BLOCKS, dtype=tl.int32))


def _block_order_id(block_order: str) -> int:
    if block_order == "sequential":
        return 0
    if block_order == "recent_first":
        return 1
    if block_order == "sink_recent_first":
        return 2
    raise ValueError("block_order must be 'sequential', 'recent_first', or 'sink_recent_first'")


def gate0_seed_only_attention_triton_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    return_raw_stats: bool = False,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Run calibrated seed-only decode for MHA or true-GQA tensors."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value)):
        raise RuntimeError("gate0_seed_only_attention_triton_forward requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if query.shape[1] != 1:
        raise ValueError("seed-only Gate-0 only supports query_len == 1")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query/key/value must have matching batch and dim")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0 or middle_seed_blocks < 0:
        raise ValueError("seed block counts must be non-negative")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    batch, _seq_q, heads, dim = query.shape
    seq_k = key.shape[1]
    kv_heads = key.shape[2]
    group_size = heads // kv_heads
    num_blocks = triton.cdiv(seq_k, block_size)
    if sink_blocks + recent_blocks > num_blocks:
        raise ValueError("sink_blocks + recent_blocks must not exceed K blocks")
    middle_blocks = num_blocks - sink_blocks - recent_blocks
    if middle_seed_blocks > middle_blocks:
        raise ValueError("middle_seed_blocks must be within the middle block count")

    output = torch.empty_like(query)
    raw_stats = (
        torch.empty(batch, heads, 4, device=query.device, dtype=torch.int32)
        if return_raw_stats
        else torch.empty(1, device=query.device, dtype=torch.int32)
    )
    score_scale = 1.0 / math.sqrt(dim)
    recent_start = num_blocks - recent_blocks
    _seed_only_kernel[(batch, heads)](
        query,
        key,
        value,
        output,
        raw_stats,
        N=seq_k,
        H=heads,
        H_KV=kv_heads,
        GROUP_SIZE=group_size,
        D=dim,
        NUM_BLOCKS=num_blocks,
        TILE_N=block_size,
        SCALE=score_scale,
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        RECENT_START=int(recent_start),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        PV_USE_BF16=value.dtype is torch.bfloat16,
        HAS_STATS=return_raw_stats,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, raw_stats if return_raw_stats else None


def gate0_seed_only_selected_attention_triton_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_heads: torch.Tensor,
    *,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    return_raw_stats: bool = False,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Run seed-only decode for arbitrary global Q heads.

    ``query`` keeps the full Q-head axis, while ``key``/``value`` keep true-GQA
    KV heads. ``selected_heads`` contains global Q-head ids. The output has
    shape ``[batch, 1, len(selected_heads), dim]`` in the same order as
    ``selected_heads``.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, selected_heads)):
        raise RuntimeError("gate0_seed_only_selected_attention_triton_forward requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if query.shape[1] != 1:
        raise ValueError("seed-only selected Gate-0 only supports query_len == 1")
    if selected_heads.dim() != 1 or selected_heads.numel() <= 0:
        raise ValueError("selected_heads must be a non-empty rank-1 tensor")
    if selected_heads.dtype != torch.int32:
        raise ValueError("selected_heads must use torch.int32 dtype")
    if selected_heads.device != query.device:
        raise ValueError("selected_heads must be on the same CUDA device as query")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query/key/value must have matching batch and dim")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0 or middle_seed_blocks < 0:
        raise ValueError("seed block counts must be non-negative")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    selected_heads = selected_heads.contiguous()
    batch, _seq_q, q_heads, dim = query.shape
    seq_k = key.shape[1]
    kv_heads = key.shape[2]
    group_size = q_heads // kv_heads
    min_head = int(selected_heads.min().item())
    max_head = int(selected_heads.max().item())
    if min_head < 0 or max_head >= q_heads:
        raise ValueError(f"selected_heads must be within [0, {q_heads})")
    num_blocks = triton.cdiv(seq_k, block_size)
    if sink_blocks + recent_blocks > num_blocks:
        raise ValueError("sink_blocks + recent_blocks must not exceed K blocks")
    middle_blocks = num_blocks - sink_blocks - recent_blocks
    if middle_seed_blocks > middle_blocks:
        raise ValueError("middle_seed_blocks must be within the middle block count")

    selected = int(selected_heads.numel())
    output = torch.empty(batch, 1, selected, dim, device=query.device, dtype=query.dtype)
    raw_stats = (
        torch.empty(batch, selected, 5, device=query.device, dtype=torch.int32)
        if return_raw_stats
        else torch.empty(1, device=query.device, dtype=torch.int32)
    )
    score_scale = 1.0 / math.sqrt(dim)
    recent_start = num_blocks - recent_blocks
    _seed_only_selected_kernel[(batch, selected)](
        query,
        key,
        value,
        selected_heads,
        output,
        raw_stats,
        N=seq_k,
        H_Q=q_heads,
        H_KV=kv_heads,
        SELECTED=selected,
        GROUP_SIZE=group_size,
        D=dim,
        NUM_BLOCKS=num_blocks,
        TILE_N=block_size,
        SCALE=score_scale,
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        RECENT_START=int(recent_start),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        PV_USE_BF16=value.dtype is torch.bfloat16,
        HAS_STATS=return_raw_stats,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, raw_stats if return_raw_stats else None
