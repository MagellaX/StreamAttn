"""Experimental Triton forward kernel for two-gate certified attention.

This module is intentionally narrow:
- contiguous Q/K/V tensors in [batch, seq, heads, dim]
- same Q/K/V head count
- no dropout, masks, ALiBi, or varlen
- optional causal masking
- centroid/radius summaries, no outlier summaries

It exists to make the PyTorch reference path kernel-shaped. Keep the reference
implementation in ``stream_attention.certified`` as the oracle.
"""

import math
from typing import Optional, Tuple

import torch

from stream_attention.certified.summaries import BlockSummaries, build_block_summaries

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _certified_fwd_kernel(
        Q,
        K,
        V,
        Centroid,
        Radius,
        MaxVNorm,
        Out,
        RawStats,
        B: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        ERROR_BUDGET: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        VALUE_BOUND: tl.constexpr,
        HAS_STATS: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        off_b = tl.program_id(1)
        off_h = tl.program_id(2)

        offs_m = q_block * TILE_M + tl.arange(0, TILE_M)
        offs_d = tl.arange(0, D)
        row_mask = offs_m < M

        q_ptrs = (
            Q
            + off_b * M * H * D
            + offs_m[:, None] * H * D
            + off_h * D
            + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        q_norm = tl.sqrt(tl.sum(q * q, axis=1))

        running_max = tl.full([TILE_M], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([TILE_M], dtype=tl.float32)
        acc_num = tl.zeros([TILE_M, D], dtype=tl.float32)

        pre_count = tl.zeros([], dtype=tl.int32)
        post_count = tl.zeros([], dtype=tl.int32)
        compute_count = tl.zeros([], dtype=tl.int32)

        for block_idx in range(0, NUM_BLOCKS):
            start_n = block_idx * TILE_N
            offs_n = start_n + tl.arange(0, TILE_N)
            block_len_i: tl.constexpr = min(TILE_N, N - start_n)
            col_mask = tl.arange(0, TILE_N) < block_len_i
            block_len = tl.full([], block_len_i, dtype=tl.float32)

            if IS_CAUSAL:
                valid_row = row_mask & (offs_m >= start_n)
                full_allowed = row_mask & (offs_m >= (start_n + block_len_i - 1))
            else:
                valid_row = row_mask
                full_allowed = row_mask

            c_ptrs = (
                Centroid
                + off_b * H * NUM_BLOCKS * D
                + off_h * NUM_BLOCKS * D
                + block_idx * D
                + offs_d
            )
            centroid = tl.load(c_ptrs).to(tl.float32)
            radius = tl.load(Radius + off_b * H * NUM_BLOCKS + off_h * NUM_BLOCKS + block_idx).to(tl.float32)
            max_v_norm = tl.load(MaxVNorm + off_b * H * NUM_BLOCKS + off_h * NUM_BLOCKS + block_idx).to(tl.float32)

            upper = (tl.sum(q * centroid[None, :], axis=1) + q_norm * radius) * SCALE
            has_state = acc_den > 0.0
            can_skip_pre = (
                valid_row
                & full_allowed
                & has_state
                & (upper <= running_max)
                & (ERROR_BUDGET > 0.0)
            )
            den_bound = block_len * tl.exp(upper - running_max)
            den_bound = tl.where(can_skip_pre, den_bound, 0.0)

            current_out = acc_num / tl.maximum(acc_den, 1.0)[:, None]
            out_norm = tl.sqrt(tl.sum(current_out * current_out, axis=1))
            if VALUE_BOUND:
                rho = den_bound / (acc_den + den_bound)
                skip_pre = can_skip_pre & (rho * (max_v_norm + out_norm) <= ERROR_BUDGET)
            else:
                skip_pre = can_skip_pre & (den_bound <= ERROR_BUDGET * acc_den)

            pre_count += tl.sum(skip_pre.to(tl.int32), axis=0)
            k_ptrs = (
                K
                + off_b * N * H * D
                + offs_n[:, None] * H * D
                + off_h * D
                + offs_d[None, :]
            )
            k_tile = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0).to(tl.float32)
            qk = tl.dot(q, tl.trans(k_tile)) * SCALE
            qk = tl.where(col_mask[None, :], qk, -float("inf"))

            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
            qk = tl.where((valid_row & ~skip_pre)[:, None], qk, -float("inf"))

            tile_max = tl.max(qk, axis=1)
            can_skip_post = (
                (valid_row & ~skip_pre)
                & has_state
                & (tile_max <= running_max)
                & (ERROR_BUDGET > 0.0)
            )
            post_den_bound = block_len * tl.exp(tile_max - running_max)
            post_den_bound = tl.where(can_skip_post, post_den_bound, 0.0)
            if VALUE_BOUND:
                rho_post = post_den_bound / (acc_den + post_den_bound)
                skip_post = can_skip_post & (
                    rho_post * (max_v_norm + out_norm) <= ERROR_BUDGET
                )
            else:
                skip_post = can_skip_post & (post_den_bound <= ERROR_BUDGET * acc_den)

            post_count += tl.sum(skip_post.to(tl.int32), axis=0)
            compute_row = valid_row & ~skip_pre & ~skip_post
            compute_count += tl.sum(compute_row.to(tl.int32), axis=0)

            qk = tl.where(compute_row[:, None], qk, -float("inf"))
            tile_max = tl.max(qk, axis=1)
            tile_valid = tile_max > -float("inf")
            prev_valid = acc_den > 0.0
            new_valid = prev_valid | tile_valid
            new_max = tl.maximum(running_max, tile_max)
            safe_new_max = tl.where(new_valid, new_max, 0.0)
            correction = tl.where(
                prev_valid,
                tl.exp(running_max - safe_new_max),
                0.0,
            )

            p = tl.exp(qk - safe_new_max[:, None])
            p = tl.where(qk > -float("inf"), p, 0.0)

            v_ptrs = (
                V
                + off_b * N * H * D
                + offs_n[:, None] * H * D
                + off_h * D
                + offs_d[None, :]
            )
            v_tile = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0).to(tl.float32)

            acc_num = acc_num * correction[:, None] + tl.dot(p, v_tile)
            acc_den = acc_den * correction + tl.sum(p, axis=1)
            running_max = tl.where(new_valid, new_max, running_max)

        out = acc_num / acc_den[:, None]
        out = tl.where(acc_den[:, None] > 0.0, out, 0.0)
        out_ptrs = (
            Out
            + off_b * M * H * D
            + offs_m[:, None] * H * D
            + off_h * D
            + offs_d[None, :]
        )
        tl.store(out_ptrs, out, mask=row_mask[:, None])

        if HAS_STATS:
            stats_base = ((off_b * H + off_h) * ((M + TILE_M - 1) // TILE_M) + q_block) * 3
            tl.store(RawStats + stats_base + 0, pre_count)
            tl.store(RawStats + stats_base + 1, post_count)
            tl.store(RawStats + stats_base + 2, compute_count)


def certified_attention_triton_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    error_budget: float = 1e-3,
    block_size: int = 64,
    tile_size_q: int = 64,
    summaries: Optional[BlockSummaries] = None,
    skip_predicate: str = "value_bound",
    return_raw_stats: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the experimental Triton certified forward kernel.

    Returns ``(output, raw_stats)``. ``raw_stats`` has shape
    ``[batch, heads, q_blocks, 3]`` with pre-skip, post-skip, and compute
    row-block counts when requested.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("certified_attention_triton_forward requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("restricted Triton path requires matching batch, heads, and dim")
    if skip_predicate not in {"mass", "value_bound"}:
        raise ValueError("skip_predicate must be 'mass' or 'value_bound'")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    batch, seq_q, heads, dim = query.shape
    seq_k = key.shape[1]

    if summaries is None:
        summaries = build_block_summaries(key, value, block_size=block_size)
    if summaries.outlier_keys is not None:
        raise ValueError("experimental Triton path does not support outlier summaries yet")
    if summaries.block_size != block_size or summaries.seq_len != seq_k:
        raise ValueError("summary shape does not match key/block_size")

    output = torch.empty_like(query)
    q_blocks = triton.cdiv(seq_q, tile_size_q)
    raw_stats = (
        torch.empty(batch, heads, q_blocks, 3, device=query.device, dtype=torch.int32)
        if return_raw_stats
        else torch.empty(1, device=query.device, dtype=torch.int32)
    )

    grid = (q_blocks, batch, heads)
    _certified_fwd_kernel[grid](
        query,
        key,
        value,
        summaries.centroid.contiguous(),
        summaries.radius.contiguous(),
        summaries.max_value_norm.contiguous(),
        output,
        raw_stats,
        B=batch,
        M=seq_q,
        N=seq_k,
        H=heads,
        D=dim,
        NUM_BLOCKS=triton.cdiv(seq_k, block_size),
        TILE_M=tile_size_q,
        TILE_N=block_size,
        SCALE=1.0 / math.sqrt(dim),
        ERROR_BUDGET=float(error_budget),
        IS_CAUSAL=bool(causal),
        VALUE_BOUND=skip_predicate == "value_bound",
        HAS_STATS=return_raw_stats,
        num_warps=4,
        num_stages=3,
    )
    return output, raw_stats if return_raw_stats else None
