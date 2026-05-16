"""Gate-1 post-QK sparse attention forward kernel.

Gate 1 is the first production-oriented sparse step:
compute QK for each K block, use the block-local max with online-softmax state,
and skip V load/PV when the block is negligible for the whole query tile.

This module deliberately does not use K summaries. Gate 0 summary routing should
be implemented as a scheduler/worklist path, not as the first hot-kernel bet.

Modal A10G smoke currently shows correct skip telemetry but little latency
change, which likely means Triton's dynamic branch is not enough by itself for
production speed. Keep this as a correctness/smoke kernel while developing a
scheduler/worklist or CUDA/CuTe implementation for the hot path.
"""

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


def build_value_norm_bounds(value: torch.Tensor, *, block_size: int) -> torch.Tensor:
    """Return max ||V|| per [batch, head, K-block]."""

    if value.dim() != 4:
        raise ValueError("value must have shape [batch, seq, heads, dim]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    batch, seq_len, heads, _ = value.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    value_bh = value.permute(0, 2, 1, 3).contiguous().float()
    bounds = torch.empty(batch, heads, num_blocks, device=value.device, dtype=torch.float32)

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        bounds[:, :, block_idx] = torch.linalg.vector_norm(
            value_bh[:, :, start:end, :],
            dim=-1,
        ).amax(dim=-1)
    return bounds


if TRITON_AVAILABLE:

    @triton.jit
    def _gate1_fwd_kernel(
        Q,
        K,
        V,
        MaxVNorm,
        Out,
        RawStats,
        M: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        ERROR_BUDGET: tl.constexpr,
        POST_QK_THRESHOLD: tl.constexpr,
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

        running_max = tl.full([TILE_M], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([TILE_M], dtype=tl.float32)
        acc_num = tl.zeros([TILE_M, D], dtype=tl.float32)

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
            else:
                valid_row = row_mask

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
            qk = tl.where(valid_row[:, None], qk, -float("inf"))

            tile_max = tl.max(qk, axis=1)
            has_state = acc_den > 0.0
            can_skip = (
                valid_row
                & has_state
                & (tile_max <= running_max - POST_QK_THRESHOLD)
                & (ERROR_BUDGET > 0.0)
            )

            den_bound = block_len * tl.exp(tile_max - running_max)
            den_bound = tl.where(can_skip, den_bound, 0.0)
            if VALUE_BOUND:
                max_v_norm = tl.load(
                    MaxVNorm + off_b * H * NUM_BLOCKS + off_h * NUM_BLOCKS + block_idx
                ).to(tl.float32)
                current_out = acc_num / tl.maximum(acc_den, 1.0)[:, None]
                out_norm = tl.sqrt(tl.sum(current_out * current_out, axis=1))
                rho = den_bound / (acc_den + den_bound)
                skip_row = can_skip & (rho * (max_v_norm + out_norm) <= ERROR_BUDGET)
            else:
                skip_row = can_skip & (den_bound <= ERROR_BUDGET * acc_den)

            compute_row = valid_row & ~skip_row
            post_count += tl.sum(skip_row.to(tl.int32), axis=0)
            compute_count += tl.sum(compute_row.to(tl.int32), axis=0)
            cta_needs_pv = tl.sum(compute_row.to(tl.int32), axis=0) > 0

            if cta_needs_pv:
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
            stats_base = ((off_b * H + off_h) * ((M + TILE_M - 1) // TILE_M) + q_block) * 2
            tl.store(RawStats + stats_base + 0, post_count)
            tl.store(RawStats + stats_base + 1, compute_count)


def gate1_attention_triton_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    error_budget: float = 1e-3,
    block_size: int = 64,
    tile_size_q: int = 64,
    value_norm_bounds: Optional[torch.Tensor] = None,
    skip_predicate: str = "value_bound",
    post_qk_threshold: float = 0.0,
    return_raw_stats: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run Gate-1 post-QK skip forward attention.

    Returns ``(output, raw_stats)``. ``raw_stats`` has shape
    ``[batch, heads, q_blocks, 2]`` with post-skip and compute row-block counts.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("gate1_attention_triton_forward requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("restricted Gate-1 path requires matching batch, heads, and dim")
    if skip_predicate not in {"mass", "value_bound"}:
        raise ValueError("skip_predicate must be 'mass' or 'value_bound'")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    batch, seq_q, heads, dim = query.shape
    seq_k = key.shape[1]

    if value_norm_bounds is None:
        value_norm_bounds = build_value_norm_bounds(value, block_size=block_size)
    value_norm_bounds = value_norm_bounds.contiguous()

    output = torch.empty_like(query)
    q_blocks = triton.cdiv(seq_q, tile_size_q)
    raw_stats = (
        torch.empty(batch, heads, q_blocks, 2, device=query.device, dtype=torch.int32)
        if return_raw_stats
        else torch.empty(1, device=query.device, dtype=torch.int32)
    )

    grid = (q_blocks, batch, heads)
    _gate1_fwd_kernel[grid](
        query,
        key,
        value,
        value_norm_bounds,
        output,
        raw_stats,
        M=seq_q,
        N=seq_k,
        H=heads,
        D=dim,
        NUM_BLOCKS=triton.cdiv(seq_k, block_size),
        TILE_M=tile_size_q,
        TILE_N=block_size,
        SCALE=1.0 / math.sqrt(dim),
        ERROR_BUDGET=float(error_budget),
        POST_QK_THRESHOLD=float(post_qk_threshold),
        IS_CAUSAL=bool(causal),
        VALUE_BOUND=skip_predicate == "value_bound",
        HAS_STATS=return_raw_stats,
        num_warps=4,
        num_stages=3,
    )
    return output, raw_stats if return_raw_stats else None
