"""Mass-only Gate-1 Triton forward kernel.

This is the low-overhead BLASST-style path: compute QK for every K block, use
online-softmax state to skip negligible blocks, and avoid all value-bound
metadata/state. It intentionally duplicates some structure from
``gate1_fwd_triton.py`` so profiler runs can compare generic versus
mass-specialized codegen.
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


if TRITON_AVAILABLE:

    @triton.jit
    def _gate1_mass_fwd_kernel(
        Q,
        K,
        V,
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
        LOG_ERROR_BUDGET: tl.constexpr,
        POST_QK_THRESHOLD: tl.constexpr,
        FORCE_MODE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
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
        q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

        running_max = tl.full([TILE_M], -float("inf"), dtype=tl.float32)
        running_lse = tl.full([TILE_M], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([TILE_M], dtype=tl.float32)
        acc_num = tl.zeros([TILE_M, D], dtype=tl.float32)

        post_count = tl.zeros([], dtype=tl.int32)
        compute_count = tl.zeros([], dtype=tl.int32)
        cta_total_count = tl.zeros([], dtype=tl.int32)
        cta_skipped_count = tl.zeros([], dtype=tl.int32)
        cta_pv_count = tl.zeros([], dtype=tl.int32)

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
            cta_has_valid = tl.sum(valid_row.to(tl.int32), axis=0) > 0
            if HAS_STATS:
                cta_total_count += cta_has_valid.to(tl.int32)

            k_ptrs = (
                K
                + off_b * N * H * D
                + offs_n[:, None] * H * D
                + off_h * D
                + offs_d[None, :]
            )
            k_tile = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
            qk = tl.where(col_mask[None, :], qk, -float("inf"))
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
            qk = tl.where(valid_row[:, None], qk, -float("inf"))
            tile_max = tl.max(qk, axis=1)

            if FORCE_MODE == 4 or FORCE_MODE == 6:
                if HAS_STATS:
                    post_count += tl.sum(valid_row.to(tl.int32), axis=0)
                    cta_skipped_count += cta_has_valid.to(tl.int32)
            elif FORCE_MODE == 7:
                running_max = tl.maximum(running_max, tile_max)
                running_lse = running_max
                if HAS_STATS:
                    post_count += tl.sum(valid_row.to(tl.int32), axis=0)
                    cta_skipped_count += cta_has_valid.to(tl.int32)
            else:
                if FORCE_MODE == 5:
                    skip_row = tl.full([TILE_M], False, dtype=tl.int1)
                else:
                    has_state = acc_den > 0.0
                    can_skip = (
                        valid_row
                        & has_state
                        & (tile_max <= running_max - POST_QK_THRESHOLD)
                        & (ERROR_BUDGET > 0.0)
                    )
                    if FORCE_MODE == 9:
                        den_bound = block_len * tl.exp(tile_max - running_max)
                        den_bound = tl.where(can_skip, den_bound, 0.0)
                        skip_row = can_skip & (den_bound <= ERROR_BUDGET * acc_den)
                    else:
                        log_tile_mass = tile_max + tl.log(block_len)
                        skip_row = can_skip & (
                            log_tile_mass <= running_lse + LOG_ERROR_BUDGET
                        )

                    if FORCE_MODE == 1:
                        skip_row = tl.full([TILE_M], False, dtype=tl.int1)
                    elif FORCE_MODE == 2:
                        skip_row = valid_row
                    elif FORCE_MODE == 3:
                        skip_row = tl.full([TILE_M], False, dtype=tl.int1)

                compute_row = valid_row & ~skip_row
                if HAS_STATS:
                    post_count += tl.sum(skip_row.to(tl.int32), axis=0)
                    compute_count += tl.sum(compute_row.to(tl.int32), axis=0)
                cta_needs_pv = tl.sum(compute_row.to(tl.int32), axis=0) > 0
                if HAS_STATS:
                    cta_pv_count += cta_needs_pv.to(tl.int32)
                    cta_skipped_count += (cta_has_valid & ~cta_needs_pv).to(tl.int32)

                if cta_needs_pv and ((FORCE_MODE == 8) or (FORCE_MODE == 9)):
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
                    acc_den = acc_den * correction + tl.sum(p, axis=1)
                    running_max = tl.where(new_valid, new_max, running_max)
                    running_lse = tl.where(
                        new_valid,
                        running_max + tl.log(tl.maximum(acc_den, 1.0e-20)),
                        running_lse,
                    )
                elif cta_needs_pv:
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
                    v_tile = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)
                    if PV_USE_BF16:
                        p_dot = p.to(tl.bfloat16)
                        v_dot = v_tile.to(tl.bfloat16)
                    else:
                        p_dot = p.to(tl.float16)
                        v_dot = v_tile.to(tl.float16)
                    acc_num = acc_num * correction[:, None] + tl.dot(
                        p_dot,
                        v_dot,
                        out_dtype=tl.float32,
                    )
                    acc_den = acc_den * correction + tl.sum(p, axis=1)
                    running_max = tl.where(new_valid, new_max, running_max)
                    running_lse = tl.where(
                        new_valid,
                        running_max + tl.log(tl.maximum(acc_den, 1.0e-20)),
                        running_lse,
                    )

        if FORCE_MODE == 7:
            diagnostic = tl.where(running_lse > -float("inf"), running_lse, 0.0)
            out = tl.where(offs_d[None, :] == 0, diagnostic[:, None], 0.0)
        elif (FORCE_MODE == 8) or (FORCE_MODE == 9):
            diagnostic = tl.where(running_lse > -float("inf"), running_lse, 0.0)
            out = tl.where(offs_d[None, :] == 0, diagnostic[:, None], 0.0)
        else:
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
            stats_base = ((off_b * H + off_h) * ((M + TILE_M - 1) // TILE_M) + q_block) * 6
            tl.store(RawStats + stats_base + 0, post_count)
            tl.store(RawStats + stats_base + 1, compute_count)
            tl.store(RawStats + stats_base + 2, cta_total_count)
            tl.store(RawStats + stats_base + 3, cta_skipped_count)
            tl.store(RawStats + stats_base + 4, cta_pv_count)
            tl.store(RawStats + stats_base + 5, tl.full([], FORCE_MODE, dtype=tl.int32))


def gate1_mass_attention_triton_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    error_budget: float = 1e-3,
    block_size: int = 64,
    tile_size_q: int = 64,
    post_qk_threshold: float = 0.0,
    force_mode: int = 0,
    return_raw_stats: bool = False,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the mass-only Gate-1 post-QK skip forward kernel."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("gate1_mass_attention_triton_forward requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("restricted Gate-1 path requires matching batch, heads, and dim")
    if force_mode not in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
        raise ValueError("force_mode must be an integer in [0, 9]")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    batch, seq_q, heads, dim = query.shape
    seq_k = key.shape[1]

    output = torch.empty_like(query)
    q_blocks = triton.cdiv(seq_q, tile_size_q)
    raw_stats = (
        torch.empty(batch, heads, q_blocks, 6, device=query.device, dtype=torch.int32)
        if return_raw_stats
        else torch.empty(1, device=query.device, dtype=torch.int32)
    )

    grid = (q_blocks, batch, heads)
    _gate1_mass_fwd_kernel[grid](
        query,
        key,
        value,
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
        LOG_ERROR_BUDGET=math.log(max(float(error_budget), 1.0e-20)),
        POST_QK_THRESHOLD=float(post_qk_threshold),
        FORCE_MODE=int(force_mode),
        IS_CAUSAL=bool(causal),
        PV_USE_BF16=value.dtype is torch.bfloat16,
        HAS_STATS=return_raw_stats,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, raw_stats if return_raw_stats else None
