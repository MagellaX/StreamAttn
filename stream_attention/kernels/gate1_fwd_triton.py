"""Gate-1 post-QK sparse attention forward kernel.

Gate 1 is the first production-oriented sparse step:
compute QK for each K block, use the block-local max with online-softmax state,
and skip V load/PV when the block is negligible for the whole query tile.

This module deliberately does not use K summaries. Gate 0 summary routing should
be implemented as a scheduler/worklist path, not as the first hot-kernel bet.

Modal A10G smoke currently shows correct row-skip telemetry but little latency
change, so this kernel exposes debug force modes and CTA-level counters to
verify whether V load/PV work is actually elided. The early force modes bypass
the Gate-1 predicate machinery so timing can isolate QK-only versus dense
compute. Keep this as a correctness/diagnostic kernel while developing
scheduler/worklist or CUDA/CuTe implementations for the hot path.
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
        LOG_ERROR_BUDGET: tl.constexpr,
        POST_QK_THRESHOLD: tl.constexpr,
        FORCE_MODE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        VALUE_BOUND: tl.constexpr,
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
        seen_v_norm_bound = tl.zeros([TILE_M], dtype=tl.float32)

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

                    log_tile_mass = tile_max + tl.log(block_len)
                    if VALUE_BOUND:
                        max_v_norm = tl.load(
                            MaxVNorm
                            + off_b * H * NUM_BLOCKS
                            + off_h * NUM_BLOCKS
                            + block_idx
                        ).to(tl.float32)
                        value_scale = tl.maximum(
                            max_v_norm + seen_v_norm_bound,
                            1.0e-20,
                        )
                        skip_row = can_skip & (
                            log_tile_mass + tl.log(value_scale)
                            <= running_lse + LOG_ERROR_BUDGET
                        )
                    elif FORCE_MODE == 9:
                        den_bound = block_len * tl.exp(tile_max - running_max)
                        den_bound = tl.where(can_skip, den_bound, 0.0)
                        skip_row = can_skip & (den_bound <= ERROR_BUDGET * acc_den)
                    else:
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

                if cta_needs_pv and (FORCE_MODE == 8 or FORCE_MODE == 9):
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
                    if VALUE_BOUND:
                        seen_v_norm_bound = tl.where(
                            compute_row,
                            tl.maximum(seen_v_norm_bound, max_v_norm),
                            seen_v_norm_bound,
                        )

        if FORCE_MODE == 7:
            diagnostic = tl.where(
                running_lse > -float("inf"),
                running_lse,
                0.0,
            )
            out = tl.where(
                offs_d[None, :] == 0,
                diagnostic[:, None],
                0.0,
            )
        elif (FORCE_MODE == 8) or (FORCE_MODE == 9):
            diagnostic = tl.where(
                running_lse > -float("inf"),
                running_lse,
                0.0,
            )
            out = tl.where(
                offs_d[None, :] == 0,
                diagnostic[:, None],
                0.0,
            )
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
    force_mode: int = 0,
    return_raw_stats: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run Gate-1 post-QK skip forward attention.

    ``force_mode`` is for diagnostics:
    ``0`` normal, ``1`` never skip after predicate, ``2`` force all valid rows
    to skip after predicate, ``3`` force all valid rows to compute after
    predicate, ``4`` early all-skip after QK, ``5`` dense compute without
    Gate-1 predicate overhead, ``6`` legacy early-skip alias for ``4``, ``7``
    true QK scan, ``8`` QK + log-domain predicate without PV, and ``9`` QK +
    old exp predicate without PV.

    Returns ``(output, raw_stats)``. ``raw_stats`` has shape
    ``[batch, heads, q_blocks, 6]``:
    row skips, row computes, CTA tiles total, CTA PV skipped, CTA PV executed,
    and force mode.
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
    if force_mode not in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
        raise ValueError("force_mode must be an integer in [0, 9]")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    batch, seq_q, heads, dim = query.shape
    seq_k = key.shape[1]

    uses_value_bound_in_kernel = (
        skip_predicate == "value_bound"
        and error_budget > 0.0
        and force_mode not in {4, 5, 6, 7, 8, 9}
    )
    needs_value_norm_bounds = uses_value_bound_in_kernel
    if value_norm_bounds is None and needs_value_norm_bounds:
        value_norm_bounds = build_value_norm_bounds(value, block_size=block_size)
    elif value_norm_bounds is None:
        value_norm_bounds = torch.empty(1, device=value.device, dtype=torch.float32)
    value_norm_bounds = value_norm_bounds.contiguous()

    output = torch.empty_like(query)
    q_blocks = triton.cdiv(seq_q, tile_size_q)
    raw_stats = (
        torch.empty(batch, heads, q_blocks, 6, device=query.device, dtype=torch.int32)
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
        LOG_ERROR_BUDGET=math.log(max(float(error_budget), 1.0e-20)),
        POST_QK_THRESHOLD=float(post_qk_threshold),
        FORCE_MODE=int(force_mode),
        IS_CAUSAL=bool(causal),
        VALUE_BOUND=uses_value_bound_in_kernel,
        PV_USE_BF16=value.dtype is torch.bfloat16,
        HAS_STATS=return_raw_stats,
        num_warps=4,
        num_stages=3,
    )
    return output, raw_stats if return_raw_stats else None
