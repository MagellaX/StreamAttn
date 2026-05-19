"""Inline Gate-0 projection filter inside a Gate-1 decode loop.

This is a narrow research kernel, not a production runtime. It targets
single-token contiguous-KV decode and tests whether calibrated projection
metadata is cheaper when consumed inside the streaming attention loop instead
of through a standalone scan/mask kernel.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


INLINE_PROJECTION_STATS = {
    "projection_skipped_blocks": 0,
    "projection_computed_blocks": 1,
    "gate1_post_qk_skipped_blocks": 2,
    "pv_executed_blocks": 3,
    "total_blocks": 4,
    "seed_computed_blocks": 5,
    "middle_blocks": 6,
    "mode": 7,
}


if TRITON_AVAILABLE:

    @triton.jit
    def _gate1_inline_projection_fwd_kernel(
        Q,
        K,
        V,
        QProj,
        Projection,
        ProjMin,
        ProjMax,
        Out,
        RawStats,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        RANK: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        PROJ_SCORE_SCALE: tl.constexpr,
        ERROR_BUDGET: tl.constexpr,
        LOG_ERROR_BUDGET: tl.constexpr,
        POST_QK_THRESHOLD: tl.constexpr,
        FILTER_MARGIN: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        RECENT_START: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        COMPUTE_QPROJ: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
        HAS_STATS: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)

        offs_d = tl.arange(0, D)
        q_ptrs = Q + off_b * H * D + off_h * D + offs_d
        q = tl.load(q_ptrs)

        offs_r = tl.arange(0, RANK)
        if COMPUTE_QPROJ:
            projection_ptrs = Projection + offs_r[:, None] * D + offs_d[None, :]
            projection_tile = tl.load(projection_ptrs).to(tl.float32)
            q_proj = tl.sum(projection_tile * q[None, :].to(tl.float32), axis=1)
        else:
            q_proj_ptrs = QProj + off_b * H * RANK + off_h * RANK + offs_r
            q_proj = tl.load(q_proj_ptrs).to(tl.float32)

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        running_lse = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)

        projection_skipped = tl.zeros([], dtype=tl.int32)
        projection_computed = tl.zeros([], dtype=tl.int32)
        gate1_post_qk_skipped = tl.zeros([], dtype=tl.int32)
        pv_executed = tl.zeros([], dtype=tl.int32)
        total_blocks = tl.zeros([], dtype=tl.int32)
        seed_computed = tl.zeros([], dtype=tl.int32)
        middle_blocks = tl.zeros([], dtype=tl.int32)

        recent_count = NUM_BLOCKS - RECENT_START

        for iter_idx in range(0, NUM_BLOCKS):
            if BLOCK_ORDER == 0:  # sequential
                block_idx = iter_idx
            elif BLOCK_ORDER == 1:  # recent first, then older blocks
                block_idx = NUM_BLOCKS - 1 - iter_idx
            else:  # sink_recent_first
                if iter_idx < SINK_BLOCKS:
                    block_idx = iter_idx
                elif iter_idx < SINK_BLOCKS + recent_count:
                    block_idx = NUM_BLOCKS - 1 - (iter_idx - SINK_BLOCKS)
                else:
                    middle_pos = iter_idx - SINK_BLOCKS - recent_count
                    block_idx = RECENT_START - 1 - middle_pos

            start_n = block_idx * TILE_N
            block_len_i = min(TILE_N, N - start_n)
            is_middle = (block_idx >= SINK_BLOCKS) and (block_idx < RECENT_START)
            is_seed = not is_middle
            if BLOCK_ORDER == 0:
                middle_order_pos = block_idx - SINK_BLOCKS
            else:
                middle_order_pos = RECENT_START - 1 - block_idx
            is_middle_seed = is_middle and (middle_order_pos < MIDDLE_SEED_BLOCKS)

            offs_n = start_n + tl.arange(0, TILE_N)
            col_mask = tl.arange(0, TILE_N) < block_len_i
            block_len = tl.full([], block_len_i, dtype=tl.float32)
            total_blocks += 1
            if is_middle:
                middle_blocks += 1

            projection_skip = tl.full([], False, dtype=tl.int1)
            if is_middle and not is_middle_seed:
                has_state = acc_den > 0.0
                metadata_base = block_idx * RANK
                metadata_offset = off_b * H * NUM_BLOCKS * RANK + off_h * NUM_BLOCKS * RANK + metadata_base + offs_r
                chosen_ptrs = tl.where(
                    q_proj >= 0.0,
                    ProjMax + metadata_offset,
                    ProjMin + metadata_offset,
                )
                chosen = tl.load(chosen_ptrs).to(tl.float32)
                proj_score = tl.sum(q_proj * chosen, axis=0) * PROJ_SCORE_SCALE
                current_lse = running_max + tl.log(tl.maximum(acc_den, 1.0e-20))
                projection_skip = (
                    has_state
                    & (ERROR_BUDGET > 0.0)
                    & (proj_score + tl.log(block_len) <= current_lse + LOG_ERROR_BUDGET + FILTER_MARGIN)
                )

            if projection_skip:
                projection_skipped += 1
            else:
                if is_middle:
                    projection_computed += 1
                if is_seed or is_middle_seed:
                    seed_computed += 1

                k_ptrs = (
                    K
                    + off_b * N * H * D
                    + offs_n[:, None] * H * D
                    + off_h * D
                    + offs_d[None, :]
                )
                k_tile = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)
                qk = tl.sum(q[None, :].to(tl.float32) * k_tile.to(tl.float32), axis=1) * SCALE
                qk = tl.where(col_mask, qk, -float("inf"))
                tile_max = tl.max(qk, axis=0)

                post_qk_skip = tl.full([], False, dtype=tl.int1)
                if is_middle:
                    has_state = acc_den > 0.0
                    can_skip = (
                        has_state
                        & (tile_max <= running_max - POST_QK_THRESHOLD)
                        & (ERROR_BUDGET > 0.0)
                    )
                    current_lse = running_max + tl.log(tl.maximum(acc_den, 1.0e-20))
                    post_qk_skip = can_skip & (
                        tile_max + tl.log(block_len) <= current_lse + LOG_ERROR_BUDGET
                    )

                if post_qk_skip:
                    gate1_post_qk_skipped += 1
                else:
                    tile_valid = tile_max > -float("inf")
                    prev_valid = acc_den > 0.0
                    new_valid = prev_valid | tile_valid
                    new_max = tl.maximum(running_max, tile_max)
                    safe_new_max = tl.where(new_valid, new_max, 0.0)
                    correction = tl.where(prev_valid, tl.exp(running_max - safe_new_max), 0.0)
                    p = tl.exp(qk - safe_new_max)
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
                        weighted = p[:, None].to(tl.bfloat16) * v_tile.to(tl.bfloat16)
                    else:
                        weighted = p[:, None].to(tl.float16) * v_tile.to(tl.float16)
                    acc_num = acc_num * correction + tl.sum(weighted.to(tl.float32), axis=0)
                    acc_den = acc_den * correction + tl.sum(p, axis=0)
                    running_max = tl.where(new_valid, new_max, running_max)
                    running_lse = tl.where(
                        new_valid,
                        running_max + tl.log(tl.maximum(acc_den, 1.0e-20)),
                        running_lse,
                    )
                    pv_executed += 1

        out = acc_num / acc_den
        out = tl.where(acc_den > 0.0, out, 0.0)
        out_ptrs = Out + off_b * H * D + off_h * D + offs_d
        tl.store(out_ptrs, out)

        if HAS_STATS:
            stats_base = (off_b * H + off_h) * 8
            tl.store(RawStats + stats_base + 0, projection_skipped)
            tl.store(RawStats + stats_base + 1, projection_computed)
            tl.store(RawStats + stats_base + 2, gate1_post_qk_skipped)
            tl.store(RawStats + stats_base + 3, pv_executed)
            tl.store(RawStats + stats_base + 4, total_blocks)
            tl.store(RawStats + stats_base + 5, seed_computed)
            tl.store(RawStats + stats_base + 6, middle_blocks)
            tl.store(RawStats + stats_base + 7, tl.full([], 1, dtype=tl.int32))


def _block_order_id(block_order: str) -> int:
    if block_order == "sequential":
        return 0
    if block_order == "recent_first":
        return 1
    if block_order == "sink_recent_first":
        return 2
    raise ValueError("block_order must be 'sequential', 'recent_first', or 'sink_recent_first'")


def gate1_inline_projection_attention_triton_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_proj: Optional[torch.Tensor],
    proj_min: torch.Tensor,
    proj_max: torch.Tensor,
    *,
    projection: Optional[torch.Tensor] = None,
    compute_qproj: bool = False,
    error_budget: float = 1e-3,
    filter_margin: float = 0.0,
    block_size: int = 16,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 0,
    block_order: str = "sink_recent_first",
    post_qk_threshold: float = 0.0,
    return_raw_stats: bool = False,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the inline calibrated projection + Gate-1 prototype.

    The prototype only supports ``query_len == 1`` and matching MHA-shaped
    ``query/key/value`` tensors. By default ``q_proj`` must be precomputed from
    the query. If ``compute_qproj`` is set, ``projection`` must be a
    ``[rank, dim]`` projection matrix and q-projection is fused into the
    inline kernel. ``proj_min/proj_max`` must contain projection metadata over
    K blocks.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    cuda_tensors = [query, key, value, proj_min, proj_max]
    if compute_qproj:
        if projection is None:
            raise ValueError("projection must be provided when compute_qproj=True")
        cuda_tensors.append(projection)
    else:
        if q_proj is None:
            raise ValueError("q_proj must be provided when compute_qproj=False")
        cuda_tensors.append(q_proj)
    if not all(t.is_cuda for t in cuda_tensors):
        raise RuntimeError("gate1_inline_projection_attention_triton_forward requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if query.shape[1] != 1:
        raise ValueError("inline projection prototype only supports query_len == 1")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query/key/value must have matching batch, heads, and dim")
    if not compute_qproj and q_proj is not None and q_proj.dim() != 4:
        raise ValueError("q_proj must have shape [batch, heads, 1, rank]")
    if compute_qproj and projection is not None and projection.dim() != 2:
        raise ValueError("projection must have shape [rank, dim]")
    if proj_min.dim() != 4 or proj_max.dim() != 4 or proj_min.shape != proj_max.shape:
        raise ValueError("projection metadata must have matching [batch, heads, blocks, rank] shapes")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0:
        raise ValueError("sink_blocks and recent_blocks must be non-negative")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if q_proj is not None:
        q_proj = q_proj.contiguous()
    if projection is not None:
        projection = projection.contiguous()
    proj_min = proj_min.contiguous()
    proj_max = proj_max.contiguous()

    batch, _seq_q, heads, dim = query.shape
    seq_k = key.shape[1]
    num_blocks = triton.cdiv(seq_k, block_size)
    if compute_qproj:
        assert projection is not None
        rank = projection.shape[0]
        if projection.shape[1] != dim:
            raise ValueError("projection shape does not match query dim")
    else:
        assert q_proj is not None
        rank = q_proj.shape[3]
        if q_proj.shape[:3] != (batch, heads, 1):
            raise ValueError("q_proj shape does not match query batch/head/query dimensions")
    if proj_min.shape != (batch, heads, num_blocks, rank):
        raise ValueError("projection metadata shape does not match key block layout")
    if sink_blocks + recent_blocks > num_blocks:
        raise ValueError("sink_blocks + recent_blocks must not exceed the number of K blocks")
    middle_blocks = num_blocks - sink_blocks - recent_blocks
    if middle_seed_blocks < 0 or middle_seed_blocks > middle_blocks:
        raise ValueError("middle_seed_blocks must be within the middle block count")

    output = torch.empty_like(query)
    raw_stats = (
        torch.empty(batch, heads, 8, device=query.device, dtype=torch.int32)
        if return_raw_stats
        else torch.empty(1, device=query.device, dtype=torch.int32)
    )

    score_scale = 1.0 / math.sqrt(dim)
    projection_score_scale = (float(dim) / float(rank)) * score_scale
    recent_start = num_blocks - recent_blocks
    grid = (batch, heads)
    q_proj_arg = q_proj if q_proj is not None else projection
    projection_arg = projection if projection is not None else q_proj_arg
    assert q_proj_arg is not None
    assert projection_arg is not None
    _gate1_inline_projection_fwd_kernel[grid](
        query,
        key,
        value,
        q_proj_arg,
        projection_arg,
        proj_min,
        proj_max,
        output,
        raw_stats,
        N=seq_k,
        H=heads,
        D=dim,
        RANK=rank,
        NUM_BLOCKS=num_blocks,
        TILE_N=block_size,
        SCALE=score_scale,
        PROJ_SCORE_SCALE=projection_score_scale,
        ERROR_BUDGET=float(error_budget),
        LOG_ERROR_BUDGET=math.log(max(float(error_budget), 1.0e-20)),
        POST_QK_THRESHOLD=float(post_qk_threshold),
        FILTER_MARGIN=float(filter_margin),
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        RECENT_START=int(recent_start),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        COMPUTE_QPROJ=compute_qproj,
        PV_USE_BF16=value.dtype is torch.bfloat16,
        HAS_STATS=return_raw_stats,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, raw_stats if return_raw_stats else None
