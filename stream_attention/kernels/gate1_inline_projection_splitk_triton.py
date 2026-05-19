"""Split-K inline Gate-0 projection prototype.

This is a narrow science kernel for single-token contiguous-KV decode. It keeps
the inline projection filter but restores KV parallelism by:

1. computing a seed online-softmax state for sink/recent/seed blocks;
2. processing remaining middle blocks in parallel chunks;
3. merging seed and chunk states.

The chunk projection threshold uses the seed LSE as a static conservative
threshold. This avoids cross-chunk dependence and is intentionally simpler than
the serial inline prototype.

Two seed strategies are exposed:

``separate``
    Three kernels: seed, chunk, merge.

``recompute_seed``
    Two kernels: each chunk recomputes the small seed state for its threshold,
    chunk 0 also writes the seed state, then merge combines seed + chunks.
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


SPLITK_PROJECTION_STATS = {
    "projection_skipped_blocks": 0,
    "projection_computed_blocks": 1,
    "gate1_post_qk_skipped_blocks": 2,
    "pv_executed_blocks": 3,
    "middle_blocks": 4,
    "seed_blocks": 5,
    "chunks": 6,
    "mode": 7,
}


if TRITON_AVAILABLE:

    @triton.jit
    def _seed_kernel(
        Q,
        K,
        V,
        QProj,
        Projection,
        SeedMax,
        SeedDen,
        SeedNum,
        QProjOut,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        RANK: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        RECENT_START: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        COMPUTE_QPROJ: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)

        offs_r = tl.arange(0, RANK)
        if COMPUTE_QPROJ:
            projection_tile = tl.load(Projection + offs_r[:, None] * D + offs_d[None, :]).to(tl.float32)
            q_proj = tl.sum(projection_tile * q[None, :].to(tl.float32), axis=1)
        else:
            q_proj = tl.load(QProj + off_b * H * RANK + off_h * RANK + offs_r).to(tl.float32)
        tl.store(QProjOut + off_b * H * RANK + off_h * RANK + offs_r, q_proj)

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)
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
                + off_b * N * H * D
                + offs_n[:, None] * H * D
                + off_h * D
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
                + off_b * N * H * D
                + offs_n[:, None] * H * D
                + off_h * D
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

        state_base = off_b * H + off_h
        tl.store(SeedMax + state_base, running_max)
        tl.store(SeedDen + state_base, acc_den)
        tl.store(SeedNum + state_base * D + offs_d, acc_num)

    @triton.jit
    def _chunk_kernel(
        Q,
        K,
        V,
        QProj,
        ProjMin,
        ProjMax,
        SeedMax,
        SeedDen,
        ChunkMax,
        ChunkDen,
        ChunkNum,
        RawStats,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        RANK: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        CHUNK_BLOCKS: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        PROJ_SCORE_SCALE: tl.constexpr,
        ERROR_BUDGET: tl.constexpr,
        LOG_ERROR_BUDGET: tl.constexpr,
        FILTER_MARGIN: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_START: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
        HAS_STATS: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_c = tl.program_id(2)
        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, RANK)
        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)
        q_proj = tl.load(QProj + off_b * H * RANK + off_h * RANK + offs_r).to(tl.float32)
        seed_den = tl.load(SeedDen + off_b * H + off_h)
        seed_max = tl.load(SeedMax + off_b * H + off_h)
        seed_lse = seed_max + tl.log(tl.maximum(seed_den, 1.0e-20))

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)
        projection_skipped = tl.zeros([], dtype=tl.int32)
        projection_computed = tl.zeros([], dtype=tl.int32)
        gate1_post_qk_skipped = tl.zeros([], dtype=tl.int32)
        pv_executed = tl.zeros([], dtype=tl.int32)
        middle_seen = tl.zeros([], dtype=tl.int32)

        for local_idx in range(0, CHUNK_BLOCKS):
            logical = off_c * CHUNK_BLOCKS + local_idx
            valid_block = logical < (RECENT_START - SINK_BLOCKS - MIDDLE_SEED_BLOCKS)
            if BLOCK_ORDER == 0:  # sequential
                block_idx = SINK_BLOCKS + MIDDLE_SEED_BLOCKS + logical
            else:  # recent_first / sink_recent_first
                block_idx = RECENT_START - 1 - MIDDLE_SEED_BLOCKS - logical

            start_n = block_idx * TILE_N
            block_len_i = min(TILE_N, N - start_n)
            block_len = tl.full([], block_len_i, dtype=tl.float32)
            middle_seen += tl.where(valid_block, 1, 0)

            metadata_offset = off_b * H * NUM_BLOCKS * RANK + off_h * NUM_BLOCKS * RANK + block_idx * RANK + offs_r
            chosen = tl.load(
                tl.where(q_proj >= 0.0, ProjMax + metadata_offset, ProjMin + metadata_offset),
                mask=valid_block,
                other=0.0,
            ).to(tl.float32)
            proj_score = tl.sum(q_proj * chosen, axis=0) * PROJ_SCORE_SCALE
            projection_skip = (
                valid_block
                & (seed_den > 0.0)
                & (ERROR_BUDGET > 0.0)
                & (proj_score + tl.log(block_len) <= seed_lse + LOG_ERROR_BUDGET + FILTER_MARGIN)
            )

            if projection_skip:
                projection_skipped += 1
            else:
                if valid_block:
                    projection_computed += 1
                offs_n = start_n + tl.arange(0, TILE_N)
                col_mask = (tl.arange(0, TILE_N) < block_len_i) & valid_block
                k_tile = tl.load(
                    K
                    + off_b * N * H * D
                    + offs_n[:, None] * H * D
                    + off_h * D
                    + offs_d[None, :],
                    mask=col_mask[:, None],
                    other=0.0,
                )
                qk = tl.sum(q[None, :].to(tl.float32) * k_tile.to(tl.float32), axis=1) * SCALE
                qk = tl.where(col_mask, qk, -float("inf"))
                tile_max = tl.max(qk, axis=0)
                post_qk_skip = (
                    valid_block
                    & (seed_den > 0.0)
                    & (ERROR_BUDGET > 0.0)
                    & (tile_max + tl.log(block_len) <= seed_lse + LOG_ERROR_BUDGET)
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
                    v_tile = tl.load(
                        V
                        + off_b * N * H * D
                        + offs_n[:, None] * H * D
                        + off_h * D
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
                    pv_executed += tl.where(valid_block, 1, 0)

        chunk_state = (off_b * H * NUM_CHUNKS + off_h * NUM_CHUNKS + off_c)
        tl.store(ChunkMax + chunk_state, running_max)
        tl.store(ChunkDen + chunk_state, acc_den)
        tl.store(ChunkNum + chunk_state * D + offs_d, acc_num)

        if HAS_STATS:
            stats_base = chunk_state * 8
            tl.store(RawStats + stats_base + 0, projection_skipped)
            tl.store(RawStats + stats_base + 1, projection_computed)
            tl.store(RawStats + stats_base + 2, gate1_post_qk_skipped)
            tl.store(RawStats + stats_base + 3, pv_executed)
            tl.store(RawStats + stats_base + 4, middle_seen)
            tl.store(
                RawStats + stats_base + 5,
                tl.full([], SINK_BLOCKS + (NUM_BLOCKS - RECENT_START) + MIDDLE_SEED_BLOCKS, dtype=tl.int32),
            )
            tl.store(RawStats + stats_base + 6, tl.full([], NUM_CHUNKS, dtype=tl.int32))
            tl.store(RawStats + stats_base + 7, tl.full([], 1, dtype=tl.int32))

    @triton.jit
    def _chunk_recompute_seed_kernel(
        Q,
        K,
        V,
        QProj,
        Projection,
        ProjMin,
        ProjMax,
        ChunkMax,
        ChunkDen,
        ChunkNum,
        RawStats,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        RANK: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        CHUNK_BLOCKS: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        PROJ_SCORE_SCALE: tl.constexpr,
        ERROR_BUDGET: tl.constexpr,
        LOG_ERROR_BUDGET: tl.constexpr,
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
        off_c = tl.program_id(2)
        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, RANK)
        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)
        if COMPUTE_QPROJ:
            projection_tile = tl.load(Projection + offs_r[:, None] * D + offs_d[None, :]).to(tl.float32)
            q_proj = tl.sum(projection_tile * q[None, :].to(tl.float32), axis=1)
        else:
            q_proj = tl.load(QProj + off_b * H * RANK + off_h * RANK + offs_r).to(tl.float32)

        seed_max = tl.full([], -float("inf"), dtype=tl.float32)
        seed_den = tl.zeros([], dtype=tl.float32)
        seed_num = tl.zeros([D], dtype=tl.float32)
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
                + off_b * N * H * D
                + offs_n[:, None] * H * D
                + off_h * D
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            qk = tl.sum(q[None, :].to(tl.float32) * k_tile.to(tl.float32), axis=1) * SCALE
            qk = tl.where(col_mask, qk, -float("inf"))
            tile_max = tl.max(qk, axis=0)
            tile_valid = tile_max > -float("inf")
            prev_valid = seed_den > 0.0
            new_valid = prev_valid | tile_valid
            new_max = tl.maximum(seed_max, tile_max)
            safe_new_max = tl.where(new_valid, new_max, 0.0)
            correction = tl.where(prev_valid, tl.exp(seed_max - safe_new_max), 0.0)
            p = tl.exp(qk - safe_new_max)
            p = tl.where(qk > -float("inf"), p, 0.0)
            v_tile = tl.load(
                V
                + off_b * N * H * D
                + offs_n[:, None] * H * D
                + off_h * D
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            if PV_USE_BF16:
                weighted = p[:, None].to(tl.bfloat16) * v_tile.to(tl.bfloat16)
            else:
                weighted = p[:, None].to(tl.float16) * v_tile.to(tl.float16)
            seed_num = seed_num * correction + tl.sum(weighted.to(tl.float32), axis=0)
            seed_den = seed_den * correction + tl.sum(p, axis=0)
            seed_max = tl.where(new_valid, new_max, seed_max)

        total_states = NUM_CHUNKS + 1
        state_base = off_b * H * total_states + off_h * total_states
        tl.store(ChunkMax + state_base, seed_max, mask=off_c == 0)
        tl.store(ChunkDen + state_base, seed_den, mask=off_c == 0)
        tl.store(ChunkNum + state_base * D + offs_d, seed_num, mask=off_c == 0)

        seed_lse = seed_max + tl.log(tl.maximum(seed_den, 1.0e-20))
        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)
        projection_skipped = tl.zeros([], dtype=tl.int32)
        projection_computed = tl.zeros([], dtype=tl.int32)
        gate1_post_qk_skipped = tl.zeros([], dtype=tl.int32)
        pv_executed = tl.zeros([], dtype=tl.int32)
        middle_seen = tl.zeros([], dtype=tl.int32)

        for local_idx in range(0, CHUNK_BLOCKS):
            logical = off_c * CHUNK_BLOCKS + local_idx
            valid_block = logical < (RECENT_START - SINK_BLOCKS - MIDDLE_SEED_BLOCKS)
            if BLOCK_ORDER == 0:  # sequential
                block_idx = SINK_BLOCKS + MIDDLE_SEED_BLOCKS + logical
            else:  # recent_first / sink_recent_first
                block_idx = RECENT_START - 1 - MIDDLE_SEED_BLOCKS - logical

            start_n = block_idx * TILE_N
            block_len_i = min(TILE_N, N - start_n)
            block_len = tl.full([], block_len_i, dtype=tl.float32)
            middle_seen += tl.where(valid_block, 1, 0)

            metadata_offset = off_b * H * NUM_BLOCKS * RANK + off_h * NUM_BLOCKS * RANK + block_idx * RANK + offs_r
            chosen = tl.load(
                tl.where(q_proj >= 0.0, ProjMax + metadata_offset, ProjMin + metadata_offset),
                mask=valid_block,
                other=0.0,
            ).to(tl.float32)
            proj_score = tl.sum(q_proj * chosen, axis=0) * PROJ_SCORE_SCALE
            projection_skip = (
                valid_block
                & (seed_den > 0.0)
                & (ERROR_BUDGET > 0.0)
                & (proj_score + tl.log(block_len) <= seed_lse + LOG_ERROR_BUDGET + FILTER_MARGIN)
            )

            if projection_skip:
                projection_skipped += 1
            else:
                if valid_block:
                    projection_computed += 1
                offs_n = start_n + tl.arange(0, TILE_N)
                col_mask = (tl.arange(0, TILE_N) < block_len_i) & valid_block
                k_tile = tl.load(
                    K
                    + off_b * N * H * D
                    + offs_n[:, None] * H * D
                    + off_h * D
                    + offs_d[None, :],
                    mask=col_mask[:, None],
                    other=0.0,
                )
                qk = tl.sum(q[None, :].to(tl.float32) * k_tile.to(tl.float32), axis=1) * SCALE
                qk = tl.where(col_mask, qk, -float("inf"))
                tile_max = tl.max(qk, axis=0)
                post_qk_skip = (
                    valid_block
                    & (seed_den > 0.0)
                    & (ERROR_BUDGET > 0.0)
                    & (tile_max + tl.log(block_len) <= seed_lse + LOG_ERROR_BUDGET)
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
                    v_tile = tl.load(
                        V
                        + off_b * N * H * D
                        + offs_n[:, None] * H * D
                        + off_h * D
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
                    pv_executed += tl.where(valid_block, 1, 0)

        chunk_state = state_base + off_c + 1
        tl.store(ChunkMax + chunk_state, running_max)
        tl.store(ChunkDen + chunk_state, acc_den)
        tl.store(ChunkNum + chunk_state * D + offs_d, acc_num)

        if HAS_STATS:
            stats_base = (off_b * H * NUM_CHUNKS + off_h * NUM_CHUNKS + off_c) * 8
            tl.store(RawStats + stats_base + 0, projection_skipped)
            tl.store(RawStats + stats_base + 1, projection_computed)
            tl.store(RawStats + stats_base + 2, gate1_post_qk_skipped)
            tl.store(RawStats + stats_base + 3, pv_executed)
            tl.store(RawStats + stats_base + 4, middle_seen)
            tl.store(
                RawStats + stats_base + 5,
                tl.full([], SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS, dtype=tl.int32),
            )
            tl.store(RawStats + stats_base + 6, tl.full([], NUM_CHUNKS, dtype=tl.int32))
            tl.store(RawStats + stats_base + 7, tl.full([], 2, dtype=tl.int32))

    @triton.jit
    def _merge_kernel(
        SeedMax,
        SeedDen,
        SeedNum,
        ChunkMax,
        ChunkDen,
        ChunkNum,
        Out,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        offs_d = tl.arange(0, D)
        state_base = off_b * H + off_h
        running_max = tl.load(SeedMax + state_base)
        acc_den = tl.load(SeedDen + state_base)
        acc_num = tl.load(SeedNum + state_base * D + offs_d)

        for chunk in range(0, NUM_CHUNKS):
            chunk_state = off_b * H * NUM_CHUNKS + off_h * NUM_CHUNKS + chunk
            chunk_den = tl.load(ChunkDen + chunk_state)
            chunk_max = tl.load(ChunkMax + chunk_state)
            chunk_num = tl.load(ChunkNum + chunk_state * D + offs_d)
            chunk_valid = chunk_den > 0.0
            prev_valid = acc_den > 0.0
            new_valid = prev_valid | chunk_valid
            new_max = tl.maximum(running_max, chunk_max)
            safe_new_max = tl.where(new_valid, new_max, 0.0)
            lhs = tl.where(prev_valid, tl.exp(running_max - safe_new_max), 0.0)
            rhs = tl.where(chunk_valid, tl.exp(chunk_max - safe_new_max), 0.0)
            acc_num = acc_num * lhs + chunk_num * rhs
            acc_den = acc_den * lhs + chunk_den * rhs
            running_max = tl.where(new_valid, new_max, running_max)

        out = acc_num / acc_den
        out = tl.where(acc_den > 0.0, out, 0.0)
        tl.store(Out + off_b * H * D + off_h * D + offs_d, out)

    @triton.jit
    def _merge_states_kernel(
        StateMax,
        StateDen,
        StateNum,
        Out,
        H: tl.constexpr,
        D: tl.constexpr,
        TOTAL_STATES: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        offs_d = tl.arange(0, D)
        state_base = off_b * H * TOTAL_STATES + off_h * TOTAL_STATES
        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)

        for state in range(0, TOTAL_STATES):
            state_idx = state_base + state
            state_den = tl.load(StateDen + state_idx)
            state_max = tl.load(StateMax + state_idx)
            state_num = tl.load(StateNum + state_idx * D + offs_d)
            state_valid = state_den > 0.0
            prev_valid = acc_den > 0.0
            new_valid = prev_valid | state_valid
            new_max = tl.maximum(running_max, state_max)
            safe_new_max = tl.where(new_valid, new_max, 0.0)
            lhs = tl.where(prev_valid, tl.exp(running_max - safe_new_max), 0.0)
            rhs = tl.where(state_valid, tl.exp(state_max - safe_new_max), 0.0)
            acc_num = acc_num * lhs + state_num * rhs
            acc_den = acc_den * lhs + state_den * rhs
            running_max = tl.where(new_valid, new_max, running_max)

        out = acc_num / acc_den
        out = tl.where(acc_den > 0.0, out, 0.0)
        tl.store(Out + off_b * H * D + off_h * D + offs_d, out)


def _block_order_id(block_order: str) -> int:
    if block_order == "sequential":
        return 0
    if block_order == "recent_first":
        return 1
    if block_order == "sink_recent_first":
        return 2
    raise ValueError("block_order must be 'sequential', 'recent_first', or 'sink_recent_first'")


def _seed_strategy_id(seed_strategy: str) -> int:
    if seed_strategy == "separate":
        return 0
    if seed_strategy == "recompute_seed":
        return 1
    raise ValueError("seed_strategy must be 'separate' or 'recompute_seed'")


def gate1_inline_projection_splitk_attention_triton_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_proj: Optional[torch.Tensor],
    proj_min: torch.Tensor,
    proj_max: torch.Tensor,
    *,
    projection: Optional[torch.Tensor] = None,
    compute_qproj: bool = False,
    num_chunks: int = 4,
    error_budget: float = 1e-3,
    filter_margin: float = 0.0,
    block_size: int = 16,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 0,
    block_order: str = "recent_first",
    seed_strategy: str = "separate",
    return_raw_stats: bool = False,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the split-K inline projection prototype."""

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
        raise RuntimeError("gate1_inline_projection_splitk_attention_triton_forward requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if query.shape[1] != 1:
        raise ValueError("split-K inline projection prototype only supports query_len == 1")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query/key/value must have matching batch, heads, and dim")
    if proj_min.dim() != 4 or proj_max.dim() != 4 or proj_min.shape != proj_max.shape:
        raise ValueError("projection metadata must have matching [batch, heads, blocks, rank] shapes")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0:
        raise ValueError("sink_blocks and recent_blocks must be non-negative")
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    seed_strategy_id = _seed_strategy_id(seed_strategy)

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

    nonseed_middle = middle_blocks - middle_seed_blocks
    chunk_blocks = max(1, triton.cdiv(nonseed_middle, num_chunks))
    output = torch.empty_like(query)
    total_states = num_chunks + 1 if seed_strategy_id == 1 else num_chunks
    seed_max = torch.empty(batch, heads, device=query.device, dtype=torch.float32)
    seed_den = torch.empty(batch, heads, device=query.device, dtype=torch.float32)
    seed_num = torch.empty(batch, heads, dim, device=query.device, dtype=torch.float32)
    q_proj_out = torch.empty(batch, heads, rank, device=query.device, dtype=torch.float32)
    chunk_max = torch.empty(batch, heads, total_states, device=query.device, dtype=torch.float32)
    chunk_den = torch.empty(batch, heads, total_states, device=query.device, dtype=torch.float32)
    chunk_num = torch.empty(batch, heads, total_states, dim, device=query.device, dtype=torch.float32)
    raw_stats = (
        torch.empty(batch, heads, num_chunks, 8, device=query.device, dtype=torch.int32)
        if return_raw_stats
        else torch.empty(1, device=query.device, dtype=torch.int32)
    )

    score_scale = 1.0 / math.sqrt(dim)
    projection_score_scale = (float(dim) / float(rank)) * score_scale
    recent_start = num_blocks - recent_blocks
    q_proj_arg = q_proj if q_proj is not None else projection
    projection_arg = projection if projection is not None else q_proj_arg
    assert q_proj_arg is not None
    assert projection_arg is not None
    block_order_id = _block_order_id(block_order)

    if seed_strategy_id == 0:
        _seed_kernel[(batch, heads)](
            query,
            key,
            value,
            q_proj_arg,
            projection_arg,
            seed_max,
            seed_den,
            seed_num,
            q_proj_out,
            N=seq_k,
            H=heads,
            D=dim,
            RANK=rank,
            NUM_BLOCKS=num_blocks,
            TILE_N=block_size,
            SCALE=score_scale,
            SINK_BLOCKS=int(sink_blocks),
            RECENT_BLOCKS=int(recent_blocks),
            RECENT_START=int(recent_start),
            MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
            BLOCK_ORDER=block_order_id,
            COMPUTE_QPROJ=compute_qproj,
            PV_USE_BF16=value.dtype is torch.bfloat16,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _chunk_kernel[(batch, heads, num_chunks)](
            query,
            key,
            value,
            q_proj_out,
            proj_min,
            proj_max,
            seed_max,
            seed_den,
            chunk_max,
            chunk_den,
            chunk_num,
            raw_stats,
            N=seq_k,
            H=heads,
            D=dim,
            RANK=rank,
            NUM_BLOCKS=num_blocks,
            NUM_CHUNKS=num_chunks,
            CHUNK_BLOCKS=chunk_blocks,
            TILE_N=block_size,
            SCALE=score_scale,
            PROJ_SCORE_SCALE=projection_score_scale,
            ERROR_BUDGET=float(error_budget),
            LOG_ERROR_BUDGET=math.log(max(float(error_budget), 1.0e-20)),
            FILTER_MARGIN=float(filter_margin),
            SINK_BLOCKS=int(sink_blocks),
            RECENT_START=int(recent_start),
            MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
            BLOCK_ORDER=block_order_id,
            PV_USE_BF16=value.dtype is torch.bfloat16,
            HAS_STATS=return_raw_stats,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _merge_kernel[(batch, heads)](
            seed_max,
            seed_den,
            seed_num,
            chunk_max,
            chunk_den,
            chunk_num,
            output,
            H=heads,
            D=dim,
            NUM_CHUNKS=num_chunks,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _chunk_recompute_seed_kernel[(batch, heads, num_chunks)](
            query,
            key,
            value,
            q_proj_arg,
            projection_arg,
            proj_min,
            proj_max,
            chunk_max,
            chunk_den,
            chunk_num,
            raw_stats,
            N=seq_k,
            H=heads,
            D=dim,
            RANK=rank,
            NUM_BLOCKS=num_blocks,
            NUM_CHUNKS=num_chunks,
            CHUNK_BLOCKS=chunk_blocks,
            TILE_N=block_size,
            SCALE=score_scale,
            PROJ_SCORE_SCALE=projection_score_scale,
            ERROR_BUDGET=float(error_budget),
            LOG_ERROR_BUDGET=math.log(max(float(error_budget), 1.0e-20)),
            FILTER_MARGIN=float(filter_margin),
            SINK_BLOCKS=int(sink_blocks),
            RECENT_BLOCKS=int(recent_blocks),
            RECENT_START=int(recent_start),
            MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
            BLOCK_ORDER=block_order_id,
            COMPUTE_QPROJ=compute_qproj,
            PV_USE_BF16=value.dtype is torch.bfloat16,
            HAS_STATS=return_raw_stats,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _merge_states_kernel[(batch, heads)](
            chunk_max,
            chunk_den,
            chunk_num,
            output,
            H=heads,
            D=dim,
            TOTAL_STATES=total_states,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return output, raw_stats if return_raw_stats else None
