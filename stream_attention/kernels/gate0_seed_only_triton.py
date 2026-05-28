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
    def _seed_only_dynamic_n_kernel(
        Q,
        K,
        V,
        Out,
        N,
        RECENT_START,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_kv_h = off_h // GROUP_SIZE
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)

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
            block_len_i = tl.minimum(TILE_N, N - start_n)
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

    @triton.jit
    def _seed_only_dynamic_n_bhnd_kernel(
        Q,
        K,
        V,
        Out,
        N,
        RECENT_START,
        K_STRIDE_B,
        K_STRIDE_H,
        K_STRIDE_N,
        K_STRIDE_D,
        V_STRIDE_B,
        V_STRIDE_H,
        V_STRIDE_N,
        V_STRIDE_D,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_kv_h = off_h // GROUP_SIZE
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)

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
            block_len_i = tl.minimum(TILE_N, N - start_n)
            offs_n = start_n + tl.arange(0, TILE_N)
            col_mask = tl.arange(0, TILE_N) < block_len_i
            k_tile = tl.load(
                K
                + off_b * K_STRIDE_B
                + off_kv_h * K_STRIDE_H
                + offs_n[:, None] * K_STRIDE_N
                + offs_d[None, :] * K_STRIDE_D,
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
                + off_b * V_STRIDE_B
                + off_kv_h * V_STRIDE_H
                + offs_n[:, None] * V_STRIDE_N
                + offs_d[None, :] * V_STRIDE_D,
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

    @triton.jit
    def _seed_only_cachepos_bhnd_kernel(
        Q,
        K,
        V,
        Out,
        CachePosition,
        K_STRIDE_B: tl.constexpr,
        K_STRIDE_H: tl.constexpr,
        K_STRIDE_N: tl.constexpr,
        V_STRIDE_B: tl.constexpr,
        V_STRIDE_H: tl.constexpr,
        V_STRIDE_N: tl.constexpr,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_kv_h = off_h // GROUP_SIZE
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)

        n = tl.load(CachePosition).to(tl.int32) + 1
        num_blocks = (n + TILE_N - 1) // TILE_N
        recent_start = num_blocks - RECENT_BLOCKS

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)

        for iter_idx in range(0, SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS):
            if iter_idx < SINK_BLOCKS:
                block_idx = tl.full([], iter_idx, dtype=tl.int32)
            elif iter_idx < SINK_BLOCKS + RECENT_BLOCKS:
                recent_pos = tl.full([], iter_idx - SINK_BLOCKS, dtype=tl.int32)
                block_idx = recent_start + recent_pos
            else:
                middle_pos = tl.full([], iter_idx - SINK_BLOCKS - RECENT_BLOCKS, dtype=tl.int32)
                if BLOCK_ORDER == 0:  # sequential
                    block_idx = SINK_BLOCKS + middle_pos
                else:  # recent_first / sink_recent_first
                    block_idx = recent_start - 1 - middle_pos

            start_n = block_idx * TILE_N
            block_len_i = tl.minimum(TILE_N, n - start_n)
            offs_n = start_n + tl.arange(0, TILE_N)
            col_mask = tl.arange(0, TILE_N) < block_len_i
            k_tile = tl.load(
                K
                + off_b * K_STRIDE_B
                + off_kv_h * K_STRIDE_H
                + offs_n[:, None] * K_STRIDE_N
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
                + off_b * V_STRIDE_B
                + off_kv_h * V_STRIDE_H
                + offs_n[:, None] * V_STRIDE_N
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

    @triton.jit
    def _pack_seed_cache_bhsd_kernel(
        K,
        V,
        KSeed,
        VSeed,
        N: tl.constexpr,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        TILE_N: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        RECENT_START: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        seed_block = tl.program_id(2)
        off_kv_h = off_h // GROUP_SIZE
        offs_n = tl.arange(0, TILE_N)
        offs_d = tl.arange(0, D)

        if seed_block < SINK_BLOCKS:
            block_idx = seed_block
        elif seed_block < SINK_BLOCKS + RECENT_BLOCKS:
            recent_pos = seed_block - SINK_BLOCKS
            block_idx = RECENT_START + recent_pos
        else:
            middle_pos = seed_block - SINK_BLOCKS - RECENT_BLOCKS
            if BLOCK_ORDER == 0:  # sequential
                block_idx = SINK_BLOCKS + middle_pos
            else:  # recent_first / sink_recent_first
                block_idx = RECENT_START - 1 - middle_pos

        src_n = block_idx * TILE_N + offs_n
        dst_s = seed_block * TILE_N + offs_n
        col_mask = src_n < N
        k_tile = tl.load(
            K
            + off_b * N * H_KV * D
            + src_n[:, None] * H_KV * D
            + off_kv_h * D
            + offs_d[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        v_tile = tl.load(
            V
            + off_b * N * H_KV * D
            + src_n[:, None] * H_KV * D
            + off_kv_h * D
            + offs_d[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        seed_tokens = (SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS) * TILE_N
        dst_base = off_b * H * seed_tokens * D + off_h * seed_tokens * D
        tl.store(
            KSeed + dst_base + dst_s[:, None] * D + offs_d[None, :],
            k_tile,
            mask=col_mask[:, None],
        )
        tl.store(
            VSeed + dst_base + dst_s[:, None] * D + offs_d[None, :],
            v_tile,
            mask=col_mask[:, None],
        )

    @triton.jit
    def _refresh_packed_seed_cache_recent_bhsd_kernel(
        K,
        V,
        KSeed,
        VSeed,
        N: tl.constexpr,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        TILE_N: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        RECENT_START: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        recent_pos = tl.program_id(2)
        off_kv_h = off_h // GROUP_SIZE
        offs_n = tl.arange(0, TILE_N)
        offs_d = tl.arange(0, D)

        src_block = RECENT_START + recent_pos
        dst_block = SINK_BLOCKS + recent_pos
        src_n = src_block * TILE_N + offs_n
        dst_s = dst_block * TILE_N + offs_n
        col_mask = src_n < N
        k_tile = tl.load(
            K
            + off_b * N * H_KV * D
            + src_n[:, None] * H_KV * D
            + off_kv_h * D
            + offs_d[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        v_tile = tl.load(
            V
            + off_b * N * H_KV * D
            + src_n[:, None] * H_KV * D
            + off_kv_h * D
            + offs_d[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        seed_tokens = (SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS) * TILE_N
        dst_base = off_b * H * seed_tokens * D + off_h * seed_tokens * D
        tl.store(
            KSeed + dst_base + dst_s[:, None] * D + offs_d[None, :],
            k_tile,
            mask=col_mask[:, None],
        )
        tl.store(
            VSeed + dst_base + dst_s[:, None] * D + offs_d[None, :],
            v_tile,
            mask=col_mask[:, None],
        )

    @triton.jit
    def _seed_only_packed_bhsd_kernel(
        Q,
        KSeed,
        VSeed,
        Out,
        H: tl.constexpr,
        D: tl.constexpr,
        SEED_TOKENS: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)

        for start_s in range(0, SEED_TOKENS, TILE_N):
            offs_s = start_s + tl.arange(0, TILE_N)
            col_mask = offs_s < SEED_TOKENS
            k_tile = tl.load(
                KSeed
                + off_b * H * SEED_TOKENS * D
                + off_h * SEED_TOKENS * D
                + offs_s[:, None] * D
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
                VSeed
                + off_b * H * SEED_TOKENS * D
                + off_h * SEED_TOKENS * D
                + offs_s[:, None] * D
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

    @triton.jit
    def _packed_ring_append_seed_only_bhsd_kernel(
        Q,
        KCurrent,
        VCurrent,
        KSeed,
        VSeed,
        Out,
        RingWriteIndex,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        SEED_TOKENS: tl.constexpr,
        TILE_N: tl.constexpr,
        SINK_TOKENS: tl.constexpr,
        RECENT_TOKENS: tl.constexpr,
        SCALE: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_kv_h = off_h // GROUP_SIZE
        offs_d = tl.arange(0, D)

        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)
        k_current = tl.load(KCurrent + off_b * H_KV * D + off_kv_h * D + offs_d)
        v_current = tl.load(VCurrent + off_b * H_KV * D + off_kv_h * D + offs_d)

        ring_idx = tl.load(RingWriteIndex).to(tl.int32) % RECENT_TOKENS
        write_s = SINK_TOKENS + ring_idx
        seed_base = off_b * H * SEED_TOKENS * D + off_h * SEED_TOKENS * D
        tl.store(KSeed + seed_base + write_s * D + offs_d, k_current)
        tl.store(VSeed + seed_base + write_s * D + offs_d, v_current)

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)

        for start_s in range(0, SEED_TOKENS, TILE_N):
            offs_s = start_s + tl.arange(0, TILE_N)
            col_mask = offs_s < SEED_TOKENS
            k_tile = tl.load(
                KSeed
                + seed_base
                + offs_s[:, None] * D
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
                VSeed
                + seed_base
                + offs_s[:, None] * D
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

    @triton.jit
    def _rope_append_seed_only_cachepos_bhnd_kernel(
        QRaw,
        KRaw,
        VRaw,
        Cos,
        Sin,
        KCache,
        VCache,
        Out,
        CachePosition,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_H: tl.constexpr,
        Q_STRIDE_S: tl.constexpr,
        K_RAW_STRIDE_B: tl.constexpr,
        K_RAW_STRIDE_H: tl.constexpr,
        K_RAW_STRIDE_S: tl.constexpr,
        V_RAW_STRIDE_B: tl.constexpr,
        V_RAW_STRIDE_H: tl.constexpr,
        V_RAW_STRIDE_S: tl.constexpr,
        COS_STRIDE_B: tl.constexpr,
        COS_STRIDE_S: tl.constexpr,
        SIN_STRIDE_B: tl.constexpr,
        SIN_STRIDE_S: tl.constexpr,
        K_STRIDE_B: tl.constexpr,
        K_STRIDE_H: tl.constexpr,
        K_STRIDE_N: tl.constexpr,
        V_STRIDE_B: tl.constexpr,
        V_STRIDE_H: tl.constexpr,
        V_STRIDE_N: tl.constexpr,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_kv_h = off_h // GROUP_SIZE
        offs_d = tl.arange(0, D)
        half = D // 2
        pair_d = (offs_d + half) % D
        rotate_sign = tl.where(offs_d < half, -1.0, 1.0)

        q_raw = tl.load(QRaw + off_b * Q_STRIDE_B + off_h * Q_STRIDE_H + offs_d)
        q_pair = tl.load(QRaw + off_b * Q_STRIDE_B + off_h * Q_STRIDE_H + pair_d)
        k_raw = tl.load(KRaw + off_b * K_RAW_STRIDE_B + off_kv_h * K_RAW_STRIDE_H + offs_d)
        k_pair = tl.load(KRaw + off_b * K_RAW_STRIDE_B + off_kv_h * K_RAW_STRIDE_H + pair_d)
        v_current = tl.load(VRaw + off_b * V_RAW_STRIDE_B + off_kv_h * V_RAW_STRIDE_H + offs_d)
        cos = tl.load(Cos + off_b * COS_STRIDE_B + offs_d)
        sin = tl.load(Sin + off_b * SIN_STRIDE_B + offs_d)

        q = q_raw * cos + (q_pair * rotate_sign) * sin
        k_current = k_raw * cos + (k_pair * rotate_sign) * sin
        pos = tl.load(CachePosition).to(tl.int32)
        n = pos + 1

        write_kv = (off_h % GROUP_SIZE) == 0
        tl.store(
            KCache
            + off_b * K_STRIDE_B
            + off_kv_h * K_STRIDE_H
            + pos * K_STRIDE_N
            + offs_d,
            k_current,
            mask=write_kv,
        )
        tl.store(
            VCache
            + off_b * V_STRIDE_B
            + off_kv_h * V_STRIDE_H
            + pos * V_STRIDE_N
            + offs_d,
            v_current,
            mask=write_kv,
        )

        num_blocks = (n + TILE_N - 1) // TILE_N
        recent_start = num_blocks - RECENT_BLOCKS

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)

        for iter_idx in range(0, SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS):
            if iter_idx < SINK_BLOCKS:
                block_idx = tl.full([], iter_idx, dtype=tl.int32)
            elif iter_idx < SINK_BLOCKS + RECENT_BLOCKS:
                recent_pos = tl.full([], iter_idx - SINK_BLOCKS, dtype=tl.int32)
                block_idx = recent_start + recent_pos
            else:
                middle_pos = tl.full([], iter_idx - SINK_BLOCKS - RECENT_BLOCKS, dtype=tl.int32)
                if BLOCK_ORDER == 0:  # sequential
                    block_idx = SINK_BLOCKS + middle_pos
                else:  # recent_first / sink_recent_first
                    block_idx = recent_start - 1 - middle_pos

            start_n = block_idx * TILE_N
            block_len_i = tl.minimum(TILE_N, n - start_n)
            offs_n = start_n + tl.arange(0, TILE_N)
            col_mask = tl.arange(0, TILE_N) < block_len_i
            current_mask = offs_n == pos
            k_tile = tl.load(
                KCache
                + off_b * K_STRIDE_B
                + off_kv_h * K_STRIDE_H
                + offs_n[:, None] * K_STRIDE_N
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            k_tile = tl.where(current_mask[:, None], k_current[None, :], k_tile)
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
                VCache
                + off_b * V_STRIDE_B
                + off_kv_h * V_STRIDE_H
                + offs_n[:, None] * V_STRIDE_N
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            v_tile = tl.where(current_mask[:, None], v_current[None, :], v_tile)
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

    @triton.jit
    def _seed_only_selected_kv_major_kernel(
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
                + off_b * H_KV * N * D
                + off_kv_h * N * D
                + offs_n[:, None] * D
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
                + off_b * H_KV * N * D
                + off_kv_h * N * D
                + offs_n[:, None] * D
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

    @triton.jit
    def _seed_only_split_seed_partial_kernel(
        Q,
        K,
        V,
        PartialM,
        PartialL,
        PartialNum,
        N: tl.constexpr,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        CHUNKS: tl.constexpr,
        TILE_SEED: tl.constexpr,
        TILE_SEED_PAD: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        RECENT_START: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        PV_USE_BF16: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_c = tl.program_id(2)
        off_kv_h = off_h // GROUP_SIZE
        offs_seed_pad = tl.arange(0, TILE_SEED_PAD)
        offs_seed = off_c * TILE_SEED + offs_seed_pad
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H * D + off_h * D + offs_d)

        seed_blocks = SINK_BLOCKS + RECENT_BLOCKS + MIDDLE_SEED_BLOCKS
        seed_slots = seed_blocks * BLOCK_SIZE
        block_pos = offs_seed // BLOCK_SIZE
        block_offset = offs_seed - block_pos * BLOCK_SIZE
        middle_pos = block_pos - SINK_BLOCKS - RECENT_BLOCKS
        recent_pos = block_pos - SINK_BLOCKS
        sequential_middle_block = SINK_BLOCKS + middle_pos
        recent_first_middle_block = RECENT_START - 1 - middle_pos
        block_idx = tl.where(block_pos < SINK_BLOCKS, block_pos, sequential_middle_block)
        block_idx = tl.where(
            (block_pos >= SINK_BLOCKS) & (block_pos < SINK_BLOCKS + RECENT_BLOCKS),
            RECENT_START + recent_pos,
            block_idx,
        )
        if BLOCK_ORDER != 0:
            block_idx = tl.where(
                block_pos >= SINK_BLOCKS + RECENT_BLOCKS,
                recent_first_middle_block,
                block_idx,
            )
        token_idx = block_idx * BLOCK_SIZE + block_offset
        col_mask = (offs_seed_pad < TILE_SEED) & (offs_seed < seed_slots) & (token_idx >= 0) & (token_idx < N)

        k_tile = tl.load(
            K
            + off_b * N * H_KV * D
            + token_idx[:, None] * H_KV * D
            + off_kv_h * D
            + offs_d[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        qk = tl.sum(q[None, :].to(tl.float32) * k_tile.to(tl.float32), axis=1) * SCALE
        qk = tl.where(col_mask, qk, -float("inf"))
        tile_max = tl.max(qk, axis=0)
        valid = tile_max > -float("inf")
        safe_max = tl.where(valid, tile_max, 0.0)
        p = tl.exp(qk - safe_max)
        p = tl.where(qk > -float("inf"), p, 0.0)
        tile_l = tl.sum(p, axis=0)
        v_tile = tl.load(
            V
            + off_b * N * H_KV * D
            + token_idx[:, None] * H_KV * D
            + off_kv_h * D
            + offs_d[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        if PV_USE_BF16:
            weighted = p[:, None].to(tl.bfloat16) * v_tile.to(tl.bfloat16)
        else:
            weighted = p[:, None].to(tl.float16) * v_tile.to(tl.float16)
        tile_num = tl.sum(weighted.to(tl.float32), axis=0)

        state_base = (off_b * H + off_h) * CHUNKS + off_c
        tl.store(PartialM + state_base, tl.where(valid, tile_max, -float("inf")))
        tl.store(PartialL + state_base, tl.where(valid, tile_l, 0.0))
        tl.store(PartialNum + state_base * D + offs_d, tl.where(valid, tile_num, 0.0))

    @triton.jit
    def _seed_only_split_seed_merge_kernel(
        PartialM,
        PartialL,
        PartialNum,
        Out,
        H: tl.constexpr,
        D: tl.constexpr,
        CHUNKS: tl.constexpr,
        CHUNKS_TILE: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        offs_c = tl.arange(0, CHUNKS_TILE)
        offs_d = tl.arange(0, D)
        state_base = (off_b * H + off_h) * CHUNKS
        chunk_mask = offs_c < CHUNKS
        m_i = tl.load(PartialM + state_base + offs_c, mask=chunk_mask, other=-float("inf"))
        l_i = tl.load(PartialL + state_base + offs_c, mask=chunk_mask, other=0.0)
        m = tl.max(m_i, axis=0)
        valid = m > -float("inf")
        safe_m = tl.where(valid, m, 0.0)
        alpha = tl.where(chunk_mask & (l_i > 0.0), tl.exp(m_i - safe_m), 0.0)
        denom = tl.sum(l_i * alpha, axis=0)
        num_i = tl.load(
            PartialNum + (state_base + offs_c[:, None]) * D + offs_d[None, :],
            mask=chunk_mask[:, None],
            other=0.0,
        )
        num = tl.sum(alpha[:, None] * num_i, axis=0)
        out = num / denom
        out = tl.where(denom > 0.0, out, 0.0)
        tl.store(Out + off_b * H * D + off_h * D + offs_d, out)


def _block_order_id(block_order: str) -> int:
    if block_order == "sequential":
        return 0
    if block_order == "recent_first":
        return 1
    if block_order == "sink_recent_first":
        return 2
    raise ValueError("block_order must be 'sequential', 'recent_first', or 'sink_recent_first'")


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _validate_seed_only_tensors(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
) -> tuple[int, int, int, int, int, int]:
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
    batch, _seq_q, heads, dim = query.shape
    seq_k = key.shape[1]
    kv_heads = key.shape[2]
    num_blocks = triton.cdiv(seq_k, block_size)
    if sink_blocks + recent_blocks > num_blocks:
        raise ValueError("sink_blocks + recent_blocks must not exceed K blocks")
    middle_blocks = num_blocks - sink_blocks - recent_blocks
    if middle_seed_blocks > middle_blocks:
        raise ValueError("middle_seed_blocks must be within the middle block count")
    return batch, heads, seq_k, kv_heads, dim, num_blocks


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


def gate0_seed_only_attention_triton_forward_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    *,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run seed-only decode into a caller-provided final output buffer."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, output)):
        raise RuntimeError("gate0_seed_only_attention_triton_forward_out requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if query.shape[1] != 1:
        raise ValueError("seed-only Gate-0 only supports query_len == 1")
    if output.shape != query.shape:
        raise ValueError("output must have the same shape as query")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
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

    score_scale = 1.0 / math.sqrt(dim)
    recent_start = num_blocks - recent_blocks
    _seed_only_kernel[(batch, heads)](
        query,
        key,
        value,
        output,
        output,
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
        HAS_STATS=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def gate0_seed_only_attention_triton_forward_out_dynamic_n(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    *,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run seed-only decode with dynamic KV length.

    This is intended for actual `use_cache=True` generation where `seq_k`
    increases every token.  The regular launch specializes on `N` and
    `RECENT_START`; this variant keeps those values as runtime scalars so a
    steady-state decode loop does not JIT-compile a new kernel per token.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, output)):
        raise RuntimeError("gate0_seed_only_attention_triton_forward_out_dynamic_n requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if query.shape[1] != 1:
        raise ValueError("seed-only Gate-0 only supports query_len == 1")
    if output.shape != query.shape:
        raise ValueError("output must have the same shape as query")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
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

    score_scale = 1.0 / math.sqrt(dim)
    recent_start = num_blocks - recent_blocks
    _seed_only_dynamic_n_kernel[(batch, heads)](
        query,
        key,
        value,
        output,
        seq_k,
        int(recent_start),
        H=heads,
        H_KV=kv_heads,
        GROUP_SIZE=group_size,
        D=dim,
        TILE_N=block_size,
        SCALE=score_scale,
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        PV_USE_BF16=value.dtype is torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def gate0_seed_only_attention_triton_forward_out_dynamic_n_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    *,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run seed-only decode against Hugging Face cache layout `[B, Hkv, N, D]`.

    This avoids materializing a full `[B, N, Hkv, D]` transpose during actual
    model decode.  KV strides are passed as runtime scalars so dynamic-cache
    sequence growth does not force per-token recompilation.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, output)):
        raise RuntimeError("gate0_seed_only_attention_triton_forward_out_dynamic_n_bhnd requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query must be [batch, 1, heads, dim], key/value must be [batch, kv_heads, seq, dim]")
    if query.shape[1] != 1:
        raise ValueError("seed-only Gate-0 only supports query_len == 1")
    if output.shape != query.shape:
        raise ValueError("output must have the same shape as query")
    if not query.is_contiguous() or not output.is_contiguous():
        raise ValueError("query and output must be contiguous")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query/key/value must have matching batch and dim")
    if query.shape[2] % key.shape[1] != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0 or middle_seed_blocks < 0:
        raise ValueError("seed block counts must be non-negative")

    batch, _seq_q, heads, dim = query.shape
    kv_heads = key.shape[1]
    seq_k = key.shape[2]
    group_size = heads // kv_heads
    num_blocks = triton.cdiv(seq_k, block_size)
    if sink_blocks + recent_blocks > num_blocks:
        raise ValueError("sink_blocks + recent_blocks must not exceed K blocks")
    middle_blocks = num_blocks - sink_blocks - recent_blocks
    if middle_seed_blocks > middle_blocks:
        raise ValueError("middle_seed_blocks must be within the middle block count")

    score_scale = 1.0 / math.sqrt(dim)
    recent_start = num_blocks - recent_blocks
    _seed_only_dynamic_n_bhnd_kernel[(batch, heads)](
        query,
        key,
        value,
        output,
        seq_k,
        int(recent_start),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        H=heads,
        H_KV=kv_heads,
        GROUP_SIZE=group_size,
        D=dim,
        TILE_N=block_size,
        SCALE=score_scale,
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        PV_USE_BF16=value.dtype is torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def gate0_seed_only_attention_triton_forward_out_cachepos_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    cache_position: torch.Tensor,
    *,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run seed-only decode from `[B, Hkv, N, D]` cache using CUDA cache position.

    This is the model-decode steady-state path: KV length is read from
    `cache_position` on device, so generation steps with growing cache lengths
    reuse the same compiled kernel.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, output, cache_position)):
        raise RuntimeError("gate0_seed_only_attention_triton_forward_out_cachepos_bhnd requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query must be [batch, 1, heads, dim], key/value must be [batch, kv_heads, seq, dim]")
    if query.shape[1] != 1:
        raise ValueError("seed-only Gate-0 only supports query_len == 1")
    if output.shape != query.shape:
        raise ValueError("output must have the same shape as query")
    if not query.is_contiguous() or not output.is_contiguous():
        raise ValueError("query and output must be contiguous")
    if not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("key/value must be contiguous for cache-position BHND decode")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query/key/value must have matching batch and dim")
    if query.shape[2] % key.shape[1] != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    if cache_position.numel() < 1:
        raise ValueError("cache_position must contain at least one element")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0 or middle_seed_blocks < 0:
        raise ValueError("seed block counts must be non-negative")

    batch, _seq_q, heads, dim = query.shape
    kv_heads = key.shape[1]
    group_size = heads // kv_heads
    score_scale = 1.0 / math.sqrt(dim)
    _seed_only_cachepos_bhnd_kernel[(batch, heads)](
        query,
        key,
        value,
        output,
        cache_position,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        H=heads,
        H_KV=kv_heads,
        GROUP_SIZE=group_size,
        D=dim,
        TILE_N=block_size,
        SCALE=score_scale,
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        PV_USE_BF16=value.dtype is torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def make_gate0_seed_only_packed_workspace(query: torch.Tensor, *, seed_tokens: int) -> dict[str, torch.Tensor]:
    """Allocate head-private packed seed K/V buffers.

    The packed layout is ``[batch, q_heads, seed_tokens, dim]``.  It
    intentionally duplicates GQA seed K/V per Q head so the probe can test
    whether coalesced per-head seed reads beat indirection through the full
    shared KV cache.
    """

    if query.dim() != 4:
        raise ValueError("query must be [batch, 1, heads, dim]")
    if seed_tokens <= 0:
        raise ValueError("seed_tokens must be positive")
    batch, _seq_q, heads, dim = query.shape
    shape = (batch, heads, int(seed_tokens), dim)
    return {
        "k_seed": torch.empty(shape, device=query.device, dtype=query.dtype),
        "v_seed": torch.empty(shape, device=query.device, dtype=query.dtype),
    }


def gate0_pack_seed_cache_bhsd(
    key: torch.Tensor,
    value: torch.Tensor,
    k_seed: torch.Tensor,
    v_seed: torch.Tensor,
    *,
    q_heads: int,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack full NHD K/V cache into head-private ``[B,Hq,S,D]`` seed K/V."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (key, value, k_seed, v_seed)):
        raise RuntimeError("gate0_pack_seed_cache_bhsd requires CUDA tensors")
    if key.dim() != 4 or value.dim() != 4:
        raise ValueError("key/value must be [batch, seq, kv_heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("key/value must be contiguous NHD tensors")
    if k_seed.shape != v_seed.shape:
        raise ValueError("k_seed and v_seed must have the same shape")
    if k_seed.dim() != 4:
        raise ValueError("k_seed/v_seed must be [batch, q_heads, seed_tokens, dim]")
    batch, seq_k, kv_heads, dim = key.shape
    seed_blocks = sink_blocks + recent_blocks + middle_seed_blocks
    seed_tokens = seed_blocks * block_size
    if k_seed.shape != (batch, int(q_heads), seed_tokens, dim):
        raise ValueError("packed seed workspace must be [batch, q_heads, seed_tokens, dim]")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be a multiple of kv_heads")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0 or middle_seed_blocks < 0:
        raise ValueError("seed block counts must be non-negative")
    num_blocks = triton.cdiv(seq_k, block_size)
    if sink_blocks + recent_blocks > num_blocks:
        raise ValueError("sink_blocks + recent_blocks must not exceed K blocks")
    middle_blocks = num_blocks - sink_blocks - recent_blocks
    if middle_seed_blocks > middle_blocks:
        raise ValueError("middle_seed_blocks must be within the middle block count")
    recent_start = num_blocks - recent_blocks
    _pack_seed_cache_bhsd_kernel[(batch, int(q_heads), seed_blocks)](
        key,
        value,
        k_seed,
        v_seed,
        N=seq_k,
        H=int(q_heads),
        H_KV=kv_heads,
        GROUP_SIZE=int(q_heads) // kv_heads,
        D=dim,
        TILE_N=block_size,
        NUM_BLOCKS=num_blocks,
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        RECENT_START=int(recent_start),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return k_seed, v_seed


def gate0_refresh_packed_seed_cache_recent_bhsd(
    key: torch.Tensor,
    value: torch.Tensor,
    k_seed: torch.Tensor,
    v_seed: torch.Tensor,
    *,
    q_heads: int,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refresh only the recent region in packed ``[B,Hq,S,D]`` seed K/V.

    This probes the persistent packed-cache strategy: sink and middle seed
    regions can stay fixed while the recent seed blocks are maintained
    incrementally instead of rebuilding the whole packed seed cache.
    """

    if recent_blocks <= 0:
        return k_seed, v_seed
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (key, value, k_seed, v_seed)):
        raise RuntimeError("gate0_refresh_packed_seed_cache_recent_bhsd requires CUDA tensors")
    if key.dim() != 4 or value.dim() != 4:
        raise ValueError("key/value must be [batch, seq, kv_heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("key/value must be contiguous NHD tensors")
    if k_seed.shape != v_seed.shape:
        raise ValueError("k_seed and v_seed must have the same shape")
    if k_seed.dim() != 4:
        raise ValueError("k_seed/v_seed must be [batch, q_heads, seed_tokens, dim]")
    batch, seq_k, kv_heads, dim = key.shape
    seed_blocks = sink_blocks + recent_blocks + middle_seed_blocks
    seed_tokens = seed_blocks * block_size
    if k_seed.shape != (batch, int(q_heads), seed_tokens, dim):
        raise ValueError("packed seed workspace must be [batch, q_heads, seed_tokens, dim]")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be a multiple of kv_heads")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0 or middle_seed_blocks < 0:
        raise ValueError("seed block counts must be non-negative")
    num_blocks = triton.cdiv(seq_k, block_size)
    if sink_blocks + recent_blocks > num_blocks:
        raise ValueError("sink_blocks + recent_blocks must not exceed K blocks")
    middle_blocks = num_blocks - sink_blocks - recent_blocks
    if middle_seed_blocks > middle_blocks:
        raise ValueError("middle_seed_blocks must be within the middle block count")
    recent_start = num_blocks - recent_blocks
    _refresh_packed_seed_cache_recent_bhsd_kernel[(batch, int(q_heads), int(recent_blocks))](
        key,
        value,
        k_seed,
        v_seed,
        N=seq_k,
        H=int(q_heads),
        H_KV=kv_heads,
        GROUP_SIZE=int(q_heads) // kv_heads,
        D=dim,
        TILE_N=block_size,
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        RECENT_START=int(recent_start),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return k_seed, v_seed


def gate0_seed_only_attention_packed_bhsd_triton_forward_out(
    query: torch.Tensor,
    k_seed: torch.Tensor,
    v_seed: torch.Tensor,
    output: torch.Tensor,
    *,
    block_size: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run seed-only decode from head-private packed ``[B,Hq,S,D]`` K/V."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, k_seed, v_seed, output)):
        raise RuntimeError("gate0_seed_only_attention_packed_bhsd_triton_forward_out requires CUDA tensors")
    if query.dim() != 4 or query.shape[1] != 1:
        raise ValueError("query must be [batch, 1, heads, dim]")
    if output.shape != query.shape:
        raise ValueError("output must have the same shape as query")
    if not query.is_contiguous() or not output.is_contiguous():
        raise ValueError("query and output must be contiguous")
    if k_seed.shape != v_seed.shape or k_seed.dim() != 4:
        raise ValueError("k_seed/v_seed must be matching [batch, heads, seed_tokens, dim] tensors")
    if not k_seed.is_contiguous() or not v_seed.is_contiguous():
        raise ValueError("k_seed/v_seed must be contiguous")
    batch, _seq_q, heads, dim = query.shape
    if k_seed.shape[0] != batch or k_seed.shape[1] != heads or k_seed.shape[3] != dim:
        raise ValueError("packed seed tensors must match query batch/heads/dim")
    seed_tokens = int(k_seed.shape[2])
    if seed_tokens <= 0 or seed_tokens % block_size != 0:
        raise ValueError("seed_tokens must be a positive multiple of block_size")
    score_scale = 1.0 / math.sqrt(dim)
    _seed_only_packed_bhsd_kernel[(batch, heads)](
        query,
        k_seed,
        v_seed,
        output,
        H=heads,
        D=dim,
        SEED_TOKENS=seed_tokens,
        TILE_N=block_size,
        SCALE=score_scale,
        PV_USE_BF16=v_seed.dtype is torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def gate0_seed_only_packed_ring_append_triton_forward_out(
    query: torch.Tensor,
    key_current: torch.Tensor,
    value_current: torch.Tensor,
    k_seed: torch.Tensor,
    v_seed: torch.Tensor,
    output: torch.Tensor,
    ring_write_index: torch.Tensor,
    *,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Append current K/V to a packed recent ring and run seed attention.

    The packed cache layout is ``[B,Hq,S,D]`` with the recent ring occupying
    slots ``[sink_blocks * block_size : (sink_blocks + recent_blocks) *
    block_size)``.  The seed attention reads the ring as an unordered set,
    which is valid for softmax attention over paired K/V seed tokens.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    tensors = (query, key_current, value_current, k_seed, v_seed, output, ring_write_index)
    if not all(t.is_cuda for t in tensors):
        raise RuntimeError("gate0_seed_only_packed_ring_append_triton_forward_out requires CUDA tensors")
    if query.dim() != 4 or query.shape[1] != 1:
        raise ValueError("query must be [batch, 1, heads, dim]")
    if key_current.dim() != 4 or value_current.dim() != 4:
        raise ValueError("key_current/value_current must be [batch, 1, kv_heads, dim]")
    if key_current.shape != value_current.shape:
        raise ValueError("key_current and value_current must have the same shape")
    if key_current.shape[1] != 1:
        raise ValueError("key_current/value_current only support one current token")
    if output.shape != query.shape:
        raise ValueError("output must have the same shape as query")
    if not query.is_contiguous() or not key_current.is_contiguous() or not value_current.is_contiguous():
        raise ValueError("query/key_current/value_current must be contiguous")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if k_seed.shape != v_seed.shape or k_seed.dim() != 4:
        raise ValueError("k_seed/v_seed must be matching [batch, heads, seed_tokens, dim] tensors")
    if not k_seed.is_contiguous() or not v_seed.is_contiguous():
        raise ValueError("k_seed/v_seed must be contiguous")
    if ring_write_index.numel() < 1:
        raise ValueError("ring_write_index must contain at least one element")
    batch, _seq_q, heads, dim = query.shape
    kv_heads = key_current.shape[2]
    if key_current.shape[0] != batch or key_current.shape[3] != dim:
        raise ValueError("current K/V must match query batch and dim")
    if heads % kv_heads != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    seed_blocks = sink_blocks + recent_blocks + middle_seed_blocks
    seed_tokens = seed_blocks * block_size
    if recent_blocks <= 0:
        raise ValueError("recent_blocks must be positive for ring append")
    if k_seed.shape != (batch, heads, seed_tokens, dim):
        raise ValueError("packed seed tensors must be [batch, q_heads, seed_tokens, dim]")
    score_scale = 1.0 / math.sqrt(dim)
    _packed_ring_append_seed_only_bhsd_kernel[(batch, heads)](
        query,
        key_current,
        value_current,
        k_seed,
        v_seed,
        output,
        ring_write_index,
        H=heads,
        H_KV=kv_heads,
        GROUP_SIZE=heads // kv_heads,
        D=dim,
        SEED_TOKENS=seed_tokens,
        TILE_N=block_size,
        SINK_TOKENS=sink_blocks * block_size,
        RECENT_TOKENS=recent_blocks * block_size,
        SCALE=score_scale,
        PV_USE_BF16=v_seed.dtype is torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def gate0_seed_only_rope_append_triton_forward_out_cachepos_bhnd(
    query_raw: torch.Tensor,
    key_raw: torch.Tensor,
    value_raw: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    cache_position: torch.Tensor,
    *,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Apply Qwen RoPE, append current K/V, and run seed-only decode.

    Inputs use Qwen decode layout before RoPE:

    * `query_raw`: `[batch, q_heads, 1, dim]`
    * `key_raw`/`value_raw`: `[batch, kv_heads, 1, dim]`
    * `cos`/`sin`: `[batch, 1, dim]` or compatible contiguous decode slice
    * `key_cache`/`value_cache`: `[batch, kv_heads, max_len, dim]`

    The kernel stores the current K/V for future steps. For the current step it
    also overrides reads at `cache_position` with the just-computed K/V, avoiding
    inter-CTA ordering assumptions inside the single fused launch.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    tensors = (query_raw, key_raw, value_raw, cos, sin, key_cache, value_cache, output, cache_position)
    if not all(t.is_cuda for t in tensors):
        raise RuntimeError("fused RoPE append seed-only decode requires CUDA tensors")
    if query_raw.dim() != 4 or key_raw.dim() != 4 or value_raw.dim() != 4:
        raise ValueError("query_raw/key_raw/value_raw must be [batch, heads, 1, dim]")
    if query_raw.shape[2] != 1 or key_raw.shape[2] != 1 or value_raw.shape[2] != 1:
        raise ValueError("fused RoPE append seed-only decode only supports query_len == 1")
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError("key_cache/value_cache must be [batch, kv_heads, seq, dim]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must have the same shape")
    if key_raw.shape != value_raw.shape:
        raise ValueError("key_raw and value_raw must have the same shape")
    if output.shape != (query_raw.shape[0], 1, query_raw.shape[1], query_raw.shape[3]):
        raise ValueError("output must be [batch, 1, q_heads, dim]")
    if query_raw.shape[0] != key_cache.shape[0] or query_raw.shape[3] != key_cache.shape[3]:
        raise ValueError("query/cache batch and dim must match")
    if query_raw.shape[1] % key_cache.shape[1] != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    if key_raw.shape[1] != key_cache.shape[1]:
        raise ValueError("current key/value KV heads must match cache KV heads")
    if cache_position.numel() < 1:
        raise ValueError("cache_position must contain at least one element")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if cos.dim() != 3 or sin.dim() != 3:
        raise ValueError("cos/sin must be [batch_or_1, 1, dim]")
    if cos.shape[-1] != query_raw.shape[3] or sin.shape[-1] != query_raw.shape[3]:
        raise ValueError("cos/sin head dimension must match query head dimension")
    if cos.shape[0] not in (1, query_raw.shape[0]) or sin.shape[0] not in (1, query_raw.shape[0]):
        raise ValueError("cos/sin batch dimension must be 1 or match query batch")

    batch, heads, _seq_q, dim = query_raw.shape
    kv_heads = key_cache.shape[1]
    group_size = heads // kv_heads
    score_scale = 1.0 / math.sqrt(dim)
    cos_stride_b = 0 if cos.shape[0] == 1 else cos.stride(0)
    sin_stride_b = 0 if sin.shape[0] == 1 else sin.stride(0)
    _rope_append_seed_only_cachepos_bhnd_kernel[(batch, heads)](
        query_raw,
        key_raw,
        value_raw,
        cos,
        sin,
        key_cache,
        value_cache,
        output,
        cache_position,
        query_raw.stride(0),
        query_raw.stride(1),
        query_raw.stride(2),
        key_raw.stride(0),
        key_raw.stride(1),
        key_raw.stride(2),
        value_raw.stride(0),
        value_raw.stride(1),
        value_raw.stride(2),
        cos_stride_b,
        cos.stride(1) if cos.dim() > 2 else 0,
        sin_stride_b,
        sin.stride(1) if sin.dim() > 2 else 0,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        H=heads,
        H_KV=kv_heads,
        GROUP_SIZE=group_size,
        D=dim,
        TILE_N=block_size,
        SCALE=score_scale,
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        PV_USE_BF16=value_cache.dtype is torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def gate0_seed_only_attention_triton_forward_out_prechecked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    *,
    seq_k: int,
    q_heads: int,
    kv_heads: int,
    group_size: int,
    dim: int,
    num_blocks: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    recent_start: int,
    middle_seed_blocks: int,
    block_order_id: int,
    score_scale: float,
    pv_use_bf16: bool,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run seed-only decode using a caller-validated fixed-shape launch plan.

    This intentionally skips Python-side tensor validation and shape arithmetic.
    It is only for planned serving paths that already checked policy, shape,
    dtype, layout, and CUDA availability before entering the steady-state loop.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    _seed_only_kernel[(query.shape[0], q_heads)](
        query,
        key,
        value,
        output,
        output,
        N=int(seq_k),
        H=int(q_heads),
        H_KV=int(kv_heads),
        GROUP_SIZE=int(group_size),
        D=int(dim),
        NUM_BLOCKS=int(num_blocks),
        TILE_N=int(block_size),
        SCALE=float(score_scale),
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        RECENT_START=int(recent_start),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=int(block_order_id),
        PV_USE_BF16=bool(pv_use_bf16),
        HAS_STATS=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def make_gate0_seed_only_split_seed_workspace(
    query: torch.Tensor,
    *,
    seed_chunks: int,
) -> dict[str, torch.Tensor]:
    """Allocate partial online-softmax state for split-seed decode."""

    if seed_chunks <= 0:
        raise ValueError("seed_chunks must be positive")
    if query.dim() != 4 or query.shape[1] != 1:
        raise ValueError("query must be [batch, 1, heads, dim]")
    batch, _seq_q, heads, dim = query.shape
    device = query.device
    return {
        "partial_m": torch.empty(batch, heads, seed_chunks, device=device, dtype=torch.float32),
        "partial_l": torch.empty(batch, heads, seed_chunks, device=device, dtype=torch.float32),
        "partial_num": torch.empty(
            batch,
            heads,
            seed_chunks,
            dim,
            device=device,
            dtype=torch.float32,
        ),
    }


def gate0_seed_only_split_seed_triton_partial(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_num: torch.Tensor,
    *,
    seed_tile_tokens: int = 64,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute split-seed partial online-softmax states.

    Grid shape is ``[batch, q_head, seed_chunk]``.  The partial tensors must
    have shapes ``[B, Hq, C]``, ``[B, Hq, C]``, and ``[B, Hq, C, D]``.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, partial_m, partial_l, partial_num)):
        raise RuntimeError("gate0_seed_only_split_seed_triton_partial requires CUDA tensors")
    if seed_tile_tokens <= 0:
        raise ValueError("seed_tile_tokens must be positive")
    batch, heads, seq_k, kv_heads, dim, num_blocks = _validate_seed_only_tensors(
        query,
        key,
        value,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
    )
    seed_slots = (sink_blocks + recent_blocks + middle_seed_blocks) * block_size
    seed_chunks = triton.cdiv(seed_slots, seed_tile_tokens)
    expected_m = (batch, heads, seed_chunks)
    expected_num = (batch, heads, seed_chunks, dim)
    if partial_m.shape != expected_m or partial_l.shape != expected_m:
        raise ValueError(f"partial_m/partial_l must have shape {expected_m}")
    if partial_num.shape != expected_num:
        raise ValueError(f"partial_num must have shape {expected_num}")
    if not (partial_m.is_contiguous() and partial_l.is_contiguous() and partial_num.is_contiguous()):
        raise ValueError("partial tensors must be contiguous")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    score_scale = 1.0 / math.sqrt(dim)
    recent_start = num_blocks - recent_blocks
    _seed_only_split_seed_partial_kernel[(batch, heads, seed_chunks)](
        query,
        key,
        value,
        partial_m,
        partial_l,
        partial_num,
        N=seq_k,
        H=heads,
        H_KV=kv_heads,
        GROUP_SIZE=heads // kv_heads,
        D=dim,
        CHUNKS=seed_chunks,
        TILE_SEED=int(seed_tile_tokens),
        TILE_SEED_PAD=_next_power_of_2(seed_tile_tokens),
        SCALE=score_scale,
        BLOCK_SIZE=int(block_size),
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        RECENT_START=int(recent_start),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        PV_USE_BF16=value.dtype is torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return partial_m, partial_l, partial_num


def gate0_seed_only_split_seed_triton_merge(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_num: torch.Tensor,
    output: torch.Tensor,
    *,
    num_warps: int = 1,
    num_stages: int = 3,
) -> torch.Tensor:
    """Merge split-seed partial online-softmax states into final output."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (partial_m, partial_l, partial_num, output)):
        raise RuntimeError("gate0_seed_only_split_seed_triton_merge requires CUDA tensors")
    if partial_m.dim() != 3 or partial_l.shape != partial_m.shape:
        raise ValueError("partial_m and partial_l must be [batch, heads, chunks]")
    if partial_num.dim() != 4 or partial_num.shape[:3] != partial_m.shape:
        raise ValueError("partial_num must be [batch, heads, chunks, dim]")
    batch, heads, seed_chunks = partial_m.shape
    dim = partial_num.shape[3]
    if output.shape != (batch, 1, heads, dim):
        raise ValueError(f"output must have shape {(batch, 1, heads, dim)}")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if not (partial_m.is_contiguous() and partial_l.is_contiguous() and partial_num.is_contiguous()):
        raise ValueError("partial tensors must be contiguous")

    _seed_only_split_seed_merge_kernel[(batch, heads)](
        partial_m,
        partial_l,
        partial_num,
        output,
        H=heads,
        D=dim,
        CHUNKS=seed_chunks,
        CHUNKS_TILE=_next_power_of_2(seed_chunks),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def gate0_seed_only_split_seed_triton_forward_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    *,
    seed_tile_tokens: int = 64,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    workspace: dict[str, torch.Tensor] | None = None,
    partial_num_warps: int = 4,
    partial_num_stages: int = 3,
    merge_num_warps: int = 1,
    merge_num_stages: int = 3,
) -> torch.Tensor:
    """Run two-kernel head-private split-seed decode into ``output``."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, output)):
        raise RuntimeError("gate0_seed_only_split_seed_triton_forward_out requires CUDA tensors")
    batch, heads, _seq_k, _kv_heads, dim, _num_blocks = _validate_seed_only_tensors(
        query,
        key,
        value,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
    )
    if output.shape != (batch, 1, heads, dim):
        raise ValueError("output must have the same shape as query")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    seed_slots = (sink_blocks + recent_blocks + middle_seed_blocks) * block_size
    seed_chunks = triton.cdiv(seed_slots, seed_tile_tokens)
    if workspace is None:
        workspace = make_gate0_seed_only_split_seed_workspace(query, seed_chunks=seed_chunks)
    partial_m = workspace["partial_m"]
    partial_l = workspace["partial_l"]
    partial_num = workspace["partial_num"]
    gate0_seed_only_split_seed_triton_partial(
        query,
        key,
        value,
        partial_m,
        partial_l,
        partial_num,
        seed_tile_tokens=seed_tile_tokens,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        num_warps=partial_num_warps,
        num_stages=partial_num_stages,
    )
    gate0_seed_only_split_seed_triton_merge(
        partial_m,
        partial_l,
        partial_num,
        output,
        num_warps=merge_num_warps,
        num_stages=merge_num_stages,
    )
    return output


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
    validate_heads: bool = True,
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
    if validate_heads:
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


def gate0_seed_only_selected_attention_kv_major_triton_forward(
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
    validate_heads: bool = True,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Run selected-head seed-only decode with KV-major K/V layout.

    ``key`` and ``value`` must be contiguous ``[batch, kv_heads, seq, dim]``.
    This probes whether StreamAttn's sparse path needs KV-head-contiguous cache
    pages rather than the usual interleaved ``[batch, seq, kv_heads, dim]``
    layout.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, selected_heads)):
        raise RuntimeError("gate0_seed_only_selected_attention_kv_major_triton_forward requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query must be [batch, seq, heads, dim], key/value must be [batch, kv_heads, seq, dim]")
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
    if query.shape[2] % key.shape[1] != 0:
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
    kv_heads = key.shape[1]
    seq_k = key.shape[2]
    group_size = q_heads // kv_heads
    if validate_heads:
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
    _seed_only_selected_kv_major_kernel[(batch, selected)](
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
