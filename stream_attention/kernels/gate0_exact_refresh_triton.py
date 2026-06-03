"""Native exact decode refresh kernels for StreamAttn verifier routes."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Set

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _exact_refresh_rows_bhnd_kernel(
        Q,
        K,
        V,
        Out,
        Rows,
        CachePosition,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_H: tl.constexpr,
        Q_STRIDE_S: tl.constexpr,
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
        N_MAX: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        off_h = tl.program_id(1)
        off_b = tl.load(Rows + row_idx).to(tl.int32)
        off_kv_h = off_h // GROUP_SIZE
        offs_d = tl.arange(0, D)

        q = tl.load(Q + off_b * Q_STRIDE_B + off_h * Q_STRIDE_H + offs_d)
        n = tl.minimum(tl.load(CachePosition).to(tl.int32) + 1, N_MAX)

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)

        for block_idx in range(0, NUM_BLOCKS):
            offs_n = block_idx * TILE_N + tl.arange(0, TILE_N)
            col_mask = offs_n < n
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
            acc_num = acc_num * correction + tl.sum(p[:, None] * v_tile.to(tl.float32), axis=0)
            acc_den = acc_den * correction + tl.sum(p, axis=0)
            running_max = tl.where(new_valid, new_max, running_max)

        out = acc_num / acc_den
        out = tl.where(acc_den > 0.0, out, 0.0)
        tl.store(Out + off_b * H * D + off_h * D + offs_d, out)

    @triton.jit
    def _exact_refresh_rows_splitk_partial_kernel(
        Q,
        K,
        V,
        Rows,
        PartialM,
        PartialL,
        PartialNum,
        CachePosition,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_H: tl.constexpr,
        Q_STRIDE_S: tl.constexpr,
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
        N_MAX: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        CHUNKS: tl.constexpr,
        CHUNK_BLOCKS: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        off_h = tl.program_id(1)
        off_c = tl.program_id(2)
        off_b = tl.load(Rows + row_idx).to(tl.int32)
        off_kv_h = off_h // GROUP_SIZE
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * Q_STRIDE_B + off_h * Q_STRIDE_H + offs_d)
        n = tl.minimum(tl.load(CachePosition).to(tl.int32) + 1, N_MAX)

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        acc_den = tl.zeros([], dtype=tl.float32)
        acc_num = tl.zeros([D], dtype=tl.float32)

        for local_block in range(0, CHUNK_BLOCKS):
            block_idx = off_c * CHUNK_BLOCKS + local_block
            offs_n = block_idx * TILE_N + tl.arange(0, TILE_N)
            block_valid = block_idx < NUM_BLOCKS
            col_mask = (offs_n < n) & block_valid
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
            acc_num = acc_num * correction + tl.sum(p[:, None] * v_tile.to(tl.float32), axis=0)
            acc_den = acc_den * correction + tl.sum(p, axis=0)
            running_max = tl.where(new_valid, new_max, running_max)

        state_base = (row_idx * H + off_h) * CHUNKS + off_c
        tl.store(PartialM + state_base, running_max)
        tl.store(PartialL + state_base, acc_den)
        tl.store(PartialNum + state_base * D + offs_d, acc_num)

    @triton.jit
    def _exact_refresh_rows_splitk_merge_kernel(
        Rows,
        PartialM,
        PartialL,
        PartialNum,
        Out,
        H: tl.constexpr,
        D: tl.constexpr,
        CHUNKS: tl.constexpr,
        CHUNKS_TILE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        off_h = tl.program_id(1)
        off_b = tl.load(Rows + row_idx).to(tl.int32)
        offs_c = tl.arange(0, CHUNKS_TILE)
        offs_d = tl.arange(0, D)
        chunk_mask = offs_c < CHUNKS
        state_base = (row_idx * H + off_h) * CHUNKS
        m_i = tl.load(PartialM + state_base + offs_c, mask=chunk_mask, other=-float("inf")).to(tl.float32)
        l_i = tl.load(PartialL + state_base + offs_c, mask=chunk_mask, other=0.0).to(tl.float32)
        m = tl.max(m_i, axis=0)
        valid = m > -float("inf")
        safe_m = tl.where(valid, m, 0.0)
        factors = tl.exp(m_i - safe_m)
        factors = tl.where(chunk_mask & (l_i > 0.0), factors, 0.0)
        den = tl.sum(l_i * factors, axis=0)
        num_i = tl.load(
            PartialNum + (state_base + offs_c[:, None]) * D + offs_d[None, :],
            mask=chunk_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        num = tl.sum(num_i * factors[:, None], axis=0)
        out = num / den
        out = tl.where(den > 0.0, out, 0.0)
        tl.store(Out + off_b * H * D + off_h * D + offs_d, out)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def make_exact_refresh_splitk_workspace(
    *,
    row_count: int,
    heads: int,
    chunks: int,
    dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    return {
        "partial_m": torch.empty(row_count, heads, chunks, device=device, dtype=torch.float32),
        "partial_l": torch.empty(row_count, heads, chunks, device=device, dtype=torch.float32),
        "partial_num": torch.empty(row_count, heads, chunks, dim, device=device, dtype=torch.float32),
    }


def _rows_tensor(
    *,
    batch: int,
    row_indices: Optional[torch.Tensor | Sequence[int] | Set[int]],
    device: torch.device,
) -> torch.Tensor:
    if row_indices is None:
        return torch.arange(batch, device=device, dtype=torch.int64)
    if isinstance(row_indices, torch.Tensor):
        rows = row_indices.to(device=device, dtype=torch.int64).contiguous()
    else:
        valid_rows = sorted({int(row) for row in row_indices if 0 <= int(row) < int(batch)})
        rows = torch.tensor(valid_rows, device=device, dtype=torch.int64)
    if rows.numel() == 0:
        return rows
    return rows[(rows >= 0) & (rows < int(batch))].contiguous()


def exact_decode_attention_rows_triton_forward_out_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    cache_position: torch.Tensor,
    *,
    row_indices: Optional[torch.Tensor | Sequence[int] | Set[int]] = None,
    tile_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run exact M=1 decode attention for selected batch rows.

    ``query`` is ``[B, Hq, 1, D]`` and K/V are native BHND
    ``[B, Hkv, N, D]`` caches. ``output`` is ``[B, 1, Hq, D]`` and only
    selected rows are overwritten. The prefix length is read from
    ``cache_position`` to avoid recompiling per decode step.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, output, cache_position)):
        raise RuntimeError("exact_decode_attention_rows_triton_forward_out_bhnd requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query must be [batch, heads, 1, dim], key/value must be [batch, kv_heads, seq, dim]")
    if query.shape[2] != 1:
        raise ValueError("exact refresh only supports query_len == 1")
    if output.shape != (query.shape[0], 1, query.shape[1], query.shape[3]):
        raise ValueError("output must be [batch, 1, heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have matching shape")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query/key/value must have matching batch and dim")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if cache_position.numel() < 1:
        raise ValueError("cache_position must contain at least one element")
    if tile_n <= 0:
        raise ValueError("tile_n must be positive")

    batch, heads, _query_len, dim = query.shape
    kv_heads = key.shape[1]
    rows = _rows_tensor(batch=batch, row_indices=row_indices, device=query.device)
    if rows.numel() == 0:
        return output
    score_scale = 1.0 / math.sqrt(float(dim))
    n_max = int(key.shape[2])
    num_blocks = triton.cdiv(n_max, int(tile_n))
    _exact_refresh_rows_bhnd_kernel[(int(rows.numel()), int(heads))](
        query,
        key,
        value,
        output,
        rows,
        cache_position,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        H=int(heads),
        H_KV=int(kv_heads),
        GROUP_SIZE=int(heads // kv_heads),
        D=int(dim),
        N_MAX=n_max,
        NUM_BLOCKS=int(num_blocks),
        TILE_N=int(tile_n),
        SCALE=score_scale,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return output


def exact_decode_attention_rows_splitk_triton_forward_out_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    cache_position: torch.Tensor,
    *,
    row_indices: Optional[torch.Tensor | Sequence[int] | Set[int]] = None,
    tile_n: int = 64,
    splits: int = 16,
    workspace: Optional[dict[str, torch.Tensor]] = None,
    partial_num_warps: int = 4,
    partial_num_stages: int = 3,
    merge_num_warps: int = 1,
    merge_num_stages: int = 3,
) -> torch.Tensor:
    """Run split-K exact M=1 decode attention for selected batch rows."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (query, key, value, output, cache_position)):
        raise RuntimeError("exact_decode_attention_rows_splitk_triton_forward_out_bhnd requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query must be [batch, heads, 1, dim], key/value must be [batch, kv_heads, seq, dim]")
    if query.shape[2] != 1:
        raise ValueError("exact refresh only supports query_len == 1")
    if output.shape != (query.shape[0], 1, query.shape[1], query.shape[3]):
        raise ValueError("output must be [batch, 1, heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have matching shape")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query/key/value must have matching batch and dim")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if cache_position.numel() < 1:
        raise ValueError("cache_position must contain at least one element")
    if tile_n <= 0:
        raise ValueError("tile_n must be positive")
    if splits <= 0:
        raise ValueError("splits must be positive")

    batch, heads, _query_len, dim = query.shape
    kv_heads = key.shape[1]
    rows = _rows_tensor(batch=batch, row_indices=row_indices, device=query.device)
    row_count = int(rows.numel())
    if row_count == 0:
        return output
    if workspace is None:
        workspace = make_exact_refresh_splitk_workspace(
            row_count=row_count,
            heads=int(heads),
            chunks=int(splits),
            dim=int(dim),
            device=query.device,
        )
    partial_m = workspace["partial_m"]
    partial_l = workspace["partial_l"]
    partial_num = workspace["partial_num"]
    expected_m = (row_count, int(heads), int(splits))
    expected_num = (row_count, int(heads), int(splits), int(dim))
    if partial_m.shape != expected_m or partial_l.shape != expected_m:
        raise ValueError(f"partial_m/partial_l must have shape {expected_m}")
    if partial_num.shape != expected_num:
        raise ValueError(f"partial_num must have shape {expected_num}")
    if not (partial_m.is_contiguous() and partial_l.is_contiguous() and partial_num.is_contiguous()):
        raise ValueError("split-K partial workspace tensors must be contiguous")

    score_scale = 1.0 / math.sqrt(float(dim))
    n_max = int(key.shape[2])
    num_blocks = triton.cdiv(n_max, int(tile_n))
    chunk_blocks = triton.cdiv(num_blocks, int(splits))
    _exact_refresh_rows_splitk_partial_kernel[(row_count, int(heads), int(splits))](
        query,
        key,
        value,
        rows,
        partial_m,
        partial_l,
        partial_num,
        cache_position,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        H=int(heads),
        H_KV=int(kv_heads),
        GROUP_SIZE=int(heads // kv_heads),
        D=int(dim),
        N_MAX=n_max,
        NUM_BLOCKS=int(num_blocks),
        CHUNKS=int(splits),
        CHUNK_BLOCKS=int(chunk_blocks),
        TILE_N=int(tile_n),
        SCALE=score_scale,
        num_warps=int(partial_num_warps),
        num_stages=int(partial_num_stages),
    )
    _exact_refresh_rows_splitk_merge_kernel[(row_count, int(heads))](
        rows,
        partial_m,
        partial_l,
        partial_num,
        output,
        H=int(heads),
        D=int(dim),
        CHUNKS=int(splits),
        CHUNKS_TILE=_next_power_of_2(int(splits)),
        num_warps=int(merge_num_warps),
        num_stages=int(merge_num_stages),
    )
    return output
