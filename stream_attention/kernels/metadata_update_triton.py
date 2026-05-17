"""Triton incremental metadata updates for KV-cache decode."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _update_value_norm_bounds_kernel(
        Bounds,
        NewV,
        H: tl.constexpr,
        D: tl.constexpr,
        NEW_SEQ: tl.constexpr,
        TOTAL_SEQ_LEN: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        FIRST_BLOCK: tl.constexpr,
        START_POS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        touched_block_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)

        block_idx = FIRST_BLOCK + touched_block_idx
        block_start = block_idx * BLOCK_N
        block_end = tl.minimum(block_start + BLOCK_N, TOTAL_SEQ_LEN)
        local_start = tl.maximum(0, block_start - START_POS)
        local_end = tl.minimum(NEW_SEQ, block_end - START_POS)
        local_count = tl.maximum(0, local_end - local_start)

        offs_n = local_start + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        token_mask = tl.arange(0, BLOCK_N) < local_count
        ptrs = (
            NewV
            + batch_idx * NEW_SEQ * H * D
            + offs_n[:, None] * H * D
            + head_idx * D
            + offs_d[None, :]
        )
        values = tl.load(ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        norms_sq = tl.sum(values * values, axis=1)
        norms_sq = tl.where(token_mask, norms_sq, 0.0)
        new_bound = tl.sqrt(tl.max(norms_sq, axis=0))

        bounds_ptr = Bounds + batch_idx * H * NUM_BLOCKS + head_idx * NUM_BLOCKS + block_idx
        old_bound = tl.load(bounds_ptr).to(tl.float32)
        tl.store(bounds_ptr, tl.maximum(old_bound, new_bound))


def update_value_norm_bounds_triton_(
    value_norm_bounds: torch.Tensor,
    new_value: torch.Tensor,
    *,
    start_pos: int,
    total_seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Update max ``||V||`` bounds for newly appended KV-cache values in-place.

    ``value_norm_bounds`` uses ``[batch, heads, blocks]`` layout, while
    ``new_value`` uses ``[batch, new_seq, heads, dim]`` layout. Only blocks
    touched by ``[start_pos, start_pos + new_seq)`` are launched.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (value_norm_bounds.is_cuda and new_value.is_cuda):
        raise RuntimeError("update_value_norm_bounds_triton_ requires CUDA tensors")
    if value_norm_bounds.dim() != 3:
        raise ValueError("value_norm_bounds must have shape [batch, heads, blocks]")
    if new_value.dim() != 4:
        raise ValueError("new_value must have shape [batch, new_seq, heads, dim]")
    if not value_norm_bounds.is_contiguous():
        raise ValueError("value_norm_bounds must be contiguous for in-place Triton update")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if total_seq_len < 0:
        raise ValueError("total_seq_len must be non-negative")

    batch, new_seq, heads, dim = new_value.shape
    if new_seq == 0:
        return value_norm_bounds
    if start_pos + new_seq > total_seq_len:
        raise ValueError("new_value extends beyond total_seq_len")

    expected_blocks = (total_seq_len + block_size - 1) // block_size
    if tuple(value_norm_bounds.shape) != (batch, heads, expected_blocks):
        raise ValueError(
            "value_norm_bounds shape does not match new_value, total_seq_len, and block_size"
        )

    new_value = new_value.contiguous()
    first_block = start_pos // block_size
    last_block = (start_pos + new_seq - 1) // block_size
    touched_blocks = last_block - first_block + 1
    _update_value_norm_bounds_kernel[(touched_blocks, heads, batch)](
        value_norm_bounds,
        new_value,
        H=heads,
        D=dim,
        NEW_SEQ=new_seq,
        TOTAL_SEQ_LEN=total_seq_len,
        NUM_BLOCKS=expected_blocks,
        FIRST_BLOCK=first_block,
        START_POS=start_pos,
        BLOCK_N=block_size,
        num_warps=4,
    )
    return value_norm_bounds
