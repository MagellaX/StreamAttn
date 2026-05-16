"""Triton metadata builders for StreamAttn runtime routing."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _value_norm_bounds_kernel(
        V,
        Out,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        block_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)

        offs_n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        mask = offs_n[:, None] < N
        ptrs = (
            V
            + batch_idx * N * H * D
            + offs_n[:, None] * H * D
            + head_idx * D
            + offs_d[None, :]
        )
        values = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        norms_sq = tl.sum(values * values, axis=1)
        norms_sq = tl.where(offs_n < N, norms_sq, 0.0)
        max_norm_sq = tl.max(norms_sq, axis=0)
        bound = tl.sqrt(max_norm_sq)
        tl.store(
            Out + batch_idx * H * NUM_BLOCKS + head_idx * NUM_BLOCKS + block_idx,
            bound,
        )


def build_value_norm_bounds_triton(
    value: torch.Tensor,
    *,
    block_size: int,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build max ||V|| per ``[batch, head, K-block]`` with Triton."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not value.is_cuda:
        raise RuntimeError("build_value_norm_bounds_triton requires a CUDA tensor")
    if value.dim() != 4:
        raise ValueError("value must have shape [batch, seq, heads, dim]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    value = value.contiguous()
    batch, seq_len, heads, dim = value.shape
    num_blocks = triton.cdiv(seq_len, block_size)
    if output is None:
        output = torch.empty(
            batch,
            heads,
            num_blocks,
            device=value.device,
            dtype=torch.float32,
        )
    elif output.shape != (batch, heads, num_blocks):
        raise ValueError("output must have shape [batch, heads, blocks]")

    _value_norm_bounds_kernel[(num_blocks, heads, batch)](
        value,
        output,
        N=seq_len,
        H=heads,
        D=dim,
        NUM_BLOCKS=num_blocks,
        BLOCK_N=block_size,
        num_warps=4,
    )
    return output
