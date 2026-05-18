"""Triton Gate-0 centroid/radius summary scan.

This kernel is intentionally narrow: it scans cached centroid/radius summaries
for contiguous-KV decode and writes per-block score upper bounds. It does not
build worklists, apply skip predicates, or handle outlier summaries yet.
"""

from __future__ import annotations

import math
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
    def _gate0_summary_scan_kernel(
        Q,
        Centroids,
        Radii,
        Out,
        M: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        block_group = tl.program_id(0)
        qh_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)

        q_idx = qh_idx % M
        head_idx = qh_idx // M

        offs_blocks = block_group * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
        offs_d = tl.arange(0, D)
        block_mask = offs_blocks < NUM_BLOCKS

        q_ptrs = (
            Q
            + batch_idx * M * H * D
            + q_idx * H * D
            + head_idx * D
            + offs_d
        )
        q = tl.load(q_ptrs).to(tl.float32)
        q_norm = tl.sqrt(tl.sum(q * q, axis=0))

        centroid_ptrs = (
            Centroids
            + batch_idx * H * NUM_BLOCKS * D
            + head_idx * NUM_BLOCKS * D
            + offs_blocks[:, None] * D
            + offs_d[None, :]
        )
        centroids = tl.load(centroid_ptrs, mask=block_mask[:, None], other=0.0).to(tl.float32)
        dot = tl.sum(centroids * q[None, :], axis=1)

        radius_ptrs = (
            Radii
            + batch_idx * H * NUM_BLOCKS
            + head_idx * NUM_BLOCKS
            + offs_blocks
        )
        radius = tl.load(radius_ptrs, mask=block_mask, other=0.0).to(tl.float32)
        upper = (dot + q_norm * radius) * SCALE

        out_ptrs = (
            Out
            + batch_idx * H * M * NUM_BLOCKS
            + head_idx * M * NUM_BLOCKS
            + q_idx * NUM_BLOCKS
            + offs_blocks
        )
        tl.store(out_ptrs, upper, mask=block_mask)


def gate0_summary_scan_triton(
    query: torch.Tensor,
    centroids: torch.Tensor,
    radii: torch.Tensor,
    *,
    scale: Optional[float] = None,
    blocks_per_program: int = 32,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute centroid/radius upper bounds with Triton.

    Args:
        query: ``[batch, query_len, heads, dim]`` tensor.
        centroids: ``[batch, heads, blocks, dim]`` tensor.
        radii: ``[batch, heads, blocks]`` tensor.
        scale: Attention scale. Defaults to ``1 / sqrt(dim)``.
        blocks_per_program: Number of summary blocks scanned per program.
        output: Optional preallocated ``[batch, heads, query_len, blocks]`` tensor.

    Returns:
        Per-block upper bounds in ``[batch, heads, query_len, blocks]`` layout.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (query.is_cuda and centroids.is_cuda and radii.is_cuda):
        raise RuntimeError("gate0_summary_scan_triton requires CUDA tensors")
    if query.dim() != 4:
        raise ValueError("query must have shape [batch, query_len, heads, dim]")
    if centroids.dim() != 4:
        raise ValueError("centroids must have shape [batch, heads, blocks, dim]")
    if radii.dim() != 3:
        raise ValueError("radii must have shape [batch, heads, blocks]")
    if blocks_per_program <= 0:
        raise ValueError("blocks_per_program must be positive")

    query = query.contiguous()
    centroids = centroids.contiguous()
    radii = radii.contiguous()
    batch, query_len, heads, dim = query.shape
    if centroids.shape[0] != batch or centroids.shape[1] != heads or centroids.shape[3] != dim:
        raise ValueError("centroids shape does not match query")
    num_blocks = centroids.shape[2]
    if radii.shape != (batch, heads, num_blocks):
        raise ValueError("radii shape does not match centroids")

    if output is None:
        output = torch.empty(
            batch,
            heads,
            query_len,
            num_blocks,
            device=query.device,
            dtype=torch.float32,
        )
    elif output.shape != (batch, heads, query_len, num_blocks):
        raise ValueError("output must have shape [batch, heads, query_len, blocks]")
    elif not output.is_cuda:
        raise RuntimeError("output must be a CUDA tensor")

    scale_value = (1.0 / math.sqrt(dim)) if scale is None else float(scale)
    grid = (triton.cdiv(num_blocks, blocks_per_program), query_len * heads, batch)
    _gate0_summary_scan_kernel[grid](
        query,
        centroids,
        radii,
        output,
        M=query_len,
        H=heads,
        D=dim,
        NUM_BLOCKS=num_blocks,
        BLOCKS_PER_PROGRAM=blocks_per_program,
        SCALE=scale_value,
        num_warps=4,
    )
    return output
