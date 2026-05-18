"""Triton Gate-0 centroid/radius summary scan.

This kernel is intentionally narrow: it scans cached centroid/radius summaries
for contiguous-KV decode and writes per-block score upper bounds. It does not
build worklists or apply skip predicates. A second narrow path handles exactly
two split outliers per block for bound-tightness experiments.
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


    @triton.jit
    def _gate0_summary_scan_outlier2_kernel(
        Q,
        Centroids,
        Radii,
        OutlierKeys,
        OutlierMask,
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
        dot_centroid = tl.sum(centroids * q[None, :], axis=1)

        radius_ptrs = (
            Radii
            + batch_idx * H * NUM_BLOCKS
            + head_idx * NUM_BLOCKS
            + offs_blocks
        )
        radius = tl.load(radius_ptrs, mask=block_mask, other=0.0).to(tl.float32)
        residual = (dot_centroid + q_norm * radius) * SCALE

        outlier_base = (
            OutlierKeys
            + batch_idx * H * NUM_BLOCKS * 2 * D
            + head_idx * NUM_BLOCKS * 2 * D
            + offs_blocks[:, None] * 2 * D
            + offs_d[None, :]
        )
        outlier0 = tl.load(outlier_base, mask=block_mask[:, None], other=0.0).to(tl.float32)
        outlier1 = tl.load(outlier_base + D, mask=block_mask[:, None], other=0.0).to(tl.float32)
        dot0 = tl.sum(outlier0 * q[None, :], axis=1) * SCALE
        dot1 = tl.sum(outlier1 * q[None, :], axis=1) * SCALE

        mask_base = (
            OutlierMask
            + batch_idx * H * NUM_BLOCKS * 2
            + head_idx * NUM_BLOCKS * 2
            + offs_blocks * 2
        )
        mask0 = tl.load(mask_base, mask=block_mask, other=0)
        mask1 = tl.load(mask_base + 1, mask=block_mask, other=0)
        neg_inf = -float("inf")
        dot0 = tl.where(mask0 & block_mask, dot0, neg_inf)
        dot1 = tl.where(mask1 & block_mask, dot1, neg_inf)
        upper = tl.maximum(residual, tl.maximum(dot0, dot1))

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
    outlier_keys: Optional[torch.Tensor] = None,
    outlier_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    blocks_per_program: int = 32,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute centroid/radius upper bounds with Triton.

    Args:
        query: ``[batch, query_len, heads, dim]`` tensor.
        centroids: ``[batch, heads, blocks, dim]`` tensor.
        radii: ``[batch, heads, blocks]`` tensor.
        outlier_keys: Optional ``[batch, heads, blocks, 2, dim]`` tensor.
        outlier_mask: Optional ``[batch, heads, blocks, 2]`` tensor.
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
    if (outlier_keys is None) != (outlier_mask is None):
        raise ValueError("outlier_keys and outlier_mask must be provided together")
    use_outlier2 = outlier_keys is not None
    if use_outlier2:
        if not (outlier_keys.is_cuda and outlier_mask.is_cuda):
            raise RuntimeError("outlier tensors must be CUDA tensors")
        if outlier_keys.shape != (batch, heads, num_blocks, 2, dim):
            raise ValueError("outlier_keys must have shape [batch, heads, blocks, 2, dim]")
        if outlier_mask.shape != (batch, heads, num_blocks, 2):
            raise ValueError("outlier_mask must have shape [batch, heads, blocks, 2]")
        outlier_keys = outlier_keys.contiguous()
        outlier_mask = outlier_mask.contiguous()

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
    if use_outlier2:
        _gate0_summary_scan_outlier2_kernel[grid](
            query,
            centroids,
            radii,
            outlier_keys,
            outlier_mask,
            output,
            M=query_len,
            H=heads,
            D=dim,
            NUM_BLOCKS=num_blocks,
            BLOCKS_PER_PROGRAM=blocks_per_program,
            SCALE=scale_value,
            num_warps=4,
        )
    else:
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
