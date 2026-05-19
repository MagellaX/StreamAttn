"""Triton Gate-0 projection candidate scan.

This scan consumes precomputed projection metadata with shape
``[batch, heads, blocks, rank]`` and projected queries with shape
``[batch, heads, query_len, rank]``. It is a candidate-filter prototype, not a
certified bound and not a Gate-0 runtime.
"""

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
    def _gate0_projection_scan_kernel(
        QProj,
        ProjMin,
        ProjMax,
        Out,
        M: tl.constexpr,
        H: tl.constexpr,
        RANK: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        SCAN_START: tl.constexpr,
        SCAN_END: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
    ):
        block_group = tl.program_id(0)
        qh_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)

        q_idx = qh_idx % M
        head_idx = qh_idx // M

        offs_blocks = SCAN_START + block_group * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
        offs_r = tl.arange(0, RANK)
        block_mask = offs_blocks < SCAN_END

        q_ptrs = (
            QProj
            + batch_idx * H * M * RANK
            + head_idx * M * RANK
            + q_idx * RANK
            + offs_r
        )
        q_proj = tl.load(q_ptrs).to(tl.float32)

        metadata_base = (
            batch_idx * H * NUM_BLOCKS * RANK
            + head_idx * NUM_BLOCKS * RANK
            + offs_blocks[:, None] * RANK
            + offs_r[None, :]
        )
        chosen_ptrs = tl.where(q_proj[None, :] >= 0.0, ProjMax + metadata_base, ProjMin + metadata_base)
        chosen = tl.load(chosen_ptrs, mask=block_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q_proj[None, :] * chosen, axis=1) * SCORE_SCALE

        out_ptrs = (
            Out
            + batch_idx * H * M * NUM_BLOCKS
            + head_idx * M * NUM_BLOCKS
            + q_idx * NUM_BLOCKS
            + offs_blocks
        )
        tl.store(out_ptrs, scores, mask=block_mask)


def gate0_projection_scan_triton(
    q_proj: torch.Tensor,
    proj_min: torch.Tensor,
    proj_max: torch.Tensor,
    *,
    dim: int,
    scan_start: int = 0,
    scan_end: Optional[int] = None,
    blocks_per_program: int = 32,
    output: Optional[torch.Tensor] = None,
    clear_output: bool = True,
) -> torch.Tensor:
    """Compute projection candidate scores with Triton.

    Args:
        q_proj: ``[batch, heads, query_len, rank]`` projected query tensor.
        proj_min: ``[batch, heads, blocks, rank]`` per-block projection minima.
        proj_max: ``[batch, heads, blocks, rank]`` per-block projection maxima.
        dim: Original attention head dimension used for score scaling.
        scan_start: First block index to scan.
        scan_end: Exclusive end block index. Defaults to all blocks.
        blocks_per_program: Number of metadata blocks scanned per program.
        output: Optional reusable ``[batch, heads, query_len, blocks]`` tensor.
        clear_output: Fill the output with ``+inf`` before scanning. Disable
            only when the caller provides a reusable buffer whose unscanned
            blocks are already initialized.

    Returns:
        Candidate scores in ``[batch, heads, query_len, blocks]`` layout.
        Unscanned blocks are ``+inf``.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (q_proj.is_cuda and proj_min.is_cuda and proj_max.is_cuda):
        raise RuntimeError("gate0_projection_scan_triton requires CUDA tensors")
    if q_proj.dim() != 4:
        raise ValueError("q_proj must have shape [batch, heads, query_len, rank]")
    if proj_min.dim() != 4 or proj_max.dim() != 4:
        raise ValueError("projection metadata must have shape [batch, heads, blocks, rank]")
    if proj_min.shape != proj_max.shape:
        raise ValueError("proj_min and proj_max must have matching shapes")
    if blocks_per_program <= 0:
        raise ValueError("blocks_per_program must be positive")

    q_proj = q_proj.contiguous()
    proj_min = proj_min.contiguous()
    proj_max = proj_max.contiguous()
    batch, heads, query_len, rank = q_proj.shape
    if proj_min.shape[0] != batch or proj_min.shape[1] != heads or proj_min.shape[3] != rank:
        raise ValueError("projection metadata shape does not match q_proj")
    num_blocks = proj_min.shape[2]
    if scan_end is None:
        scan_end = num_blocks
    if scan_start < 0 or scan_end < scan_start or scan_end > num_blocks:
        raise ValueError("invalid scan range")

    if output is None:
        output = torch.empty(
            batch,
            heads,
            query_len,
            num_blocks,
            device=q_proj.device,
            dtype=torch.float32,
        )
    elif output.shape != (batch, heads, query_len, num_blocks):
        raise ValueError("output must have shape [batch, heads, query_len, blocks]")
    elif not output.is_cuda:
        raise RuntimeError("output must be a CUDA tensor")
    elif not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if clear_output:
        output.fill_(float("inf"))

    scan_blocks = scan_end - scan_start
    if scan_blocks == 0:
        return output

    score_scale = (float(dim) / float(rank)) / (float(dim) ** 0.5)
    grid = (triton.cdiv(scan_blocks, blocks_per_program), query_len * heads, batch)
    _gate0_projection_scan_kernel[grid](
        q_proj,
        proj_min,
        proj_max,
        output,
        M=query_len,
        H=heads,
        RANK=rank,
        NUM_BLOCKS=num_blocks,
        SCAN_START=scan_start,
        SCAN_END=scan_end,
        BLOCKS_PER_PROGRAM=blocks_per_program,
        SCORE_SCALE=score_scale,
        num_warps=4,
    )
    return output
