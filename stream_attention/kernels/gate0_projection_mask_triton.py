"""Triton Gate-0 projection threshold mask scan.

This is the mask-output counterpart to ``gate0_projection_scan_triton``. It
evaluates the projection candidate score, applies the skip threshold inside the
kernel, and writes a uint8 skip mask. It is still a candidate-filter prototype,
not a certified Gate-0 runtime.
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
    def _gate0_projection_mask_kernel(
        QProj,
        ProjMin,
        ProjMax,
        Thresholds,
        HasState,
        BlockLogLengths,
        SkipMask,
        M: tl.constexpr,
        H: tl.constexpr,
        RANK: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        SCAN_START: tl.constexpr,
        SCAN_END: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        FILTER_MARGIN: tl.constexpr,
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

        row_base = batch_idx * H * M * NUM_BLOCKS + head_idx * M * NUM_BLOCKS + q_idx * NUM_BLOCKS
        thresholds = tl.load(Thresholds + row_base + offs_blocks, mask=block_mask, other=-float("inf")).to(tl.float32)
        has_state = tl.load(HasState + row_base + offs_blocks, mask=block_mask, other=0) != 0
        log_lengths = tl.load(BlockLogLengths + offs_blocks, mask=block_mask, other=float("inf")).to(tl.float32)

        skip = has_state & (scores + log_lengths <= thresholds + FILTER_MARGIN)
        tl.store(SkipMask + row_base + offs_blocks, skip.to(tl.uint8), mask=block_mask)


    @triton.jit
    def _gate0_projection_mask_static_threshold_kernel(
        QProj,
        ProjMin,
        ProjMax,
        StaticThresholds,
        BlockLogLengths,
        SkipMask,
        M: tl.constexpr,
        H: tl.constexpr,
        RANK: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        SCAN_START: tl.constexpr,
        SCAN_END: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        FILTER_MARGIN: tl.constexpr,
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

        threshold = tl.load(StaticThresholds + batch_idx * H * M + head_idx * M + q_idx).to(tl.float32)
        log_lengths = tl.load(BlockLogLengths + offs_blocks, mask=block_mask, other=float("inf")).to(tl.float32)

        row_base = batch_idx * H * M * NUM_BLOCKS + head_idx * M * NUM_BLOCKS + q_idx * NUM_BLOCKS
        skip = block_mask & (scores + log_lengths <= threshold + FILTER_MARGIN)
        tl.store(SkipMask + row_base + offs_blocks, skip.to(tl.uint8), mask=block_mask)


def gate0_projection_mask_triton(
    q_proj: torch.Tensor,
    proj_min: torch.Tensor,
    proj_max: torch.Tensor,
    thresholds: torch.Tensor,
    has_state: torch.Tensor,
    block_log_lengths: torch.Tensor,
    *,
    dim: int,
    filter_margin: float,
    scan_start: int = 0,
    scan_end: Optional[int] = None,
    blocks_per_program: int = 32,
    output: Optional[torch.Tensor] = None,
    clear_output: bool = True,
) -> torch.Tensor:
    """Compute a projection candidate skip mask with Triton.

    Args:
        q_proj: ``[batch, heads, query_len, rank]`` projected query tensor.
        proj_min: ``[batch, heads, blocks, rank]`` per-block projection minima.
        proj_max: ``[batch, heads, blocks, rank]`` per-block projection maxima.
        thresholds: ``[batch, heads, query_len, blocks]`` skip thresholds.
        has_state: ``[batch, heads, query_len, blocks]`` valid-state mask.
        block_log_lengths: ``[blocks]`` log of valid tokens per block.
        dim: Original attention head dimension used for score scaling.
        filter_margin: Additive margin applied to the skip threshold.
        scan_start: First block index to scan.
        scan_end: Exclusive end block index. Defaults to all blocks.
        blocks_per_program: Number of metadata blocks scanned per program.
        output: Optional reusable uint8 ``[batch, heads, query_len, blocks]`` tensor.
        clear_output: Fill the output with zeros before scanning. Disable only
            when the caller provides a reusable buffer whose unscanned blocks are
            already initialized to zero.

    Returns:
        ``uint8`` skip mask in ``[batch, heads, query_len, blocks]`` layout.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (q_proj.is_cuda and proj_min.is_cuda and proj_max.is_cuda):
        raise RuntimeError("gate0_projection_mask_triton requires CUDA tensors")
    if not (thresholds.is_cuda and has_state.is_cuda and block_log_lengths.is_cuda):
        raise RuntimeError("threshold, state, and block length tensors must be CUDA tensors")
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
    thresholds = thresholds.contiguous()
    has_state = has_state.contiguous()
    block_log_lengths = block_log_lengths.contiguous()

    batch, heads, query_len, rank = q_proj.shape
    if proj_min.shape[0] != batch or proj_min.shape[1] != heads or proj_min.shape[3] != rank:
        raise ValueError("projection metadata shape does not match q_proj")
    num_blocks = proj_min.shape[2]
    expected = (batch, heads, query_len, num_blocks)
    if thresholds.shape != expected or has_state.shape != expected:
        raise ValueError("thresholds and has_state must have shape [batch, heads, query_len, blocks]")
    if block_log_lengths.shape != (num_blocks,):
        raise ValueError("block_log_lengths must have shape [blocks]")
    if scan_end is None:
        scan_end = num_blocks
    if scan_start < 0 or scan_end < scan_start or scan_end > num_blocks:
        raise ValueError("invalid scan range")

    created_output = output is None
    if output is None:
        output = torch.empty(expected, device=q_proj.device, dtype=torch.uint8)
    elif output.shape != expected:
        raise ValueError("output must have shape [batch, heads, query_len, blocks]")
    elif output.dtype != torch.uint8:
        raise ValueError("output must have dtype torch.uint8")
    elif not output.is_cuda:
        raise RuntimeError("output must be a CUDA tensor")
    elif not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if clear_output or created_output:
        output.zero_()

    scan_blocks = scan_end - scan_start
    if scan_blocks == 0:
        return output

    score_scale = (float(dim) / float(rank)) / (float(dim) ** 0.5)
    grid = (triton.cdiv(scan_blocks, blocks_per_program), query_len * heads, batch)
    _gate0_projection_mask_kernel[grid](
        q_proj,
        proj_min,
        proj_max,
        thresholds,
        has_state,
        block_log_lengths,
        output,
        M=query_len,
        H=heads,
        RANK=rank,
        NUM_BLOCKS=num_blocks,
        SCAN_START=scan_start,
        SCAN_END=scan_end,
        BLOCKS_PER_PROGRAM=blocks_per_program,
        SCORE_SCALE=score_scale,
        FILTER_MARGIN=float(filter_margin),
        num_warps=4,
    )
    return output


def gate0_projection_mask_static_threshold_triton(
    q_proj: torch.Tensor,
    proj_min: torch.Tensor,
    proj_max: torch.Tensor,
    static_thresholds: torch.Tensor,
    block_log_lengths: torch.Tensor,
    *,
    dim: int,
    filter_margin: float,
    scan_start: int = 0,
    scan_end: Optional[int] = None,
    blocks_per_program: int = 32,
    output: Optional[torch.Tensor] = None,
    clear_output: bool = True,
) -> torch.Tensor:
    """Compute a projection skip mask with one threshold per row/head."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (q_proj.is_cuda and proj_min.is_cuda and proj_max.is_cuda):
        raise RuntimeError("gate0_projection_mask_static_threshold_triton requires CUDA tensors")
    if not (static_thresholds.is_cuda and block_log_lengths.is_cuda):
        raise RuntimeError("static threshold and block length tensors must be CUDA tensors")
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
    static_thresholds = static_thresholds.contiguous()
    block_log_lengths = block_log_lengths.contiguous()

    batch, heads, query_len, rank = q_proj.shape
    if proj_min.shape[0] != batch or proj_min.shape[1] != heads or proj_min.shape[3] != rank:
        raise ValueError("projection metadata shape does not match q_proj")
    num_blocks = proj_min.shape[2]
    if static_thresholds.shape != (batch, heads, query_len):
        raise ValueError("static_thresholds must have shape [batch, heads, query_len]")
    if block_log_lengths.shape != (num_blocks,):
        raise ValueError("block_log_lengths must have shape [blocks]")
    if scan_end is None:
        scan_end = num_blocks
    if scan_start < 0 or scan_end < scan_start or scan_end > num_blocks:
        raise ValueError("invalid scan range")

    expected = (batch, heads, query_len, num_blocks)
    created_output = output is None
    if output is None:
        output = torch.empty(expected, device=q_proj.device, dtype=torch.uint8)
    elif output.shape != expected:
        raise ValueError("output must have shape [batch, heads, query_len, blocks]")
    elif output.dtype != torch.uint8:
        raise ValueError("output must have dtype torch.uint8")
    elif not output.is_cuda:
        raise RuntimeError("output must be a CUDA tensor")
    elif not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if clear_output or created_output:
        output.zero_()

    scan_blocks = scan_end - scan_start
    if scan_blocks == 0:
        return output

    score_scale = (float(dim) / float(rank)) / (float(dim) ** 0.5)
    grid = (triton.cdiv(scan_blocks, blocks_per_program), query_len * heads, batch)
    _gate0_projection_mask_static_threshold_kernel[grid](
        q_proj,
        proj_min,
        proj_max,
        static_thresholds,
        block_log_lengths,
        output,
        M=query_len,
        H=heads,
        RANK=rank,
        NUM_BLOCKS=num_blocks,
        SCAN_START=scan_start,
        SCAN_END=scan_end,
        BLOCKS_PER_PROGRAM=blocks_per_program,
        SCORE_SCALE=score_scale,
        FILTER_MARGIN=float(filter_margin),
        num_warps=4,
    )
    return output
