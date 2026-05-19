"""Triton Gate-0 projection threshold scan with packed bitmask output.

The kernel evaluates the same calibrated projection candidate score as the
uint8 mask kernel, but writes one int32 word per 32 KV blocks. Bit ``i`` in word
``w`` corresponds to absolute block ``w * 32 + i``.
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
    def _gate0_projection_bitmask_kernel(
        QProj,
        ProjMin,
        ProjMax,
        Thresholds,
        HasState,
        BlockLogLengths,
        PackedMask,
        M: tl.constexpr,
        H: tl.constexpr,
        RANK: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        NUM_WORDS: tl.constexpr,
        START_WORD: tl.constexpr,
        SCAN_START: tl.constexpr,
        SCAN_END: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        FILTER_MARGIN: tl.constexpr,
    ):
        word_idx = START_WORD + tl.program_id(0)
        qh_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)

        q_idx = qh_idx % M
        head_idx = qh_idx // M

        offs_bits = tl.arange(0, 32)
        offs_blocks = word_idx * 32 + offs_bits
        block_mask = (offs_blocks >= SCAN_START) & (offs_blocks < SCAN_END) & (offs_blocks < NUM_BLOCKS)
        offs_r = tl.arange(0, RANK)

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
        mins = tl.load(ProjMin + metadata_base, mask=block_mask[:, None], other=0.0).to(tl.float32)
        maxs = tl.load(ProjMax + metadata_base, mask=block_mask[:, None], other=0.0).to(tl.float32)
        chosen = tl.where(q_proj[None, :] >= 0.0, maxs, mins)
        scores = tl.sum(q_proj[None, :] * chosen, axis=1) * SCORE_SCALE

        row_base_blocks = batch_idx * H * M * NUM_BLOCKS + head_idx * M * NUM_BLOCKS + q_idx * NUM_BLOCKS
        thresholds = tl.load(Thresholds + row_base_blocks + offs_blocks, mask=block_mask, other=-float("inf")).to(
            tl.float32
        )
        has_state = tl.load(HasState + row_base_blocks + offs_blocks, mask=block_mask, other=0) != 0
        log_lengths = tl.load(BlockLogLengths + offs_blocks, mask=block_mask, other=float("inf")).to(tl.float32)

        skip = block_mask & has_state & (scores + log_lengths <= thresholds + FILTER_MARGIN)
        bit_values = (skip.to(tl.int32) << offs_bits).to(tl.int32)
        packed = tl.sum(bit_values, axis=0)

        row_base_words = batch_idx * H * M * NUM_WORDS + head_idx * M * NUM_WORDS + q_idx * NUM_WORDS
        tl.store(PackedMask + row_base_words + word_idx, packed)


def gate0_projection_bitmask_triton(
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
    output: Optional[torch.Tensor] = None,
    clear_output: bool = True,
) -> torch.Tensor:
    """Compute a packed projection candidate skip bitmask with Triton."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (q_proj.is_cuda and proj_min.is_cuda and proj_max.is_cuda):
        raise RuntimeError("gate0_projection_bitmask_triton requires CUDA tensors")
    if not (thresholds.is_cuda and has_state.is_cuda and block_log_lengths.is_cuda):
        raise RuntimeError("threshold, state, and block length tensors must be CUDA tensors")
    if q_proj.dim() != 4:
        raise ValueError("q_proj must have shape [batch, heads, query_len, rank]")
    if proj_min.dim() != 4 or proj_max.dim() != 4:
        raise ValueError("projection metadata must have shape [batch, heads, blocks, rank]")
    if proj_min.shape != proj_max.shape:
        raise ValueError("proj_min and proj_max must have matching shapes")

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
    expected_blocks = (batch, heads, query_len, num_blocks)
    if thresholds.shape != expected_blocks or has_state.shape != expected_blocks:
        raise ValueError("thresholds and has_state must have shape [batch, heads, query_len, blocks]")
    if block_log_lengths.shape != (num_blocks,):
        raise ValueError("block_log_lengths must have shape [blocks]")
    if scan_end is None:
        scan_end = num_blocks
    if scan_start < 0 or scan_end < scan_start or scan_end > num_blocks:
        raise ValueError("invalid scan range")

    num_words = (num_blocks + 31) // 32
    expected_words = (batch, heads, query_len, num_words)
    created_output = output is None
    if output is None:
        output = torch.empty(expected_words, device=q_proj.device, dtype=torch.int32)
    elif output.shape != expected_words:
        raise ValueError("output must have shape [batch, heads, query_len, ceil(blocks / 32)]")
    elif output.dtype != torch.int32:
        raise ValueError("output must have dtype torch.int32")
    elif not output.is_cuda:
        raise RuntimeError("output must be a CUDA tensor")
    elif not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if clear_output or created_output:
        output.zero_()

    if scan_end == scan_start:
        return output

    score_scale = (float(dim) / float(rank)) / (float(dim) ** 0.5)
    start_word = scan_start // 32
    end_word = (scan_end + 31) // 32
    grid = (end_word - start_word, query_len * heads, batch)
    _gate0_projection_bitmask_kernel[grid](
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
        NUM_WORDS=num_words,
        START_WORD=start_word,
        SCAN_START=scan_start,
        SCAN_END=scan_end,
        SCORE_SCALE=score_scale,
        FILTER_MARGIN=float(filter_margin),
        num_warps=4,
    )
    return output
