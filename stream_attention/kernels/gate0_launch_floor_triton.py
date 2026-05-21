"""Launch-floor diagnostic kernels for Gate-0 decode backend work.

These kernels are intentionally not runtime candidates.  They isolate the
cost of progressively more realistic selected-head sparse launches so the
benchmarks can decide whether separate StreamAttn kernels can ever compose
with a FlashInfer-quality exact decode path.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _empty_selected_kernel(
        SelectedHeads,
        Scratch,
        SELECTED: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_s = tl.program_id(1)
        head = tl.load(SelectedHeads + off_s)
        tl.store(Scratch + off_b * SELECTED + off_s, head + off_b)

    @triton.jit
    def _q_only_selected_kernel(
        Q,
        SelectedHeads,
        Scratch,
        H_Q: tl.constexpr,
        SELECTED: tl.constexpr,
        D: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_s = tl.program_id(1)
        q_head = tl.load(SelectedHeads + off_s)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H_Q * D + q_head * D + offs_d)
        acc = tl.sum(q.to(tl.float32), axis=0)
        tl.store(Scratch + off_b * SELECTED + off_s, acc)

    @triton.jit
    def _qkv_no_softmax_selected_kernel(
        Q,
        K,
        V,
        SelectedHeads,
        Scratch,
        N: tl.constexpr,
        H_Q: tl.constexpr,
        H_KV: tl.constexpr,
        SELECTED: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        TILE_N: tl.constexpr,
        SINK_BLOCKS: tl.constexpr,
        RECENT_BLOCKS: tl.constexpr,
        RECENT_START: tl.constexpr,
        MIDDLE_SEED_BLOCKS: tl.constexpr,
        BLOCK_ORDER: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        off_b = tl.program_id(0)
        off_s = tl.program_id(1)
        q_head = tl.load(SelectedHeads + off_s)
        kv_head = q_head // GROUP_SIZE
        offs_d = tl.arange(0, D)
        q = tl.load(Q + off_b * H_Q * D + q_head * D + offs_d)
        acc = tl.zeros([], dtype=tl.float32)

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
                + kv_head * D
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            v_tile = tl.load(
                V
                + off_b * N * H_KV * D
                + offs_n[:, None] * H_KV * D
                + kv_head * D
                + offs_d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
            qk = tl.sum(q[None, :].to(tl.float32) * k_tile.to(tl.float32), axis=1) * SCALE
            qk = tl.where(col_mask, qk, 0.0)
            # Include a tiny V contribution so the V load cannot be optimized away.
            v_sum = tl.sum(v_tile.to(tl.float32), axis=1)
            acc += tl.sum(qk + v_sum * 0.000001, axis=0)

        tl.store(Scratch + off_b * SELECTED + off_s, acc)


def _block_order_id(block_order: str) -> int:
    if block_order == "sequential":
        return 0
    if block_order == "recent_first":
        return 1
    if block_order == "sink_recent_first":
        return 2
    raise ValueError("block_order must be 'sequential', 'recent_first', or 'sink_recent_first'")


def _validate_selected_inputs(
    query: torch.Tensor,
    selected_heads: torch.Tensor,
) -> tuple[int, int, int, int]:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not query.is_cuda or not selected_heads.is_cuda:
        raise RuntimeError("launch-floor kernels require CUDA tensors")
    if query.dim() != 4 or query.shape[1] != 1:
        raise ValueError("query must be [batch, 1, q_heads, dim]")
    if selected_heads.dim() != 1 or selected_heads.numel() <= 0:
        raise ValueError("selected_heads must be a non-empty rank-1 tensor")
    if selected_heads.dtype != torch.int32:
        raise ValueError("selected_heads must use torch.int32 dtype")
    if selected_heads.device != query.device:
        raise ValueError("selected_heads must be on the same CUDA device as query")
    batch, _seq, q_heads, dim = query.shape
    return int(batch), int(q_heads), int(dim), int(selected_heads.numel())


def gate0_launch_floor_empty_triton_forward(
    query: torch.Tensor,
    selected_heads: torch.Tensor,
) -> torch.Tensor:
    """Launch a selected-head kernel that only reads head ids and writes scratch."""

    batch, _q_heads, _dim, selected = _validate_selected_inputs(query, selected_heads)
    selected_heads = selected_heads.contiguous()
    scratch = torch.empty(batch, selected, device=query.device, dtype=torch.float32)
    _empty_selected_kernel[(batch, selected)](
        selected_heads,
        scratch,
        SELECTED=selected,
        num_warps=1,
        num_stages=1,
    )
    return scratch


def gate0_launch_floor_q_only_triton_forward(
    query: torch.Tensor,
    selected_heads: torch.Tensor,
    *,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Launch a selected-head kernel that loads/reduces Q only."""

    batch, q_heads, dim, selected = _validate_selected_inputs(query, selected_heads)
    query = query.contiguous()
    selected_heads = selected_heads.contiguous()
    scratch = torch.empty(batch, selected, device=query.device, dtype=torch.float32)
    _q_only_selected_kernel[(batch, selected)](
        query,
        selected_heads,
        scratch,
        H_Q=q_heads,
        SELECTED=selected,
        D=dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return scratch


def gate0_launch_floor_qkv_no_softmax_triton_forward(
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
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Load Q/K/V seed blocks and do QK-style reductions without softmax/PV."""

    batch, q_heads, dim, selected = _validate_selected_inputs(query, selected_heads)
    if not key.is_cuda or not value.is_cuda:
        raise RuntimeError("key/value must be CUDA tensors")
    if key.dim() != 4 or value.dim() != 4:
        raise ValueError("key/value must be [batch, seq, kv_heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if key.shape[0] != batch or key.shape[3] != dim:
        raise ValueError("query/key/value must have matching batch and dim")
    if q_heads % key.shape[2] != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0 or middle_seed_blocks < 0:
        raise ValueError("seed block counts must be non-negative")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    selected_heads = selected_heads.contiguous()
    seq_k = int(key.shape[1])
    kv_heads = int(key.shape[2])
    group_size = q_heads // kv_heads
    num_blocks = triton.cdiv(seq_k, block_size)
    if sink_blocks + recent_blocks > num_blocks:
        raise ValueError("sink_blocks + recent_blocks must not exceed K blocks")
    middle_blocks = num_blocks - sink_blocks - recent_blocks
    if middle_seed_blocks > middle_blocks:
        raise ValueError("middle_seed_blocks must be within the middle block count")

    scratch = torch.empty(batch, selected, device=query.device, dtype=torch.float32)
    _qkv_no_softmax_selected_kernel[(batch, selected)](
        query,
        key,
        value,
        selected_heads,
        scratch,
        N=seq_k,
        H_Q=q_heads,
        H_KV=kv_heads,
        SELECTED=selected,
        GROUP_SIZE=group_size,
        D=dim,
        TILE_N=block_size,
        SCALE=1.0 / math.sqrt(dim),
        SINK_BLOCKS=int(sink_blocks),
        RECENT_BLOCKS=int(recent_blocks),
        RECENT_START=int(num_blocks - recent_blocks),
        MIDDLE_SEED_BLOCKS=int(middle_seed_blocks),
        BLOCK_ORDER=_block_order_id(block_order),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return scratch

