"""Compact exact-repair kernels for Gate-0 KV-group speculation.

This is a research kernel for tiny repair sets.  It computes exact attention
for selected repair rows without padding them to a 16-row WGMMA tile.  The
kernel is split-K over KV chunks, then merges online-softmax partial states.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_repair_chunk_kernel(
    Q,
    K,
    V,
    PartialNum,
    PartialM,
    PartialL,
    N: tl.constexpr,
    D: tl.constexpr,
    TOKENS_PER_CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n = chunk * TOKENS_PER_CHUNK + offs_n
    n_mask = n < tl.minimum((chunk + 1) * TOKENS_PER_CHUNK, N)
    d_mask = offs_d < D

    q = tl.load(Q + row * D + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    k = tl.load(K + n[:, None] * D + offs_d[None, :], mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(
        tl.float32
    )
    scores = tl.sum(k * q[None, :], axis=1) * (1.0 / tl.sqrt(D + 0.0))
    scores = tl.where(n_mask, scores, -float("inf"))
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    l = tl.sum(p, axis=0)
    v = tl.load(V + n[:, None] * D + offs_d[None, :], mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(
        tl.float32
    )
    num = tl.sum(p[:, None] * v, axis=0)

    partial_base = (row * tl.num_programs(1) + chunk) * D
    tl.store(PartialNum + partial_base + offs_d, num, mask=d_mask)
    tl.store(PartialM + row * tl.num_programs(1) + chunk, m)
    tl.store(PartialL + row * tl.num_programs(1) + chunk, l)


@triton.jit
def _compact_repair_merge_kernel(
    PartialNum,
    PartialM,
    PartialL,
    Out,
    NUM_CHUNKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d_block = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C)
    offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    c_mask = offs_c < NUM_CHUNKS
    d_mask = offs_d < D

    m_vals = tl.load(PartialM + row * NUM_CHUNKS + offs_c, mask=c_mask, other=-float("inf")).to(tl.float32)
    m = tl.max(m_vals, axis=0)
    l_vals = tl.load(PartialL + row * NUM_CHUNKS + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    weights = tl.exp(m_vals - m) * l_vals
    den = tl.sum(weights, axis=0)
    vals = tl.load(
        PartialNum + (row * NUM_CHUNKS + offs_c[:, None]) * D + offs_d[None, :],
        mask=c_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    num = tl.sum(vals * tl.exp(m_vals[:, None] - m), axis=0)
    out = num / den
    tl.store(Out + row * D + offs_d, out, mask=d_mask)


def compact_repair_splitk_triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_chunks: int = 256,
    block_d: int = 32,
) -> torch.Tensor:
    """Exact attention for repair rows.

    Args:
        q: [1, R, D], contiguous selected repair Q rows.
        k/v: [1, N, 1, D], contiguous true-GQA KV group.
    """

    if q.dim() != 3 or k.dim() != 4 or v.shape != k.shape:
        raise ValueError("expected q [1,R,D] and k/v [1,N,1,D]")
    if q.shape[0] != 1 or k.shape[0] != 1 or k.shape[2] != 1:
        raise ValueError("compact repair v0 supports B=1 and one KV head")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q/k/v must be contiguous")
    _, repair_rows, dim = q.shape
    _, kv_len, _, kv_dim = k.shape
    if dim != kv_dim:
        raise ValueError("Q/K head_dim mismatch")
    if dim not in (64, 128):
        raise ValueError("compact repair v0 supports D=64 or D=128")
    if repair_rows <= 0:
        return torch.empty_like(q)
    if kv_len % num_chunks != 0:
        raise ValueError("num_chunks must divide kv_len")

    tokens_per_chunk = kv_len // num_chunks
    block_n = triton.next_power_of_2(tokens_per_chunk)
    if block_n > 256:
        raise ValueError("tokens per chunk too large for compact repair v0")
    partial_num = torch.empty((repair_rows, num_chunks, dim), device=q.device, dtype=torch.float32)
    partial_m = torch.empty((repair_rows, num_chunks), device=q.device, dtype=torch.float32)
    partial_l = torch.empty((repair_rows, num_chunks), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)

    _compact_repair_chunk_kernel[(repair_rows, num_chunks)](
        q,
        k,
        v,
        partial_num,
        partial_m,
        partial_l,
        kv_len,
        dim,
        tokens_per_chunk,
        BLOCK_N=block_n,
        BLOCK_D=dim,
        num_warps=4,
    )
    block_c = triton.next_power_of_2(num_chunks)
    _compact_repair_merge_kernel[(repair_rows, triton.cdiv(dim, block_d))](
        partial_num,
        partial_m,
        partial_l,
        out,
        num_chunks,
        dim,
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out
