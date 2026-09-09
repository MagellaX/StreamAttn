"""Experimental two-gate attention with cumulative omission accounting.

Contiguous BSHD FP16/BF16, compact MHA/GQA, equal QK/V dimensions, optional
top-left causal mask. No dropout, paged storage, or arbitrary mask support.
This is a diagnostic native path, not a performance-promoted serving backend.
The omission bound excludes floating-point arithmetic and output-cast error.
"""

import math
from typing import Optional, Tuple

import torch

from stream_attention.certified.summaries import BlockSummaries, build_block_summaries

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _within_budget(total_bound, acc_den, value_radius,
                       EPS: tl.constexpr, VALUE_BOUND: tl.constexpr):
        if VALUE_BOUND:
            fraction = total_bound / tl.maximum(acc_den + total_bound, 1.0e-30)
            return 2.0 * value_radius * fraction <= EPS
        return total_bound <= EPS * acc_den

    @triton.jit
    def _certified_fwd_kernel(
        Q, K, V, Centroid, Radius, MaxVNorm, Out, RawStats, ErrorBound, Retained,
        M: tl.constexpr, N: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
        D: tl.constexpr, NUM_BLOCKS: tl.constexpr, SUMMARY_WIDTH: tl.constexpr,
        TILE_M: tl.constexpr, TILE_N: tl.constexpr, SCALE: tl.constexpr,
        EPS: tl.constexpr, IS_CAUSAL: tl.constexpr, VALUE_BOUND: tl.constexpr,
        ENABLE_PRE: tl.constexpr, ENABLE_POST: tl.constexpr,
        ROWWISE_OMISSIONS: tl.constexpr,
        MATERIALIZE_SKIPPED: tl.constexpr, HAS_STATS: tl.constexpr,
        HAS_BOUND: tl.constexpr,
        HAS_SUPPORT: tl.constexpr,
    ):
        qb, b, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        kh = h // (HQ // HKV)
        rows = qb * TILE_M + tl.arange(0, TILE_M)
        cols = tl.arange(0, TILE_N)
        ds = tl.arange(0, D)
        row_mask = rows < M
        q = tl.load(Q + ((b * M + rows[:, None]) * HQ + h) * D + ds[None, :],
                    mask=row_mask[:, None], other=0.0)
        qf = q.to(tl.float32)
        q_norm = tl.sqrt(tl.sum(qf * qf, axis=1))
        m = tl.full([TILE_M], -float("inf"), tl.float32)
        den = tl.zeros([TILE_M], tl.float32)
        num = tl.zeros([TILE_M, D], tl.float32)
        omitted = tl.zeros([TILE_M], tl.float32)

        value_radius = tl.full([], 0.0, tl.float32)
        if EPS > 0.0 and (VALUE_BOUND or HAS_BOUND):
            ids = tl.arange(0, SUMMARY_WIDTH)
            bounds = tl.load(MaxVNorm + (b * HKV + kh) * NUM_BLOCKS + ids,
                             mask=ids < NUM_BLOCKS, other=0.0)
            value_radius = tl.max(bounds.to(tl.float32), axis=0)

        pre_count = tl.full([], 0, tl.int32)
        post_count = tl.full([], 0, tl.int32)
        compute_count = tl.full([], 0, tl.int32)
        valid_tiles = tl.full([], 0, tl.int32)
        qk_tiles = tl.full([], 0, tl.int32)
        pv_tiles = tl.full([], 0, tl.int32)

        for block in range(NUM_BLOCKS):
            positions = block * TILE_N + cols
            block_len = tl.minimum(TILE_N, N - block * TILE_N)
            valid = row_mask
            full_allowed = row_mask
            if IS_CAUSAL:
                valid = row_mask & (rows >= block * TILE_N)
                full_allowed = row_mask & (rows >= block * TILE_N + block_len - 1)
            if HAS_STATS:
                valid_tiles += (tl.sum(valid.to(tl.int32), 0) > 0).to(tl.int32)

            skip_pre = tl.full([TILE_M], False, tl.int1)
            if ENABLE_PRE and EPS > 0.0:
                base = (b * HKV + kh) * NUM_BLOCKS + block
                center = tl.load(Centroid + base * D + ds).to(tl.float32)
                radius = tl.load(Radius + base).to(tl.float32)
                upper = (tl.sum(qf * center[None, :], 1) + q_norm * radius) * SCALE
                eligible = valid & full_allowed & (den > 0.0) & (upper <= m)
                mass = tl.where(eligible, block_len * tl.exp(upper - m), 0.0)
                skip_pre = eligible & _within_budget(
                    omitted + mass, den, value_radius, EPS, VALUE_BOUND)
                if not ROWWISE_OMISSIONS:
                    unanimous = tl.sum((valid & ~skip_pre).to(tl.int32), 0) == 0
                    skip_pre = skip_pre & unanimous
                omitted += tl.where(skip_pre, mass, 0.0)

            needs_k = valid & ~skip_pre
            retained = tl.full([TILE_M], False, tl.int1)
            if HAS_STATS:
                pre_count += tl.sum(skip_pre.to(tl.int32), 0)
            # Predicate the whole load/MMA region, not just its resulting scores.
            if (tl.sum(needs_k.to(tl.int32), 0) > 0) or MATERIALIZE_SKIPPED:
                kt = tl.load(K + ((b * N + positions[:, None]) * HKV + kh) * D + ds[None, :],
                             mask=(positions < N)[:, None], other=0.0)
                scores = tl.dot(q, tl.trans(kt), out_dtype=tl.float32) * SCALE
                scores = tl.where((positions < N)[None, :] & needs_k[:, None],
                                  scores, -float("inf"))
                if IS_CAUSAL:
                    scores = tl.where(rows[:, None] >= positions[None, :],
                                      scores, -float("inf"))
                if HAS_STATS:
                    qk_tiles += 1
                tile_max = tl.max(scores, 1)
                skip_post = tl.full([TILE_M], False, tl.int1)
                if ENABLE_POST and EPS > 0.0:
                    eligible = needs_k & (den > 0.0) & (tile_max <= m)
                    mass = tl.where(eligible, block_len * tl.exp(tile_max - m), 0.0)
                    skip_post = eligible & _within_budget(
                        omitted + mass, den, value_radius, EPS, VALUE_BOUND)
                    if not ROWWISE_OMISSIONS:
                        unanimous = tl.sum((needs_k & ~skip_post).to(tl.int32), 0) == 0
                        skip_post = skip_post & unanimous
                    omitted += tl.where(skip_post, mass, 0.0)
                compute = needs_k & ~skip_post
                retained = compute
                if HAS_STATS:
                    post_count += tl.sum(skip_post.to(tl.int32), 0)
                    compute_count += tl.sum(compute.to(tl.int32), 0)

                if (tl.sum(compute.to(tl.int32), 0) > 0) or MATERIALIZE_SKIPPED:
                    scores = tl.where(compute[:, None], scores, -float("inf"))
                    tile_max = tl.max(scores, 1)
                    new_m = tl.maximum(m, tile_max)
                    safe_m = tl.where(new_m > -float("inf"), new_m, 0.0)
                    correction = tl.where(den > 0.0, tl.exp(m - safe_m), 0.0)
                    p = tl.exp(scores - safe_m[:, None])
                    vt = tl.load(V + ((b * N + positions[:, None]) * HKV + kh) * D + ds[None, :],
                                 mask=(positions < N)[:, None], other=0.0)
                    num = num * correction[:, None] + tl.dot(p.to(vt.dtype), vt, out_dtype=tl.float32)
                    den = den * correction + tl.sum(p, 1)
                    omitted *= correction
                    m = new_m
                    if HAS_STATS:
                        pv_tiles += 1
            if HAS_SUPPORT:
                tl.store(Retained + ((b * HQ + h) * M + rows) * NUM_BLOCKS + block,
                         retained, mask=row_mask)

        output = num / tl.maximum(den, 1.0e-30)[:, None]
        offset = (b * M + rows) * HQ + h
        tl.store(Out + offset[:, None] * D + ds[None, :], output, mask=row_mask[:, None])
        if HAS_BOUND:
            error = 2.0 * value_radius * omitted / tl.maximum(den + omitted, 1.0e-30)
            tl.store(ErrorBound + offset, error, mask=row_mask)
        if HAS_STATS:
            base = ((b * HQ + h) * tl.cdiv(M, TILE_M) + qb) * 6
            tl.store(RawStats + base, pre_count)
            tl.store(RawStats + base + 1, post_count)
            tl.store(RawStats + base + 2, compute_count)
            tl.store(RawStats + base + 3, valid_tiles)
            tl.store(RawStats + base + 4, qk_tiles)
            tl.store(RawStats + base + 5, pv_tiles)


def certified_attention_triton_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    error_budget: float = 1e-3,
    block_size: int = 64,
    tile_size_q: int = 64,
    summaries: Optional[BlockSummaries] = None,
    skip_predicate: str = "value_bound",
    enable_summary_gate: bool = True,
    enable_post_qk_gate: bool = True,
    return_raw_stats: bool = False,
    out: Optional[torch.Tensor] = None,
    raw_stats_out: Optional[torch.Tensor] = None,
    error_bound_out: Optional[torch.Tensor] = None,
    materialize_skipped_work: bool = False,
    rowwise_omissions: bool = False,
    retained_blocks_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the two-gate experiment; supplied buffers allow fixed-address replay.

    Raw stats end in six fields: pre/post/computed row-blocks, valid CTA tiles,
    executed QK tiles, executed PV tiles. Mixed rows can force physical tile work
    despite logical skips in the legacy ``rowwise_omissions`` diagnostic. By
    default omissions are charged only when every valid row permits region
    bypass. Counters describe executed regions, not measured HBM traffic.
    materialize_skipped_work is a same-support mask-only diagnostic control.

    value_bound uses a cumulative row-L2 omission budget; mass instead bounds
    omitted/retained partition mass. error_bound_out contains only the omission
    bound, excluding floating-point error. Cached summaries must match current
    K/V contents, including after input mutation. Optional retained_blocks_out
    records the actual [B, Hq, M, NB] support for untimed numerical diagnostics.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("certified_attention_triton_forward requires CUDA tensors")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, seq, heads, dim]")
    if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K/V batch/dimensions must match; K/V shapes must match")
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Q/K/V must have the same FP16 or BF16 dtype")
    if key.device != query.device or value.device != query.device:
        raise ValueError("Q/K/V must be on the same device")
    batch, seq_q, heads, dim = query.shape
    seq_k, kv_heads = key.shape[1:3]
    if min(batch, seq_q, seq_k, heads, kv_heads) <= 0 or heads % kv_heads:
        raise ValueError("nonempty inputs and a positive integral GQA group are required")
    if any(x < 16 or x > 256 or x & (x - 1) for x in (dim, block_size, tile_size_q)):
        raise ValueError("dim and tile sizes must be powers of two in [16, 256]")
    if not math.isfinite(error_budget) or error_budget < 0:
        raise ValueError("error_budget must be finite and non-negative")
    if skip_predicate not in {"mass", "value_bound"}:
        raise ValueError("skip_predicate must be 'mass' or 'value_bound'")

    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    if summaries is None:
        summaries = build_block_summaries(key, value, block_size=block_size)
    blocks = triton.cdiv(seq_k, block_size)
    if summaries.outlier_keys is not None:
        raise ValueError("experimental Triton path does not support outlier summaries")
    if (error_budget > 0 and (skip_predicate == "value_bound" or error_bound_out is not None)
            and not summaries.has_value_bounds):
        raise ValueError("value-bound metadata is missing; key-only summaries cannot certify V")
    if (summaries.block_size != block_size or summaries.seq_len != seq_k
            or summaries.centroid.shape != (batch, kv_heads, blocks, dim)
            or summaries.radius.shape != (batch, kv_heads, blocks)
            or summaries.max_value_norm.shape != (batch, kv_heads, blocks)):
        raise ValueError("summary shape does not match K/V")
    for tensor in (summaries.centroid, summaries.radius, summaries.max_value_norm):
        if tensor.device != query.device or tensor.dtype != torch.float32:
            raise ValueError("summaries must be FP32 on the Q/K/V device")

    def buffer(tensor, shape, dtype, name):
        if tensor is None:
            return torch.empty(shape, device=query.device, dtype=dtype)
        if (tensor.shape != shape or tensor.dtype != dtype or tensor.device != query.device
                or not tensor.is_contiguous()):
            raise ValueError(f"invalid {name} buffer")
        return tensor

    output = buffer(out, query.shape, query.dtype, "output")
    stats_shape = (batch, heads, triton.cdiv(seq_q, tile_size_q), 6)
    raw = buffer(raw_stats_out, stats_shape, torch.int32, "stats") if return_raw_stats else output
    bound = (buffer(error_bound_out, query.shape[:-1], torch.float32, "bound")
             if error_bound_out is not None else output)
    support = (buffer(retained_blocks_out, (batch, heads, seq_q, blocks), torch.bool, "support")
               if retained_blocks_out is not None else output)
    _certified_fwd_kernel[(triton.cdiv(seq_q, tile_size_q), batch, heads)](
        query, key, value, summaries.centroid.contiguous(), summaries.radius.contiguous(),
        summaries.max_value_norm.contiguous(), output, raw, bound, support,
        M=seq_q, N=seq_k, HQ=heads, HKV=kv_heads, D=dim, NUM_BLOCKS=blocks,
        SUMMARY_WIDTH=triton.next_power_of_2(blocks), TILE_M=tile_size_q,
        TILE_N=block_size, SCALE=1.0 / math.sqrt(dim), EPS=float(error_budget),
        IS_CAUSAL=bool(causal), VALUE_BOUND=skip_predicate == "value_bound",
        ENABLE_PRE=enable_summary_gate, ENABLE_POST=enable_post_qk_gate,
        ROWWISE_OMISSIONS=rowwise_omissions,
        MATERIALIZE_SKIPPED=materialize_skipped_work, HAS_STATS=return_raw_stats,
        HAS_BOUND=error_bound_out is not None, HAS_SUPPORT=retained_blocks_out is not None,
        num_warps=4, num_stages=1,
    )
    return output, raw if return_raw_stats else None
