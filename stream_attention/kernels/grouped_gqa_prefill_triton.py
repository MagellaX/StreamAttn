"""Experimental KV-group-owned exact GQA prefill floor."""

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
    def _grouped_gqa_prefill_kernel(
        Q,
        K,
        V,
        Out,
        Lse,
        stride_qb,
        stride_qm,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kn,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vn,
        stride_vh,
        stride_vd,
        stride_ob,
        stride_om,
        stride_oh,
        stride_od,
        stride_lb,
        stride_lh,
        stride_lm,
        q_start,
        M: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HEADS_PER_PROGRAM: tl.constexpr,
        GROUPS_PER_KV: tl.constexpr,
        GROUPED_ROWS: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        SCALE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        INPUT_BF16: tl.constexpr,
        INPUT_FP16: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        off_b = tl.program_id(1)
        off_head_group = tl.program_id(2)

        off_kvh = off_head_group // GROUPS_PER_KV
        subgroup = off_head_group % GROUPS_PER_KV
        head_base = off_kvh * GROUP_SIZE + subgroup * HEADS_PER_PROGRAM

        offs_grouped_m = tl.arange(0, GROUPED_ROWS)
        local_h = offs_grouped_m // TILE_M
        local_m = offs_grouped_m % TILE_M
        offs_m = pid_m * TILE_M + local_m
        offs_h = head_base + local_h
        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, TILE_N)

        q_ptrs = (
            Q
            + off_b * stride_qb
            + offs_m[:, None] * stride_qm
            + offs_h[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        running_max = tl.full([GROUPED_ROWS], -float("inf"), tl.float32)
        acc_den = tl.zeros([GROUPED_ROWS], tl.float32)
        acc_num = tl.zeros([GROUPED_ROWS, D], tl.float32)
        has_valid = tl.zeros([GROUPED_ROWS], tl.int1)

        end_n = N
        if IS_CAUSAL:
            # A query tile cannot consume keys beyond its latest absolute row.
            # Truncating the stream here removes upper-triangular K/V loads and
            # QK/PV work while leaving the online-softmax recurrence unchanged.
            end_n = tl.minimum(N, (pid_m + 1) * TILE_M + q_start)
        for start_n in range(0, end_n, TILE_N):
            cols = start_n + offs_n
            kv_mask = (cols[:, None] < N) & (offs_d[None, :] < D)
            k_ptrs = (
                K
                + off_b * stride_kb
                + cols[:, None] * stride_kn
                + off_kvh * stride_kh
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V
                + off_b * stride_vb
                + cols[:, None] * stride_vn
                + off_kvh * stride_vh
                + offs_d[None, :] * stride_vd
            )
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * SCALE
            valid = (offs_m[:, None] < M) & (cols[None, :] < N)
            if IS_CAUSAL:
                valid &= (offs_m[:, None] + q_start) >= cols[None, :]
            qk = tl.where(valid, qk, -float("inf"))

            tile_max = tl.max(qk, axis=1)
            tile_valid = tile_max > -float("inf")
            new_valid = has_valid | tile_valid
            candidate_max = tl.maximum(running_max, tile_max)
            safe_previous = tl.where(has_valid, running_max, 0.0)
            safe_new = tl.where(new_valid, candidate_max, 0.0)
            correction = tl.where(
                has_valid,
                tl.exp(safe_previous - safe_new),
                0.0,
            )
            probabilities = tl.where(
                valid,
                tl.exp(qk - safe_new[:, None]),
                0.0,
            )

            if INPUT_BF16:
                probabilities_for_pv = probabilities.to(tl.bfloat16)
            elif INPUT_FP16:
                probabilities_for_pv = probabilities.to(tl.float16)
            else:
                probabilities_for_pv = probabilities
            acc_num = acc_num * correction[:, None] + tl.dot(probabilities_for_pv, v)
            acc_den = acc_den * correction + tl.sum(probabilities, axis=1)
            running_max = tl.where(new_valid, candidate_max, running_max)
            has_valid = new_valid

        inv_den = tl.where(acc_den > 0.0, 1.0 / acc_den, 0.0)
        output = acc_num * inv_den[:, None]
        out_ptrs = (
            Out
            + off_b * stride_ob
            + offs_m[:, None] * stride_om
            + offs_h[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(out_ptrs, output.to(Out.dtype.element_ty), mask=q_mask)

        lse = tl.where(
            acc_den > 0.0,
            running_max + tl.log(acc_den),
            -float("inf"),
        )
        lse_ptrs = (
            Lse
            + off_b * stride_lb
            + offs_h * stride_lh
            + offs_m * stride_lm
        )
        tl.store(lse_ptrs, lse, mask=offs_m < M)


def grouped_gqa_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    heads_per_program: int,
    tile_m: int,
    tile_n: int = 64,
    causal: bool = True,
    q_start: int = 0,
    return_lse: bool = False,
    num_warps: int | None = None,
    num_stages: int = 2,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the experimental grouped-KV exact forward kernel.

    Inputs use compact BSHD GQA layout. Each program owns one subset of query
    heads belonging to a single KV head and shares every loaded K/V tile across
    that subset.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for grouped GQA prefill")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("grouped GQA prefill requires CUDA tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-4 BSHD tensors")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must share one CUDA device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must share one dtype")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("grouped GQA prefill supports FP16, BF16, or FP32")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query and K/V batch/head dimensions must match")
    q_heads = int(query.shape[2])
    kv_heads = int(key.shape[2])
    if q_heads % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    group_size = q_heads // kv_heads
    if heads_per_program not in (1, 2, 4, 8):
        raise ValueError("heads_per_program must be one of 1, 2, 4, or 8")
    if group_size % heads_per_program:
        raise ValueError("heads_per_program must divide the GQA group size")
    grouped_rows = heads_per_program * tile_m
    if grouped_rows <= 0 or grouped_rows & (grouped_rows - 1):
        raise ValueError("heads_per_program * tile_m must be a power of two")
    if tile_n <= 0 or tile_n & (tile_n - 1):
        raise ValueError("tile_n must be a power of two")
    if num_warps is not None and num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if num_stages not in (1, 2, 3, 4):
        raise ValueError("num_stages must be between 1 and 4")
    head_dim = int(query.shape[3])
    if head_dim not in (16, 32, 64, 128):
        raise ValueError("head_dim must be 16, 32, 64, or 128")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    batch, query_len = int(query.shape[0]), int(query.shape[1])
    kv_len = int(key.shape[1])
    if output is None:
        output = torch.empty_like(query)
    elif (
        output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous and match query shape/dtype/device")
    expected_lse_shape = (batch, q_heads, query_len)
    if lse is None:
        lse = torch.empty(
            expected_lse_shape,
            device=query.device,
            dtype=torch.float32,
        )
    elif (
        tuple(lse.shape) != expected_lse_shape
        or lse.dtype != torch.float32
        or lse.device != query.device
        or not lse.is_contiguous()
    ):
        raise ValueError("lse must be contiguous FP32 [batch,q_heads,query_len]")
    groups_per_kv = group_size // heads_per_program
    grid = (
        triton.cdiv(query_len, tile_m),
        batch,
        kv_heads * groups_per_kv,
    )
    launch_warps = num_warps or (8 if grouped_rows >= 128 else 4)
    _grouped_gqa_prefill_kernel[grid](
        query,
        key,
        value,
        output,
        lse,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *lse.stride(),
        q_start,
        M=query_len,
        N=kv_len,
        D=head_dim,
        GROUP_SIZE=group_size,
        HEADS_PER_PROGRAM=heads_per_program,
        GROUPS_PER_KV=groups_per_kv,
        GROUPED_ROWS=grouped_rows,
        TILE_M=tile_m,
        TILE_N=tile_n,
        SCALE=1.0 / math.sqrt(head_dim),
        IS_CAUSAL=causal,
        INPUT_BF16=query.dtype == torch.bfloat16,
        INPUT_FP16=query.dtype == torch.float16,
        num_warps=launch_warps,
        num_stages=num_stages,
    )
    if return_lse:
        return output, lse
    return output


def effective_kv_reuse(
    *,
    heads_per_program: int,
    tile_m: int,
    reference_tile_m: int,
) -> float:
    """Return K/V scan reduction relative to a Q-head-owned reference."""

    if min(heads_per_program, tile_m, reference_tile_m) <= 0:
        raise ValueError("reuse inputs must be positive")
    return heads_per_program * tile_m / reference_tile_m


__all__ = ["TRITON_AVAILABLE", "effective_kv_reuse", "grouped_gqa_prefill"]
