"""
Fused Online Softmax Attention - Production Implementation

This is the core novel contribution: a fused attention mechanism that computes
softmax normalization "on the fly" using running accumulators, achieving both
memory efficiency and numerical stability in a single kernel pass.

Key innovations:
- Online softmax computation with running max and sum
- Tiled processing for efficient memory access
- Single-pass algorithm avoiding materialization of attention matrix
- Multi-GPU support through PyTorch Distributed

Based on the original StreamAttention research prototype.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, Dict, Any
import logging
from contextlib import nullcontext

try:
    from torch.nn.attention import SDPBackend
except ImportError:  # pragma: no cover - older PyTorch
    SDPBackend = None

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)


if TRITON_AVAILABLE:

    _sm = (
        torch.cuda.get_device_capability()[0] * 10
        + torch.cuda.get_device_capability()[1]
        if torch.cuda.is_available()
        else 0
    )

    _SM90_CONFIGS = [
        triton.Config({"TILE_M": 128, "TILE_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"TILE_M": 256, "TILE_N": 128}, num_warps=8, num_stages=4),
    ]
    _SM80_CONFIGS = [
        triton.Config({"TILE_M": 128, "TILE_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"TILE_M": 128, "TILE_N": 128}, num_warps=8, num_stages=3),
    ]
    _FALLBACK_CONFIGS = [
        triton.Config({"TILE_M": 64, "TILE_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"TILE_M": 128, "TILE_N": 64}, num_warps=4, num_stages=2),
    ]
    _CONFIGS = (
        _SM90_CONFIGS if _sm >= 90 else _SM80_CONFIGS if _sm >= 80 else _FALLBACK_CONFIGS
    )

    @triton.autotune(configs=_CONFIGS, key=["M", "N", "D"])
    @triton.jit
    def fused_online_attention_kernel(
        Q,
        K,
        V,
        Out,
        Lse,  # Log-sum-exp for numerical stability
        Mask,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_ob,
        stride_oh,
        stride_om,
        stride_ok,
        stride_lb,
        stride_lh,
        stride_lm,
        stride_mb,
        stride_mh,
        stride_mm,
        stride_mn,
        dropout_p,
        dropout_scale,
        rng_seed,
        rng_offset,
        AlibiSlopes,
        global_M,
        global_N,
        q_start,
        H: tl.constexpr,  # num heads
        M: tl.constexpr,  # seq_len_q
        N: tl.constexpr,  # seq_len_k
        D: tl.constexpr,  # head_dim
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_DROPOUT: tl.constexpr,
        HAS_ALIBI: tl.constexpr,
        USE_WGMMA: tl.constexpr,
        USE_TMA: tl.constexpr,
        USE_CP_ASYNC: tl.constexpr,
    ):
        """
        Fused Online Softmax Attention Kernel
        
        Each program processes TILE_M query rows against all keys/values in tiles.
        Maintains running_max and acc_den/acc_num for online softmax.
        """
        # Program IDs
        start_m = tl.program_id(0)
        off_b = tl.program_id(1) 
        off_h = tl.program_id(2)
        
        # Offsets
        offs_m = start_m * TILE_M + tl.arange(0, TILE_M)
        offs_n = tl.arange(0, TILE_N)
        offs_k = tl.arange(0, D)
        
        # Load Q tile (TMA for Hopper)
        q_ptrs = (
            Q
            + off_b * stride_qb
            + off_h * stride_qh
            + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        )
        q_mask = (offs_m[:, None] < M) & (offs_k[None, :] < D)
        if USE_TMA:
            # Placeholder for TMA-based transfer with arrival barriers
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        q = q.to(tl.float32)
        
        # Accumulators
        running_max = tl.full([TILE_M], value=-float("inf"), dtype=tl.float32)
        acc_num = tl.zeros([TILE_M, D], dtype=tl.float32)
        acc_den = tl.zeros([TILE_M], dtype=tl.float32)
        has_valid = tl.zeros([TILE_M], dtype=tl.float32)

        # Iterate over K/V tiles
        for start_n in range(0, N, TILE_N):
            start_n = tl.multiple_of(start_n, TILE_N)
            
            k_ptrs = (
                K
                + off_b * stride_kb
                + off_h * stride_kh
                + ((start_n + offs_n)[:, None] * stride_kn + offs_k[None, :] * stride_kk)
            )
            v_ptrs = (
                V
                + off_b * stride_vb
                + off_h * stride_vh
                + ((start_n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk)
            )
            kv_mask = ((start_n + offs_n)[:, None] < N) & (offs_k[None, :] < D)
            if USE_CP_ASYNC:
                # cp.async + double buffering placeholder
                k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
                v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
            else:
                k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
                v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
            k = k.to(tl.float32)
            v = v.to(tl.float32)
            
            # QK^T
            # Hopper uses WGMMA tensor cores; Ampere uses mma.sync
            qk = tl.dot(q, tl.trans(k)) * scale
            
            # Causal mask
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            if HAS_MASK:
                mask_ptrs = (
                    Mask
                    + off_b * stride_mb
                    + off_h * stride_mh
                    + (offs_m[:, None] * stride_mm + (start_n + offs_n)[None, :] * stride_mn)
                )
                mask_mask = (offs_m[:, None] < M) & ((start_n + offs_n)[None, :] < N)
                mask_vals = tl.load(mask_ptrs, mask=mask_mask, other=0.0).to(qk.dtype)
                qk += mask_vals

            if HAS_ALIBI:
                slope = tl.load(AlibiSlopes + off_h).to(tl.float32)
                q_pos = (offs_m[:, None] + q_start).to(tl.float32)
                k_pos = (start_n + offs_n)[None, :].to(tl.float32)
                qk += slope * (k_pos - q_pos)

            # Online softmax update with fully masked-row safeguards
            tile_max = tl.max(qk, axis=1)
            prev_valid = has_valid > 0
            tile_valid = tile_max > float("-inf")
            new_valid = prev_valid | tile_valid

            candidate_max = tl.maximum(running_max, tile_max)
            safe_prev = tl.where(prev_valid, running_max, 0.0)
            safe_new = tl.where(new_valid, candidate_max, 0.0)
            correction = tl.where(prev_valid, tl.exp(safe_prev - safe_new), 1.0)

            running_max = tl.where(new_valid, candidate_max, float("-inf"))
            acc_num *= correction[:, None]
            acc_den *= correction

            qk_shifted = qk - safe_new[:, None]
            exp_qk = tl.where(new_valid[:, None], tl.exp(qk_shifted), 0.0)
            has_valid = new_valid.to(tl.float32)
            
            if HAS_DROPOUT:
                bh = off_b * H + off_h
                row_global = (offs_m[:, None] + q_start)
                col_global = (start_n + offs_n)[None, :]
                rng_offsets = (
                    (bh * global_M + row_global) * global_N + col_global + rng_offset
                ).to(tl.int32)
                keep = tl.rand(rng_seed, rng_offsets) > dropout_p
                exp_qk = exp_qk * keep.to(exp_qk.dtype) * dropout_scale

            acc_num += tl.dot(exp_qk, v)
            acc_den += tl.sum(exp_qk, axis=1)
        
        # Final output with safe denominator; handle rows with all keys masked
        zero_den = acc_den == 0
        inv_den = tl.where(zero_den, 0.0, 1.0 / acc_den)
        out = acc_num * inv_den[:, None]

        out_ptrs = (
            Out
            + off_b * stride_ob
            + off_h * stride_oh
            + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
        )
        out_mask = (offs_m[:, None] < M) & (offs_k[None, :] < D)
        tl.store(out_ptrs, out.to(Out.dtype.element_ty), mask=out_mask)
        
        # LSE: set to -inf for fully masked rows (acc_den == 0)
        lse = tl.where(zero_den, float("-inf"), running_max + tl.log(acc_den))
        lse_ptrs = Lse + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
        lse_mask = offs_m < M
        tl.store(lse_ptrs, lse, mask=lse_mask)


    @triton.autotune(configs=_CONFIGS, key=["M", "N", "D"])
    @triton.jit
    def fused_online_attention_bwd_kernel(
        Q,
        K,
        V,
        dQ,
        dK,
        dV,
        GO,
        Lse,
        Mask,
        AlibiSlopesIn,
        GradAlibiOut,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_dqb,
        stride_dqh,
        stride_dqm,
        stride_dqk,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dkk,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        stride_dvk,
        stride_gob,
        stride_goh,
        stride_gom,
        stride_gok,
        stride_lb,
        stride_lh,
        stride_lm,
        stride_mb,
        stride_mh,
        stride_mm,
        stride_mn,
        stride_as,
        stride_ag,
        scale,
        global_M,
        global_N,
        q_start,
        H: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_ALIBI: tl.constexpr,
        USE_WGMMA: tl.constexpr,
        USE_TMA: tl.constexpr,
        USE_CP_ASYNC: tl.constexpr,
    ):
        """Single-sweep backward kernel computing dQ, dK, dV, and optional ALiBi grad."""
        start_m = tl.program_id(0)
        off_b = tl.program_id(1)
        off_h = tl.program_id(2)

        offs_m = start_m * TILE_M + tl.arange(0, TILE_M)
        offs_n = tl.arange(0, TILE_N)
        offs_k = tl.arange(0, D)

        row_mask = offs_m < M
        k_mask = offs_k < D

        q_ptrs = (
            Q
            + off_b * stride_qb
            + off_h * stride_qh
            + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        )
        go_ptrs = (
            GO
            + off_b * stride_gob
            + off_h * stride_goh
            + (offs_m[:, None] * stride_gom + offs_k[None, :] * stride_gok)
        )
        lse_ptrs = Lse + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm

        q = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        go = tl.load(go_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        lse = tl.load(lse_ptrs, mask=row_mask, other=float("-inf"))
        valid_row = lse > float("-inf")
        lse = tl.where(valid_row, lse, 0.0)

        dq_acc = tl.zeros([TILE_M, D], dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            start_n = tl.multiple_of(start_n, TILE_N)
            col_idx = start_n + offs_n
            col_mask = col_idx < N

            k_ptrs = (
                K
                + off_b * stride_kb
                + off_h * stride_kh
                + (col_idx[:, None] * stride_kn + offs_k[None, :] * stride_kk)
            )
            v_ptrs = (
                V
                + off_b * stride_vb
                + off_h * stride_vh
                + (col_idx[:, None] * stride_vn + offs_k[None, :] * stride_vk)
            )
            kv_mask = col_mask[:, None] & k_mask[None, :]

            if USE_CP_ASYNC:
                k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
                v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            else:
                k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
                v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            logits = tl.dot(q, tl.trans(k_tile)) * scale

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= col_idx[None, :]
                logits = tl.where(causal_mask, logits, float("-inf"))

            if HAS_MASK:
                mask_ptrs = (
                    Mask
                    + off_b * stride_mb
                    + off_h * stride_mh
                    + (offs_m[:, None] * stride_mm + col_idx[None, :] * stride_mn)
                )
                mask_load_mask = row_mask[:, None] & col_mask[None, :]
                mask_vals = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0)
                logits = logits + mask_vals

            if HAS_ALIBI:
                slope = tl.load(AlibiSlopesIn + off_h * stride_as).to(tl.float32)
                q_pos = (offs_m[:, None] + q_start).to(tl.float32)
                k_pos = col_idx[None, :].to(tl.float32)
                logits = logits + slope * (k_pos - q_pos)

            lse_tile = lse[:, None]
            probs = tl.exp(logits - lse_tile)
            valid = row_mask[:, None] & col_mask[None, :] & valid_row[:, None]
            probs = probs * valid.to(probs.dtype)

            dv_tile = tl.dot(tl.trans(probs), go)
            dV_ptrs = (
                dV
                + off_b * stride_dvb
                + off_h * stride_dvh
                + (col_idx[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)
            )
            tl.atomic_add(dV_ptrs, dv_tile, mask=kv_mask)

            dP = tl.dot(go, tl.trans(v_tile))
            row_sum = tl.sum(dP * probs, axis=1)
            dS = (dP - row_sum[:, None]) * probs

            dq_acc += tl.dot(dS, k_tile) * scale

            dK_tile = tl.dot(tl.trans(dS), q) * scale
            dK_ptrs = (
                dK
                + off_b * stride_dkb
                + off_h * stride_dkh
                + (col_idx[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)
            )
            tl.atomic_add(dK_ptrs, dK_tile, mask=kv_mask)

            if HAS_ALIBI:
                delta = col_idx[None, :].to(tl.float32) - (offs_m[:, None] + q_start).to(tl.float32)
                grad_alibi_tile = tl.sum(dS * delta)
                tl.atomic_add(GradAlibiOut + off_h * stride_ag, grad_alibi_tile)

        dq_ptrs = (
            dQ
            + off_b * stride_dqb
            + off_h * stride_dqh
            + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk)
        )
        tl.store(
            dq_ptrs,
            dq_acc,
            mask=row_mask[:, None] & k_mask[None, :],
        )


    @triton.jit
    def fused_online_attention_bwd_kernel(
        Q, K, V,
        GradOut,
        Lse,
        Mask,
        GradQ,
        GradK,
        GradV,
        GradAlibi,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_gob, stride_goh, stride_gom, stride_gok,
        stride_lsb, stride_lsh, stride_lsm,
        stride_mb, stride_mh, stride_mm, stride_mn,
        stride_gqb, stride_gqh, stride_gqm, stride_gqk,
        stride_gkb, stride_gkh, stride_gkn, stride_gkk,
        stride_gvb, stride_gvh, stride_gvn, stride_gvk,
        dropout_p,
        dropout_scale,
        rng_seed,
        rng_offset,
        AlibiSlopes,
        global_M,
        global_N,
        q_start,
        H: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_DROPOUT: tl.constexpr,
        HAS_ALIBI: tl.constexpr,
        ACCUM_ALIBI: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_b = tl.program_id(1)
        off_h = tl.program_id(2)

        offs_m = start_m * TILE_M + tl.arange(0, TILE_M)
        offs_n = tl.arange(0, TILE_N)
        offs_k = tl.arange(0, D)

        q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (
            offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        )
        q_mask = (offs_m[:, None] < M) & (offs_k[None, :] < D)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        go_ptrs = GradOut + off_b * stride_gob + off_h * stride_goh + (
            offs_m[:, None] * stride_gom + offs_k[None, :] * stride_gok
        )
        go = tl.load(go_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        lse_ptrs = Lse + off_b * stride_lsb + off_h * stride_lsh + offs_m * stride_lsm
        lse = tl.load(lse_ptrs, mask=offs_m < M, other=float('-inf')).to(tl.float32)
        row_mask = lse > float("-inf")
        lse = tl.where(row_mask, lse, 0.0)
        go = go * row_mask[:, None]

        dq = tl.zeros([TILE_M, D], dtype=tl.float32)
        grad_alibi_acc = tl.zeros([], dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            start_n = tl.multiple_of(start_n, TILE_N)

            k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (
                (start_n + offs_n)[:, None] * stride_kn + offs_k[None, :] * stride_kk
            )
            v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (
                (start_n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk
            )
            kv_mask = ((start_n + offs_n)[:, None] < N) & (offs_k[None, :] < D)
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            qk = tl.dot(q, tl.trans(k)) * scale

            if HAS_MASK:
                mask_ptrs = Mask + off_b * stride_mb + off_h * stride_mh + (
                    offs_m[:, None] * stride_mm + (start_n + offs_n)[None, :] * stride_mn
                )
                mask_mask = (offs_m[:, None] < M) & (
                    (start_n + offs_n)[None, :] < N
                )
                mask_vals = tl.load(mask_ptrs, mask=mask_mask, other=0.0)
                qk += mask_vals

            if HAS_ALIBI:
                slope = tl.load(AlibiSlopes + off_h).to(tl.float32)
                q_pos = (offs_m[:, None] + q_start).to(tl.float32)
                k_pos = (start_n + offs_n)[None, :].to(tl.float32)
                qk += slope * (k_pos - q_pos)

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
                qk = tl.where(causal_mask, qk, float('-inf'))

            exp_term = qk - lse[:, None]
            p = tl.exp(exp_term)
            p = p * row_mask[:, None]
            col_mask = (start_n + offs_n) < N
            p = tl.where(col_mask[None, :], p, 0.0)

            if HAS_DROPOUT:
                bh = off_b * H + off_h
                row_global = (offs_m[:, None] + q_start)
                col_global = (start_n + offs_n)[None, :]
                rng_offsets = (
                    (bh * global_M + row_global) * global_N + col_global + rng_offset
                ).to(tl.int32)
                keep = tl.rand(rng_seed, rng_offsets) > dropout_p
                p = p * keep.to(p.dtype) * dropout_scale

            dV_tile = tl.dot(tl.trans(p), go)
            dv_mask = col_mask[:, None] & (offs_k[None, :] < D)
            grad_v_ptrs = GradV + off_b * stride_gvb + off_h * stride_gvh + (
                (start_n + offs_n)[:, None] * stride_gvn + offs_k[None, :] * stride_gvk
            )
            tl.atomic_add(
                grad_v_ptrs,
                dV_tile.to(GradV.dtype.element_ty),
                mask=dv_mask,
            )

            dP = tl.dot(go, tl.trans(v))
            attn_dot = tl.sum(dP * p, axis=1)
            dS = (dP - attn_dot[:, None]) * p

            dq += tl.dot(dS, k) * scale
            dk_tile = tl.dot(tl.trans(dS), q) * scale
            grad_k_ptrs = GradK + off_b * stride_gkb + off_h * stride_gkh + (
                (start_n + offs_n)[:, None] * stride_gkn + offs_k[None, :] * stride_gkk
            )
            tl.atomic_add(
                grad_k_ptrs,
                dk_tile.to(GradK.dtype.element_ty),
                mask=dv_mask,
            )

            if HAS_ALIBI:
                q_pos = (offs_m[:, None] + q_start).to(tl.float32)
                k_pos = (start_n + offs_n)[None, :].to(tl.float32)
                delta = k_pos - q_pos
                grad_alibi_acc += tl.sum(dS * delta)

        grad_q_ptrs = GradQ + off_b * stride_gqb + off_h * stride_gqh + (
            offs_m[:, None] * stride_gqm + offs_k[None, :] * stride_gqk
        )
        tl.store(
            grad_q_ptrs,
            dq.to(GradQ.dtype.element_ty),
            mask=q_mask,
        )

        if ACCUM_ALIBI:
            tl.atomic_add(GradAlibi + off_h, grad_alibi_acc)

class FusedOnlineAttention(nn.Module):
    def _mask_supported_for_triton(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_len_q: int,
        seq_len_k: int,
    ) -> bool:
        try:
            _ = self._prepare_triton_mask(
                attention_mask,
                batch_size,
                self.num_heads,
                seq_len_q,
                seq_len_k,
                attention_mask.device,
            )
            return True
        except Exception:
            return False

    def _can_use_triton_backward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        go: torch.Tensor,
        lse: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        alibi_used: bool,
    ) -> bool:
        """Return True when we can run the Triton backward implementation."""
        if not TRITON_AVAILABLE:
            return False
        if not (q.is_cuda and k.is_cuda and v.is_cuda and go.is_cuda):
            return False
        if self.world_size > 1:
            # Multi-host/GPU path still relies on Python fallback.
            return False
        # Dropout is disabled in autograd path (enforced in forward).
        batch_size, num_heads, seq_len_q, _ = q.shape
        seq_len_k = k.shape[2]
        if attention_mask is not None:
            try:
                self._prepare_triton_mask(
                    attention_mask,
                    batch_size,
                    num_heads,
                    seq_len_q,
                    seq_len_k,
                    q.device,
                )
            except Exception:
                return False
        if alibi_used and not hasattr(self, "scale"):
            return False
        return True

    def _backward_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        go: torch.Tensor,
        lse: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        full_seq_len_q: int,
        q_start: int,
        causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_k = k.shape[2]

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        go = go.contiguous()
        lse = lse.contiguous()

        dQ = torch.zeros_like(q, dtype=torch.float32)
        dK = torch.zeros_like(k, dtype=torch.float32)
        dV = torch.zeros_like(v, dtype=torch.float32)

        mask_tensor = None
        stride_mb = stride_mh = stride_mm = stride_mn = 0
        if attention_mask is not None:
            mask_tensor = self._prepare_triton_mask(
                attention_mask,
                batch_size,
                num_heads,
                seq_len_q,
                seq_len_k,
                q.device,
            )
            stride_mb, stride_mh, stride_mm, stride_mn = mask_tensor.stride()
        mask_ptr = mask_tensor if mask_tensor is not None else q

        has_alibi = alibi_slopes is not None
        if has_alibi:
            alibi_in = alibi_slopes.to(q.device, dtype=torch.float32).contiguous()
            grad_alibi = torch.zeros_like(alibi_in)
            stride_as = alibi_in.stride(0)
            stride_ag = grad_alibi.stride(0)
        else:
            alibi_in = torch.empty(0, device=q.device, dtype=torch.float32)
            grad_alibi = torch.empty(0, device=q.device, dtype=torch.float32)
            stride_as = stride_ag = 0

        grid = (
            triton.cdiv(seq_len_q, self.tile_size_q),
            batch_size,
            num_heads,
        )

        fused_online_attention_bwd_kernel[grid](
            q,
            k,
            v,
            dQ,
            dK,
            dV,
            go,
            lse,
            mask_ptr,
            alibi_in,
            grad_alibi,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            dQ.stride(3),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dK.stride(3),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            dV.stride(3),
            go.stride(0),
            go.stride(1),
            go.stride(2),
            go.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            stride_mb,
            stride_mh,
            stride_mm,
            stride_mn,
            stride_as,
            stride_ag,
            self.scale,
            full_seq_len_q,
            seq_len_k,
            q_start,
            H=num_heads,
            M=seq_len_q,
            N=seq_len_k,
            D=head_dim,
            TILE_M=self.tile_size_q,
            TILE_N=self.tile_size_k,
            IS_CAUSAL=causal,
            HAS_MASK=mask_tensor is not None,
            HAS_ALIBI=has_alibi,
            USE_WGMMA=self.sm >= 90,
            USE_TMA=self.sm >= 90,
            USE_CP_ASYNC=self.sm >= 80 and self.sm < 90,
        )

        grad_alibi_out: Optional[torch.Tensor]
        if has_alibi:
            grad_alibi_out = grad_alibi
        else:
            grad_alibi_out = None

        return dQ, dK, dV, grad_alibi_out

    """
    Production-ready Fused Online Attention module
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        tile_size_q: int = 128,
        tile_size_k: int = 64,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.tile_size_q = tile_size_q
        self.tile_size_k = tile_size_k
        self.dropout = dropout
        self.scale = scale or (1.0 / math.sqrt(head_dim))
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.deterministic = False
        self._det_seed: Optional[int] = None
        self._det_offset: int = 0
        if self.device.type == "cuda":
            cap = torch.cuda.get_device_capability(self.device)
            self.sm = cap[0] * 10 + cap[1]
        else:
            self.sm = 0
        self.verify = os.getenv("STREAM_ATTN_VERIFY", "0") in ("1", "true", "True", "yes", "on")
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(
            f"FusedOnlineAttention initialized: heads={num_heads}, dim={head_dim}, tile_q={tile_size_q}, tile_k={tile_size_k}, world_size={self.world_size}, sm={self.sm}, triton={TRITON_AVAILABLE}"
        )

    def set_deterministic(self, enabled: bool, seed: Optional[int] = None):
        """Enable/disable deterministic mode for Triton dropout RNG."""
        self.deterministic = enabled
        if enabled:
            if seed is None:
                seed = torch.initial_seed()
            self._det_seed = int(seed & 0xFFFFFFFF)
            self._det_offset = 0
        else:
            self._det_seed = None
            self._det_offset = 0

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        return_lse: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        deterministic: Optional[bool] = None,
    ) -> torch.Tensor:
        batch_size, seq_len_q, num_heads_q, head_dim_q = query.shape
        _, seq_len_k, num_heads_k, head_dim_k = key.shape
        assert num_heads_q == num_heads_k == self.num_heads
        assert head_dim_q == head_dim_k == self.head_dim

        if alibi_slopes is not None:
            if not isinstance(alibi_slopes, torch.Tensor):
                raise ValueError('alibi_slopes must be a Tensor of shape [num_heads]')
            if alibi_slopes.numel() != self.num_heads:
                raise ValueError(
                    f'alibi_slopes must have length {self.num_heads}, got {alibi_slopes.numel()}'
                )

        full_seq_len_q = seq_len_q
        start_idx = 0
        if self.world_size > 1:
            queries_per_gpu = seq_len_q // self.world_size
            start_idx = self.rank * queries_per_gpu
            end_idx = (
                start_idx + queries_per_gpu if self.rank < self.world_size - 1 else seq_len_q
            )
            query = query[:, start_idx:end_idx]
            seq_len_q = query.shape[1]
        
        effective_dropout = dropout_p if self.training else 0.0
        deterministic_mode = self.deterministic if deterministic is None else deterministic

        mask_supported = attention_mask is None
        if not mask_supported:
            dims = attention_mask.dim()
            if dims == 2:
                mask_supported = attention_mask.shape[0] == batch_size and attention_mask.shape[1] == seq_len_k
            elif dims == 3:
                mask_supported = (
                    attention_mask.shape[0] == batch_size
                    and attention_mask.shape[1] == seq_len_q
                    and attention_mask.shape[2] == seq_len_k
                )
            elif dims == 4:
                mask_supported = attention_mask.shape[0] == batch_size and attention_mask.shape[-1] == seq_len_k
            else:
                mask_supported = False

        use_triton = (
            TRITON_AVAILABLE
            and query.is_cuda
            and key.is_cuda
            and value.is_cuda
            and mask_supported
        )

        requires_grad = (
            torch.is_grad_enabled()
            and (query.requires_grad or key.requires_grad or value.requires_grad)
        )

        if use_triton and requires_grad:
            if effective_dropout != 0.0 or return_lse:
                use_triton = False
            else:
                output = FusedOnlineAttentionFunction.apply(
                    self,
                    query,
                    key,
                    value,
                    bool(causal),
                    attention_mask,
                    float(effective_dropout),
                    alibi_slopes,
                    deterministic_mode,
                    full_seq_len_q,
                    start_idx,
                )
                return output

        if use_triton:
            return self._forward_triton(
                query,
                key,
                value,
                causal=causal,
                attention_mask=attention_mask,
                dropout_p=effective_dropout,
                alibi_slopes=alibi_slopes,
                deterministic_mode=deterministic_mode,
                full_seq_len_q=full_seq_len_q,
                q_start=start_idx,
                return_lse=return_lse,
                metadata=None,
            )
        else:
            # Fallback to PyTorch SDPA
            q = query.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_heads, seq_len_q, self.head_dim
            )
            k = key.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_heads, seq_len_k, self.head_dim
            )
            v = value.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_heads, seq_len_k, self.head_dim
            )

            add_mask = None
            if attention_mask is not None:
                attn_mask_bh = self._prepare_attn_mask(
                    attention_mask,
                    batch_size,
                    self.num_heads,
                    seq_len_q,
                    seq_len_k,
                    q.device,
                    q.dtype,
                )
                if attn_mask_bh.dtype == torch.bool:
                    add_mask = torch.where(
                        attn_mask_bh,
                        torch.full((1,), float('-inf'), dtype=q.dtype, device=q.device),
                        torch.zeros(1, dtype=q.dtype, device=q.device),
                    )
                else:
                    add_mask = attn_mask_bh.to(q.dtype)

            if alibi_slopes is not None:
                slopes = alibi_slopes.to(q.device, dtype=torch.float32)
                pos_q = torch.arange(seq_len_q, device=q.device, dtype=torch.float32)
                pos_k = torch.arange(seq_len_k, device=q.device, dtype=torch.float32)
                delta = pos_k.unsqueeze(0) - pos_q.unsqueeze(1)
                bias_h = slopes.view(self.num_heads, 1, 1) * delta
                bias_bh = bias_h.unsqueeze(0).expand(batch_size, self.num_heads, seq_len_q, seq_len_k)
                bias_bh = bias_bh.reshape(batch_size * self.num_heads, seq_len_q, seq_len_k).to(q.dtype)
                add_mask = bias_bh if add_mask is None else add_mask + bias_bh

            is_causal = causal
            if add_mask is not None and causal:
                tri = torch.triu(
                    torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=q.device),
                    diagonal=1,
                ).unsqueeze(0)
                tri = tri.expand(batch_size * self.num_heads, seq_len_q, seq_len_k)
                tri_add = torch.where(
                    tri,
                    torch.full((1,), float('-inf'), dtype=q.dtype, device=q.device),
                    torch.zeros(1, dtype=q.dtype, device=q.device),
                )
                add_mask = add_mask + tri_add
                is_causal = False

            sdpa_kwargs = dict(attn_mask=add_mask, is_causal=is_causal, dropout_p=effective_dropout)

            sdpa_ctx = nullcontext()
            if q.is_cuda:
                try:
                    sdpa_ctx = torch.backends.cuda.sdp_kernel(
                        enable_math=True,
                        enable_flash=False,
                        enable_mem_efficient=False,
                    )
                except Exception:  # pragma: no cover - environment dependent
                    sdpa_ctx = nullcontext()
            with sdpa_ctx:
                out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)

            out = (
                out.reshape(batch_size, self.num_heads, seq_len_q, self.head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            return (out, None) if return_lse else out


    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = attention_mask
        if mask.dtype == torch.bool:
            mask = mask.to(torch.float32)
            mask = mask.masked_fill(mask > 0, float('-inf'))
        else:
            mask = mask.to(dtype)
        if mask.dim() == 2:
            mask = mask.view(batch_size, 1, 1, seq_len_k)
        elif mask.dim() == 3:
            mask = mask.view(batch_size, 1, seq_len_q, seq_len_k)
        elif mask.dim() == 4:
            pass
        else:
            raise ValueError(
                "Unsupported attention_mask shape. Expected 2D, 3D, or 4D tensor."
            )
        bh_mask = mask.expand(
            batch_size,
            num_heads,
            mask.shape[-2] if mask.dim() == 4 else seq_len_q,
            seq_len_k,
        )
        bh_mask = bh_mask.reshape(
            batch_size * num_heads, bh_mask.shape[-2], bh_mask.shape[-1]
        ).to(device)
        return bh_mask

    def _prepare_triton_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = attention_mask
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            mask = mask[:, None, :, :]
        elif mask.dim() != 4:
            raise ValueError("Unsupported attention_mask shape. Expected 2D, 3D, or 4D tensor.")

        mask = mask.to(device)
        mask = mask.expand(batch_size, num_heads, seq_len_q, seq_len_k).contiguous()
        if mask.dtype == torch.bool:
            mask = mask.to(torch.float32)
            mask = mask.masked_fill(mask > 0, float('-inf'))
        else:
            mask = mask.to(torch.float32)
        return mask


    def _prepare_triton_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = attention_mask
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            mask = mask[:, None, :, :]
        elif mask.dim() != 4:
            raise ValueError("Unsupported attention_mask shape. Expected 2D, 3D, or 4D tensor.")

        mask = mask.to(device)
        mask = mask.expand(batch_size, num_heads, seq_len_q, seq_len_k).contiguous()
        if mask.dtype == torch.bool:
            mask = mask.to(torch.float32)
            mask = mask.masked_fill(mask > 0, float('-inf'))
        else:
            mask = mask.to(torch.float32)
        return mask

    def _forward_triton(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
        attention_mask: Optional[torch.Tensor],
        dropout_p: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic_mode: bool,
        full_seq_len_q: int,
        q_start: int,
        return_lse: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        batch_size, seq_len_q = query.shape[0], query.shape[1]
        seq_len_k = key.shape[1]

        output = torch.empty_like(query)
        lse = torch.empty(
            (batch_size, self.num_heads, seq_len_q),
            dtype=torch.float32,
            device=query.device,
        )

        grid = lambda meta: (
            triton.cdiv(seq_len_q, meta["TILE_M"]),
            batch_size,
            self.num_heads,
        )

        mask_tensor = None
        has_mask = attention_mask is not None
        if has_mask:
            mask_tensor = self._prepare_triton_mask(
                attention_mask,
                batch_size,
                self.num_heads,
                seq_len_q,
                seq_len_k,
                query.device,
            )
            mask_ptr = mask_tensor
            stride_mb, stride_mh, stride_mm, stride_mn = mask_ptr.stride()
        else:
            mask_ptr = query
            stride_mb = stride_mh = stride_mm = stride_mn = 0

        has_alibi = alibi_slopes is not None
        if has_alibi:
            alibi_ptr = alibi_slopes.to(query.device, dtype=torch.float32).contiguous()
        else:
            alibi_ptr = query

        has_dropout = dropout_p > 0.0
        if has_dropout:
            if deterministic_mode:
                if self._det_seed is None:
                    self._det_seed = int(torch.initial_seed() & 0xFFFFFFFF)
                    self._det_offset = 0
                rng_seed = self._det_seed
                rng_offset = self._det_offset
                consumed = batch_size * self.num_heads * full_seq_len_q * seq_len_k
                self._det_offset += consumed
            else:
                rng_seed = int(
                    torch.randint(
                        0,
                        2**31,
                        (1,),
                        device=query.device,
                        dtype=torch.int64,
                    ).item()
                )
                rng_offset = 0
            dropout_scale = 1.0 / (1.0 - dropout_p)
        else:
            rng_seed = 0
            rng_offset = 0
            dropout_scale = 1.0

        fused_online_attention_kernel[grid](
            query,
            key,
            value,
            output,
            lse,
            mask_ptr,
            query.stride(0),
            query.stride(2),
            query.stride(1),
            query.stride(3),
            key.stride(0),
            key.stride(2),
            key.stride(1),
            key.stride(3),
            value.stride(0),
            value.stride(2),
            value.stride(1),
            value.stride(3),
            output.stride(0),
            output.stride(2),
            output.stride(1),
            output.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            stride_mb,
            stride_mh,
            stride_mm,
            stride_mn,
            float(dropout_p),
            dropout_scale,
            int(rng_seed),
            int(rng_offset),
            alibi_ptr,
            full_seq_len_q,
            seq_len_k,
            q_start,
            H=self.num_heads,
            M=seq_len_q,
            N=seq_len_k,
            D=self.head_dim,
            scale=self.scale,
            IS_CAUSAL=causal,
            HAS_MASK=has_mask,
            HAS_DROPOUT=has_dropout,
            HAS_ALIBI=has_alibi,
            USE_WGMMA=self.sm >= 90,
            USE_TMA=self.sm >= 90,
            USE_CP_ASYNC=self.sm >= 80 and self.sm < 90,
        )

        if self.world_size > 1:
            output_list = [torch.empty_like(output) for _ in range(self.world_size)]
            dist.all_gather(output_list, output)
            output = torch.cat(output_list, dim=1)

        if self.verify:
            self._verify_output(
                query,
                key,
                value,
                output,
                causal,
                attention_mask,
                dropout_p,
                alibi_slopes,
            )

        if metadata is not None:
            metadata["mask"] = mask_tensor
            metadata["has_mask"] = has_mask
            metadata["has_dropout"] = has_dropout
            metadata["dropout_p"] = float(dropout_p)
            metadata["dropout_scale"] = float(dropout_scale)
            metadata["rng_seed"] = int(rng_seed)
            metadata["rng_offset"] = int(rng_offset)
            metadata["has_alibi"] = has_alibi
            metadata["full_seq_len_q"] = int(full_seq_len_q)
            metadata["q_start"] = int(q_start)

        return (output, lse) if return_lse else output

    def _verify_output(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        causal: bool,
        attention_mask: Optional[torch.Tensor],
        dropout_p: float,
        alibi_slopes: Optional[torch.Tensor],
    ) -> None:
        """Compare Triton output against PyTorch reference."""
        if dropout_p > 0.0:
            # Skip verification when dropout introduces randomness.
            return

        bsz, sq, _, _ = query.shape
        sk = key.shape[1]
        q = query.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, sq, self.head_dim)
        k = key.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, sk, self.head_dim)
        v = value.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, sk, self.head_dim)

        add_mask = None
        if attention_mask is not None:
            attn_mask_bh = self._prepare_attn_mask(
                attention_mask,
                bsz,
                self.num_heads,
                sq,
                sk,
                q.device,
                q.dtype,
            )
            if attn_mask_bh.dtype == torch.bool:
                add_mask = torch.where(
                    attn_mask_bh,
                    torch.full((1,), float('-inf'), dtype=q.dtype, device=q.device),
                    torch.zeros(1, dtype=q.dtype, device=q.device),
                )
            else:
                add_mask = attn_mask_bh.to(q.dtype)

        if alibi_slopes is not None:
            slopes = alibi_slopes.to(q.device, dtype=torch.float32)
            pos_q = torch.arange(sq, device=q.device, dtype=torch.float32)
            pos_k = torch.arange(sk, device=q.device, dtype=torch.float32)
            delta = pos_k.unsqueeze(0) - pos_q.unsqueeze(1)
            bias_h = slopes.view(self.num_heads, 1, 1) * delta
            bias_bh = bias_h.unsqueeze(0).expand(bsz, self.num_heads, sq, sk)
            bias_bh = bias_bh.reshape(bsz * self.num_heads, sq, sk).to(q.dtype)
            add_mask = bias_bh if add_mask is None else add_mask + bias_bh

        is_causal = causal
        if add_mask is not None and causal:
            tri = torch.triu(
                torch.ones(sq, sk, dtype=torch.bool, device=q.device), diagonal=1
            ).unsqueeze(0)
            tri = tri.expand(bsz * self.num_heads, sq, sk)
            tri_add = torch.where(
                tri,
                torch.full((1,), float('-inf'), dtype=q.dtype, device=q.device),
                torch.zeros(1, dtype=q.dtype, device=q.device),
            )
            add_mask = add_mask + tri_add
            is_causal = False

        sdpa_kwargs = dict(attn_mask=add_mask, is_causal=is_causal, dropout_p=0.0)
        ref = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        ref = (
            ref.reshape(bsz, self.num_heads, sq, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    @torch.no_grad()
    def benchmark(
        self, seq_len: int, batch_size: int = 1, warmup: int = 10, iterations: int = 100
    ) -> Dict[str, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.dtype if device.type == "cuda" else torch.float32
        nh = self.num_heads
        hd = self.head_dim
        q = torch.randn(batch_size, seq_len, nh, hd, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for _ in range(warmup):
            _ = self.forward(q, k, v, causal=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        import time
        start = time.time()
        for _ in range(iterations):
            _ = self.forward(q, k, v, causal=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / iterations
        flops = 4.0 * batch_size * nh * seq_len * seq_len * hd
        tflops = flops / elapsed / 1e12
        bytes_per_el = torch.tensor([], dtype=dtype).element_size()
        memory_bytes = 3 * batch_size * seq_len * nh * hd * bytes_per_el
        bandwidth = memory_bytes / elapsed / 1e9
        return {
            "time_ms": elapsed * 1000.0,
            "tflops": tflops,
            "bandwidth_gb_s": bandwidth,
            "seq_len": seq_len,
            "batch_size": batch_size,
        }


class FusedOnlineAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module: "FusedOnlineAttention",
        query,
        key,
        value,
        causal: bool,
        attention_mask: Optional[torch.Tensor],
        dropout_p: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic_mode: bool,
        full_seq_len_q: int,
        q_start: int,
    ):
        metadata: Dict[str, Any] = {}
        output, lse = module._forward_triton(
            query,
            key,
            value,
            causal=causal,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            alibi_slopes=alibi_slopes,
            deterministic_mode=deterministic_mode,
            full_seq_len_q=full_seq_len_q,
            q_start=q_start,
            return_lse=True,
            metadata=metadata,
        )

        ctx.module = module
        ctx.causal = bool(causal)
        ctx._metadata = metadata
        ctx.dropout_p = float(dropout_p)
        ctx.training_flag = module.training

        if alibi_slopes is not None:
            alibi_tensor = alibi_slopes.to(query.device)
        else:
            alibi_tensor = query.new_empty(0, device=query.device)

        ctx.save_for_backward(query, key, value, lse, alibi_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        module: FusedOnlineAttention = ctx.module
        query, key, value, lse, alibi_tensor = ctx.saved_tensors
        metadata = getattr(ctx, "_metadata", None)
        dropout_p = float(ctx.dropout_p)

        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        nh = module.num_heads
        hd = module.head_dim

        grad_output = grad_output.contiguous()
        use_triton = (
            metadata is not None
            and TRITON_AVAILABLE
            and grad_output.is_cuda
            and query.is_cuda
            and module.world_size == 1
        )

        grad_query = grad_key = grad_value = grad_alibi = None

        if use_triton:
            has_mask = metadata["has_mask"]
            mask_tensor = metadata["mask"] if has_mask else query
            if has_mask:
                mask_tensor = mask_tensor.contiguous()
                stride_mb, stride_mh, stride_mm, stride_mn = mask_tensor.stride()
            else:
                stride_mb = stride_mh = stride_mm = stride_mn = 0

            has_dropout = metadata["has_dropout"]
            dropout_scale = float(metadata["dropout_scale"]) if has_dropout else 1.0
            rng_seed = int(metadata["rng_seed"]) if has_dropout else 0
            rng_offset = int(metadata["rng_offset"]) if has_dropout else 0

            has_alibi = metadata["has_alibi"]
            accum_alibi = has_alibi and ctx.needs_input_grad[7]
            if has_alibi:
                alibi_float = alibi_tensor.to(torch.float32).contiguous()
            else:
                alibi_float = query

            grad_alibi_buffer = (
                torch.zeros(module.num_heads, device=query.device, dtype=torch.float32)
                if accum_alibi
                else torch.empty(1, device=query.device, dtype=torch.float32)
            )

            query_ = query.contiguous()
            key_ = key.contiguous()
            value_ = value.contiguous()
            lse_ = lse.contiguous()

            grad_query = torch.empty_like(query_)
            grad_key = torch.zeros_like(key_)
            grad_value = torch.zeros_like(value_)

            seq_len_q = query_.shape[1]
            seq_len_k = key_.shape[1]

            grid = (
                triton.cdiv(seq_len_q, module.tile_size_q),
                query_.shape[0],
                module.num_heads,
            )

            fused_online_attention_bwd_kernel[grid](
                query_,
                key_,
                value_,
                grad_output,
                lse_,
                mask_tensor,
                grad_query,
                grad_key,
                grad_value,
                grad_alibi_buffer,
                query_.stride(0),
                query_.stride(2),
                query_.stride(1),
                query_.stride(3),
                key_.stride(0),
                key_.stride(2),
                key_.stride(1),
                key_.stride(3),
                value_.stride(0),
                value_.stride(2),
                value_.stride(1),
                value_.stride(3),
                grad_output.stride(0),
                grad_output.stride(2),
                grad_output.stride(1),
                grad_output.stride(3),
                lse_.stride(0),
                lse_.stride(1),
                lse_.stride(2),
                stride_mb,
                stride_mh,
                stride_mm,
                stride_mn,
                grad_query.stride(0),
                grad_query.stride(2),
                grad_query.stride(1),
                grad_query.stride(3),
                grad_key.stride(0),
                grad_key.stride(2),
                grad_key.stride(1),
                grad_key.stride(3),
                grad_value.stride(0),
                grad_value.stride(2),
                grad_value.stride(1),
                grad_value.stride(3),
                float(metadata["dropout_p"]) if has_dropout else 0.0,
                dropout_scale,
                rng_seed,
                rng_offset,
                alibi_float,
                int(metadata["full_seq_len_q"]),
                seq_len_k,
                int(metadata["q_start"]),
                H=module.num_heads,
                M=seq_len_q,
                N=seq_len_k,
                D=module.head_dim,
                TILE_M=module.tile_size_q,
                TILE_N=module.tile_size_k,
                scale=module.scale,
                IS_CAUSAL=ctx.causal,
                HAS_MASK=has_mask,
                HAS_DROPOUT=has_dropout,
                HAS_ALIBI=has_alibi,
                ACCUM_ALIBI=accum_alibi,
            )

            if accum_alibi:
                grad_alibi = grad_alibi_buffer.to(alibi_tensor.dtype)
            else:
                grad_alibi = None
        else:
            bsz, seq_len_q, _, _ = query.shape
            nh = module.num_heads
            hd = module.head_dim

            q_ref = query.permute(0, 2, 1, 3).reshape(bsz * nh, seq_len_q, hd).detach()
            k_ref = key.permute(0, 2, 1, 3).reshape(bsz * nh, seq_len_k, hd).detach()
            v_ref = value.permute(0, 2, 1, 3).reshape(bsz * nh, seq_len_k, hd).detach()
            q_ref.requires_grad = True
            k_ref.requires_grad = True
            v_ref.requires_grad = True

            add_mask = None
            if attention_mask is not None:
                attn_mask_bh = module._prepare_attn_mask(
                    attention_mask,
                    bsz,
                    nh,
                    seq_len_q,
                    seq_len_k,
                    q_ref.device,
                    q_ref.dtype,
                )
                if attn_mask_bh.dtype == torch.bool:
                    add_mask = torch.where(
                        attn_mask_bh,
                        torch.full((1,), float('-inf'), dtype=q_ref.dtype, device=q_ref.device),
                        torch.zeros(1, dtype=q_ref.dtype, device=q_ref.device),
                    )
                else:
                    add_mask = attn_mask_bh.to(q_ref.dtype)

            slopes_ref = None
            if alibi_tensor.numel() > 0:
                slopes_ref = alibi_tensor.detach().to(q_ref.dtype)
                slopes_ref.requires_grad = ctx.needs_input_grad[7]
                pos_q = torch.arange(seq_len_q, device=q_ref.device, dtype=q_ref.dtype)
                pos_k = torch.arange(seq_len_k, device=q_ref.device, dtype=q_ref.dtype)
                delta = pos_k.unsqueeze(0) - pos_q.unsqueeze(1)
                bias_h = slopes_ref.view(nh, 1, 1) * delta
                bias_bh = bias_h.unsqueeze(0).expand(bsz, nh, seq_len_q, seq_len_k)
                bias_bh = bias_bh.reshape(bsz * nh, seq_len_q, seq_len_k)
                add_mask = bias_bh if add_mask is None else add_mask + bias_bh

            is_causal = bool(ctx.causal)
            if add_mask is not None and ctx.causal:
                tri = torch.triu(
                    torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=q_ref.device),
                    diagonal=1,
                ).unsqueeze(0)
                tri = tri.expand(bsz * nh, seq_len_q, seq_len_k)
                tri_add = torch.where(
                    tri,
                    torch.full((1,), float('-inf'), dtype=q_ref.dtype, device=q_ref.device),
                    torch.zeros(1, dtype=q_ref.dtype, device=q_ref.device),
                )
                add_mask = add_mask + tri_add
                is_causal = False

            dropout = ctx.dropout_p if ctx.training_flag else 0.0
            sdpa_ctx = nullcontext()
            if q_ref.is_cuda:
                try:
                    sdpa_ctx = torch.backends.cuda.sdp_kernel(
                        enable_math=True,
                        enable_flash=False,
                        enable_mem_efficient=False,
                    )
                except Exception:  # pragma: no cover - environment dependent
                    sdpa_ctx = nullcontext()

            go = grad_output.permute(0, 2, 1, 3).reshape(bsz * nh, seq_len_q, hd)

            inputs = [q_ref, k_ref, v_ref]
            if slopes_ref is not None and ctx.needs_input_grad[7]:
                inputs.append(slopes_ref)

            with sdpa_ctx:
                y = F.scaled_dot_product_attention(
                    q_ref,
                    k_ref,
                    v_ref,
                    attn_mask=add_mask,
                    is_causal=is_causal,
                    dropout_p=dropout,
                )
            grads = torch.autograd.grad(y, inputs, go, allow_unused=False)

            grad_q = grads[0].reshape(bsz, nh, seq_len_q, hd).permute(0, 2, 1, 3).contiguous()
            grad_k = grads[1].reshape(bsz, nh, seq_len_k, hd).permute(0, 2, 1, 3).contiguous()
            grad_v = grads[2].reshape(bsz, nh, seq_len_k, hd).permute(0, 2, 1, 3).contiguous()
            if slopes_ref is not None and ctx.needs_input_grad[7]:
                grad_alibi = grads[-1]
            else:
                grad_alibi = None

        if grad_alibi is not None and grad_alibi.numel() > 0:
            grad_alibi = grad_alibi.to(alibi_tensor.dtype)
        else:
            grad_alibi = None

        return (
            None,
            grad_query,
            grad_key,
            grad_value,
            None,
            None,
            None,
            grad_alibi,
            None,
            None,
            None,
        )

def create_fused_online_attention(
    num_heads: int, head_dim: int, **kwargs
) -> FusedOnlineAttention:
    return FusedOnlineAttention(num_heads, head_dim, **kwargs) 



