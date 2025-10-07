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
from typing import Optional, Tuple, Dict
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
        TILE_K: tl.constexpr,
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

        # Accumulators
        running_max = tl.full([TILE_M], value=-float("inf"), dtype=tl.float32)
        acc_num = tl.zeros([TILE_M, D], dtype=tl.float32)
        acc_den = tl.zeros([TILE_M], dtype=tl.float32)

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
                v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            else:
                k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
                v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

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
                mask_vals = tl.load(mask_ptrs, mask=mask_mask, other=0.0)
                qk += mask_vals

            if HAS_ALIBI:
                slope = tl.load(AlibiSlopes + off_h).to(tl.float32)
                q_pos = (offs_m[:, None] + q_start).to(tl.float32)
                k_pos = (start_n + offs_n)[None, :].to(tl.float32)
                qk += slope * (k_pos - q_pos)

            # Online softmax update
            tile_max = tl.max(qk, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            correction = tl.exp(running_max - new_max)
            acc_num *= correction[:, None]
            acc_den *= correction

            exp_qk = tl.exp(qk - new_max[:, None])

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
            running_max = new_max

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


class FusedOnlineAttention(nn.Module):
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

        mask_supported = (attention_mask is None) or (
            attention_mask.dim() == 2
            and attention_mask.shape[0] == batch_size
            and attention_mask.shape[1] == seq_len_k
        )

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
        attn_mask_bh = None
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
                sdpa_ctx = torch.nn.attention.sdpa_kernel(
                    SDPBackend.FLASH_ATTENTION
                )
            except (AttributeError, TypeError):
                try:
                    sdpa_ctx = torch.backends.cuda.sdp_kernel(
                        enable_math=True,
                        enable_flash=True,
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
        if mask.dtype == torch.float16 or mask.dtype == torch.bfloat16:
            mask = mask.to(dtype)
        if mask.dtype == torch.bool:
            pass
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

        mask_ptr = query
        stride_mb = stride_mh = stride_mm = stride_mn = 0
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

        has_alibi = alibi_slopes is not None
        if has_alibi:
            if not isinstance(alibi_slopes, torch.Tensor):
                raise ValueError("alibi_slopes must be a Tensor of shape [num_heads]")
            if alibi_slopes.numel() != self.num_heads:
                raise ValueError(
                    f"alibi_slopes must have length {self.num_heads}, got {alibi_slopes.numel()}"
                )
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
            TILE_K=self.head_dim,
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
                    torch.full((1,), float("-inf"), dtype=q.dtype, device=q.device),
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
                torch.full((1,), float("-inf"), dtype=q.dtype, device=q.device),
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
        alibi_slopes: Optional[torch.Tensor],
        deterministic_mode: bool,
        full_seq_len_q: int,
        q_start: int,
    ):
        output, lse = module._forward_triton(
            query,
            key,
            value,
            causal=causal,
            attention_mask=attention_mask,
            dropout_p=0.0,
            alibi_slopes=alibi_slopes,
            deterministic_mode=deterministic_mode,
            full_seq_len_q=full_seq_len_q,
            q_start=q_start,
            return_lse=True,
        )

        if alibi_slopes is not None:
            alibi_tensor = alibi_slopes.to(query.device)
        else:
            alibi_tensor = query.new_empty(0)

        ctx.module = module
        ctx.causal = bool(causal)
        ctx.scale = module.scale
        ctx.tile_size_q = module.tile_size_q
        ctx.batch_size = query.shape[0]
        ctx.num_heads = query.shape[2]
        ctx.seq_len_q = query.shape[1]
        ctx.seq_len_k = key.shape[1]
        ctx.head_dim = query.shape[3]
        ctx.attention_mask = attention_mask
        ctx.alibi_used = alibi_slopes is not None

        ctx.save_for_backward(query, key, value, lse, alibi_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        module: FusedOnlineAttention = ctx.module
        query, key, value, lse, alibi_tensor = ctx.saved_tensors

        batch_size = ctx.batch_size
        num_heads = ctx.num_heads
        seq_len_q = ctx.seq_len_q
        seq_len_k = ctx.seq_len_k
        head_dim = ctx.head_dim
        scale = ctx.scale
        tile_m = ctx.tile_size_q
        causal = ctx.causal

        grad_output = grad_output.contiguous()

        q = query.permute(0, 2, 1, 3).contiguous()
        k = key.permute(0, 2, 1, 3).contiguous()
        v = value.permute(0, 2, 1, 3).contiguous()
        go = grad_output.permute(0, 2, 1, 3).contiguous()
        lse_bh = lse.to(torch.float32)

        q_float = q.to(torch.float32)
        k_float = k.to(torch.float32)
        v_float = v.to(torch.float32)
        go_float = go.to(torch.float32)

        dQ = torch.zeros_like(q_float)
        dK = torch.zeros_like(k_float)
        dV = torch.zeros_like(v_float)

        attention_mask = ctx.attention_mask
        mask_bh = None
        if attention_mask is not None:
            mask_tensor = module._prepare_triton_mask(
                attention_mask,
                batch_size,
                num_heads,
                seq_len_q,
                seq_len_k,
                query.device,
            )
            mask_bh = mask_tensor

        alibi_used = ctx.alibi_used
        if alibi_used:
            slopes = alibi_tensor.to(torch.float32)
            grad_alibi = torch.zeros_like(slopes)
        else:
            slopes = None
            grad_alibi = None

        pos_q = torch.arange(seq_len_q, device=query.device, dtype=torch.float32)
        pos_k = torch.arange(seq_len_k, device=query.device, dtype=torch.float32)

        for b in range(batch_size):
            for h in range(num_heads):
                q_b = q_float[b, h]
                k_b = k_float[b, h]
                v_b = v_float[b, h]
                go_b = go_float[b, h]
                lse_b = lse_bh[b, h]
                mask_b = mask_bh[b, h] if mask_bh is not None else None
                slope = slopes[h] if alibi_used else None

                for m_start in range(0, seq_len_q, tile_m):
                    m_end = min(m_start + tile_m, seq_len_q)
                    q_tile = q_b[m_start:m_end]
                    go_tile = go_b[m_start:m_end]
                    lse_tile = lse_b[m_start:m_end]

                    logits = torch.matmul(q_tile, k_b.transpose(0, 1)) * scale
                    if mask_b is not None:
                        logits = logits + mask_b[m_start:m_end]
                    if alibi_used:
                        delta = pos_k.unsqueeze(0) - pos_q[m_start:m_end].unsqueeze(1)
                        logits = logits + slope * delta
                    if causal:
                        row_idx = pos_q[m_start:m_end].unsqueeze(1)
                        causal_mask = pos_k.unsqueeze(0) > row_idx
                        logits = logits.masked_fill(causal_mask, float('-inf'))

                    exp_term = logits - lse_tile.unsqueeze(1)
                    probs = torch.exp(exp_term)
                    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))

                    dV[b, h] += torch.matmul(probs.transpose(0, 1), go_tile)

                    dP = torch.matmul(go_tile, v_b)
                    attn_dot = (dP * probs).sum(dim=1, keepdim=True)
                    dS = (dP - attn_dot) * probs

                    dQ[b, h, m_start:m_end] += torch.matmul(dS, k_b) * scale
                    dK[b, h] += torch.matmul(dS.transpose(0, 1), q_tile) * scale

                    if alibi_used:
                        delta = pos_k.unsqueeze(0) - pos_q[m_start:m_end].unsqueeze(1)
                        grad_alibi[h] += torch.sum(dS * delta)

        grad_query = dQ.to(query.dtype).permute(0, 2, 1, 3).contiguous()
        grad_key = dK.to(key.dtype).permute(0, 2, 1, 3).contiguous()
        grad_value = dV.to(value.dtype).permute(0, 2, 1, 3).contiguous()

        if alibi_used:
            grad_alibi_slopes = grad_alibi.to(alibi_tensor.dtype)
        else:
            grad_alibi_slopes = None

        return (
            None,
            grad_query,
            grad_key,
            grad_value,
            None,
            None,
            grad_alibi_slopes,
            None,
            None,
            None,
        )
def create_fused_online_attention(
    num_heads: int, head_dim: int, **kwargs
) -> FusedOnlineAttention:
    return FusedOnlineAttention(num_heads, head_dim, **kwargs)
