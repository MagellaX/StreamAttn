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
        # Optional key padding mask [B, N] (bool as int32 1:valid,0:masked)
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
        stride_mn,
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

            # Key padding mask: valid_k=1 means keep; 0 means masked out
            if HAS_MASK:
                mask_ptrs = Mask + off_b * stride_mb + (start_n + offs_n) * stride_mn
                valid_k = tl.load(mask_ptrs, mask=(start_n + offs_n) < N, other=0)
                qk = tl.where(valid_k[None, :] != 0, qk, float("-inf"))

            # Online softmax update
            tile_max = tl.max(qk, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            correction = tl.exp(running_max - new_max)
            acc_num *= correction[:, None]
            acc_den *= correction

            exp_qk = tl.exp(qk - new_max[:, None])
            acc_num += tl.dot(exp_qk, v)
            acc_den += tl.sum(exp_qk, axis=1)
            running_max = new_max

        # Final output with safe denominator; handle rows with all keys masked
        denom_safe = tl.where(acc_den > 0, acc_den, 1.0)
        out = acc_num / denom_safe[:, None]

        out_ptrs = (
            Out
            + off_b * stride_ob
            + off_h * stride_oh
            + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
        )
        out_mask = (offs_m[:, None] < M) & (offs_k[None, :] < D)
        tl.store(out_ptrs, out.to(Out.dtype.element_ty), mask=out_mask)

        # LSE: set to -inf for fully masked rows (acc_den == 0)
        lse = tl.where(acc_den > 0, running_max + tl.log(acc_den), float("-inf"))
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        return_lse: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        batch_size, seq_len_q, num_heads_q, head_dim_q = query.shape
        _, seq_len_k, num_heads_k, head_dim_k = key.shape
        assert num_heads_q == num_heads_k == self.num_heads
        assert head_dim_q == head_dim_k == self.head_dim

        if self.world_size > 1:
            queries_per_gpu = seq_len_q // self.world_size
            start_idx = self.rank * queries_per_gpu
            end_idx = (
                start_idx + queries_per_gpu if self.rank < self.world_size - 1 else seq_len_q
            )
            query = query[:, start_idx:end_idx]
            seq_len_q = query.shape[1]

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
            and (dropout_p == 0.0 or not self.training)
        )

        if use_triton and (
            torch.is_grad_enabled()
            and (query.requires_grad or key.requires_grad or value.requires_grad)
        ):
            if return_lse:
                use_triton = False
            else:
                return FusedOnlineAttentionAutogradFn.apply(
                    self,
                    query,
                    key,
                    value,
                    bool(causal),
                    attention_mask,
                    float(dropout_p),
                )

        if use_triton:
            return self._forward_triton(
                query,
                key,
                value,
                causal=causal,
                attention_mask=attention_mask,
                dropout_p=dropout_p,
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

            if attn_mask_bh is not None:
                if attn_mask_bh.dtype == torch.bool:
                    add_mask = torch.where(
                        attn_mask_bh,
                        torch.full((1,), float("-inf"), dtype=q.dtype, device=q.device),
                        torch.zeros(1, dtype=q.dtype, device=q.device),
                    )
                else:
                    add_mask = attn_mask_bh
                if causal:
                    tri = torch.triu(
                        torch.ones(
                            seq_len_q, seq_len_k, dtype=torch.bool, device=q.device
                        ),
                        diagonal=1,
                    ).unsqueeze(0)
                    tri_add = torch.where(
                        tri,
                        torch.full((1,), float("-inf"), dtype=q.dtype, device=q.device),
                        torch.zeros(1, dtype=q.dtype, device=q.device),
                    )
                    add_mask = add_mask + tri_add
                sdpa_kwargs = dict(
                    attn_mask=add_mask,
                    is_causal=False,
                    dropout_p=(dropout_p if self.training else 0.0),
                )
            else:
                sdpa_kwargs = dict(
                    attn_mask=None,
                    is_causal=causal,
                    dropout_p=(dropout_p if self.training else 0.0),
                )

            if q.is_cuda:
                with torch.backends.cuda.sdp_kernel(
                    enable_math=True, enable_flash=True, enable_mem_efficient=False
                ):
                    out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
            else:
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

    def _forward_triton(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
        attention_mask: Optional[torch.Tensor],
        dropout_p: float,
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
        has_mask = attention_mask is not None
        if has_mask:
            if attention_mask.dtype == torch.bool:
                mask_norm = (~attention_mask).to(torch.int32)
            else:
                mask_norm = (attention_mask == 0).to(torch.int32)
            if (
                mask_norm.dim() != 2
                or mask_norm.shape[0] != batch_size
                or mask_norm.shape[1] != seq_len_k
            ):
                raise ValueError(
                    "attention_mask must be shape [batch, seq_len_k] for fused Triton path"
                )
            mask_norm = mask_norm.contiguous().to(query.device)
            mask_ptr = mask_norm
            stride_mb, stride_mn = mask_ptr.stride(0), mask_ptr.stride(1)
        else:
            mask_ptr = output  # dummy
            stride_mb, stride_mn = 0, 0
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
            stride_mn,
            H=self.num_heads,
            M=seq_len_q,
            N=seq_len_k,
            D=self.head_dim,
            TILE_K=self.head_dim,
            scale=self.scale,
            IS_CAUSAL=causal,
            HAS_MASK=has_mask,
            USE_WGMMA=self.sm >= 90,
            USE_TMA=self.sm >= 90,
            USE_CP_ASYNC=self.sm >= 80 and self.sm < 90,
        )
        if self.world_size > 1:
            output_list = [torch.empty_like(output) for _ in range(self.world_size)]
            dist.all_gather(output_list, output)
            output = torch.cat(output_list, dim=1)
        if self.verify:
            self._verify_output(query, key, value, output, causal, attention_mask, dropout_p)
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
    ) -> None:
        """Compare Triton output against PyTorch reference."""
        bsz, sq, _, _ = query.shape
        sk = key.shape[1]
        q = query.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, sq, self.head_dim)
        k = key.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, sk, self.head_dim)
        v = value.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, sk, self.head_dim)
        attn_mask_bh = None
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
        sdpa_kwargs = dict(
            attn_mask=attn_mask_bh,
            is_causal=causal if attn_mask_bh is None else False,
            dropout_p=0.0,
        )
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


class FusedOnlineAttentionAutogradFn(torch.autograd.Function):
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
    ):
        ctx.module = module
        ctx.causal = bool(causal)
        ctx.has_mask = attention_mask is not None
        ctx.attention_mask = attention_mask
        ctx.save_for_backward(query, key, value)
        with torch.no_grad():
            out = module._forward_triton(
                query,
                key,
                value,
                causal=causal,
                attention_mask=attention_mask,
                dropout_p=dropout_p,
                return_lse=False,
            )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        module: FusedOnlineAttention = ctx.module
        query, key, value = ctx.saved_tensors
        bsz, sq, nh, hd = query.shape
        sk = key.shape[1]

        q = (
            query.detach().requires_grad_(True).permute(0, 2, 1, 3).reshape(bsz * nh, sq, hd)
        )
        k = (
            key.detach().requires_grad_(True).permute(0, 2, 1, 3).reshape(bsz * nh, sk, hd)
        )
        v = (
            value.detach().requires_grad_(True).permute(0, 2, 1, 3).reshape(bsz * nh, sk, hd)
        )

        attn_mask_bh = None
        if ctx.has_mask and ctx.attention_mask is not None:
            mask = ctx.attention_mask
            if mask.dtype != torch.bool:
                mask = mask == 0  # numeric: 0 valid
            else:
                mask = ~mask  # boolean: True means masked -> invert
            mask = mask.view(bsz, 1, 1, sk).expand(bsz, 1, sq, sk)
            attn_mask_bh = mask.expand(bsz, nh, sq, sk).reshape(bsz * nh, sq, sk).to(q.device)

        add_mask = None
        if attn_mask_bh is not None:
            add_mask = torch.where(
                attn_mask_bh,
                torch.zeros(1, dtype=q.dtype, device=q.device),
                torch.full((1,), float("-inf"), dtype=q.dtype, device=q.device),
            )
            if ctx.causal:
                tri = torch.triu(
                    torch.ones(sq, sk, dtype=torch.bool, device=q.device), diagonal=1
                ).unsqueeze(0)
                tri_add = torch.where(
                    tri,
                    torch.full((1,), float("-inf"), dtype=q.dtype, device=q.device),
                    torch.zeros(1, dtype=q.dtype, device=q.device),
                )
                add_mask = add_mask + tri_add

        if q.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_math=True, enable_flash=True, enable_mem_efficient=False
            ):
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=(add_mask if add_mask is not None else None),
                    is_causal=(False if add_mask is not None else ctx.causal),
                    dropout_p=0.0,
                )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=(add_mask if add_mask is not None else None),
                is_causal=(False if add_mask is not None else ctx.causal),
                dropout_p=0.0,
            )

        y = y.reshape(bsz, nh, sq, hd).permute(0, 2, 1, 3).contiguous()
        grads = torch.autograd.grad(y, (q, k, v), grad_out, allow_unused=False)
        dq = grads[0].reshape(bsz, nh, sq, hd).permute(0, 2, 1, 3).contiguous()
        dk = grads[1].reshape(bsz, nh, sk, hd).permute(0, 2, 1, 3).contiguous()
        dv = grads[2].reshape(bsz, nh, sk, hd).permute(0, 2, 1, 3).contiguous()
        return None, dq, dk, dv, None, None, None


def create_fused_online_attention(
    num_heads: int, head_dim: int, **kwargs
) -> FusedOnlineAttention:
    return FusedOnlineAttention(num_heads, head_dim, **kwargs)
