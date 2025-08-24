"""Multihead-style wrapper around FusedOnlineAttention.

This module provides :class:`StreamMultiheadAttention`, mirroring the
constructor and forward signature of :class:`torch.nn.MultiheadAttention`
while delegating the actual attention computation to
:class:`~stream_attention.core.fused_online_attention.FusedOnlineAttention`.

A convenience factory :func:`create_stream_attention` is also provided to
instantiate the optimal backend automatically. If Triton kernels are
available, the fused implementation is used; otherwise it falls back to
PyTorch's native ``nn.MultiheadAttention`` which relies on SDPA.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .fused_online_attention import FusedOnlineAttention, TRITON_AVAILABLE


class StreamMultiheadAttention(nn.Module):
    """Drop-in replacement for :class:`torch.nn.MultiheadAttention`.

    Parameters mirror those of ``nn.MultiheadAttention`` but the module uses
    :class:`FusedOnlineAttention` internally when available.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.inner = FusedOnlineAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            dropout=dropout,
            dtype=dtype or torch.float32,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention.

        Arguments are equivalent to ``nn.MultiheadAttention``. Masks other than
        ``key_padding_mask`` are currently unsupported and will raise.
        """

        if attn_mask is not None:
            raise NotImplementedError(
                "attn_mask is not supported in StreamMultiheadAttention"
            )

        if need_weights:
            raise NotImplementedError(
                "need_weights=True is not supported; set need_weights=False (default)"
            )

        if not self.batch_first:
            # (L, N, E) -> (N, L, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        bsz, tgt_len, _ = query.shape
        src_len = key.shape[1]

        q = self.q_proj(query).view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(bsz, src_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(bsz, src_len, self.num_heads, self.head_dim)

        attn_mask_inner = key_padding_mask if key_padding_mask is not None else None
        attn_output = self.inner(
            query=q,
            key=k,
            value=v,
            causal=bool(is_causal),
            attention_mask=attn_mask_inner,
            dropout_p=(self.inner.dropout if self.training else 0.0),
        )

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, None


def create_stream_attention(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    bias: bool = True,
    batch_first: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Factory that returns an attention module using the best available backend.

    If Triton fused kernels are available, returns :class:`StreamMultiheadAttention`.
    Otherwise, falls back to PyTorch's ``nn.MultiheadAttention`` which utilises
    SDPA under the hood.
    """

    if TRITON_AVAILABLE and torch.cuda.is_available():
        return StreamMultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
    return nn.MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=dropout,
        bias=bias,
        batch_first=batch_first,
        device=device,
        dtype=dtype,
    )
