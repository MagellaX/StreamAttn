"""Optional, forced-backend micro-prefill runner factories.

Contract: contiguous CUDA q[B,M,Hq,D], k/v[B,Hkv,N,D], fp16/bf16,
no gradients, no mask/dropout, scale=D**-0.5. Each factory returns a
zero-argument callable producing [B,M,Hq,D]. Inputs remain live views:
mutating their values between calls is supported; changing storage/shape is not.
Factory work is metadata-only except FA3's preallocated output. Warmup and
all backend work (including allocations and copies below) belong to the caller.

Source provenance (read against these exact revisions):
* Standalone Dao-AILab FA3, NOT FlashInfer backend="fa3": build hopper/ from
  flash-attention v2.8.3, commit 060c9188beec3a8b62b33a3bfa6d5d2d44975fab,
  against the installed torch==2.7.1+cu128 with --no-build-isolation --no-deps.
  https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/flash_attn_interface.py
  https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/flash_api.cpp
  https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/setup.py
  Unit last strides bypass maybe_contiguous; C++ uses row/head/batch strides.
  num_splits=0 and pack_gqa=None select upstream scheduling heuristics, not a
  Python KV repack. LSE/scheduling/split workspace is still backend work.
* xformers==0.0.31 from https://download.pytorch.org/whl/cu128 targets torch2.7.1.
  https://github.com/facebookresearch/xformers/blob/v0.0.31/xformers/ops/fmha/cutlass.py
  Its cutlassF-pt uses aten::_efficient_attention_forward. For GQA it launches
  each KV head group on a stream and torch.stack copies their OUTPUTS. Thus
  inputs are not repacked, but this is NOT a copy-free or single-launch runner.
  A CUTLASS directory does not prove compiled operator availability.
* PyTorch v2.7.1 forced cuDNN SDPA passes tensor sizes/strides to cuDNN,
  without the math backend's repeat_interleave GQA preprocessing:
  https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/transformers/attention.cpp
  https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cudnn/MHA.cpp
  https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
  Its eligibility gate rejects M==1 or N==1 and requires D<=128, D%8==0.
  A passing gate does not guarantee cuDNN graph/engine support; execution must
  succeed too. Source inspection cannot rule out opaque cuDNN engine workspace.

No factory pads, casts, repeats, makes inputs contiguous, or retries another
backend. Unsupported imports, APIs, shapes and execution errors propagate.
These are noncausal baselines; rectangular causal masks differ across backends.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature

import torch


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("Expected q[B,M,Hq,D] and k/v[B,Hkv,N,D]")
    if any(t.numel() == 0 for t in (q, k, v)):
        raise ValueError("Empty micro-prefill inputs are not supported")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("K/V shapes and Q/KV batch/head dimensions must match")
    if q.shape[2] % k.shape[1] != 0:
        raise ValueError("Hq must be divisible by Hkv")
    if any(not t.is_contiguous() for t in (q, k, v)):
        raise ValueError(
            "Inputs must already be contiguous BMHD Q and HND K/V; no repack"
        )
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("Inputs must have the same fp16 or bf16 dtype; no casting")
    if q.device != k.device or q.device != v.device or not q.is_cuda:
        raise ValueError("Inputs must be on the same CUDA device")
    if any(t.requires_grad for t in (q, k, v)):
        raise ValueError(
            "These forward-only benchmark runners do not support gradients"
        )


def fa3_runner(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """Standalone v2.8.3/hopper FA3; reuse output, retain true GQA and HND storage.

    The returned output aliases a runner-owned buffer overwritten on every call.
    This intentionally requires the pinned API, not the changed main-branch API.
    """
    _validate_inputs(q, k, v)
    if q.shape[-1] % 8 or q.shape[-1] > 256:
        raise ValueError("Standalone FA3 requires D%8==0 and D<=256; no padding")
    from flash_attn_interface import _flash_attn_forward

    kwargs = dict(
        q=q,
        k=k.transpose(1, 2),
        v=v.transpose(1, 2),
        k_new=None,
        v_new=None,
        qv=None,
        out=torch.empty_like(q),
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        page_table=None,
        kv_batch_idx=None,
        leftpad_k=None,
        rotary_cos=None,
        rotary_sin=None,
        seqlens_rotary=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=True,
        scheduler_metadata=None,
        num_splits=0,
        pack_gqa=None,
        sm_margin=0,
    )
    try:
        signature(_flash_attn_forward).bind(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "Standalone FA3 requires the v2.8.3/hopper _flash_attn_forward API"
        ) from exc

    def run() -> torch.Tensor:
        return _flash_attn_forward(**kwargs)[0]

    return run


def cutlass_runner(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """Force xFormers CUTLASS FwOp with zero-stride GQA5D input views.

    Upstream GQA group launches, stream synchronization, and output stack/copy
    remain INSIDE each invocation. No native single-launch GQA claim is made.
    """
    _validate_inputs(q, k, v)
    from xformers.ops.fmha import memory_efficient_attention_forward
    from xformers.ops.fmha.common import Inputs
    from xformers.ops.fmha.cutlass import FwOp

    b, m, hq, d = q.shape
    hkv, n = k.shape[1:3]
    heads_per_group = hq // hkv
    q5 = q.view(b, m, hkv, heads_per_group, d)
    k5 = k.transpose(1, 2).unsqueeze(3).expand(b, n, hkv, heads_per_group, d)
    v5 = v.transpose(1, 2).unsqueeze(3).expand(b, n, hkv, heads_per_group, d)
    scale = d**-0.5
    if FwOp.OPERATOR is None:
        raise RuntimeError(
            "xFormers CUTLASS operator is unavailable, irrespective of source directories"
        )
    reasons = FwOp.not_supported_reasons(
        Inputs(query=q5, key=k5, value=v5, p=0.0, scale=scale)
    )
    if reasons:
        raise RuntimeError("xFormers CUTLASS unsupported: " + "; ".join(reasons))

    def run() -> torch.Tensor:
        out = memory_efficient_attention_forward(
            q5, k5, v5, p=0.0, scale=scale, op=FwOp
        )
        # view fails on an incompatible output layout rather than copying it.
        return out.view(b, m, hq, d)

    return run


def cudnn_runner(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """Force only cuDNN SDPA, preserving unequal Q/KV head counts and strides."""
    _validate_inputs(q, k, v)
    from torch.backends.cuda import SDPAParams, can_use_cudnn_attention
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    qh = q.transpose(1, 2)
    enable_gqa = q.shape[2] != k.shape[1]
    scale = q.shape[-1] ** -0.5
    params = SDPAParams(qh, k, v, None, 0.0, False, enable_gqa)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        if not can_use_cudnn_attention(params, debug=True):
            raise RuntimeError(
                "cuDNN SDPA unsupported for these inputs; no fallback or GQA expansion"
            )

    def run() -> torch.Tensor:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = scaled_dot_product_attention(
                qh,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        return out.transpose(1, 2)

    return run
