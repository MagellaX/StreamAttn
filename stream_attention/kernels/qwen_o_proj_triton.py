"""Decode-shaped Qwen output projection kernels.

These kernels target the routed decode path where the attention output is
`[batch, 1, hidden]`.  They are an experimental floor probe: PyTorch's linear
path is excellent in general, but this tiny-M decode shape is worth testing
directly before we spend time on deeper projection fusion.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _qwen_o_proj_decode_kernel(
        X,
        W,
        Bias,
        Out,
        B: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        X_STRIDE_K: tl.constexpr,
        W_STRIDE_N: tl.constexpr,
        W_STRIDE_K: tl.constexpr,
        OUT_STRIDE_B: tl.constexpr,
        OUT_STRIDE_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            x = tl.load(
                X + offs_m[:, None] * X_STRIDE_B + k_idx[None, :] * X_STRIDE_K,
                mask=(offs_m[:, None] < B) & (k_idx[None, :] < K),
                other=0.0,
            )
            # torch.nn.functional.linear uses W[out_feature, in_feature].
            w = tl.load(
                W + offs_n[None, :] * W_STRIDE_N + k_idx[:, None] * W_STRIDE_K,
                mask=(offs_n[None, :] < N) & (k_idx[:, None] < K),
                other=0.0,
            )
            acc += tl.dot(x, w)

        if HAS_BIAS:
            bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]

        tl.store(
            Out + offs_m[:, None] * OUT_STRIDE_B + offs_n[None, :] * OUT_STRIDE_N,
            acc,
            mask=(offs_m[:, None] < B) & (offs_n[None, :] < N),
        )


def qwen_o_proj_triton_forward(
    attn_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Apply `F.linear` for `[batch, 1, hidden]` decode tensors.

    `weight` follows PyTorch's layout `[out_features, in_features]`.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not attn_output.is_cuda or not weight.is_cuda or (bias is not None and not bias.is_cuda):
        raise RuntimeError("qwen_o_proj_triton_forward requires CUDA tensors")
    if attn_output.dim() != 3 or attn_output.shape[1] != 1:
        raise ValueError("attn_output must be [batch, 1, hidden]")
    if weight.dim() != 2:
        raise ValueError("weight must be [out_features, in_features]")
    batch = int(attn_output.shape[0])
    in_features = int(attn_output.shape[2])
    out_features = int(weight.shape[0])
    if int(weight.shape[1]) != in_features:
        raise ValueError("weight in_features must match attn_output hidden size")
    if bias is not None and (bias.dim() != 1 or int(bias.numel()) != out_features):
        raise ValueError("bias must be [out_features]")
    if attn_output.device != weight.device or (bias is not None and bias.device != weight.device):
        raise ValueError("attn_output, weight, and bias must be on the same CUDA device")
    if attn_output.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("attn_output must be fp16 or bf16")
    if weight.dtype != attn_output.dtype:
        raise ValueError("weight dtype must match attn_output dtype")
    if bias is not None and bias.dtype != attn_output.dtype:
        raise ValueError("bias dtype must match attn_output dtype")
    if block_m < 16 or block_n < 16 or block_k < 16:
        raise ValueError("Triton dot block sizes must be at least 16")

    x = attn_output.contiguous()
    w = weight.contiguous()
    b = bias.contiguous() if bias is not None else w
    out = torch.empty((batch, 1, out_features), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(batch, block_m), triton.cdiv(out_features, block_n))
    _qwen_o_proj_decode_kernel[grid](
        x,
        w,
        b,
        out,
        B=batch,
        K=in_features,
        N=out_features,
        X_STRIDE_B=x.stride(0),
        X_STRIDE_K=x.stride(2),
        W_STRIDE_N=w.stride(0),
        W_STRIDE_K=w.stride(1),
        OUT_STRIDE_B=out.stride(0),
        OUT_STRIDE_N=out.stride(2),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
