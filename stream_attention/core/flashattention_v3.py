import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from contextlib import nullcontext
import logging
try:
    from torch.nn.attention import SDPBackend
except ImportError:  # pragma: no cover - older PyTorch
    SDPBackend = None

logger = logging.getLogger(__name__)


def _use_flash_sdpa() -> bool:
    # Prefer flash kernel on CUDA when available
    return torch.cuda.is_available()


class FlashAttentionV3(nn.Module):
    """
    FlashAttention V3 wrapper using PyTorch SDPA backends.

    - Uses flash-kernel backed scaled_dot_product_attention on CUDA
    - Falls back to math/mem-efficient kernels elsewhere
    - API: inputs as [batch, seq_len, num_heads, head_dim]
    """

    def __init__(self, config=None):
        super().__init__()
        self.num_heads = getattr(config, "num_heads", None)
        self.head_dim = getattr(config, "head_dim", None)
        self.dropout = getattr(config, "dropout", 0.0) or 0.0
        self.dtype = (
            torch.float16 if getattr(config, "use_fp16", True) else torch.float32
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len_q, num_heads_q, head_dim_q = query.shape
        _, seq_len_kv, num_heads_k, head_dim_k = key.shape

        assert num_heads_q == num_heads_k, "Mismatched heads"
        assert head_dim_q == head_dim_k, "Mismatched head dim"

        # Prepare inputs for SDPA: [B, H, L, D]
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                mask = torch.where(attention_mask, 0.0, float("-inf")).to(q.dtype)
            else:
                mask = attention_mask.to(q.dtype)
            if mask.dim() == 2:
                attn_mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                attn_mask = mask[:, None, :, :]
            else:
                attn_mask = mask

        cpu_cast_back = None
        if q.device.type == "cpu" and q.dtype in (torch.float16, torch.bfloat16):
            cpu_cast_back = q.dtype
            q = q.float()
            k = k.float()
            v = v.float()
            if attn_mask is not None:
                attn_mask = attn_mask.float()

        sdpa_ctx = nullcontext()
        if _use_flash_sdpa() and q.device.type == "cuda":
            try:
                # Prefer the newer torch.nn.attention API when available
                sdpa_ctx = torch.nn.attention.sdpa_kernel(
                    SDPBackend.FLASH_ATTENTION
                )
            except (AttributeError, TypeError):
                # Fallback for older PyTorch releases
                try:
                    sdpa_ctx = torch.backends.cuda.sdp_kernel(
                        enable_math=True,
                        enable_flash=True,
                        enable_mem_efficient=False,
                    )
                except Exception as e:  # pragma: no cover - depends on env
                    # Gracefully degrade to default kernel selection when the
                    # CUDA SDPA context manager is unavailable or unsupported.
                    logger.debug(
                        "torch.backends.cuda.sdp_kernel unavailable or unsupported: %s",
                        e,
                    )
                    sdpa_ctx = nullcontext()

        try:
            with sdpa_ctx:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=causal,
                )
        except RuntimeError as e:  # pragma: no cover - device/kernel dependent
            # If a forced-flash configuration leads to "no available kernel",
            # retry without any forced backend so PyTorch can choose a valid one.
            logger.debug(
                "FlashAttention SDPA failed under forced settings, retrying default: %s",
                e,
            )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )





        if cpu_cast_back is not None:
            out = out.to(cpu_cast_back)

        out = out.permute(0, 2, 1, 3).contiguous()
        return out

    @torch.no_grad()
    def benchmark(
        self, seq_len: int, batch_size: int = 1, warmup: int = 10, iterations: int = 100
    ) -> Dict[str, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.dtype if device.type == "cuda" else torch.float32

        nh = self.num_heads or 8
        hd = self.head_dim or 64

        q = torch.randn(batch_size, seq_len, nh, hd, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, nh, hd, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, nh, hd, device=device, dtype=dtype)

        # Warmup
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

        # FLOPS: 4 * B * H * Q * K * D approximately for attention
        flops = (
            4.0
            * batch_size
            * (self.num_heads or nh)
            * seq_len
            * seq_len
            * (self.head_dim or hd)
        )
        tflops = flops / elapsed / 1e12
        bytes_per_el = torch.tensor([], dtype=dtype).element_size()
        memory_bytes = (
            3
            * batch_size
            * seq_len
            * (self.num_heads or nh)
            * (self.head_dim or hd)
            * bytes_per_el
        )
        bandwidth = memory_bytes / elapsed / 1e9

        return {
            "time_ms": elapsed * 1000.0,
            "tflops": tflops,
            "bandwidth_gb_s": bandwidth,
            "seq_len": seq_len,
            "batch_size": batch_size,
        }
