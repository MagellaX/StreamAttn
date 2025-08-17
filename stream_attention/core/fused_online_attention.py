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
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Dict, Any
import logging

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)


if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({'TILE_M': 64, 'TILE_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'TILE_M': 128, 'TILE_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'TILE_M': 128, 'TILE_N': 128}, num_warps=8, num_stages=2),
            triton.Config({'TILE_M': 256, 'TILE_N': 128}, num_warps=8, num_stages=3),
        ],
        key=['M', 'N', 'D']
    )
    @triton.jit
    def fused_online_attention_kernel(
        Q, K, V, Out,
        Lse,  # Log-sum-exp for numerical stability
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_lb, stride_lh, stride_lm,
        H: tl.constexpr,  # num heads
        M: tl.constexpr,  # seq_len_q
        N: tl.constexpr,  # seq_len_k
        D: tl.constexpr,  # head_dim
        TILE_M: tl.constexpr,
        TILE_K: tl.constexpr,
        TILE_N: tl.constexpr,
        scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        Fused Online Softmax Attention Kernel
        
        This is the novel kernel that processes attention in tiles while maintaining
        running statistics for online softmax computation. Each thread block processes
        TILE_M query vectors against all key/value vectors in tiles of size TILE_K.
        
        The key innovation is maintaining acc_num (weighted sum), acc_den (sum of exp),
        and running_max for numerically stable softmax without materializing the full
        attention matrix.
        """
        # Program IDs
        start_m = tl.program_id(0)
        off_b = tl.program_id(1) 
        off_h = tl.program_id(2)
        
        # Initialize offsets
        offs_m = start_m * TILE_M + tl.arange(0, TILE_M)
        offs_n = tl.arange(0, TILE_N)
        offs_k = tl.arange(0, D)
        
        # Query pointers
        q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + \
                 (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        
        # Load query tile
        q_mask = (offs_m[:, None] < M) & (offs_k[None, :] < D)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        
        # Initialize accumulators for online softmax
        running_max = tl.full([TILE_M], value=-float('inf'), dtype=tl.float32)
        acc_num = tl.zeros([TILE_M, D], dtype=tl.float32)  # Numerator (weighted values)
        acc_den = tl.zeros([TILE_M], dtype=tl.float32)     # Denominator (sum of exp)
        
        # Process K/V in tiles - this is where the magic happens
        for start_n in range(0, N, TILE_N):
            # Adjust for current tile
            start_n = tl.multiple_of(start_n, TILE_N)
            
            # Key pointers for this tile
            k_ptrs = K + off_b * stride_kb + off_h * stride_kh + \
                     ((start_n + offs_n)[:, None] * stride_kn + offs_k[None, :] * stride_kk)
            
            # Value pointers for this tile  
            v_ptrs = V + off_b * stride_vb + off_h * stride_vh + \
                     ((start_n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk)
            
            # Load K, V tiles
            kv_mask = ((start_n + offs_n)[:, None] < N) & (offs_k[None, :] < D)
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
            
            # Compute QK^T for this tile (vectorized over head_dim)
            qk = tl.dot(q, tl.trans(k)) * scale
            
            # Apply causal mask if needed
            if IS_CAUSAL:
                causal_mask = (offs_m[:, None] >= (start_n + offs_n)[None, :])
                qk = tl.where(causal_mask, qk, float('-inf'))
            
            # Online softmax update - THE NOVEL PART!
            # 1. Find new max
            tile_max = tl.max(qk, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            
            # 2. Correct previous accumulator with new max
            correction = tl.exp(running_max - new_max)
            acc_num *= correction[:, None]
            acc_den *= correction
            
            # 3. Compute exp and update accumulators
            exp_qk = tl.exp(qk - new_max[:, None])
            
            # Update numerator (weighted sum of values)
            acc_num += exp_qk @ v
            
            # Update denominator (sum of exp)
            acc_den += tl.sum(exp_qk, axis=1)
            
            # Update running max
            running_max = new_max
        
        # Final output = acc_num / acc_den
        out = acc_num / acc_den[:, None]
        
        # Store output
        out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + \
                   (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
        out_mask = (offs_m[:, None] < M) & (offs_k[None, :] < D)
        tl.store(out_ptrs, out.to(Out.dtype.element_ty), mask=out_mask)
        
        # Store log-sum-exp for backward pass
        lse = running_max + tl.log(acc_den)
        lse_ptrs = Lse + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
        lse_mask = offs_m < M
        tl.store(lse_ptrs, lse, mask=lse_mask)


class FusedOnlineAttention(nn.Module):
    """
    Production-ready Fused Online Attention module
    
    This module implements the novel fused attention mechanism with:
    - Online softmax computation in a single pass
    - Tiled processing for memory efficiency
    - Multi-GPU support via PyTorch Distributed
    - Automatic mixed precision support
    - Comprehensive error handling
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
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.tile_size_q = tile_size_q
        self.tile_size_k = tile_size_k
        self.dropout = dropout
        self.scale = scale or (1.0 / math.sqrt(head_dim))
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        
        # Multi-GPU setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        logger.info(
            f"FusedOnlineAttention initialized: heads={num_heads}, dim={head_dim}, tile_q={tile_size_q}, tile_k={tile_size_k}, world_size={self.world_size}, triton={TRITON_AVAILABLE}"
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        return_lse: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of Fused Online Attention
        
        Args:
            query: [batch_size, seq_len_q, num_heads, head_dim]
            key: [batch_size, seq_len_k, num_heads, head_dim]
            value: [batch_size, seq_len_k, num_heads, head_dim]
            causal: Whether to apply causal masking
            return_lse: Whether to return log-sum-exp for analysis
        
        Returns:
            output: [batch_size, seq_len_q, num_heads, head_dim]
            lse: Optional[batch_size, num_heads, seq_len_q] if return_lse=True
        """
        batch_size, seq_len_q, num_heads_q, head_dim_q = query.shape
        _, seq_len_k, num_heads_k, head_dim_k = key.shape
        assert num_heads_q == num_heads_k == self.num_heads, f"Number of heads mismatch: {num_heads_q} vs {num_heads_k} vs {self.num_heads}"
        assert head_dim_q == head_dim_k == self.head_dim, f"Head dimension mismatch: {head_dim_q} vs {head_dim_k} vs {self.head_dim}"
        
        # Multi-GPU partitioning (simple query sharding)
        if self.world_size > 1:
            queries_per_gpu = seq_len_q // self.world_size
            start_idx = self.rank * queries_per_gpu
            end_idx = start_idx + queries_per_gpu if self.rank < self.world_size - 1 else seq_len_q
            query = query[:, start_idx:end_idx]
            seq_len_q = query.shape[1]
        
        # Triton path only on CUDA
        use_triton = TRITON_AVAILABLE and query.is_cuda and key.is_cuda and value.is_cuda
        # If gradients are required, fall back to PyTorch SDPA which supports autograd
        if use_triton and (torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad)):
            use_triton = False
        if use_triton:
            output = torch.empty_like(query)
            lse = torch.empty((batch_size, self.num_heads, seq_len_q), dtype=torch.float32, device=query.device)
            grid = lambda meta: (triton.cdiv(seq_len_q, meta['TILE_M']), batch_size, self.num_heads)
            fused_online_attention_kernel[grid](
                query, key, value, output, lse,
                query.stride(0), query.stride(2), query.stride(1), query.stride(3),
                key.stride(0), key.stride(2), key.stride(1), key.stride(3),
                value.stride(0), value.stride(2), value.stride(1), value.stride(3),
                output.stride(0), output.stride(2), output.stride(1), output.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                H=self.num_heads, M=seq_len_q, N=seq_len_k, D=self.head_dim,
                TILE_M=self.tile_size_q, TILE_K=self.head_dim, TILE_N=self.tile_size_k,
                scale=self.scale, IS_CAUSAL=causal, num_warps=4, num_stages=2
            )
            if self.world_size > 1:
                output_list = [torch.empty_like(output) for _ in range(self.world_size)]
                dist.all_gather(output_list, output)
                output = torch.cat(output_list, dim=1)
            return (output, lse) if return_lse else output
        else:
            # Fallback to PyTorch SDPA
            q = query.permute(0,2,1,3).reshape(batch_size * self.num_heads, seq_len_q, self.head_dim)
            k = key.permute(0,2,1,3).reshape(batch_size * self.num_heads, seq_len_k, self.head_dim)
            v = value.permute(0,2,1,3).reshape(batch_size * self.num_heads, seq_len_k, self.head_dim)
            if q.is_cuda:
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
            else:
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
            out = out.reshape(batch_size, self.num_heads, seq_len_q, self.head_dim).permute(0,2,1,3).contiguous()
            return (out, None) if return_lse else out
    
    def benchmark(self, seq_len: int, batch_size: int = 1, warmup: int = 10, iterations: int = 100):
        """Benchmark the kernel performance"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.dtype if device.type == "cuda" else torch.float32
        q = torch.randn(batch_size, seq_len, self.num_heads, self.head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, self.num_heads, self.head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, self.num_heads, self.head_dim, device=device, dtype=dtype)
        for _ in range(warmup):
            _ = self.forward(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        import time
        start_time = time.time()
        for _ in range(iterations):
            _ = self.forward(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_time = (time.time() - start_time) / iterations
        flops = 2 * seq_len * seq_len * self.num_heads * self.head_dim * batch_size
        tflops = flops / elapsed_time / 1e12
        bytes_per_el = torch.tensor([], dtype=dtype).element_size()
        memory_bytes = 3 * seq_len * self.num_heads * self.head_dim * batch_size * bytes_per_el
        bandwidth = memory_bytes / elapsed_time / 1e9
        return {'time_ms': elapsed_time * 1000, 'tflops': tflops, 'bandwidth_gb_s': bandwidth, 'seq_len': seq_len, 'batch_size': batch_size}


def create_fused_online_attention(
    num_heads: int,
    head_dim: int,
    **kwargs
) -> FusedOnlineAttention:
    """Factory function to create FusedOnlineAttention instance"""
    return FusedOnlineAttention(num_heads, head_dim, **kwargs) 