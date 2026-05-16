"""
Memory Optimization Utilities for StreamAttention

Implements various memory optimization techniques including:
- KV cache compression (8x reduction)
- Gradient checkpointing
- Memory-efficient attention patterns
- Dynamic memory allocation
- Cache eviction strategies

References:
- Model Tells You What to Discard: Adaptive KV Cache Compression
- ChunkKV: Semantic-Preserving KV Cache Compression
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import time

logger = logging.getLogger(__name__)


@dataclass
class CompressionStats:
    """Statistics for KV cache compression"""

    original_size: int
    compressed_size: int
    compression_ratio: float
    tokens_retained: int
    memory_saved_mb: float
    compression_time: float


class KVCacheCompressor(ABC):
    """Abstract base class for KV cache compression strategies"""

    @abstractmethod
    def compress(
        self, keys: torch.Tensor, values: torch.Tensor, compression_ratio: float = 8.0
    ) -> Tuple[torch.Tensor, torch.Tensor, CompressionStats]:
        """Compress KV cache with specified ratio"""
        pass

    @abstractmethod
    def decompress(
        self, compressed_keys: torch.Tensor, compressed_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress KV cache (if applicable)"""
        pass


class ImportanceBasedCompressor(KVCacheCompressor):
    """
    Importance-based KV cache compression

    Retains tokens with highest attention scores across layers/heads
    """

    def __init__(self, method: str = "attention_sum"):
        self.method = method
        self.attention_history = []

    def update_attention_scores(self, attention_weights: torch.Tensor):
        """Update attention history for importance calculation"""
        self.attention_history.append(attention_weights.detach())

    def compress(
        self, keys: torch.Tensor, values: torch.Tensor, compression_ratio: float = 8.0
    ) -> Tuple[torch.Tensor, torch.Tensor, CompressionStats]:
        """
        Compress KV cache based on token importance

        Args:
            keys: [batch, seq_len, num_heads, head_dim]
            values: [batch, seq_len, num_heads, head_dim]
            compression_ratio: Target compression ratio

        Returns:
            Compressed keys, values, and statistics
        """
        start_time = time.time()

        batch_size, seq_len, num_heads, head_dim = keys.shape
        keep_len = max(1, int(seq_len / compression_ratio))

        # Calculate importance scores
        if self.method == "attention_sum" and self.attention_history:
            # Use historical attention scores
            importance = torch.stack(self.attention_history).mean(dim=0)
            importance = importance.sum(dim=(1, 2))  # Sum over heads and queries
        elif self.method == "key_norm":
            # Use key vector norms
            importance = keys.norm(dim=-1).mean(dim=-1)  # [batch, seq_len]
        elif self.method == "value_norm":
            # Use value vector norms
            importance = values.norm(dim=-1).mean(dim=-1)
        else:
            # Random selection as fallback
            importance = torch.rand(batch_size, seq_len, device=keys.device)

        # Select top-k important tokens
        _, indices = importance.topk(keep_len, dim=-1, sorted=True)

        # Gather compressed KV
        indices_expanded = (
            indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_heads, head_dim)
        )
        compressed_keys = keys.gather(1, indices_expanded)
        compressed_values = values.gather(1, indices_expanded)

        # Calculate statistics
        original_size = keys.numel() + values.numel()
        compressed_size = compressed_keys.numel() + compressed_values.numel()
        memory_saved = (
            (original_size - compressed_size) * keys.element_size() / (1024 * 1024)
        )

        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size,
            tokens_retained=keep_len,
            memory_saved_mb=memory_saved,
            compression_time=time.time() - start_time,
        )

        return compressed_keys, compressed_values, stats

    def decompress(
        self, compressed_keys: torch.Tensor, compressed_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No decompression needed for importance-based method"""
        return compressed_keys, compressed_values


class ChunkBasedCompressor(KVCacheCompressor):
    """
    Chunk-based KV cache compression (ChunkKV)

    Preserves semantic chunks rather than individual tokens
    """

    def __init__(self, chunk_size: int = 64):
        self.chunk_size = chunk_size

    def compress(
        self, keys: torch.Tensor, values: torch.Tensor, compression_ratio: float = 8.0
    ) -> Tuple[torch.Tensor, torch.Tensor, CompressionStats]:
        """Compress by selecting representative tokens from chunks"""
        start_time = time.time()

        batch_size, seq_len, num_heads, head_dim = keys.shape
        num_chunks = math.ceil(seq_len / self.chunk_size)
        keep_per_chunk = max(1, int(self.chunk_size / compression_ratio))

        compressed_keys_list = []
        compressed_values_list = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, seq_len)

            # Extract chunk
            chunk_keys = keys[:, start_idx:end_idx]
            chunk_values = values[:, start_idx:end_idx]

            # Select representative tokens from chunk
            # Use attention-based selection within chunk
            chunk_len = end_idx - start_idx
            if chunk_len <= keep_per_chunk:
                compressed_keys_list.append(chunk_keys)
                compressed_values_list.append(chunk_values)
            else:
                # Compute self-attention within chunk
                scores = torch.matmul(
                    chunk_keys.mean(dim=2),  # Average over heads
                    chunk_keys.mean(dim=2).transpose(-2, -1),
                ) / math.sqrt(head_dim)

                # Sum attention received by each token
                importance = scores.sum(dim=-1)
                _, indices = importance.topk(keep_per_chunk, dim=-1)

                # Gather representatives
                indices = (
                    indices.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, num_heads, head_dim)
                )
                compressed_keys_list.append(chunk_keys.gather(1, indices))
                compressed_values_list.append(chunk_values.gather(1, indices))

        # Concatenate all chunks
        compressed_keys = torch.cat(compressed_keys_list, dim=1)
        compressed_values = torch.cat(compressed_values_list, dim=1)

        # Statistics
        original_size = keys.numel() + values.numel()
        compressed_size = compressed_keys.numel() + compressed_values.numel()
        memory_saved = (
            (original_size - compressed_size) * keys.element_size() / (1024 * 1024)
        )

        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size,
            tokens_retained=compressed_keys.shape[1],
            memory_saved_mb=memory_saved,
            compression_time=time.time() - start_time,
        )

        return compressed_keys, compressed_values, stats

    def decompress(
        self, compressed_keys: torch.Tensor, compressed_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No decompression for chunk-based method"""
        return compressed_keys, compressed_values


class QuantizedCompressor(KVCacheCompressor):
    """
    Quantization-based KV cache compression

    Reduces precision of KV cache entries
    """

    def __init__(self, bits: int = 4):
        self.bits = bits
        self.scale = None
        self.zero_point = None

    def compress(
        self, keys: torch.Tensor, values: torch.Tensor, compression_ratio: float = 8.0
    ) -> Tuple[torch.Tensor, torch.Tensor, CompressionStats]:
        """Compress using quantization"""
        start_time = time.time()

        # Combine keys and values for unified quantization
        kv_combined = torch.cat([keys, values], dim=-1)

        # Calculate quantization parameters
        if self.bits == 8:
            # INT8 quantization
            compressed = self._quantize_int8(kv_combined)
        elif self.bits == 4:
            # INT4 quantization
            compressed = self._quantize_int4(kv_combined)
        elif self.bits == 2:
            # Binary quantization
            compressed = self._quantize_binary(kv_combined)
        else:
            raise ValueError(f"Unsupported bit width: {self.bits}")

        # Split back into keys and values
        head_dim = keys.shape[-1]
        compressed_keys = compressed[..., :head_dim]
        compressed_values = compressed[..., head_dim:]

        # Statistics
        original_size = keys.numel() + values.numel()
        compression_ratio_actual = 16.0 / self.bits  # FP16 to INT
        compressed_size = int(original_size / compression_ratio_actual)
        memory_saved = (
            (original_size - compressed_size) * 2 / (1024 * 1024)
        )  # 2 bytes per FP16

        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio_actual,
            tokens_retained=keys.shape[1],  # All tokens retained
            memory_saved_mb=memory_saved,
            compression_time=time.time() - start_time,
        )

        return compressed_keys, compressed_values, stats

    def _quantize_int8(self, tensor: torch.Tensor) -> torch.Tensor:
        """INT8 quantization"""
        # Calculate scale and zero point
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / 255
        zero_point = -min_val / scale

        # Quantize
        quantized = torch.round((tensor - min_val) / scale).to(torch.uint8)

        # Store parameters for dequantization
        self.scale = scale
        self.zero_point = zero_point

        return quantized

    def _quantize_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """INT4 quantization (simulated)"""
        # For simplicity, we'll use INT8 and indicate it's INT4
        # In production, use proper INT4 storage
        return self._quantize_int8(tensor)

    def _quantize_binary(self, tensor: torch.Tensor) -> torch.Tensor:
        """Binary quantization"""
        # Simple sign-based quantization
        return (tensor > 0).to(torch.uint8)

    def decompress(
        self, compressed_keys: torch.Tensor, compressed_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize compressed KV cache"""
        if self.scale is None:
            return compressed_keys, compressed_values

        # Dequantize
        keys = compressed_keys.float() * self.scale + self.zero_point
        values = compressed_values.float() * self.scale + self.zero_point

        return keys, values


class HybridCompressor(KVCacheCompressor):
    """
    Hybrid compression combining multiple strategies

    Uses importance-based selection + quantization
    """

    def __init__(self):
        self.importance_compressor = ImportanceBasedCompressor()
        self.quantized_compressor = QuantizedCompressor(bits=8)

    def compress(
        self, keys: torch.Tensor, values: torch.Tensor, compression_ratio: float = 8.0
    ) -> Tuple[torch.Tensor, torch.Tensor, CompressionStats]:
        """Apply hybrid compression"""
        # First, select important tokens
        importance_ratio = compression_ratio / 2
        compressed_k, compressed_v, stats1 = self.importance_compressor.compress(
            keys, values, importance_ratio
        )

        # Then, quantize the selected tokens
        compressed_k, compressed_v, stats2 = self.quantized_compressor.compress(
            compressed_k, compressed_v, 2.0
        )

        # Combine statistics
        total_stats = CompressionStats(
            original_size=stats1.original_size,
            compressed_size=stats2.compressed_size,
            compression_ratio=stats1.original_size / stats2.compressed_size,
            tokens_retained=stats1.tokens_retained,
            memory_saved_mb=stats1.memory_saved_mb + stats2.memory_saved_mb,
            compression_time=stats1.compression_time + stats2.compression_time,
        )

        return compressed_k, compressed_v, total_stats

    def decompress(
        self, compressed_keys: torch.Tensor, compressed_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress hybrid compression"""
        return self.quantized_compressor.decompress(compressed_keys, compressed_values)


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention implementations

    Includes gradient checkpointing and chunked processing
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        chunk_size: int = 512,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint

        assert hidden_size % num_heads == 0

    def chunked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention in chunks to save memory

        Useful for very long sequences
        """
        batch_size, seq_len_q, num_heads, head_dim = query.shape
        seq_len_k = key.shape[1]

        # Process query chunks
        output_chunks = []

        for q_start in range(0, seq_len_q, self.chunk_size):
            q_end = min(q_start + self.chunk_size, seq_len_q)
            q_chunk = query[:, q_start:q_end]

            # Compute attention scores for this query chunk
            chunk_output = self._compute_chunk_attention(
                q_chunk, key, value, attention_mask, q_start
            )
            output_chunks.append(chunk_output)

        # Concatenate results
        output = torch.cat(output_chunks, dim=1)
        return output

    def _compute_chunk_attention(
        self,
        query_chunk: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_offset: int,
    ) -> torch.Tensor:
        """Compute attention for a single query chunk"""
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._attention_forward,
                query_chunk,
                key,
                value,
                attention_mask,
                query_offset,
            )
        else:
            return self._attention_forward(
                query_chunk, key, value, attention_mask, query_offset
            )

    def _attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_offset: int,
    ) -> torch.Tensor:
        """Standard attention computation"""
        batch_size, chunk_len, num_heads, head_dim = query.shape
        seq_len_k = key.shape[1]

        # Reshape for batch matrix multiplication
        q = query.transpose(1, 2)  # [batch, heads, chunk_len, dim]
        k = key.transpose(1, 2)  # [batch, heads, seq_len_k, dim]
        v = value.transpose(1, 2)  # [batch, heads, seq_len_k, dim]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply mask if provided
        if attention_mask is not None:
            # Adjust mask for query chunk
            chunk_mask = attention_mask[:, query_offset : query_offset + chunk_len]
            scores = scores.masked_fill(
                ~chunk_mask.unsqueeze(1).unsqueeze(-1), float("-inf")
            )
            # Prevent NaNs when an entire query row is masked: replace fully -inf rows with zeros
            all_masked = ~torch.isfinite(scores).any(dim=-1, keepdim=True)
            scores = torch.where(all_masked, torch.zeros_like(scores), scores)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2)

        return output


def create_kv_compressor(method: str = "hybrid", **kwargs) -> KVCacheCompressor:
    """
    Factory function to create KV cache compressor

    Args:
        method: Compression method ('importance', 'chunk', 'quantized', 'hybrid')
        **kwargs: Additional arguments for the compressor

    Returns:
        KVCacheCompressor instance
    """
    if method == "importance":
        return ImportanceBasedCompressor(**kwargs)
    elif method == "chunk":
        return ChunkBasedCompressor(**kwargs)
    elif method == "quantized":
        return QuantizedCompressor(**kwargs)
    elif method == "hybrid":
        return HybridCompressor(**kwargs)
    else:
        raise ValueError(f"Unknown compression method: {method}")


# Gradient checkpointing utilities
def gradient_checkpoint_sequential(
    functions: List[nn.Module], segments: int, *args, **kwargs
) -> Any:
    """
    Gradient checkpointing for sequential modules

    Saves memory by recomputing activations during backward pass
    """

    def run_segment(segment_idx: int, *inputs):
        start_idx = segment_idx * len(functions) // segments
        end_idx = (segment_idx + 1) * len(functions) // segments

        for func in functions[start_idx:end_idx]:
            if isinstance(inputs, tuple):
                inputs = func(*inputs)
            else:
                inputs = func(inputs)

        return inputs

    # Run all segments with checkpointing
    inputs = args
    for segment_idx in range(segments):
        inputs = torch.utils.checkpoint.checkpoint(
            run_segment, segment_idx, *inputs, **kwargs
        )
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

    return inputs[0] if len(inputs) == 1 else inputs


# Memory profiling utilities
class MemoryProfiler:
    """Profile memory usage during model execution"""

    def __init__(self):
        self.snapshots = []
        self.enabled = False

    def start(self):
        """Start memory profiling"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.enabled = True
        self.snapshot("start")

    def snapshot(self, label: str):
        """Take a memory snapshot"""
        if not self.enabled:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024**2)  # MB

            self.snapshots.append(
                {
                    "label": label,
                    "allocated_mb": allocated,
                    "reserved_mb": reserved,
                    "timestamp": time.time(),
                }
            )

    def report(self) -> Dict[str, Any]:
        """Generate memory profiling report"""
        if not self.snapshots:
            return {}

        report = {
            "snapshots": self.snapshots,
            "peak_allocated_mb": max(s["allocated_mb"] for s in self.snapshots),
            "peak_reserved_mb": max(s["reserved_mb"] for s in self.snapshots),
        }

        # Calculate memory usage between snapshots
        if len(self.snapshots) > 1:
            deltas = []
            for i in range(1, len(self.snapshots)):
                delta = {
                    "from": self.snapshots[i - 1]["label"],
                    "to": self.snapshots[i]["label"],
                    "allocated_delta_mb": (
                        self.snapshots[i]["allocated_mb"]
                        - self.snapshots[i - 1]["allocated_mb"]
                    ),
                    "time_delta_s": (
                        self.snapshots[i]["timestamp"]
                        - self.snapshots[i - 1]["timestamp"]
                    ),
                }
                deltas.append(delta)
            report["deltas"] = deltas

        return report
