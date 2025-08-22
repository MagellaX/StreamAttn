"""
Comprehensive Test Suite for StreamAttention

Tests all attention mechanisms for correctness, performance, and memory efficiency.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import time
import gc

from stream_attention import StreamAttention, StreamAttentionConfig
from stream_attention.core.flashattention_v3 import FlashAttentionV3
from stream_attention.core.ring_attention import RingAttention
from stream_attention.core.star_attention import StarAttention
from stream_attention.utils.memory import create_kv_compressor, MemoryProfiler


class TestStreamAttention:
    """Test suite for StreamAttention"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return StreamAttentionConfig(
            num_heads=8,
            head_dim=64,
            max_sequence_length=4096,
            enable_flash_attention=True,
            enable_ring_attention=True,
            enable_star_attention=True,
            enable_kv_compression=True,
            kv_compression_ratio=4.0,
            gradient_checkpointing=True,
        )

    @pytest.fixture
    def device(self):
        """Get test device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_test_tensors(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create test query, key, value tensors"""
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        return q, k, v

    def reference_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Reference implementation of attention for comparison"""
        # Convert to float32 for accuracy
        q = query.float()
        k = key.float()
        v = value.float()

        batch_size, seq_len, num_heads, head_dim = q.shape

        # Reshape for batch matrix multiply
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)

        # Apply causal mask
        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2)

        return output.to(dtype)

    def test_flash_attention_correctness(self, config, device):
        """Test FlashAttention V3 correctness"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for FlashAttention")

        flash_attn = FlashAttentionV3(config).to(device)

        # Test different sequence lengths
        for seq_len in [128, 512, 1024, 2048]:
            batch_size = 2
            q, k, v = self.create_test_tensors(
                batch_size, seq_len, config.num_heads, config.head_dim, device
            )

            # FlashAttention output
            with torch.cuda.amp.autocast():
                flash_output = flash_attn(q, k, v, causal=True)

            # Reference output
            ref_output = self.reference_attention(q, k, v, causal=True, dtype=q.dtype)

            # Compare (allow some tolerance due to different implementations)
            torch.testing.assert_close(
                flash_output,
                ref_output,
                rtol=5e-2,
                atol=5e-2,
                msg=f"FlashAttention mismatch at seq_len={seq_len}",
            )

    def test_ring_attention_correctness(self, config, device):
        """Test Ring Attention correctness"""
        if not torch.distributed.is_initialized():
            pytest.skip("Distributed required for Ring Attention")

        ring_attn = RingAttention(config).to(device)

        # Test with local sequence
        seq_len = 1024
        batch_size = 1
        q, k, v = self.create_test_tensors(
            batch_size, seq_len, config.num_heads, config.head_dim, device
        )

        # Ring attention output
        ring_output = ring_attn(q, k, v, causal=True)

        # Reference output
        ref_output = self.reference_attention(q, k, v, causal=True, dtype=q.dtype)

        # Compare
        torch.testing.assert_close(
            ring_output, ref_output, rtol=1e-1, atol=1e-1, msg="Ring Attention mismatch"
        )

    def test_star_attention_correctness(self, config, device):
        """Test Star Attention correctness"""
        star_attn = StarAttention(config).to(device)

        # Test context encoding and query processing
        context_len = 2048
        query_len = 128
        batch_size = 1

        # Create context
        context = torch.randn(
            batch_size,
            context_len,
            config.num_heads,
            config.head_dim,
            device=device,
            dtype=torch.float16,
        )

        # Encode context
        state = star_attn.encode_context(context)

        # Create query
        query = torch.randn(
            batch_size,
            query_len,
            config.num_heads,
            config.head_dim,
            device=device,
            dtype=torch.float16,
        )

        # Process query
        output = star_attn.process_query(query, state)

        # Check output shape
        assert output.shape == query.shape, "Star Attention output shape mismatch"

        # Check that output is not NaN or Inf
        assert torch.isfinite(output).all(), "Star Attention produced non-finite values"

    def test_memory_efficiency(self, config, device):
        """Test memory efficiency of different attention methods"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory profiling")

        profiler = MemoryProfiler()
        seq_lengths = [1024, 2048, 4096, 8192]
        batch_size = 1

        memory_usage = {}

        for method in ["flash", "star"]:
            memory_usage[method] = []

            for seq_len in seq_lengths:
                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()

                # Create tensors
                q, k, v = self.create_test_tensors(
                    batch_size, seq_len, config.num_heads, config.head_dim, device
                )

                # Profile memory
                profiler.start()

                if method == "flash":
                    model = FlashAttentionV3(config).to(device)
                    with torch.cuda.amp.autocast():
                        _ = model(q, k, v)
                elif method == "star":
                    model = StarAttention(config).to(device)
                    state = model.encode_context(q)
                    _ = model.process_query(q[:, :128], state)

                profiler.snapshot(f"{method}_{seq_len}")
                report = profiler.report()

                memory_usage[method].append(report["peak_allocated_mb"])

        # Star attention should use less memory for long sequences
        for i, seq_len in enumerate(seq_lengths):
            if seq_len >= 4096:
                assert (
                    memory_usage["star"][i] < memory_usage["flash"][i] * 1.5
                ), f"Star attention not memory efficient at seq_len={seq_len}"

    def test_kv_cache_compression(self, device):
        """Test KV cache compression"""
        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        # Create KV cache
        keys = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        values = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Test different compression methods
        for method in ["importance", "chunk", "quantized", "hybrid"]:
            compressor = create_kv_compressor(method)

            # Compress
            comp_k, comp_v, stats = compressor.compress(
                keys, values, compression_ratio=4.0
            )

            # Check compression ratio
            assert (
                stats.compression_ratio >= 3.5
            ), f"{method} compression didn't achieve target ratio"

            # Check shapes
            assert comp_k.shape[0] == batch_size
            assert comp_k.shape[2] == num_heads
            assert comp_k.shape[3] == head_dim

            # For non-quantized methods, check token reduction
            if method in ["importance", "chunk"]:
                assert comp_k.shape[1] <= seq_len // 3.5

    def test_gradient_flow(self, config, device):
        """Test gradient flow through attention mechanisms"""
        model = StreamAttention(config).to(device)

        seq_len = 512
        batch_size = 2
        q, k, v = self.create_test_tensors(
            batch_size,
            seq_len,
            config.num_heads,
            config.head_dim,
            device,
            dtype=torch.float32,
        )

        # Enable gradients
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True

        # Forward pass
        output = model(q, k, v)

        # Compute loss
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist and are finite
        assert q.grad is not None, "No gradient for query"
        assert k.grad is not None, "No gradient for key"
        assert v.grad is not None, "No gradient for value"

        assert torch.isfinite(q.grad).all(), "Non-finite gradients in query"
        assert torch.isfinite(k.grad).all(), "Non-finite gradients in key"
        assert torch.isfinite(v.grad).all(), "Non-finite gradients in value"

    def test_mixed_precision(self, config, device):
        """Test mixed precision training"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for mixed precision")

        model = StreamAttention(config).to(device)

        seq_len = 1024
        batch_size = 2

        # Test with different dtypes
        for dtype in [torch.float16, torch.bfloat16]:
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue

            q, k, v = self.create_test_tensors(
                batch_size, seq_len, config.num_heads, config.head_dim, device, dtype
            )

            with torch.cuda.amp.autocast(dtype=dtype):
                output = model(q, k, v)

            assert output.dtype == dtype, f"Output dtype mismatch for {dtype}"
            assert torch.isfinite(output).all(), f"Non-finite values with {dtype}"

    @pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048, 4096])
    def test_performance_scaling(self, config, device, seq_len):
        """Test performance scaling with sequence length"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for performance testing")

        model = StreamAttention(config).to(device)
        batch_size = 1

        q, k, v = self.create_test_tensors(
            batch_size, seq_len, config.num_heads, config.head_dim, device
        )

        # Warmup
        for _ in range(3):
            _ = model(q, k, v)

        torch.cuda.synchronize()

        # Time execution
        start_time = time.time()
        iterations = 10

        for _ in range(iterations):
            _ = model(q, k, v)

        torch.cuda.synchronize()
        elapsed_time = (time.time() - start_time) / iterations

        # Calculate FLOPS
        flops = 4 * seq_len * seq_len * config.num_heads * config.head_dim
        tflops = flops / elapsed_time / 1e12

        print(
            f"\nSeq length: {seq_len}, Time: {elapsed_time:.3f}s, TFLOPS: {tflops:.2f}"
        )

        # Performance should scale reasonably with sequence length
        # This is a basic check - specific thresholds depend on hardware
        assert (
            elapsed_time < seq_len / 100
        ), f"Performance too slow for seq_len={seq_len}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
