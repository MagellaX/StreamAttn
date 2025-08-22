"""
Long context example for StreamAttention

Demonstrates handling sequences up to 1M tokens
"""

import torch
import time
import gc
from stream_attention import StreamAttention, StreamAttentionConfig


def test_long_context(seq_len: int, batch_size: int = 1):
    """Test StreamAttention on long sequences"""
    print(f"\nTesting sequence length: {seq_len:,} tokens")
    print("-" * 50)

    # Configuration for long context
    config = StreamAttentionConfig(
        num_heads=32,
        head_dim=128,
        tile_size_q=256,
        tile_size_k=128,
        gradient_checkpointing=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        # Initialize model
        attention = StreamAttention(config).to(device).to(dtype)

        # Create input
        hidden_dim = config.num_heads * config.head_dim
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

        # Warmup
        print("Warming up...")
        with torch.no_grad():
            _ = attention(hidden_states=x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        print("Running benchmark...")
        start_time = time.time()

        with torch.no_grad():
            output = attention(hidden_states=x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        # Results
        total_time = end_time - start_time
        tokens_per_second = (batch_size * seq_len) / total_time

        print(f"✓ Success!")
        print(f"  Time: {total_time:.2f} seconds")
        print(f"  Throughput: {tokens_per_second:,.0f} tokens/second")

        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Memory: {memory_gb:.2f} GB")

        # Cleanup
        del x, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ Failed: {e}")


def main():
    print("StreamAttention Long Context Example")
    print("=" * 60)

    # Test different sequence lengths
    test_lengths = [
        16_384,  # 16K
        65_536,  # 64K
        131_072,  # 128K
        262_144,  # 256K
        524_288,  # 512K
        1_048_576,  # 1M
    ]

    for seq_len in test_lengths:
        test_long_context(seq_len)

    print("\n" + "=" * 60)
    print("Long context testing complete!")


if __name__ == "__main__":
    main()
