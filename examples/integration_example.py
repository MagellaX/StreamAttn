"""
StreamAttention Integration Examples

This shows how to integrate the novel fused online softmax attention
into various deep learning workflows.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, GPT2Model, GPT2Config

from stream_attention import StreamAttention, StreamAttentionConfig
from stream_attention.integration.hf import replace_gpt2_attention


def example_1_drop_in_replacement():
    """Example 1: Drop-in replacement for PyTorch MultiheadAttention"""
    print("Example 1: Drop-in replacement")

    # Standard PyTorch attention
    standard_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

    # StreamAttention replacement
    config = StreamAttentionConfig(
        num_heads=8, head_dim=64, use_qkv_projections=True  # 512 / 8 = 64
    )
    stream_attn = StreamAttention(config)

    # Test with sample input
    batch_size, seq_len = 2, 1024
    x = torch.randn(batch_size, seq_len, 512)

    # Standard attention
    standard_out, _ = standard_attn(x, x, x)

    # StreamAttention (with projections)
    stream_out = stream_attn(x)

    print(f"Input shape: {x.shape}")
    print(f"Standard output shape: {standard_out.shape}")
    print(f"Stream output shape: {stream_out.shape}")
    print("✓ Drop-in replacement successful!\n")


def example_2_huggingface_integration():
    """Example 2: Integration with HuggingFace transformers"""
    print("Example 2: HuggingFace integration")

    # Load a small GPT2 model
    model_name = "gpt2"
    model = GPT2Model.from_pretrained(model_name)

    # Get model config
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    # Create StreamAttention config
    stream_config = StreamAttentionConfig(
        num_heads=num_heads, head_dim=head_dim, use_qkv_projections=True
    )

    # Replace GPT-2 attention safely using adapter wrapper
    num_replaced = replace_gpt2_attention(model, stream_config)
    print(
        f"✓ HuggingFace model updated with StreamAttention via adapter (layers replaced: {num_replaced})\n"
    )

    # Test the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    text = "The future of AI is"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Model outputs shape: {outputs.last_hidden_state.shape}")


def example_3_custom_transformer():
    """Example 3: Building a custom transformer with StreamAttention"""
    print("Example 3: Custom transformer")

    class StreamTransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()

            # StreamAttention instead of standard attention
            self.attention = StreamAttention(
                StreamAttentionConfig(
                    num_heads=num_heads,
                    head_dim=d_model // num_heads,
                    dropout=dropout,
                    use_qkv_projections=True,
                    use_layer_norm=True,
                )
            )

            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

            # Layer norms
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x, mask=None):
            # Self-attention with residual
            attn_out = self.attention(self.norm1(x))
            x = x + attn_out

            # FFN with residual
            x = x + self.ffn(self.norm2(x))

            return x

    # Create a small transformer
    model = nn.Sequential(
        *[StreamTransformerBlock(d_model=512, num_heads=8, d_ff=2048) for _ in range(6)]
    )

    # Test it
    x = torch.randn(2, 1024, 512)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Custom transformer built successfully!\n")


def example_4_benchmark_performance():
    """Example 4: Benchmark performance improvements"""
    print("Example 4: Performance benchmarking")

    config = StreamAttentionConfig(
        num_heads=32, head_dim=128, tile_size_q=128, tile_size_k=64
    )

    stream_attn = StreamAttention(config)

    # Benchmark different sequence lengths
    results = stream_attn.benchmark_speedup(
        seq_lengths=[512, 1024, 2048, 4096, 8192], batch_size=1
    )

    print("\nPerformance Results:")
    print("Seq Length | Time (ms) | Speedup | TFLOPS | Bandwidth")
    print("-" * 60)

    for seq_len, metrics in results.items():
        print(
            f"{seq_len:10d} | {metrics['stream_time_ms']:9.2f} | "
            f"{metrics['speedup']:7.2f}x | {metrics['tflops']:6.2f} | "
            f"{metrics['bandwidth_gb_s']:6.1f} GB/s"
        )

    print("\n✓ Benchmarking complete!")


def example_5_multi_gpu():
    """Example 5: Multi-GPU usage"""
    print("\nExample 5: Multi-GPU usage")

    if not torch.distributed.is_initialized():
        print("Note: Run with torchrun for multi-GPU support:")
        print("  torchrun --nproc_per_node=2 integration_example.py")
        print("  Skipping multi-GPU example...\n")
        return

    # StreamAttention automatically handles multi-GPU
    config = StreamAttentionConfig(num_heads=32, head_dim=128, enable_distributed=True)

    model = StreamAttention(config).cuda()

    # Create input on current GPU
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    x = torch.randn(2, 4096, 4096, device=device)

    # Forward pass - automatically distributed
    output = model(x)

    print(f"GPU {local_rank}: Input shape: {x.shape}")
    print(f"GPU {local_rank}: Output shape: {output.shape}")
    print("✓ Multi-GPU processing successful!\n")


def example_6_long_context():
    """Example 6: Extreme long context handling"""
    print("Example 6: Long context handling")

    # Configure for long context
    config = StreamAttentionConfig(
        num_heads=32,
        head_dim=128,
        tile_size_q=256,  # Larger tiles for long sequences
        tile_size_k=128,
        gradient_checkpointing=True,  # Save memory
    )

    model = StreamAttention(config)

    # Test with very long sequence
    # Note: In practice, you'd use this with real data
    batch_size = 1
    seq_len = 32768  # 32K tokens
    hidden_dim = 4096

    print(f"Processing {seq_len:,} tokens...")

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)

    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()

        # Measure memory usage
        torch.cuda.reset_peak_memory_stats()

        with torch.cuda.amp.autocast():
            output = model(x)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory usage: {peak_memory:.2f} GB")
    else:
        output = model(x)

    print(f"Output shape: {output.shape}")
    print("✓ Long context processing successful!\n")


if __name__ == "__main__":
    print("StreamAttention Integration Examples")
    print("=" * 60)
    print()

    # Run examples
    example_1_drop_in_replacement()
    example_2_huggingface_integration()
    example_3_custom_transformer()
    example_4_benchmark_performance()
    example_5_multi_gpu()
    example_6_long_context()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("\nStreamAttention is ready for production use!")
    print("Check the documentation for more advanced usage patterns.")
