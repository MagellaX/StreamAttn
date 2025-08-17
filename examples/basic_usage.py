"""
Basic usage example for StreamAttention
"""

import torch
from stream_attention import StreamAttention, AttentionConfig

def main():
    # Configuration
    batch_size = 2
    seq_len = 8192
    num_heads = 32
    head_dim = 128
    hidden_dim = num_heads * head_dim
    
    print("StreamAttention Basic Usage Example")
    print("=" * 50)
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {head_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print()
    
    # Create configuration
    config = AttentionConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        max_sequence_length=100_000,  # Support up to 100K tokens
    )
    
    # Initialize StreamAttention
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attention = StreamAttention(config).to(device)
    
    print(f"Using device: {device}")
    print(f"Available attention methods: {attention._available_methods()}")
    print()
    
    # Create input tensor
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_dim,
        device=device, dtype=torch.float16
    )
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = attention(hidden_states)
        if isinstance(output, tuple):
            output = output[0]
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak memory usage: {memory_mb:.2f} MB")
    
    print("\nSuccess! StreamAttention is working correctly.")

if __name__ == "__main__":
    main() 