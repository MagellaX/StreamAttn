"""
StreamAttention - Novel Fused Online Softmax Attention

A breakthrough attention mechanism that computes softmax normalization "on the fly"
using running accumulators, achieving both memory efficiency and numerical stability
in a single kernel pass.

Key Features:
- Single-pass attention computation without materializing attention matrix
- Online softmax with running statistics for numerical stability
- Tiled processing for efficient memory access
- Multi-GPU support through PyTorch Distributed
- Easy integration with existing deep learning workflows
"""

__version__ = "1.0.0"
__author__ = "StreamAttention Team"
__license__ = "MIT"

from .core.config import StreamAttentionConfig
from .core.fused_online_attention import (
    FusedOnlineAttention,
    create_fused_online_attention,
)
from .core.attention import StreamAttention
from .core.multihead_attention import StreamMultiheadAttention, create_stream_attention
from .core.flashattention_v3 import FlashAttentionV3

# Utilities
from .utils.memory import (
    MemoryProfiler,
    create_kv_compressor,
    gradient_checkpoint_sequential,
)

# Main API
__all__ = [
    # Main modules
    "StreamAttention",
    "StreamMultiheadAttention",
    "FusedOnlineAttention",
    "StreamAttentionConfig",
    "FlashAttentionV3",
    # Factory functions
    "create_stream_attention",
    "create_fused_online_attention",
    # Utilities
    "MemoryProfiler",
    "create_kv_compressor",
    "gradient_checkpoint_sequential",
    # Version
    "__version__",
]
