"""
StreamAttention Configuration

Configuration for the novel fused online softmax attention mechanism.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import os


@dataclass
class StreamAttentionConfig:
    """
    Configuration for StreamAttention

    The key parameters control the novel online softmax algorithm's tiling
    and performance characteristics.
    """

    # Model dimensions
    num_heads: int = 32
    head_dim: int = 128

    # Tiling parameters - THE KEY TO PERFORMANCE
    tile_size_q: int = 128  # Number of queries processed per tile (TILE_M)
    tile_size_k: int = 64  # Number of keys processed per tile (TILE_N)

    # Memory and precision
    use_fp16: bool = True
    gradient_checkpointing: bool = False

    # Optional components
    use_qkv_projections: bool = True
    qkv_bias: bool = False
    use_layer_norm: bool = False
    dropout: float = 0.0

    # Multi-GPU settings
    enable_distributed: bool = True

    # Performance tuning
    num_warps: int = 4
    num_stages: int = 2

    # Additional fields for integrations/tests
    max_sequence_length: int = 65536
    enable_flash_attention: bool = True
    enable_ring_attention: bool = True
    enable_star_attention: bool = True
    enable_kv_compression: bool = True
    kv_compression_ratio: float = 4.0
    ring_attention_block_size: int = 2048
    ring_attention_overlap_size: int = 256
    star_attention_block_size: int = 2048
    star_attention_anchor_size: int = 256
    star_attention_num_hosts: int = 1

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "StreamAttentionConfig":
        """Load configuration from YAML file"""
        try:
            import yaml  # Lazy import to avoid hard dependency during module import
        except Exception as e:  # pragma: no cover - depends on env
            raise ImportError(
                "PyYAML is required for from_yaml(). Install with `pip install pyyaml`."
            ) from e

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> "StreamAttentionConfig":
        """Load configuration from environment variables"""
        config = cls()

        # Override with environment variables if present
        if os.getenv("STREAM_ATTENTION_NUM_HEADS"):
            config.num_heads = int(os.getenv("STREAM_ATTENTION_NUM_HEADS"))
        if os.getenv("STREAM_ATTENTION_HEAD_DIM"):
            config.head_dim = int(os.getenv("STREAM_ATTENTION_HEAD_DIM"))
        if os.getenv("STREAM_ATTENTION_TILE_Q"):
            config.tile_size_q = int(os.getenv("STREAM_ATTENTION_TILE_Q"))
        if os.getenv("STREAM_ATTENTION_TILE_K"):
            config.tile_size_k = int(os.getenv("STREAM_ATTENTION_TILE_K"))
        if os.getenv("STREAM_ATTENTION_MAX_SEQ_LEN"):
            config.max_sequence_length = int(os.getenv("STREAM_ATTENTION_MAX_SEQ_LEN"))
        if os.getenv("STREAM_ATTENTION_ENABLE_KV_COMPRESSION"):
            config.enable_kv_compression = os.getenv(
                "STREAM_ATTENTION_ENABLE_KV_COMPRESSION"
            ).lower() in {"1", "true", "yes"}
        if os.getenv("STREAM_ATTENTION_KV_COMPRESSION_RATIO"):
            config.kv_compression_ratio = float(
                os.getenv("STREAM_ATTENTION_KV_COMPRESSION_RATIO")
            )
        if os.getenv("STREAM_ATTENTION_RING_BLOCK_SIZE"):
            config.ring_attention_block_size = int(
                os.getenv("STREAM_ATTENTION_RING_BLOCK_SIZE")
            )
        if os.getenv("STREAM_ATTENTION_RING_OVERLAP_SIZE"):
            config.ring_attention_overlap_size = int(
                os.getenv("STREAM_ATTENTION_RING_OVERLAP_SIZE")
            )
        if os.getenv("STREAM_ATTENTION_STAR_BLOCK_SIZE"):
            config.star_attention_block_size = int(
                os.getenv("STREAM_ATTENTION_STAR_BLOCK_SIZE")
            )
        if os.getenv("STREAM_ATTENTION_STAR_ANCHOR_SIZE"):
            config.star_attention_anchor_size = int(
                os.getenv("STREAM_ATTENTION_STAR_ANCHOR_SIZE")
            )
        if os.getenv("STREAM_ATTENTION_STAR_NUM_HOSTS"):
            config.star_attention_num_hosts = int(
                os.getenv("STREAM_ATTENTION_STAR_NUM_HOSTS")
            )

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "tile_size_q": self.tile_size_q,
            "tile_size_k": self.tile_size_k,
            "use_fp16": self.use_fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_qkv_projections": self.use_qkv_projections,
            "qkv_bias": self.qkv_bias,
            "use_layer_norm": self.use_layer_norm,
            "dropout": self.dropout,
            "enable_distributed": self.enable_distributed,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "max_sequence_length": self.max_sequence_length,
            "enable_flash_attention": self.enable_flash_attention,
            "enable_ring_attention": self.enable_ring_attention,
            "enable_star_attention": self.enable_star_attention,
            "enable_kv_compression": self.enable_kv_compression,
            "kv_compression_ratio": self.kv_compression_ratio,
            "ring_attention_block_size": self.ring_attention_block_size,
            "ring_attention_overlap_size": self.ring_attention_overlap_size,
            "star_attention_block_size": self.star_attention_block_size,
            "star_attention_anchor_size": self.star_attention_anchor_size,
            "star_attention_num_hosts": self.star_attention_num_hosts,
        }

    def optimal_tile_sizes(self, seq_len: int) -> Tuple[int, int]:
        """
        Get optimal tile sizes based on sequence length

        This is a key optimization - tile sizes significantly impact performance
        """
        if seq_len <= 1024:
            return 64, 64
        elif seq_len <= 4096:
            return 128, 64
        elif seq_len <= 16384:
            return 128, 128
        else:
            return 256, 128
