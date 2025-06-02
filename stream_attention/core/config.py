"""
StreamAttention Configuration

Configuration for the novel fused online softmax attention mechanism.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import yaml
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
    tile_size_k: int = 64   # Number of keys processed per tile (TILE_N)
    
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
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "StreamAttentionConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
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
            "num_stages": self.num_stages
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