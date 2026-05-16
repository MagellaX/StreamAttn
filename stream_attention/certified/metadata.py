"""Reusable block metadata for certified and Gate-1 attention paths."""

from dataclasses import dataclass
from typing import Optional

import torch

from stream_attention.kernels.gate1_fwd_triton import build_value_norm_bounds


@dataclass
class StreamAttnMetadataCache:
    """Block metadata that should be built or updated outside hot attention.

    Tensors use ``[batch, heads, blocks, ...]`` layout. Gate-1 only needs
    ``value_norm_bounds`` today; Gate-0 will add key summaries and outlier
    metadata to the same cache object.
    """

    block_size: int
    seq_len: int
    value_norm_bounds: Optional[torch.Tensor] = None
    key_centroids: Optional[torch.Tensor] = None
    key_radii: Optional[torch.Tensor] = None
    outlier_keys: Optional[torch.Tensor] = None
    valid_blocks: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if self.value_norm_bounds is not None and self.value_norm_bounds.dim() != 3:
            raise ValueError("value_norm_bounds must have shape [batch, heads, blocks]")

    @property
    def num_blocks(self) -> int:
        return (self.seq_len + self.block_size - 1) // self.block_size

    @classmethod
    def from_value(
        cls,
        value: torch.Tensor,
        *,
        block_size: int,
        safety_margin: float = 1.0,
    ) -> "StreamAttnMetadataCache":
        """Build a cache containing conservative per-block value norm bounds."""

        if safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1.0")
        if value.dim() != 4:
            raise ValueError("value must have shape [batch, seq, heads, dim]")

        bounds = build_value_norm_bounds(value, block_size=block_size)
        if safety_margin != 1.0:
            bounds = bounds * float(safety_margin)
        return cls(
            block_size=block_size,
            seq_len=value.shape[1],
            value_norm_bounds=bounds,
        )

    def require_value_norm_bounds(self) -> torch.Tensor:
        if self.value_norm_bounds is None:
            raise ValueError("metadata cache does not contain value_norm_bounds")
        return self.value_norm_bounds

    def validate_for_value(self, value: torch.Tensor) -> None:
        """Validate that cached value metadata matches a value tensor."""

        if value.dim() != 4:
            raise ValueError("value must have shape [batch, seq, heads, dim]")
        batch, seq_len, heads, _ = value.shape
        if seq_len != self.seq_len:
            raise ValueError("value sequence length does not match metadata cache")
        if self.value_norm_bounds is None:
            return
        expected = (batch, heads, self.num_blocks)
        if tuple(self.value_norm_bounds.shape) != expected:
            raise ValueError(
                "value_norm_bounds shape does not match value tensor and block size"
            )

    def to(self, *args, **kwargs) -> "StreamAttnMetadataCache":
        """Return a copy with tensor metadata moved via ``Tensor.to``."""

        def move(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if tensor is None else tensor.to(*args, **kwargs)

        return StreamAttnMetadataCache(
            block_size=self.block_size,
            seq_len=self.seq_len,
            value_norm_bounds=move(self.value_norm_bounds),
            key_centroids=move(self.key_centroids),
            key_radii=move(self.key_radii),
            outlier_keys=move(self.outlier_keys),
            valid_blocks=move(self.valid_blocks),
        )
