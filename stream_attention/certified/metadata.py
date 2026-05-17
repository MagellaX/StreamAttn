"""Reusable block metadata for certified and Gate-1 attention paths."""

from dataclasses import dataclass
from typing import Optional

import torch

from stream_attention.kernels.gate1_fwd_triton import build_value_norm_bounds
from stream_attention.kernels.metadata_triton import (
    TRITON_AVAILABLE as METADATA_TRITON_AVAILABLE,
    build_value_norm_bounds_triton,
)
from stream_attention.kernels.metadata_update_triton import (
    TRITON_AVAILABLE as METADATA_UPDATE_TRITON_AVAILABLE,
    update_value_norm_bounds_triton_,
)


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
        use_triton: Optional[bool] = None,
    ) -> "StreamAttnMetadataCache":
        """Build a cache containing conservative per-block value norm bounds."""

        if safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1.0")
        if value.dim() != 4:
            raise ValueError("value must have shape [batch, seq, heads, dim]")

        if use_triton is None:
            use_triton = bool(value.is_cuda and METADATA_TRITON_AVAILABLE)
        if use_triton:
            bounds = build_value_norm_bounds_triton(value, block_size=block_size)
        else:
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

    def update_value_bounds_(
        self,
        new_value: torch.Tensor,
        *,
        start_pos: int,
        use_triton: Optional[bool] = None,
    ) -> "StreamAttnMetadataCache":
        """Incrementally update cached value bounds for appended KV-cache values."""

        if self.value_norm_bounds is None:
            raise ValueError("metadata cache does not contain value_norm_bounds")
        if new_value.dim() != 4:
            raise ValueError("new_value must have shape [batch, seq, heads, dim]")
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")

        batch, new_seq, heads, _ = new_value.shape
        if start_pos + new_seq > self.seq_len:
            raise ValueError("new_value extends beyond metadata cache seq_len")
        expected = (batch, heads, self.num_blocks)
        if tuple(self.value_norm_bounds.shape) != expected:
            raise ValueError(
                "value_norm_bounds shape does not match new_value and block size"
            )
        if new_seq == 0:
            return self

        if use_triton is None:
            use_triton = bool(
                new_value.is_cuda
                and self.value_norm_bounds.is_cuda
                and METADATA_UPDATE_TRITON_AVAILABLE
            )
        if use_triton:
            update_value_norm_bounds_triton_(
                self.value_norm_bounds,
                new_value,
                start_pos=start_pos,
                total_seq_len=self.seq_len,
                block_size=self.block_size,
            )
            return self

        value_bh = new_value.permute(0, 2, 1, 3).contiguous().float()
        first_block = start_pos // self.block_size
        last_block = (start_pos + new_seq - 1) // self.block_size
        for block_idx in range(first_block, last_block + 1):
            block_start = block_idx * self.block_size
            block_end = min(block_start + self.block_size, self.seq_len)
            local_start = max(0, block_start - start_pos)
            local_end = min(new_seq, block_end - start_pos)
            if local_start >= local_end:
                continue
            norms = torch.linalg.vector_norm(
                value_bh[:, :, local_start:local_end, :],
                dim=-1,
            ).amax(dim=-1)
            self.value_norm_bounds[:, :, block_idx] = torch.maximum(
                self.value_norm_bounds[:, :, block_idx],
                norms.to(self.value_norm_bounds.dtype),
            )
        return self

    def clone(self) -> "StreamAttnMetadataCache":
        """Return a deep tensor copy of this metadata cache."""

        def clone_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if tensor is None else tensor.clone()

        return StreamAttnMetadataCache(
            block_size=self.block_size,
            seq_len=self.seq_len,
            value_norm_bounds=clone_tensor(self.value_norm_bounds),
            key_centroids=clone_tensor(self.key_centroids),
            key_radii=clone_tensor(self.key_radii),
            outlier_keys=clone_tensor(self.outlier_keys),
            valid_blocks=clone_tensor(self.valid_blocks),
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
