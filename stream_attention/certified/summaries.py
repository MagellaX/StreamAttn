"""K/V block summaries for certified attention."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class BlockSummaries:
    """Centroid/radius summaries for K/V blocks.

    Shapes use ``[batch, heads, blocks, ...]`` because the certified attention
    prototype internally works in batch/head-major layout.
    """

    centroid: torch.Tensor
    radius: torch.Tensor
    max_key_norm: torch.Tensor
    max_value_norm: torch.Tensor
    block_lengths: torch.Tensor
    block_size: int
    seq_len: int
    outlier_keys: Optional[torch.Tensor] = None
    outlier_mask: Optional[torch.Tensor] = None

    @property
    def num_blocks(self) -> int:
        return int(self.centroid.shape[2])


def _validate_kv(key: torch.Tensor, value: Optional[torch.Tensor]) -> None:
    if key.dim() != 4:
        raise ValueError("key must have shape [batch, seq, heads, dim]")
    if value is not None and value.shape != key.shape:
        raise ValueError("value must have the same shape as key")


def build_block_summaries(
    key: torch.Tensor,
    value: Optional[torch.Tensor] = None,
    *,
    block_size: int = 64,
    num_outliers: int = 0,
) -> BlockSummaries:
    """Build centroid/radius summaries for K/V blocks.

    Args:
        key: Tensor with shape ``[batch, seq_k, heads, dim]``.
        value: Optional tensor with the same shape as ``key``. When omitted,
            ``max_value_norm`` is filled with zeros.
        block_size: Number of K/V tokens represented by one summary block.
        num_outliers: Number of farthest-from-centroid keys to split out from
            each residual ball. ``0`` gives plain centroid/radius summaries.

    Returns:
        ``BlockSummaries`` in batch/head-major layout.
    """

    _validate_kv(key, value)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_outliers < 0:
        raise ValueError("num_outliers must be non-negative")

    batch, seq_len, heads, dim = key.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    device = key.device

    key_bh = key.permute(0, 2, 1, 3).contiguous().float()
    value_bh = None if value is None else value.permute(0, 2, 1, 3).contiguous().float()

    centroid = torch.empty(batch, heads, num_blocks, dim, device=device, dtype=torch.float32)
    radius = torch.empty(batch, heads, num_blocks, device=device, dtype=torch.float32)
    max_key_norm = torch.empty(batch, heads, num_blocks, device=device, dtype=torch.float32)
    max_value_norm = torch.empty(batch, heads, num_blocks, device=device, dtype=torch.float32)
    block_lengths = torch.empty(num_blocks, device=device, dtype=torch.long)
    outlier_keys = None
    outlier_mask = None
    if num_outliers > 0:
        outlier_keys = torch.zeros(
            batch,
            heads,
            num_blocks,
            num_outliers,
            dim,
            device=device,
            dtype=torch.float32,
        )
        outlier_mask = torch.zeros(
            batch,
            heads,
            num_blocks,
            num_outliers,
            device=device,
            dtype=torch.bool,
        )

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        block = key_bh[:, :, start:end, :]
        raw_center = block.mean(dim=2)
        raw_centered = block - raw_center[:, :, None, :]

        if num_outliers > 0:
            selected = min(num_outliers, end - start)
            dist = torch.linalg.vector_norm(raw_centered, dim=-1)
            top_idx = torch.topk(dist, k=selected, dim=-1).indices
            gathered = torch.gather(
                block,
                2,
                top_idx[:, :, :, None].expand(batch, heads, selected, dim),
            )
            outlier_keys[:, :, block_idx, :selected, :] = gathered
            outlier_mask[:, :, block_idx, :selected] = True

            keep = torch.ones(batch, heads, end - start, device=device, dtype=torch.bool)
            keep.scatter_(2, top_idx, False)
            keep_f = keep.to(torch.float32)
            residual_count = keep_f.sum(dim=2).clamp_min(1.0)
            center = (block * keep_f[:, :, :, None]).sum(dim=2) / residual_count[:, :, None]
            centered = block - center[:, :, None, :]
            residual_dist = torch.linalg.vector_norm(centered, dim=-1)
            residual_dist = residual_dist.masked_fill(~keep, 0.0)
        else:
            center = raw_center
            residual_dist = torch.linalg.vector_norm(raw_centered, dim=-1)

        centroid[:, :, block_idx, :] = center
        radius[:, :, block_idx] = residual_dist.amax(dim=-1)
        max_key_norm[:, :, block_idx] = torch.linalg.vector_norm(block, dim=-1).amax(dim=-1)
        block_lengths[block_idx] = end - start

        if value_bh is None:
            max_value_norm[:, :, block_idx] = 0.0
        else:
            value_block = value_bh[:, :, start:end, :]
            max_value_norm[:, :, block_idx] = torch.linalg.vector_norm(
                value_block, dim=-1
            ).amax(dim=-1)

    return BlockSummaries(
        centroid=centroid,
        radius=radius,
        max_key_norm=max_key_norm,
        max_value_norm=max_value_norm,
        block_lengths=block_lengths,
        block_size=block_size,
        seq_len=seq_len,
        outlier_keys=outlier_keys,
        outlier_mask=outlier_mask,
    )
