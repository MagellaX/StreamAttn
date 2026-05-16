"""Bound calculations for certified attention."""

import math
from typing import Optional

import torch

from .summaries import BlockSummaries


def block_score_upper_bound(
    query: torch.Tensor,
    summaries: BlockSummaries,
    block_idx: int,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Upper-bound QK scores for one K block.

    Args:
        query: Tensor with shape ``[batch, heads, seq_q, dim]`` in float.
        summaries: K/V block summaries.
        block_idx: Summary block index.
        scale: Optional attention scale. Defaults to ``1 / sqrt(dim)``.

    Returns:
        Tensor with shape ``[batch, heads, seq_q]`` where each element bounds
        every scaled dot-product score between that query row and any key in
        ``block_idx``.
    """

    if query.dim() != 4:
        raise ValueError("query must have shape [batch, heads, seq_q, dim]")
    if block_idx < 0 or block_idx >= summaries.num_blocks:
        raise IndexError("block_idx is out of range")

    dim = query.shape[-1]
    scale = (1.0 / math.sqrt(dim)) if scale is None else scale

    centroid = summaries.centroid[:, :, block_idx, :]
    radius = summaries.radius[:, :, block_idx]
    dot_centroid = torch.sum(query * centroid[:, :, None, :], dim=-1)
    query_norm = torch.linalg.vector_norm(query, dim=-1)
    residual_bound = (dot_centroid + query_norm * radius[:, :, None]) * scale

    if summaries.outlier_keys is None or summaries.outlier_mask is None:
        return residual_bound

    outliers = summaries.outlier_keys[:, :, block_idx, :, :]
    outlier_mask = summaries.outlier_mask[:, :, block_idx, :]
    outlier_scores = torch.einsum("bhsd,bhod->bhso", query, outliers) * scale
    outlier_scores = outlier_scores.masked_fill(
        ~outlier_mask[:, :, None, :],
        -float("inf"),
    )
    outlier_bound = outlier_scores.amax(dim=-1)
    return torch.maximum(residual_bound, outlier_bound)
