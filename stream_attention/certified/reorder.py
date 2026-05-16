"""K/V block ordering policies for certified attention."""

from typing import Sequence, Union

import torch

from .bounds import block_score_upper_bound
from .summaries import BlockSummaries


BlockOrder = Union[str, Sequence[int]]


def resolve_block_order(
    order: BlockOrder,
    query: torch.Tensor,
    summaries: BlockSummaries,
    *,
    scale: float,
) -> list[int]:
    """Resolve a block-order policy to concrete block indices.

    Args:
        order: ``"sequential"``, ``"reverse"``, ``"sink_local"``,
            ``"summary_desc"``, or an explicit sequence of block ids.
        query: Query tensor in ``[batch, heads, seq_q, dim]`` layout.
        summaries: K/V block summaries.
        scale: Attention score scale.
    """

    num_blocks = summaries.num_blocks
    if isinstance(order, str):
        if order == "sequential":
            return list(range(num_blocks))
        if order == "reverse":
            return list(reversed(range(num_blocks)))
        if order == "sink_local":
            if num_blocks == 0:
                return []
            return [0] + list(reversed(range(1, num_blocks)))
        if order == "summary_desc":
            scores = []
            for block_idx in range(num_blocks):
                upper = block_score_upper_bound(
                    query,
                    summaries,
                    block_idx,
                    scale=scale,
                )
                scores.append(float(upper.max().item()))
            return sorted(range(num_blocks), key=lambda idx: scores[idx], reverse=True)
        raise ValueError(f"unknown block order: {order}")

    resolved = [int(idx) for idx in order]
    expected = set(range(num_blocks))
    if set(resolved) != expected or len(resolved) != num_blocks:
        raise ValueError("explicit block order must be a permutation of all block ids")
    return resolved

