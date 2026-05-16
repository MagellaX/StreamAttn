"""Certified single-GPU attention prototypes.

This package contains PyTorch reference implementations for the certified
streaming-attention path. These functions are intentionally kernel-free so the
math and telemetry can be validated before Triton/CUDA work.
"""

from .attention import (
    CertifiedAttentionOutput,
    CertifiedAttentionStats,
    certified_attention,
)
from .metadata import StreamAttnMetadataCache
from .summaries import BlockSummaries, build_block_summaries

__all__ = [
    "BlockSummaries",
    "CertifiedAttentionOutput",
    "CertifiedAttentionStats",
    "StreamAttnMetadataCache",
    "build_block_summaries",
    "certified_attention",
]
