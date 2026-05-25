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
from .certified import (
    BlockSummaries,
    CertifiedAttentionOutput,
    CertifiedAttentionStats,
    StreamAttnMetadataCache,
    build_block_summaries,
    certified_attention,
)
from .router import (
    AttentionRouteRequest,
    BackendDecision,
    ActiveCurvePoint,
    CostEntry,
    CostKey,
    Gate1CostModel,
    StreamAttnPolicy,
    StreamAttnRouter,
    normalize_device_class,
)
from .telemetry import ActiveFractionKey, ActiveFractionTelemetry, Prediction
from .gate1 import (
    Gate1RunInfo,
    Gate1Stats,
    dense_attention_forward,
    make_route_request,
    stream_attn_gate1,
    summarize_gate1_raw_stats,
    summarize_gate1_raw_stats_per_head,
)
from .gate0_fused_hybrid import (
    Gate0FusedHybridPolicy,
    Gate0FusedHybridRunInfo,
    Gate0FusedHybridStats,
    Gate0ProjectionMetadata,
    build_gate0_projection_matrix,
    build_gate0_projection_metadata,
    make_gate0_fused_hybrid_workspace,
    stream_attn_gate0_fused_hybrid,
    summarize_gate0_fused_hybrid_raw_stats,
    summarize_gate0_fused_hybrid_raw_stats_per_head,
)
from .decode import (
    DecodeCostEntry,
    DecodeCostKey,
    DecodeCostModel,
    Gate0SeedOnlyBatchedPolicy,
    Gate0SeedOnlyBatchedRunInfo,
    Gate0SeedOnlyBatchedStats,
    StreamAttnDecodePlan,
    StreamAttnDecodePolicy,
    StreamAttnDecodeWorkspace,
    StreamAttnDecodeWrapper,
    StreamAttnSeedOnlyDecodeService,
    StreamAttnServingInfo,
    decode_cost_model_from_profile_rows,
    load_packaged_gate0_seed_only_batched_policy,
    stream_attn_decode_plan,
    stream_attn_decode_run,
    stream_attn_seed_only_decode,
)

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
    "BlockSummaries",
    "CertifiedAttentionOutput",
    "CertifiedAttentionStats",
    "StreamAttnMetadataCache",
    "AttentionRouteRequest",
    "BackendDecision",
    "ActiveCurvePoint",
    "CostEntry",
    "CostKey",
    "Gate1CostModel",
    "StreamAttnPolicy",
    "StreamAttnRouter",
    "normalize_device_class",
    "ActiveFractionKey",
    "ActiveFractionTelemetry",
    "Prediction",
    "Gate1RunInfo",
    "Gate1Stats",
    # Factory functions
    "create_stream_attention",
    "create_fused_online_attention",
    "build_block_summaries",
    "certified_attention",
    "dense_attention_forward",
    "make_route_request",
    "stream_attn_gate1",
    "summarize_gate1_raw_stats",
    "summarize_gate1_raw_stats_per_head",
    "Gate0FusedHybridPolicy",
    "Gate0FusedHybridRunInfo",
    "Gate0FusedHybridStats",
    "Gate0ProjectionMetadata",
    "build_gate0_projection_matrix",
    "build_gate0_projection_metadata",
    "make_gate0_fused_hybrid_workspace",
    "stream_attn_gate0_fused_hybrid",
    "summarize_gate0_fused_hybrid_raw_stats",
    "summarize_gate0_fused_hybrid_raw_stats_per_head",
    "DecodeCostEntry",
    "DecodeCostKey",
    "DecodeCostModel",
    "Gate0SeedOnlyBatchedPolicy",
    "Gate0SeedOnlyBatchedRunInfo",
    "Gate0SeedOnlyBatchedStats",
    "StreamAttnDecodePlan",
    "StreamAttnDecodePolicy",
    "StreamAttnDecodeWorkspace",
    "StreamAttnDecodeWrapper",
    "StreamAttnSeedOnlyDecodeService",
    "StreamAttnServingInfo",
    "decode_cost_model_from_profile_rows",
    "load_packaged_gate0_seed_only_batched_policy",
    "stream_attn_decode_plan",
    "stream_attn_decode_run",
    "stream_attn_seed_only_decode",
    # Utilities
    "MemoryProfiler",
    "create_kv_compressor",
    "gradient_checkpoint_sequential",
    # Version
    "__version__",
]
