"""Runtime facade for calibrated Gate-0 fused-hybrid decode.

This is the StreamAttn-facing wrapper for the current Gate-0 research path:
calibrated projection metadata is consumed inside the split-K online attention
loop, while unsafe heads run exact mode in the same fused kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .gate1 import dense_attention_forward
from .kernels.gate1_inline_projection_splitk_triton import (
    gate1_inline_projection_splitk_attention_triton_forward,
    make_splitk_workspace,
)


@dataclass(frozen=True)
class Gate0FusedHybridPolicy:
    """Calibrated fused-hybrid Gate-0 runtime configuration."""

    head_modes: Tuple[int, ...]
    trusted_sparse_heads: Tuple[int, ...]
    exact_heads: Tuple[int, ...]
    block_size: int = 32
    sink_blocks: int = 2
    recent_blocks: int = 2
    middle_seed_blocks: int = 8
    chunk_anchor_blocks: int = 0
    block_order: str = "recent_first"
    num_chunks: int = 32
    seed_strategy: str = "recompute_seed"
    filter_margin: float = 64.0
    error_budget: float = 1e-2
    projection_kind: str = "random"
    projection_dim: int = 8
    projection_seed: int = 1
    projection_metadata_dtype: str = "fp16"
    splitk_workspace: str = "reuse"
    safety_budget_name: str = "unknown"
    model_id: str = "unknown"
    layer_id: int = -1
    kv_len_bucket: Optional[int] = None
    expected_dense_ms: Optional[float] = None
    expected_fused_hybrid_ms: Optional[float] = None
    expected_speedup_vs_dense: Optional[float] = None
    expected_max_abs_error: Optional[float] = None
    expected_mean_abs_error: Optional[float] = None

    @property
    def heads(self) -> int:
        return len(self.head_modes)

    @classmethod
    def from_entry(cls, entry: Dict[str, Any]) -> "Gate0FusedHybridPolicy":
        """Create a policy from a fused-hybrid policy artifact entry."""

        runtime = dict(entry.get("runtime") or {})
        budget = dict(entry.get("safety_budget") or {})
        quality = dict(entry.get("quality") or {})
        head_modes = tuple(int(item) for item in runtime.get("head_modes") or ())
        if not head_modes:
            trusted = set(int(item) for item in runtime.get("trusted_sparse_heads") or ())
            exact = set(int(item) for item in runtime.get("exact_heads") or ())
            head_count = max(trusted | exact) + 1 if trusted or exact else 0
            head_modes = tuple(0 if head in trusted else 1 for head in range(head_count))
        trusted_sparse_heads = tuple(
            int(item) for item in runtime.get("trusted_sparse_heads") or ()
        )
        exact_heads = tuple(int(item) for item in runtime.get("exact_heads") or ())
        return cls(
            head_modes=head_modes,
            trusted_sparse_heads=trusted_sparse_heads,
            exact_heads=exact_heads,
            block_size=int(runtime.get("block_size") or 32),
            sink_blocks=int(runtime.get("sink_blocks") or 2),
            recent_blocks=int(runtime.get("recent_blocks") or 2),
            middle_seed_blocks=int(runtime.get("middle_seed_blocks") or 8),
            chunk_anchor_blocks=int(runtime.get("chunk_anchor_blocks") or 0),
            block_order=str(runtime.get("block_order") or "recent_first"),
            num_chunks=int(runtime.get("num_chunks") or 32),
            seed_strategy=str(runtime.get("seed_strategy") or "recompute_seed"),
            filter_margin=float(runtime.get("filter_margin") or 0.0),
            error_budget=float(runtime.get("error_budget") or 0.0),
            projection_kind=str(runtime.get("projection_kind") or "random"),
            projection_dim=int(runtime.get("projection_dim") or 8),
            projection_seed=int(runtime.get("projection_seed") or 1),
            projection_metadata_dtype=str(runtime.get("projection_metadata_dtype") or "fp16"),
            splitk_workspace=str(runtime.get("splitk_workspace") or "reuse"),
            safety_budget_name=str(budget.get("name") or "unknown"),
            model_id=str(entry.get("model_id") or "unknown"),
            layer_id=int(entry.get("layer_id") if entry.get("layer_id") is not None else -1),
            kv_len_bucket=(
                int(entry["kv_len_bucket"])
                if entry.get("kv_len_bucket") is not None
                else None
            ),
            expected_dense_ms=(
                float(quality["dense_all_ms"])
                if quality.get("dense_all_ms") is not None
                else None
            ),
            expected_fused_hybrid_ms=(
                float(quality["fused_hybrid_ms"])
                if quality.get("fused_hybrid_ms") is not None
                else None
            ),
            expected_speedup_vs_dense=(
                float(quality["speedup_vs_dense_all"])
                if quality.get("speedup_vs_dense_all") is not None
                else None
            ),
            expected_max_abs_error=(
                float(quality["max_abs_error"])
                if quality.get("max_abs_error") is not None
                else None
            ),
            expected_mean_abs_error=(
                float(quality["mean_abs_error"])
                if quality.get("mean_abs_error") is not None
                else None
            ),
        )

    @classmethod
    def from_json(cls, path: Union[str, Path], *, entry_index: int = 0) -> "Gate0FusedHybridPolicy":
        """Load a policy entry from a JSON policy artifact."""

        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = payload.get("stable_entries") or payload.get("entries") or []
        if not entries:
            raise ValueError("policy JSON does not contain entries")
        return cls.from_entry(entries[entry_index])

    def head_modes_tensor(self, device: torch.device) -> torch.Tensor:
        """Return int32 head modes on the requested device."""

        return torch.tensor(self.head_modes, device=device, dtype=torch.int32)


@dataclass(frozen=True)
class Gate0ProjectionMetadata:
    """Projection metadata cache for fused-hybrid Gate-0 decode."""

    projection: torch.Tensor
    proj_min: torch.Tensor
    proj_max: torch.Tensor
    block_size: int
    projection_kind: str
    projection_seed: int
    metadata_dtype: torch.dtype

    @property
    def rank(self) -> int:
        return int(self.projection.shape[0])

    @property
    def num_blocks(self) -> int:
        return int(self.proj_min.shape[2])

    def validate_for(self, key: torch.Tensor, policy: Gate0FusedHybridPolicy) -> None:
        if key.dim() != 4:
            raise ValueError("key must have shape [batch, kv_len, heads, dim]")
        batch, seq_k, heads, dim = key.shape
        expected_blocks = (seq_k + policy.block_size - 1) // policy.block_size
        if self.projection.shape != (policy.projection_dim, dim):
            raise ValueError("projection shape does not match key dim/policy rank")
        if self.proj_min.shape != (batch, heads, expected_blocks, policy.projection_dim):
            raise ValueError("projection metadata shape does not match key/policy")
        if self.proj_max.shape != self.proj_min.shape:
            raise ValueError("proj_min and proj_max shapes differ")


@dataclass(frozen=True)
class Gate0FusedHybridStats:
    projection_skipped_blocks: int
    projection_computed_blocks: int
    gate1_post_qk_skipped_blocks: int
    pv_executed_blocks: int
    middle_blocks: int
    seed_blocks: int
    chunks: int
    mode_sum: int

    @property
    def projection_skip_fraction(self) -> float:
        return 0.0 if self.middle_blocks <= 0 else self.projection_skipped_blocks / self.middle_blocks

    @property
    def pv_executed_fraction(self) -> float:
        return 0.0 if self.middle_blocks <= 0 else self.pv_executed_blocks / self.middle_blocks


@dataclass(frozen=True)
class Gate0FusedHybridRunInfo:
    policy: Gate0FusedHybridPolicy
    stats: Optional[Gate0FusedHybridStats]
    per_head_stats: Optional[Tuple[Gate0FusedHybridStats, ...]]


def _metadata_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported projection metadata dtype: {name}")


def _hadamard_matrix(dim: int, *, device: torch.device) -> torch.Tensor:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("hadamard projection requires power-of-two dim")
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < dim:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return F.normalize(h, p=2, dim=-1)


def build_gate0_projection_matrix(
    *,
    kind: str,
    dim: int,
    rank: int,
    seed: int,
    device: Union[torch.device, str],
) -> torch.Tensor:
    """Build the fixed projection matrix used by calibrated Gate-0."""

    if rank <= 0 or rank > dim:
        raise ValueError("projection rank must be in [1, dim]")
    device = torch.device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if kind == "random":
        matrix = torch.randn(rank, dim, generator=generator, dtype=torch.float32)
        return F.normalize(matrix, p=2, dim=-1).to(device=device)
    if kind == "hadamard":
        h = _hadamard_matrix(dim, device=torch.device("cpu"))
        indices = torch.randperm(dim, generator=generator)[:rank]
        return h[indices].to(device=device)
    raise ValueError(f"unknown projection kind: {kind}")


def build_gate0_projection_metadata(
    key: torch.Tensor,
    policy: Gate0FusedHybridPolicy,
    *,
    projection: Optional[torch.Tensor] = None,
) -> Gate0ProjectionMetadata:
    """Build per-block projection min/max metadata for a KV cache."""

    if key.dim() != 4:
        raise ValueError("key must have shape [batch, kv_len, heads, dim]")
    if key.shape[2] != policy.heads:
        raise ValueError("key head count does not match policy head modes")
    projection = (
        projection
        if projection is not None
        else build_gate0_projection_matrix(
            kind=policy.projection_kind,
            dim=key.shape[-1],
            rank=policy.projection_dim,
            seed=policy.projection_seed,
            device=key.device,
        )
    )
    if projection.shape != (policy.projection_dim, key.shape[-1]):
        raise ValueError("projection shape does not match policy/key")
    k_bhnd = key.permute(0, 2, 1, 3).contiguous().float()
    k_proj = torch.einsum("bhnd,rd->bhnr", k_bhnd, projection)
    batch, heads, seq_k, rank = k_proj.shape
    num_blocks = (seq_k + policy.block_size - 1) // policy.block_size
    mins = torch.empty(batch, heads, num_blocks, rank, device=key.device, dtype=torch.float32)
    maxs = torch.empty_like(mins)
    for block_idx in range(num_blocks):
        start = block_idx * policy.block_size
        end = min(start + policy.block_size, seq_k)
        block = k_proj[:, :, start:end, :]
        mins[:, :, block_idx, :] = block.amin(dim=2)
        maxs[:, :, block_idx, :] = block.amax(dim=2)
    dtype = _metadata_dtype(policy.projection_metadata_dtype)
    return Gate0ProjectionMetadata(
        projection=projection,
        proj_min=mins.to(dtype=dtype),
        proj_max=maxs.to(dtype=dtype),
        block_size=policy.block_size,
        projection_kind=policy.projection_kind,
        projection_seed=policy.projection_seed,
        metadata_dtype=dtype,
    )


def summarize_gate0_fused_hybrid_raw_stats(raw_stats: torch.Tensor) -> Gate0FusedHybridStats:
    """Summarize raw fused-hybrid split-K counters."""

    if raw_stats.dim() != 4 or raw_stats.shape[-1] != 8:
        raise ValueError("raw_stats must have shape [batch, heads, chunks, 8]")
    totals = raw_stats.detach().sum(dim=(0, 1, 2)).cpu()
    return Gate0FusedHybridStats(
        projection_skipped_blocks=int(totals[0].item()),
        projection_computed_blocks=int(totals[1].item()),
        gate1_post_qk_skipped_blocks=int(totals[2].item()),
        pv_executed_blocks=int(totals[3].item()),
        middle_blocks=int(totals[4].item()),
        seed_blocks=int(totals[5].item()),
        chunks=int(totals[6].item()),
        mode_sum=int(totals[7].item()),
    )


def summarize_gate0_fused_hybrid_raw_stats_per_head(
    raw_stats: torch.Tensor,
) -> Tuple[Gate0FusedHybridStats, ...]:
    """Summarize fused-hybrid split-K counters independently per head."""

    if raw_stats.dim() != 4 or raw_stats.shape[-1] != 8:
        raise ValueError("raw_stats must have shape [batch, heads, chunks, 8]")
    totals = raw_stats.detach().sum(dim=(0, 2)).cpu()
    return tuple(
        Gate0FusedHybridStats(
            projection_skipped_blocks=int(row[0].item()),
            projection_computed_blocks=int(row[1].item()),
            gate1_post_qk_skipped_blocks=int(row[2].item()),
            pv_executed_blocks=int(row[3].item()),
            middle_blocks=int(row[4].item()),
            seed_blocks=int(row[5].item()),
            chunks=int(row[6].item()),
            mode_sum=int(row[7].item()),
        )
        for row in totals
    )


def make_gate0_fused_hybrid_workspace(
    query: torch.Tensor,
    policy: Gate0FusedHybridPolicy,
) -> Dict[str, torch.Tensor]:
    """Allocate reusable fused-hybrid split-K workspace buffers."""

    return make_splitk_workspace(
        query,
        rank=policy.projection_dim,
        num_chunks=policy.num_chunks,
        seed_strategy=policy.seed_strategy,
    )


def stream_attn_gate0_fused_hybrid(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    policy: Gate0FusedHybridPolicy,
    metadata: Optional[Gate0ProjectionMetadata] = None,
    build_metadata_if_missing: bool = True,
    causal: bool = False,
    return_info: bool = False,
    workspace: Optional[Dict[str, torch.Tensor]] = None,
    fallback: str = "error",
    num_warps: int = 4,
    num_stages: int = 3,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Gate0FusedHybridRunInfo]]:
    """Run calibrated fused-hybrid Gate-0 decode.

    This API is intentionally decode-only: ``query`` must be ``[B, 1, H, D]``.
    If causal masking is needed, the KV cache should already be cropped to the
    visible prefix before calling this function.
    """

    if causal:
        raise ValueError("fused-hybrid Gate-0 expects a pre-cropped decode KV cache; causal=True is not supported")
    if fallback not in {"error", "dense"}:
        raise ValueError("fallback must be 'error' or 'dense'")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must have shape [batch, seq, heads, dim]")
    if query.shape[1] != 1:
        raise ValueError("fused-hybrid Gate-0 only supports query_len == 1")
    if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query/key/value must have matching batch, heads, and dim")
    if query.shape[2] != policy.heads:
        raise ValueError("query head count does not match policy head modes")
    if not query.is_cuda:
        if fallback == "dense":
            output = dense_attention_forward(query, key, value, causal=False)
            if return_info:
                info = Gate0FusedHybridRunInfo(policy=policy, stats=None, per_head_stats=None)
                return output, info
            return output
        raise RuntimeError("fused-hybrid Gate-0 requires CUDA tensors")

    if metadata is None:
        if not build_metadata_if_missing:
            raise ValueError("projection metadata is required when build_metadata_if_missing=False")
        metadata = build_gate0_projection_metadata(key, policy)
    metadata.validate_for(key, policy)

    output, raw_stats = gate1_inline_projection_splitk_attention_triton_forward(
        query,
        key,
        value,
        None,
        metadata.proj_min,
        metadata.proj_max,
        projection=metadata.projection,
        compute_qproj=True,
        num_chunks=policy.num_chunks,
        error_budget=policy.error_budget,
        filter_margin=policy.filter_margin,
        block_size=policy.block_size,
        sink_blocks=policy.sink_blocks,
        recent_blocks=policy.recent_blocks,
        middle_seed_blocks=policy.middle_seed_blocks,
        chunk_anchor_blocks=policy.chunk_anchor_blocks,
        block_order=policy.block_order,
        seed_strategy=policy.seed_strategy,
        head_modes=policy.head_modes_tensor(query.device),
        return_raw_stats=return_info,
        workspace=workspace,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if not return_info:
        return output
    stats = summarize_gate0_fused_hybrid_raw_stats(raw_stats) if raw_stats is not None else None
    per_head_stats = (
        summarize_gate0_fused_hybrid_raw_stats_per_head(raw_stats)
        if raw_stats is not None
        else None
    )
    return output, Gate0FusedHybridRunInfo(
        policy=policy,
        stats=stats,
        per_head_stats=per_head_stats,
    )
