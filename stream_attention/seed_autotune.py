"""Analytical autotuning helpers for StreamAttn seed-only decode kernels.

The seed-only route has a different kernel economics problem than dense exact
attention.  When the scheduled seed window is tiny, duplicating seed K/V reads
per Q head can be cheaper than preserving true-GQA K/V sharing because the
extra CTAs improve occupancy while bytes remain a small fraction of exact.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SeedKernelShape:
    """Shape and seed schedule used by seed-only decode."""

    batch: int
    q_heads: int
    kv_heads: int
    kv_len: int
    dim: int
    block_size: int
    sink_blocks: int
    recent_blocks: int
    middle_seed_blocks: int
    dtype_bytes: int = 2

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "q_heads",
            "kv_heads",
            "kv_len",
            "dim",
            "block_size",
            "dtype_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.q_heads % self.kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads for true GQA")
        for name in ("sink_blocks", "recent_blocks", "middle_seed_blocks"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def group_size(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def seed_blocks(self) -> int:
        return self.sink_blocks + self.recent_blocks + self.middle_seed_blocks

    @property
    def seed_tokens(self) -> int:
        return min(self.kv_len, self.seed_blocks * self.block_size)

    @property
    def seed_token_ratio(self) -> float:
        return float(self.seed_tokens) / float(self.kv_len)

    @property
    def head_private_kv_byte_ratio(self) -> float:
        """Seed K/V bytes vs exact K/V bytes if every Q head reloads seed K/V."""

        return float(self.group_size) * self.seed_token_ratio

    @property
    def gqa_shared_kv_byte_ratio(self) -> float:
        """Seed K/V bytes vs exact K/V bytes if K/V sharing is preserved."""

        return self.seed_token_ratio

    @property
    def exact_kv_bytes(self) -> int:
        return 2 * self.batch * self.kv_heads * self.kv_len * self.dim * self.dtype_bytes

    @property
    def head_private_seed_kv_bytes(self) -> int:
        return 2 * self.batch * self.q_heads * self.seed_tokens * self.dim * self.dtype_bytes

    @property
    def gqa_shared_seed_kv_bytes(self) -> int:
        return 2 * self.batch * self.kv_heads * self.seed_tokens * self.dim * self.dtype_bytes

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            {
                "group_size": self.group_size,
                "seed_blocks": self.seed_blocks,
                "seed_tokens": self.seed_tokens,
                "seed_token_ratio": self.seed_token_ratio,
                "head_private_kv_byte_ratio": self.head_private_kv_byte_ratio,
                "gqa_shared_kv_byte_ratio": self.gqa_shared_kv_byte_ratio,
                "exact_kv_bytes": self.exact_kv_bytes,
                "head_private_seed_kv_bytes": self.head_private_seed_kv_bytes,
                "gqa_shared_seed_kv_bytes": self.gqa_shared_seed_kv_bytes,
            }
        )
        return data


@dataclass(frozen=True)
class SeedKernelCandidate:
    """One candidate seed-only kernel mode."""

    mode: str
    seed_tile_tokens: int
    seed_chunks: int
    cta_count: int
    occupancy_threshold_ctas: int
    kv_byte_ratio_vs_exact: float
    row_work_ratio_vs_exact: float
    duplicate_kv_across_q_heads: bool
    viable_duplication: bool
    viable_occupancy: bool
    recommendation_score: float

    @property
    def viable(self) -> bool:
        return self.viable_duplication and self.viable_occupancy

    def to_dict(self) -> dict:
        data = asdict(self)
        data["viable"] = self.viable
        return data


@dataclass(frozen=True)
class SeedKernelAutotuneResult:
    """Analytical autotune output for a seed-only route."""

    shape: SeedKernelShape
    sm_count: int
    target_waves: float
    duplication_byte_budget: float
    occupancy_threshold_ctas: int
    candidates: tuple[SeedKernelCandidate, ...]
    recommended_mode: str
    recommended_seed_tile_tokens: int
    decision: str

    def to_dict(self) -> dict:
        return {
            "schema": "streamattn.seed_kernel_mode_autotune.v1",
            "shape": self.shape.to_dict(),
            "sm_count": self.sm_count,
            "target_waves": self.target_waves,
            "duplication_byte_budget": self.duplication_byte_budget,
            "occupancy_threshold_ctas": self.occupancy_threshold_ctas,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "recommended_mode": self.recommended_mode,
            "recommended_seed_tile_tokens": self.recommended_seed_tile_tokens,
            "decision": self.decision,
        }


def _unique_positive(values: Iterable[int]) -> List[int]:
    return sorted({int(value) for value in values if int(value) > 0}, reverse=True)


def seed_kernel_candidates(
    shape: SeedKernelShape,
    *,
    sm_count: int = 132,
    target_waves: float = 0.75,
    seed_tile_tokens: Iterable[int] = (384, 256, 192, 128, 96, 64, 32),
    duplication_byte_budget: float = 0.15,
) -> List[SeedKernelCandidate]:
    """Return analytical candidates for seed-only kernel modes.

    ``duplication_byte_budget`` is applied to the head-private byte ratio
    ``G*S/N``.  The default keeps the Qwen L8 32K route green while rejecting
    policies that duplicate too much of the exact KV traffic.
    """

    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if target_waves <= 0.0:
        raise ValueError("target_waves must be positive")
    if duplication_byte_budget <= 0.0:
        raise ValueError("duplication_byte_budget must be positive")

    threshold = max(1, math.ceil(float(sm_count) * float(target_waves)))
    candidates: list[SeedKernelCandidate] = []

    direct_ctas = shape.batch * shape.q_heads
    duplicate_ratio = shape.head_private_kv_byte_ratio
    row_ratio = shape.seed_token_ratio
    candidates.append(
        SeedKernelCandidate(
            mode="head_private_direct_seed",
            seed_tile_tokens=shape.seed_tokens,
            seed_chunks=1,
            cta_count=direct_ctas,
            occupancy_threshold_ctas=threshold,
            kv_byte_ratio_vs_exact=duplicate_ratio,
            row_work_ratio_vs_exact=row_ratio,
            duplicate_kv_across_q_heads=True,
            viable_duplication=duplicate_ratio <= duplication_byte_budget,
            viable_occupancy=direct_ctas >= threshold,
            recommendation_score=_candidate_score(
                direct_ctas,
                threshold,
                duplicate_ratio,
                duplication_byte_budget,
                seed_chunks=1,
            ),
        )
    )

    for tile in _unique_positive(seed_tile_tokens):
        tile = min(tile, shape.seed_tokens)
        chunks = max(1, math.ceil(shape.seed_tokens / tile))
        if chunks == 1:
            continue
        ctas = shape.batch * shape.q_heads * chunks
        candidates.append(
            SeedKernelCandidate(
                mode="head_private_split_seed",
                seed_tile_tokens=tile,
                seed_chunks=chunks,
                cta_count=ctas,
                occupancy_threshold_ctas=threshold,
                kv_byte_ratio_vs_exact=duplicate_ratio,
                row_work_ratio_vs_exact=row_ratio,
                duplicate_kv_across_q_heads=True,
                viable_duplication=duplicate_ratio <= duplication_byte_budget,
                viable_occupancy=ctas >= threshold,
                recommendation_score=_candidate_score(
                    ctas,
                    threshold,
                    duplicate_ratio,
                    duplication_byte_budget,
                    seed_chunks=chunks,
                ),
            )
        )

    shared_ratio = shape.gqa_shared_kv_byte_ratio
    for tile in _unique_positive(seed_tile_tokens):
        tile = min(tile, shape.seed_tokens)
        chunks = max(1, math.ceil(shape.seed_tokens / tile))
        ctas = shape.batch * shape.kv_heads * chunks
        candidates.append(
            SeedKernelCandidate(
                mode="gqa_shared_seed",
                seed_tile_tokens=tile,
                seed_chunks=chunks,
                cta_count=ctas,
                occupancy_threshold_ctas=threshold,
                kv_byte_ratio_vs_exact=shared_ratio,
                row_work_ratio_vs_exact=row_ratio,
                duplicate_kv_across_q_heads=False,
                viable_duplication=True,
                viable_occupancy=ctas >= threshold,
                recommendation_score=_candidate_score(
                    ctas,
                    threshold,
                    shared_ratio,
                    duplication_byte_budget,
                    seed_chunks=chunks,
                )
                - 0.20,  # shared-GQA seed has lower CTA supply for small Hkv.
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (candidate.viable, candidate.recommendation_score),
        reverse=True,
    )


def _candidate_score(
    cta_count: int,
    occupancy_threshold_ctas: int,
    kv_byte_ratio: float,
    duplication_byte_budget: float,
    *,
    seed_chunks: int,
) -> float:
    occupancy = min(float(cta_count) / float(occupancy_threshold_ctas), 1.0)
    byte_headroom = max(0.0, 1.0 - kv_byte_ratio / duplication_byte_budget)
    merge_penalty = 0.04 * max(0, seed_chunks - 1)
    over_split_penalty = 0.01 * max(0.0, float(cta_count) / float(occupancy_threshold_ctas) - 2.0)
    return occupancy + byte_headroom - merge_penalty - over_split_penalty


def autotune_seed_kernel_mode(
    shape: SeedKernelShape,
    *,
    sm_count: int = 132,
    target_waves: float = 0.75,
    seed_tile_tokens: Iterable[int] = (384, 256, 192, 128, 96, 64, 32),
    duplication_byte_budget: float = 0.15,
) -> SeedKernelAutotuneResult:
    """Pick an analytical seed kernel mode for a route."""

    candidates = seed_kernel_candidates(
        shape,
        sm_count=sm_count,
        target_waves=target_waves,
        seed_tile_tokens=seed_tile_tokens,
        duplication_byte_budget=duplication_byte_budget,
    )
    viable = [candidate for candidate in candidates if candidate.viable]
    if viable:
        best = viable[0]
        decision = "seed_only_native_candidate"
    else:
        best = candidates[0]
        decision = "exact_native_until_kernel_or_policy_changes"
    return SeedKernelAutotuneResult(
        shape=shape,
        sm_count=sm_count,
        target_waves=target_waves,
        duplication_byte_budget=duplication_byte_budget,
        occupancy_threshold_ctas=max(1, math.ceil(float(sm_count) * float(target_waves))),
        candidates=tuple(candidates),
        recommended_mode=best.mode,
        recommended_seed_tile_tokens=best.seed_tile_tokens,
        decision=decision,
    )


def seed_shape_from_policy(
    policy,
    *,
    batch: Optional[int] = None,
    dtype_bytes: int = 2,
) -> SeedKernelShape:
    """Build a :class:`SeedKernelShape` from a Gate0 seed-only policy-like object."""

    policy_batch = getattr(policy, "batch", None)
    shape_batch = int(batch if batch is not None else policy_batch if policy_batch is not None else getattr(policy, "min_batch"))
    return SeedKernelShape(
        batch=shape_batch,
        q_heads=int(getattr(policy, "heads")),
        kv_heads=int(getattr(policy, "kv_heads")),
        kv_len=int(getattr(policy, "kv_len_bucket")),
        dim=int(getattr(policy, "dim")),
        block_size=int(getattr(policy, "block_size")),
        sink_blocks=int(getattr(policy, "sink_blocks")),
        recent_blocks=int(getattr(policy, "recent_blocks")),
        middle_seed_blocks=int(getattr(policy, "middle_seed_blocks")),
        dtype_bytes=int(dtype_bytes),
    )
