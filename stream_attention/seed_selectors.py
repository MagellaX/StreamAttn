"""Seed block selector utilities.

These helpers are deliberately tensor-level and side-effect free so they can be
used by diagnostic profilers first and production kernels later.  The core
problem is selecting middle seed blocks for a decode query without blindly using
fixed middle positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch


DEFAULT_SELECTOR_PROFILES = ("fixed_policy",)
SELECTOR_PROFILES = frozenset(
    {
        "fixed_policy",
        "block_mean_proxy",
        "block_l2_bound_proxy",
        "support_top2_norm",
        "support_top4_norm",
        "support_top2_norm_refine16",
        "support_top2_norm_refine32",
        "support_top4_norm_refine16",
        "support_top4_norm_refine32",
        "support_extreme2_mean",
        "support_extreme4_mean",
        "support_extreme2_mean_refine32",
        "support_extreme4_mean_refine32",
        "support_rand4",
        "support_rand8",
        "support_rand4_refine32",
        "support_rand8_refine32",
        "support_block_oracle",
        "qk_block_max",
        "exact_mass_oracle",
        "support_mass_oracle",
        "value_residual_oracle",
    }
)


@dataclass(frozen=True)
class BlockSummary:
    block_size: int
    method: str
    mean: Optional[torch.Tensor] = None
    radius: Optional[torch.Tensor] = None
    support_keys: Optional[torch.Tensor] = None
    support_key_indices: Optional[torch.Tensor] = None


def parse_selector_profiles(text: str) -> List[str]:
    profiles = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if not profiles:
        profiles = list(DEFAULT_SELECTOR_PROFILES)
    unknown = sorted(set(profiles) - SELECTOR_PROFILES)
    if unknown:
        raise ValueError(f"unknown selector profiles: {unknown}; supported={sorted(SELECTOR_PROFILES)}")
    return profiles


def policy_seed_blocks(
    *,
    seq_len: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    include_middle: bool = True,
) -> List[int]:
    num_blocks = math.ceil(seq_len / block_size)
    recent_start = num_blocks - recent_blocks
    blocks: List[int] = []
    blocks.extend(range(0, sink_blocks))
    blocks.extend(range(recent_start, num_blocks))
    if include_middle:
        if block_order == "sequential":
            blocks.extend(range(sink_blocks, sink_blocks + middle_seed_blocks))
        else:
            blocks.extend(range(recent_start - 1, recent_start - 1 - middle_seed_blocks, -1))
    seen = set()
    valid = []
    for block in blocks:
        if block < 0 or block >= num_blocks or block in seen:
            continue
        seen.add(block)
        valid.append(int(block))
    return valid


def seed_indices(
    *,
    seq_len: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
) -> torch.Tensor:
    indices: List[int] = []
    seen = set()
    for block in policy_seed_blocks(
        seq_len=seq_len,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        include_middle=True,
    ):
        start = block * block_size
        end = min(seq_len, start + block_size)
        for idx in range(start, end):
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return torch.tensor(sorted(indices), dtype=torch.long)


def seed_mask_from_blocks(
    *,
    seq_len: int,
    block_size: int,
    blocks: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
    for block in blocks:
        start = int(block) * block_size
        end = min(seq_len, start + block_size)
        if start < seq_len:
            mask[start:end] = True
    return mask


def block_scores_from_values(values: torch.Tensor, *, block_size: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    seq_len = int(values.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        scores[block] = float(values[start:end].float().sum().item())
    return scores


def block_max_scores(values: torch.Tensor, *, block_size: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    seq_len = int(values.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        scores[block] = float(values[start:end].float().max().item())
    return scores


def block_mean_proxy_scores(scores: torch.Tensor, *, block_size: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    seq_len = int(scores.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        out[block] = float(scores[start:end].float().mean().item())
    return out


def block_l2_bound_proxy_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    scores: torch.Tensor,
    *,
    block_size: int,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    seq_len = int(k.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    q_norm = torch.linalg.vector_norm(q.float())
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        k_block = k[start:end].float()
        mean = k_block.mean(dim=0)
        radius = torch.linalg.vector_norm(k_block - mean, dim=-1).max()
        out[block] = float(scores[start:end].float().mean().item() + q_norm.item() * radius.item())
    return out


def support_top_norm_scores(scores: torch.Tensor, k: torch.Tensor, *, block_size: int, top_p: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    seq_len = int(k.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    norms = torch.linalg.vector_norm(k.float(), dim=-1)
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        count = min(int(top_p), end - start)
        local = torch.topk(norms[start:end], k=count).indices + start
        out[block] = float(scores[local].float().max().item())
    return out


def support_extreme_mean_scores(q: torch.Tensor, k: torch.Tensor, *, block_size: int, top_p: int) -> Dict[int, float]:
    """Score blocks from a tiny farthest-from-mean support-key sketch.

    The selected keys are query-independent block summaries.  At runtime this
    corresponds to storing ``top_p`` extreme keys per block and evaluating only
    those keys against the decode query.
    """

    out: Dict[int, float] = {}
    seq_len = int(k.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    q_float = q.float()
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        k_block = k[start:end].float()
        mean = k_block.mean(dim=0, keepdim=True)
        distances = torch.linalg.vector_norm(k_block - mean, dim=-1)
        count = min(int(top_p), end - start)
        local = torch.topk(distances, k=count).indices
        out[block] = float(torch.matmul(k_block[local], q_float).max().item())
    return out


def _deterministic_directions(dim: int, count: int, *, device: torch.device) -> torch.Tensor:
    """Return stable pseudo-random directions without global RNG state."""

    rows = torch.arange(1, int(count) + 1, device=device, dtype=torch.float32)[:, None]
    cols = torch.arange(1, int(dim) + 1, device=device, dtype=torch.float32)[None, :]
    dirs = torch.sin(rows * cols * 12.9898) + torch.cos(rows * cols * 78.233)
    return torch.nn.functional.normalize(dirs, dim=-1)


def support_random_direction_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    directions: int,
) -> Dict[int, float]:
    """Score blocks from support keys picked by fixed random directions."""

    out: Dict[int, float] = {}
    seq_len = int(k.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    q_float = q.float()
    dirs = _deterministic_directions(int(k.shape[-1]), int(directions), device=k.device)
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        k_block = k[start:end].float()
        local = torch.matmul(k_block, dirs.T).argmax(dim=0)
        unique_local = torch.unique(local, sorted=True)
        out[block] = float(torch.matmul(k_block[unique_local], q_float).max().item())
    return out


def block_value_scores(probs: torch.Tensor, v: torch.Tensor, *, block_size: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    seq_len = int(probs.shape[0])
    num_blocks = math.ceil(seq_len / block_size)
    weighted_v = probs[:, None].float() * v.float()
    for block in range(num_blocks):
        start = block * block_size
        end = min(seq_len, start + block_size)
        if start >= end:
            continue
        scores[block] = float(torch.linalg.vector_norm(weighted_v[start:end].sum(dim=0)).item())
    return scores


def refined_block_scores(
    *,
    proxy_scores: Dict[int, float],
    exact_qk_scores: Dict[int, float],
    candidate_blocks: int,
) -> Dict[int, float]:
    candidates = {
        block
        for block, _score in sorted(
            proxy_scores.items(),
            key=lambda item: (-float(item[1]), int(item[0])),
        )[:candidate_blocks]
    }
    return {block: (exact_qk_scores[block] if block in candidates else -float("inf")) for block in proxy_scores}


def selector_cost_estimate(
    selector: str,
    *,
    seq_len: int,
    block_size: int,
) -> Dict[str, float]:
    """Estimate query-dot token work for selector comparisons."""

    num_blocks = math.ceil(seq_len / block_size)
    if selector == "fixed_policy":
        proxy_tokens = 0
        refine_tokens = 0
    elif selector == "qk_block_max":
        proxy_tokens = seq_len
        refine_tokens = 0
    elif selector in {"block_mean_proxy", "block_l2_bound_proxy"}:
        proxy_tokens = num_blocks
        refine_tokens = 0
    elif selector.startswith("support_top2") or selector.startswith("support_extreme2"):
        proxy_tokens = num_blocks * 2
        refine_tokens = (
            min(32, num_blocks) * block_size
            if selector.endswith("_refine32")
            else min(16, num_blocks) * block_size
            if selector.endswith("_refine16")
            else 0
        )
    elif selector.startswith("support_top4") or selector.startswith("support_extreme4"):
        proxy_tokens = num_blocks * 4
        refine_tokens = (
            min(32, num_blocks) * block_size
            if selector.endswith("_refine32")
            else min(16, num_blocks) * block_size
            if selector.endswith("_refine16")
            else 0
        )
    elif selector.startswith("support_rand4"):
        proxy_tokens = num_blocks * 4
        refine_tokens = min(32, num_blocks) * block_size if selector.endswith("_refine32") else 0
    elif selector.startswith("support_rand8"):
        proxy_tokens = num_blocks * 8
        refine_tokens = min(32, num_blocks) * block_size if selector.endswith("_refine32") else 0
    else:
        proxy_tokens = seq_len
        refine_tokens = 0
    estimated = proxy_tokens + refine_tokens
    return {
        "selector_proxy_dot_tokens": float(proxy_tokens),
        "selector_refine_dot_tokens": float(refine_tokens),
        "selector_estimated_dot_tokens": float(estimated),
        "selector_estimated_dot_token_ratio": 0.0 if seq_len <= 0 else float(estimated) / float(seq_len),
    }


def select_middle_blocks(
    scores: Dict[int, float],
    *,
    base_blocks: Sequence[int],
    fixed_middle_blocks: Sequence[int],
    middle_seed_blocks: int,
) -> List[int]:
    base = set(int(block) for block in base_blocks)
    selected: List[int] = []
    for block, score in sorted(scores.items(), key=lambda item: (-float(item[1]), int(item[0]))):
        if block in base or block in selected:
            continue
        selected.append(int(block))
        if len(selected) >= middle_seed_blocks:
            return selected
    for block in fixed_middle_blocks:
        if block in base or block in selected:
            continue
        selected.append(int(block))
        if len(selected) >= middle_seed_blocks:
            break
    return selected


def selector_seed_masks(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    scores: torch.Tensor,
    probs: torch.Tensor,
    v: torch.Tensor,
    support_mask: torch.Tensor,
    distractor_mask: torch.Tensor,
    policy: Any,
    selector_profiles: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    seq_len = int(probs.shape[0])
    block_size = int(policy.block_size)
    base_blocks = policy_seed_blocks(
        seq_len=seq_len,
        block_size=block_size,
        sink_blocks=int(policy.sink_blocks),
        recent_blocks=int(policy.recent_blocks),
        middle_seed_blocks=int(policy.middle_seed_blocks),
        block_order=str(policy.block_order),
        include_middle=False,
    )
    fixed_blocks = policy_seed_blocks(
        seq_len=seq_len,
        block_size=block_size,
        sink_blocks=int(policy.sink_blocks),
        recent_blocks=int(policy.recent_blocks),
        middle_seed_blocks=int(policy.middle_seed_blocks),
        block_order=str(policy.block_order),
        include_middle=True,
    )
    fixed_middle_blocks = [block for block in fixed_blocks if block not in set(base_blocks)]
    support_values = support_mask.float()
    distractor_values = distractor_mask.float()
    support_count_scores = block_scores_from_values(
        support_values - 0.25 * distractor_values,
        block_size=block_size,
    )
    qk_scores = block_max_scores(scores, block_size=block_size)
    mean_scores = block_mean_proxy_scores(scores, block_size=block_size)
    l2_bound_scores = block_l2_bound_proxy_scores(q, k, scores, block_size=block_size)
    top2_scores = support_top_norm_scores(scores, k, block_size=block_size, top_p=2)
    top4_scores = support_top_norm_scores(scores, k, block_size=block_size, top_p=4)
    extreme2_scores = support_extreme_mean_scores(q, k, block_size=block_size, top_p=2)
    extreme4_scores = support_extreme_mean_scores(q, k, block_size=block_size, top_p=4)
    rand4_scores = support_random_direction_scores(q, k, block_size=block_size, directions=4)
    rand8_scores = support_random_direction_scores(q, k, block_size=block_size, directions=8)
    exact_mass_scores = block_scores_from_values(probs, block_size=block_size)
    support_mass_scores = block_scores_from_values(
        probs * support_values - 0.25 * probs * distractor_values,
        block_size=block_size,
    )
    value_scores = block_value_scores(probs, v, block_size=block_size)
    score_by_selector = {
        "support_block_oracle": support_count_scores,
        "qk_block_max": qk_scores,
        "exact_mass_oracle": exact_mass_scores,
        "support_mass_oracle": support_mass_scores,
        "value_residual_oracle": value_scores,
        "block_mean_proxy": mean_scores,
        "block_l2_bound_proxy": l2_bound_scores,
        "support_top2_norm": top2_scores,
        "support_top4_norm": top4_scores,
        "support_top2_norm_refine16": refined_block_scores(
            proxy_scores=top2_scores,
            exact_qk_scores=qk_scores,
            candidate_blocks=16,
        ),
        "support_top2_norm_refine32": refined_block_scores(
            proxy_scores=top2_scores,
            exact_qk_scores=qk_scores,
            candidate_blocks=32,
        ),
        "support_top4_norm_refine16": refined_block_scores(
            proxy_scores=top4_scores,
            exact_qk_scores=qk_scores,
            candidate_blocks=16,
        ),
        "support_top4_norm_refine32": refined_block_scores(
            proxy_scores=top4_scores,
            exact_qk_scores=qk_scores,
            candidate_blocks=32,
        ),
        "support_extreme2_mean": extreme2_scores,
        "support_extreme4_mean": extreme4_scores,
        "support_extreme2_mean_refine32": refined_block_scores(
            proxy_scores=extreme2_scores,
            exact_qk_scores=qk_scores,
            candidate_blocks=32,
        ),
        "support_extreme4_mean_refine32": refined_block_scores(
            proxy_scores=extreme4_scores,
            exact_qk_scores=qk_scores,
            candidate_blocks=32,
        ),
        "support_rand4": rand4_scores,
        "support_rand8": rand8_scores,
        "support_rand4_refine32": refined_block_scores(
            proxy_scores=rand4_scores,
            exact_qk_scores=qk_scores,
            candidate_blocks=32,
        ),
        "support_rand8_refine32": refined_block_scores(
            proxy_scores=rand8_scores,
            exact_qk_scores=qk_scores,
            candidate_blocks=32,
        ),
    }
    qk_middle_blocks = select_middle_blocks(
        qk_scores,
        base_blocks=base_blocks,
        fixed_middle_blocks=fixed_middle_blocks,
        middle_seed_blocks=int(policy.middle_seed_blocks),
    )
    qk_blocks = list(base_blocks) + qk_middle_blocks
    out: Dict[str, Dict[str, Any]] = {}
    for selector in selector_profiles:
        if selector == "fixed_policy":
            blocks = fixed_blocks
        else:
            middle_blocks = select_middle_blocks(
                score_by_selector[selector],
                base_blocks=base_blocks,
                fixed_middle_blocks=fixed_middle_blocks,
                middle_seed_blocks=int(policy.middle_seed_blocks),
            )
            blocks = list(base_blocks) + middle_blocks
        mask = seed_mask_from_blocks(
            seq_len=seq_len,
            block_size=block_size,
            blocks=blocks,
            device=probs.device,
        )
        out[selector] = {
            "mask": mask,
            "selected_blocks": [int(block) for block in blocks],
            "middle_blocks": [int(block) for block in blocks if block not in set(base_blocks)],
            "overlap_with_fixed_blocks": len(set(blocks) & set(fixed_blocks)),
            "overlap_with_qk_block_max_blocks": len(set(blocks) & set(qk_blocks)),
            "qk_block_max_block_count": len(qk_blocks),
            "fixed_block_count": len(fixed_blocks),
            **selector_cost_estimate(selector, seq_len=seq_len, block_size=block_size),
        }
    return out
