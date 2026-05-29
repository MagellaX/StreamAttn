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


@dataclass(frozen=True)
class SeedBlockSelection:
    selector: str
    selected_blocks: List[int]
    middle_blocks: List[int]
    base_blocks: List[int]
    fixed_blocks: List[int]
    cost: Dict[str, float]


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


def seed_token_indices_from_blocks(
    *,
    seq_len: int,
    block_size: int,
    blocks: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    indices: List[int] = []
    seen = set()
    for block in blocks:
        start = int(block) * block_size
        end = min(int(seq_len), start + block_size)
        if start >= int(seq_len):
            continue
        for idx in range(start, end):
            if idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
    if not indices:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.tensor(sorted(indices), device=device, dtype=torch.long)


def _blocked_k_view(k: torch.Tensor, *, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(k.shape[0])
    dim = int(k.shape[-1])
    num_blocks = math.ceil(seq_len / block_size)
    padded_len = num_blocks * block_size
    if padded_len == seq_len:
        padded = k
    else:
        pad = torch.zeros((padded_len - seq_len, dim), device=k.device, dtype=k.dtype)
        padded = torch.cat([k, pad], dim=0)
    blocks = padded.reshape(num_blocks, block_size, dim)
    positions = torch.arange(padded_len, device=k.device).reshape(num_blocks, block_size)
    valid = positions < seq_len
    return blocks, valid


def _block_qk_scores_tensor(q: torch.Tensor, k: torch.Tensor, *, block_size: int) -> torch.Tensor:
    blocks, valid = _blocked_k_view(k, block_size=block_size)
    dots = torch.matmul(blocks.float(), q.float())
    dots = dots.masked_fill(~valid, -float("inf"))
    return dots.max(dim=1).values


def _gather_block_keys(blocks: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    dim = int(blocks.shape[-1])
    return blocks.gather(1, indices[..., None].expand(-1, -1, dim))


def _support_top_norm_scores_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    top_p: int,
) -> torch.Tensor:
    blocks, valid = _blocked_k_view(k, block_size=block_size)
    norms = torch.linalg.vector_norm(blocks.float(), dim=-1).masked_fill(~valid, -float("inf"))
    count = min(int(top_p), int(block_size))
    idx = torch.topk(norms, k=count, dim=1).indices
    keys = _gather_block_keys(blocks.float(), idx)
    return torch.matmul(keys, q.float()).max(dim=1).values


def _support_extreme_mean_scores_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    top_p: int,
) -> torch.Tensor:
    blocks, valid = _blocked_k_view(k, block_size=block_size)
    blocks_f = blocks.float()
    valid_f = valid.float()
    counts = valid_f.sum(dim=1).clamp_min(1.0)
    mean = (blocks_f * valid_f[..., None]).sum(dim=1, keepdim=True) / counts[:, None, None]
    distances = torch.linalg.vector_norm(blocks_f - mean, dim=-1).masked_fill(~valid, -float("inf"))
    count = min(int(top_p), int(block_size))
    idx = torch.topk(distances, k=count, dim=1).indices
    keys = _gather_block_keys(blocks_f, idx)
    return torch.matmul(keys, q.float()).max(dim=1).values


def _support_random_scores_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    directions: int,
) -> torch.Tensor:
    blocks, valid = _blocked_k_view(k, block_size=block_size)
    blocks_f = blocks.float()
    dirs = _deterministic_directions(int(k.shape[-1]), int(directions), device=k.device)
    projected = torch.matmul(blocks_f, dirs.T).masked_fill(~valid[..., None], -float("inf"))
    idx = projected.argmax(dim=1)
    keys = _gather_block_keys(blocks_f, idx)
    return torch.matmul(keys, q.float()).max(dim=1).values


def _mean_scores_tensor(q: torch.Tensor, k: torch.Tensor, *, block_size: int) -> torch.Tensor:
    blocks, valid = _blocked_k_view(k, block_size=block_size)
    blocks_f = blocks.float()
    valid_f = valid.float()
    counts = valid_f.sum(dim=1).clamp_min(1.0)
    mean = (blocks_f * valid_f[..., None]).sum(dim=1) / counts[:, None]
    return torch.matmul(mean, q.float())


def _l2_bound_scores_tensor(q: torch.Tensor, k: torch.Tensor, *, block_size: int) -> torch.Tensor:
    blocks, valid = _blocked_k_view(k, block_size=block_size)
    blocks_f = blocks.float()
    valid_f = valid.float()
    counts = valid_f.sum(dim=1).clamp_min(1.0)
    mean = (blocks_f * valid_f[..., None]).sum(dim=1, keepdim=True) / counts[:, None, None]
    centered = torch.linalg.vector_norm(blocks_f - mean, dim=-1).masked_fill(~valid, 0.0)
    radius = centered.max(dim=1).values
    return torch.matmul(mean.squeeze(1), q.float()) + torch.linalg.vector_norm(q.float()) * radius


def _score_tensor_for_runtime_selector(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    selector: str,
) -> torch.Tensor:
    if selector == "qk_block_max":
        return _block_qk_scores_tensor(q, k, block_size=block_size)
    if selector == "block_mean_proxy":
        return _mean_scores_tensor(q, k, block_size=block_size)
    if selector == "block_l2_bound_proxy":
        return _l2_bound_scores_tensor(q, k, block_size=block_size)
    if selector.startswith("support_top2_norm"):
        scores = _support_top_norm_scores_tensor(q, k, block_size=block_size, top_p=2)
    elif selector.startswith("support_top4_norm"):
        scores = _support_top_norm_scores_tensor(q, k, block_size=block_size, top_p=4)
    elif selector.startswith("support_extreme2_mean"):
        scores = _support_extreme_mean_scores_tensor(q, k, block_size=block_size, top_p=2)
    elif selector.startswith("support_extreme4_mean"):
        scores = _support_extreme_mean_scores_tensor(q, k, block_size=block_size, top_p=4)
    elif selector.startswith("support_rand4"):
        scores = _support_random_scores_tensor(q, k, block_size=block_size, directions=4)
    elif selector.startswith("support_rand8"):
        scores = _support_random_scores_tensor(q, k, block_size=block_size, directions=8)
    else:
        raise ValueError(f"selector {selector!r} cannot run from q/k summaries only")
    if selector.endswith("_refine16") or selector.endswith("_refine32"):
        candidate_blocks = 32 if selector.endswith("_refine32") else 16
        count = min(int(candidate_blocks), int(scores.numel()))
        candidates = torch.topk(scores, k=count).indices
        exact = torch.full_like(scores, -float("inf"))
        blocks, valid = _blocked_k_view(k, block_size=block_size)
        candidate_blocks_view = blocks.index_select(0, candidates).float()
        candidate_valid = valid.index_select(0, candidates)
        candidate_scores = torch.matmul(candidate_blocks_view, q.float()).masked_fill(
            ~candidate_valid,
            -float("inf"),
        )
        exact[candidates] = candidate_scores.max(dim=1).values
        scores = exact
    return scores


def select_seed_blocks_by_profile(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    policy: Any,
    selector: str,
) -> SeedBlockSelection:
    """Select seed blocks for one row/head from a runtime-feasible q/k proxy.

    The implementation is intentionally reference-oriented: it computes block
    summaries from the provided cache tensor so correctness can be tested before
    a production selected-block kernel exists.  The selected block interface is
    the same one a future native selector should feed.
    """

    if selector not in SELECTOR_PROFILES:
        raise ValueError(f"unknown selector profile: {selector}")
    seq_len = int(k.shape[0])
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
    if selector == "fixed_policy":
        middle_blocks = fixed_middle_blocks
        selected_blocks = fixed_blocks
    else:
        unsupported_oracles = {
            "support_block_oracle",
            "exact_mass_oracle",
            "support_mass_oracle",
            "value_residual_oracle",
        }
        if selector in unsupported_oracles:
            raise ValueError(f"selector {selector!r} requires oracle labels/probabilities")
        scores = _score_tensor_for_runtime_selector(q, k, block_size=block_size, selector=selector)
        base = set(int(block) for block in base_blocks)
        ranked = sorted(
            ((int(block), float(score)) for block, score in enumerate(scores.detach().cpu().tolist())),
            key=lambda item: (-item[1], item[0]),
        )
        middle_blocks = []
        for block, score in ranked:
            if block in base or block in middle_blocks or not math.isfinite(score):
                continue
            middle_blocks.append(block)
            if len(middle_blocks) >= int(policy.middle_seed_blocks):
                break
        if len(middle_blocks) < int(policy.middle_seed_blocks):
            for block in fixed_middle_blocks:
                if block in base or block in middle_blocks:
                    continue
                middle_blocks.append(int(block))
                if len(middle_blocks) >= int(policy.middle_seed_blocks):
                    break
        selected_blocks = list(base_blocks) + middle_blocks
    return SeedBlockSelection(
        selector=selector,
        selected_blocks=[int(block) for block in selected_blocks],
        middle_blocks=[int(block) for block in middle_blocks],
        base_blocks=[int(block) for block in base_blocks],
        fixed_blocks=[int(block) for block in fixed_blocks],
        cost=selector_cost_estimate(selector, seq_len=seq_len, block_size=block_size),
    )


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
