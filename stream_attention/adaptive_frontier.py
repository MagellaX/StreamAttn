"""Reference math for StreamAttn's adaptive output-sufficiency frontier.

These routines are intentionally implementation-agnostic.  They answer whether
an adaptive attention family can preserve the attention module output before we
spend time lowering that family into CUDA/Triton.  The runtime kernels should
only consume a method after this reference path establishes a semantic upper
bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BlockAttentionStates:
    """Normalized per-block softmax states.

    ``log_partition`` has shape ``[rows, q_heads, blocks]`` and ``output`` has
    shape ``[rows, q_heads, blocks, dim]``.  A state represents the unnormalized
    pair ``(Z_b, N_b)`` as ``(exp(log_partition_b), Z_b * output_b)``.
    """

    log_partition: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class MergedAttentionState:
    output: torch.Tensor
    log_partition: torch.Tensor
    valid: torch.Tensor
    scaled_denominator: torch.Tensor


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int, int]:
    if q.dim() != 3:
        raise ValueError("q must have shape [rows, q_heads, dim]")
    if k.dim() != 3 or v.shape != k.shape:
        raise ValueError("k and v must have matching shape [kv_heads, tokens, dim]")
    rows, q_heads, dim = map(int, q.shape)
    kv_heads, tokens, kv_dim = map(int, k.shape)
    if kv_dim != dim:
        raise ValueError("Q and K/V head dimensions must match")
    if q_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    return rows, q_heads, kv_heads, tokens, dim


def _valid_lengths_tensor(
    valid_lengths: Optional[torch.Tensor], *, rows: int, tokens: int, device: torch.device
) -> torch.Tensor:
    if valid_lengths is None:
        return torch.full((rows,), tokens, device=device, dtype=torch.long)
    lengths = valid_lengths.to(device=device, dtype=torch.long)
    if lengths.shape != (rows,):
        raise ValueError("valid_lengths must have shape [rows]")
    if bool(((lengths < 0) | (lengths > tokens)).any()):
        raise ValueError("valid_lengths must lie in [0, tokens]")
    return lengths


def _blocked_kv(
    k: torch.Tensor, v: torch.Tensor, *, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    tokens = int(k.shape[1])
    blocks = math.ceil(tokens / block_size)
    padded_tokens = blocks * block_size
    if padded_tokens != tokens:
        pad_shape = (int(k.shape[0]), padded_tokens - tokens, int(k.shape[2]))
        k = torch.cat([k, torch.zeros(pad_shape, device=k.device, dtype=k.dtype)], dim=1)
        v = torch.cat([v, torch.zeros(pad_shape, device=v.device, dtype=v.dtype)], dim=1)
    positions = torch.arange(padded_tokens, device=k.device).reshape(blocks, block_size)
    return (
        k.reshape(int(k.shape[0]), blocks, block_size, int(k.shape[2])),
        v.reshape(int(v.shape[0]), blocks, block_size, int(v.shape[2])),
        positions,
    )


def exact_block_attention_states(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    valid_lengths: Optional[torch.Tensor] = None,
) -> BlockAttentionStates:
    """Compute exact per-block online-softmax states for a reference capture."""

    rows, q_heads, kv_heads, tokens, dim = _validate_qkv(q, k, v)
    lengths = _valid_lengths_tensor(
        valid_lengths, rows=rows, tokens=tokens, device=q.device
    )
    k_blocks, v_blocks, positions = _blocked_kv(k, v, block_size=block_size)
    group_size = q_heads // kv_heads
    k_q = k_blocks.repeat_interleave(group_size, dim=0).float()
    v_q = v_blocks.repeat_interleave(group_size, dim=0).float()
    scores = torch.einsum("rhd,hbsd->rhbs", q.float(), k_q) / math.sqrt(dim)
    valid = positions[None, None, :, :] < lengths[:, None, None, None]
    scores = scores.masked_fill(~valid, -torch.inf)

    block_max = scores.amax(dim=-1)
    finite = torch.isfinite(block_max)
    shifted = scores - torch.where(finite, block_max, torch.zeros_like(block_max)).unsqueeze(-1)
    weights = torch.exp(shifted).masked_fill(~valid, 0.0)
    denominator = weights.sum(dim=-1)
    numerator = torch.einsum("rhbs,hbsd->rhbd", weights, v_q)
    output = numerator / denominator.clamp_min(1.0e-30).unsqueeze(-1)
    output = torch.where(finite.unsqueeze(-1), output, torch.zeros_like(output))
    log_partition = torch.where(
        finite,
        block_max + denominator.clamp_min(1.0e-30).log(),
        torch.full_like(block_max, -torch.inf),
    )
    return BlockAttentionStates(log_partition=log_partition, output=output)


def diagonal_gaussian_block_states(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    valid_lengths: Optional[torch.Tensor] = None,
) -> BlockAttentionStates:
    """Estimate block ``(Z, N)`` from diagonal joint K/V moments.

    For ``q_bar = q / sqrt(D)`` and an approximately joint-Gaussian block,

    ``Z ~= n exp(q_bar mu_k + .5 q_bar^2 var(k))``

    ``N / Z ~= mu_v + diag(cov(v, k)) q_bar``.

    Exact selected blocks can replace these approximate states through
    :func:`control_variate_attention`.
    """

    rows, q_heads, kv_heads, tokens, dim = _validate_qkv(q, k, v)
    lengths = _valid_lengths_tensor(
        valid_lengths, rows=rows, tokens=tokens, device=q.device
    )
    k_blocks, v_blocks, positions = _blocked_kv(k, v, block_size=block_size)
    group_size = q_heads // kv_heads
    q_bar = q.float() / math.sqrt(dim)

    log_partitions = []
    outputs = []
    for row in range(rows):
        valid = positions < lengths[row]
        valid_f = valid.float()
        count = valid_f.sum(dim=-1)
        safe_count = count.clamp_min(1.0)
        weight = valid_f[None, :, :, None]
        mu_k = (k_blocks.float() * weight).sum(dim=2) / safe_count[None, :, None]
        mu_v = (v_blocks.float() * weight).sum(dim=2) / safe_count[None, :, None]
        centered_k = (k_blocks.float() - mu_k[:, :, None, :]) * weight
        centered_v = (v_blocks.float() - mu_v[:, :, None, :]) * weight
        var_k = centered_k.square().sum(dim=2) / safe_count[None, :, None]
        cov_vk_diag = (centered_v * centered_k).sum(dim=2) / safe_count[None, :, None]

        mu_k = mu_k.repeat_interleave(group_size, dim=0)
        mu_v = mu_v.repeat_interleave(group_size, dim=0)
        var_k = var_k.repeat_interleave(group_size, dim=0)
        cov_vk_diag = cov_vk_diag.repeat_interleave(group_size, dim=0)
        qr = q_bar[row]
        mean_score = torch.einsum("hd,hbd->hb", qr, mu_k)
        score_variance = torch.einsum("hd,hbd->hb", qr.square(), var_k)
        log_z = count.clamp_min(1.0).log()[None, :] + mean_score + 0.5 * score_variance
        valid_blocks = count > 0
        log_z = torch.where(
            valid_blocks[None, :], log_z, torch.full_like(log_z, -torch.inf)
        )
        conditional_v = mu_v + cov_vk_diag * qr[:, None, :]
        conditional_v = torch.where(
            valid_blocks[None, :, None], conditional_v, torch.zeros_like(conditional_v)
        )
        log_partitions.append(log_z)
        outputs.append(conditional_v)

    return BlockAttentionStates(
        log_partition=torch.stack(log_partitions, dim=0),
        output=torch.stack(outputs, dim=0),
    )


def _validate_states(states: BlockAttentionStates) -> tuple[int, int, int, int]:
    if states.log_partition.dim() != 3 or states.output.dim() != 4:
        raise ValueError("block states must have shapes [R,H,B] and [R,H,B,D]")
    if states.output.shape[:-1] != states.log_partition.shape:
        raise ValueError("block state prefixes must match")
    return tuple(map(int, states.output.shape))


def merge_block_attention_states(
    states: BlockAttentionStates, selected: Optional[torch.Tensor] = None
) -> MergedAttentionState:
    """Merge block states using a stable generalized online-softmax reduction."""

    rows, heads, blocks, _dim = _validate_states(states)
    if selected is None:
        selected = torch.ones(
            (rows, heads, blocks), device=states.output.device, dtype=torch.bool
        )
    if selected.shape != states.log_partition.shape:
        raise ValueError("selected must have shape [rows, q_heads, blocks]")
    active_log_z = states.log_partition.masked_fill(~selected, -torch.inf)
    reference = active_log_z.amax(dim=-1)
    finite = torch.isfinite(reference)
    reference_safe = torch.where(finite, reference, torch.zeros_like(reference))
    weight = torch.exp(active_log_z - reference_safe.unsqueeze(-1))
    weight = torch.where(torch.isfinite(active_log_z), weight, torch.zeros_like(weight))
    denominator = weight.sum(dim=-1)
    numerator = torch.einsum("rhb,rhbd->rhd", weight, states.output.float())
    output = numerator / denominator.clamp_min(1.0e-30).unsqueeze(-1)
    output = torch.where(finite.unsqueeze(-1), output, torch.zeros_like(output))
    log_partition = torch.where(
        finite,
        reference_safe + denominator.clamp_min(1.0e-30).log(),
        torch.full_like(reference, -torch.inf),
    )
    return MergedAttentionState(output, log_partition, finite, denominator)


def control_variate_attention(
    exact: BlockAttentionStates,
    approximate: BlockAttentionStates,
    *,
    selected: torch.Tensor,
    sampled: Optional[torch.Tensor] = None,
    inclusion_probability: Optional[torch.Tensor] = None,
    denominator_floor: float = 1.0e-12,
) -> MergedAttentionState:
    """Merge approximate-all states with exact deterministic/sample corrections.

    ``sampled`` must exclude deterministic ``selected`` blocks.  With known
    positive inclusion probabilities, the correction is a Horvitz-Thompson
    control variate for the unnormalized numerator and denominator.  The final
    ratio is not claimed to be unbiased.
    """

    shape = _validate_states(exact)
    if _validate_states(approximate) != shape:
        raise ValueError("exact and approximate block-state shapes must match")
    if selected.shape != exact.log_partition.shape:
        raise ValueError("selected must have shape [rows, q_heads, blocks]")
    if sampled is None:
        sampled = torch.zeros_like(selected)
    if sampled.shape != selected.shape or bool((sampled & selected).any()):
        raise ValueError("sampled must match selected and exclude deterministic blocks")
    if inclusion_probability is None:
        inclusion_probability = torch.ones_like(exact.log_partition)
    if inclusion_probability.shape != exact.log_partition.shape:
        raise ValueError("inclusion_probability must match block log partitions")
    if bool((sampled & (inclusion_probability <= 0)).any()):
        raise ValueError("sampled blocks require positive inclusion probability")

    corrected = selected | sampled
    reference_logs = torch.stack(
        [
            approximate.log_partition,
            exact.log_partition.masked_fill(~corrected, -torch.inf),
        ],
        dim=0,
    )
    reference = reference_logs.amax(dim=(0, 3))
    finite_reference = torch.isfinite(reference)
    reference_safe = torch.where(
        finite_reference, reference, torch.zeros_like(reference)
    )

    def scaled(log_z: torch.Tensor) -> torch.Tensor:
        value = torch.exp(log_z - reference_safe.unsqueeze(-1))
        return torch.where(torch.isfinite(log_z), value, torch.zeros_like(value))

    approx_z = scaled(approximate.log_partition)
    exact_z = scaled(exact.log_partition)
    denominator = approx_z.sum(dim=-1)
    numerator = torch.einsum("rhb,rhbd->rhd", approx_z, approximate.output.float())

    deterministic_factor = selected.float()
    sample_factor = sampled.float() / inclusion_probability.clamp_min(1.0e-30)
    factor = deterministic_factor + sample_factor
    denominator = denominator + (factor * (exact_z - approx_z)).sum(dim=-1)
    numerator = numerator + torch.einsum(
        "rhb,rhbd->rhd",
        factor,
        exact_z.unsqueeze(-1) * exact.output.float()
        - approx_z.unsqueeze(-1) * approximate.output.float(),
    )
    valid = finite_reference & torch.isfinite(denominator) & (denominator > denominator_floor)
    output = numerator / denominator.clamp_min(denominator_floor).unsqueeze(-1)
    output = torch.where(valid.unsqueeze(-1), output, torch.zeros_like(output))
    log_partition = torch.where(
        valid,
        reference_safe + denominator.clamp_min(denominator_floor).log(),
        torch.full_like(reference, -torch.inf),
    )
    return MergedAttentionState(output, log_partition, valid, denominator)


def gqa_topk_mask(
    block_scores: torch.Tensor, *, kv_heads: int, blocks_per_group: int
) -> torch.Tensor:
    """Select one physically shared block route per true-GQA KV group."""

    if block_scores.dim() != 3:
        raise ValueError("block_scores must have shape [rows, q_heads, blocks]")
    rows, q_heads, blocks = map(int, block_scores.shape)
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if blocks_per_group < 0 or blocks_per_group > blocks:
        raise ValueError("blocks_per_group must lie in [0, blocks]")
    group_size = q_heads // kv_heads
    grouped = block_scores.reshape(rows, kv_heads, group_size, blocks).amax(dim=2)
    indices = torch.topk(grouped, k=blocks_per_group, dim=-1).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask.repeat_interleave(group_size, dim=1)


def post_wo_gqa_greedy_mask(
    exact: BlockAttentionStates,
    *,
    full_output: torch.Tensor,
    o_proj_weight: torch.Tensor,
    kv_heads: int,
    blocks_per_group: int,
    base_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Greedily minimize post-``W_O`` error with physical GQA-shared routes.

    This is an offline semantic oracle.  Every selected block is shared by all
    Q heads belonging to the same KV head, so its budget corresponds to
    physical KV-page reads rather than independent per-head selections.
    """

    rows, q_heads, blocks, dim = _validate_states(exact)
    if full_output.shape != (rows, q_heads, dim):
        raise ValueError("full_output must have shape [rows, q_heads, dim]")
    if o_proj_weight.shape[1] != q_heads * dim:
        raise ValueError("o_proj_weight input width must equal q_heads * dim")
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if base_mask is None:
        selected = torch.zeros(
            (rows, q_heads, blocks), device=full_output.device, dtype=torch.bool
        )
    else:
        if base_mask.shape != exact.log_partition.shape:
            raise ValueError("base_mask must match block-state log partitions")
        selected = base_mask.clone()
    group_size = q_heads // kv_heads
    target_projected = F.linear(full_output.flatten(1).float(), o_proj_weight.float())

    for row in range(rows):
        row_mask = selected[row : row + 1]
        row_states = BlockAttentionStates(
            exact.log_partition[row : row + 1], exact.output[row : row + 1]
        )
        current = merge_block_attention_states(row_states, row_mask)
        current_output = current.output[0]
        current_log_z = current.log_partition[0]
        current_projected = F.linear(
            current_output.flatten().float(), o_proj_weight.float()
        )
        for _round in range(blocks_per_group):
            for group in range(kv_heads):
                heads = slice(group * group_size, (group + 1) * group_size)
                already = row_mask[0, heads].any(dim=0)
                candidates = torch.nonzero(~already, as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                old_log_z = current_log_z[heads]
                block_log_z = exact.log_partition[row, heads].index_select(1, candidates)
                block_output = exact.output[row, heads].index_select(1, candidates)
                common = torch.maximum(old_log_z[:, None], block_log_z)
                old_weight = torch.exp(old_log_z[:, None] - common)
                block_weight = torch.exp(block_log_z - common)
                denominator = old_weight + block_weight
                candidate_group_output = (
                    old_weight[..., None] * current_output[heads, None, :]
                    + block_weight[..., None] * block_output
                ) / denominator.clamp_min(1.0e-30)[..., None]
                # [candidate, group_head, dim], matching the contiguous W_O slice.
                candidate_group_output = candidate_group_output.permute(1, 0, 2)
                delta = candidate_group_output - current_output[heads][None, :, :]
                start = group * group_size * dim
                end = (group + 1) * group_size * dim
                projected_delta = F.linear(
                    delta.flatten(1), o_proj_weight[:, start:end].float()
                )
                errors = (
                    current_projected[None, :]
                    + projected_delta
                    - target_projected[row : row + 1]
                ).square().sum(dim=-1)
                errors = torch.where(
                    torch.isfinite(block_log_z).all(dim=0),
                    errors,
                    torch.full_like(errors, torch.inf),
                )
                best = int(torch.argmin(errors).item())
                block = int(candidates[best].item())
                row_mask[:, heads, block] = True
                current_output[heads] = candidate_group_output[best]
                current_log_z[heads] = torch.logaddexp(
                    old_log_z, exact.log_partition[row, heads, block]
                )
                current_projected = current_projected + projected_delta[best]
        selected[row : row + 1] = row_mask
    return selected


def poisson_tail_sample(
    priority: torch.Tensor,
    *,
    selected: torch.Tensor,
    expected_samples: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independently sample tail blocks with a fixed expected sample budget."""

    if priority.shape != selected.shape or priority.dim() != 3:
        raise ValueError("priority and selected must match [rows, groups, blocks]")
    if expected_samples < 0:
        raise ValueError("expected_samples must be non-negative")
    eligible = ~selected
    weight = priority.float().clamp_min(0.0)
    weight = torch.where(eligible, weight, torch.zeros_like(weight))
    probability = torch.zeros_like(weight)
    for index in torch.cartesian_prod(
        torch.arange(weight.shape[0]), torch.arange(weight.shape[1])
    ):
        row, group = map(int, index.tolist())
        active = eligible[row, group]
        count = int(active.sum().item())
        if count == 0 or expected_samples == 0:
            continue
        if expected_samples >= count:
            probability[row, group, active] = 1.0
            continue
        local = weight[row, group, active]
        if float(local.sum().item()) <= 0.0:
            local = torch.ones_like(local)
        local = local.clamp_min(local.mean() * 1.0e-6)
        lo = 0.0
        hi = float(expected_samples / local.min().item())
        for _ in range(50):
            scale = 0.5 * (lo + hi)
            total = torch.clamp(local * scale, max=1.0).sum().item()
            if total < expected_samples:
                lo = scale
            else:
                hi = scale
        probability[row, group, active] = torch.clamp(local * hi, max=1.0)
    draw = torch.rand(
        probability.shape,
        device=probability.device,
        generator=generator,
        dtype=probability.dtype,
    )
    sampled = eligible & (draw < probability)
    return sampled, probability
