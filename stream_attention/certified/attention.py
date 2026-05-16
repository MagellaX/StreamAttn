"""Single-GPU certified attention prototype."""

from dataclasses import dataclass
import math
from typing import Literal, Optional, Union

import torch

from .bounds import block_score_upper_bound
from .reorder import BlockOrder, resolve_block_order
from .summaries import BlockSummaries, build_block_summaries


@dataclass(frozen=True)
class CertifiedAttentionStats:
    """Telemetry from certified attention."""

    skipped_row_blocks: int
    skipped_pre_k_row_blocks: int
    skipped_post_qk_row_blocks: int
    computed_row_blocks: int
    masked_row_blocks: int
    total_row_blocks: int
    skip_fraction: float
    max_error_bound: float
    mean_error_bound: float
    row_error_bound: torch.Tensor
    lse: torch.Tensor


@dataclass(frozen=True)
class CertifiedAttentionOutput:
    """Output wrapper returned when ``return_stats=True``."""

    output: torch.Tensor
    stats: CertifiedAttentionStats


def _validate_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must have shape [batch, seq, heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key must have the same batch size")
    if query.shape[2] != key.shape[2]:
        raise ValueError("certified_attention currently requires Q/K/V to have the same head count")
    if query.shape[3] != key.shape[3]:
        raise ValueError("query and key must have the same head dimension")


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.where(den > 0, num / den, torch.zeros_like(num))


def _current_output_norm(acc_num: torch.Tensor, acc_den: torch.Tensor) -> torch.Tensor:
    out = _safe_div(acc_num, acc_den[..., None])
    return torch.linalg.vector_norm(out, dim=-1)


def _skip_from_den_bound(
    den_bound: torch.Tensor,
    value_bound: torch.Tensor,
    acc_num: torch.Tensor,
    acc_den: torch.Tensor,
    error_budget: float,
    predicate: str,
) -> torch.Tensor:
    if predicate == "mass":
        return den_bound <= error_budget * acc_den
    if predicate == "value_bound":
        rho = _safe_div(den_bound, acc_den + den_bound)
        out_norm = _current_output_norm(acc_num, acc_den)
        return rho * (value_bound + out_norm) <= error_budget
    raise ValueError(f"unknown skip predicate: {predicate}")


def certified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    error_budget: float = 1e-3,
    block_size: int = 64,
    summaries: Optional[BlockSummaries] = None,
    num_summary_outliers: int = 0,
    skip_predicate: Literal["mass", "value_bound"] = "value_bound",
    block_order: BlockOrder = "sequential",
    enable_summary_gate: bool = True,
    enable_post_qk_gate: bool = True,
    summary_threshold: float = 0.0,
    post_qk_threshold: float = 0.0,
    return_stats: bool = False,
) -> Union[torch.Tensor, CertifiedAttentionOutput]:
    """Compute attention with certified K/V block skipping.

    This is a PyTorch reference path for validating the certified-attention
    math. It streams K/V blocks, maintains online-softmax state, and skips a
    row/block only when the selected predicate certifies the skipped block is
    below ``error_budget``.

    Gate 0 uses K/V summaries before full K/V loading. Gate 1 computes QK,
    then applies a BLASST-style local-max test before loading V/PV work.

    ``error_budget=0`` disables skipping and returns exact tiled attention up to
    normal floating-point differences.
    """

    _validate_inputs(query, key, value)
    if error_budget < 0:
        raise ValueError("error_budget must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    batch, seq_q, heads, dim = query.shape
    seq_k = key.shape[1]
    scale = 1.0 / math.sqrt(dim)
    out_dtype = query.dtype
    device = query.device

    if summaries is None:
        summaries = build_block_summaries(
            key,
            value,
            block_size=block_size,
            num_outliers=num_summary_outliers,
        )
    elif summaries.block_size != block_size:
        raise ValueError("summaries.block_size must match block_size")
    elif summaries.seq_len != seq_k:
        raise ValueError("summaries.seq_len must match key sequence length")

    q = query.permute(0, 2, 1, 3).contiguous().float()
    k = key.permute(0, 2, 1, 3).contiguous().float()
    v = value.permute(0, 2, 1, 3).contiguous().float()

    running_max = torch.full((batch, heads, seq_q), -float("inf"), device=device)
    acc_den = torch.zeros(batch, heads, seq_q, device=device)
    acc_num = torch.zeros(batch, heads, seq_q, dim, device=device)
    skipped_den_bound = torch.zeros(batch, heads, seq_q, device=device)

    query_pos = torch.arange(seq_q, device=device)
    skipped_row_blocks = 0
    skipped_pre_k_row_blocks = 0
    skipped_post_qk_row_blocks = 0
    computed_row_blocks = 0
    masked_row_blocks = 0
    order = resolve_block_order(block_order, q, summaries, scale=scale)

    for block_idx in order:
        start = block_idx * block_size
        end = min(start + block_size, seq_k)
        block_len = int(summaries.block_lengths[block_idx].item())
        key_pos = torch.arange(start, end, device=device)

        if causal:
            valid_row = query_pos >= start
            full_block_allowed = query_pos >= (end - 1)
        else:
            valid_row = torch.ones(seq_q, device=device, dtype=torch.bool)
            full_block_allowed = valid_row

        valid_row_bh = valid_row.view(1, 1, seq_q).expand(batch, heads, seq_q)
        full_allowed_bh = full_block_allowed.view(1, 1, seq_q).expand(
            batch, heads, seq_q
        )

        block_value_bound = summaries.max_value_norm[:, :, block_idx][:, :, None]
        upper = block_score_upper_bound(q, summaries, block_idx, scale=scale)
        has_state = torch.isfinite(running_max) & (acc_den > 0)
        if enable_summary_gate:
            can_skip = (
                valid_row_bh
                & full_allowed_bh
                & has_state
                & (upper <= running_max - summary_threshold)
                & (error_budget > 0)
            )
        else:
            can_skip = torch.zeros_like(valid_row_bh)

        den_bound = torch.zeros_like(acc_den)
        den_bound[can_skip] = block_len * torch.exp(
            upper[can_skip] - running_max[can_skip]
        )
        skip_pre = can_skip & _skip_from_den_bound(
            den_bound,
            block_value_bound,
            acc_num,
            acc_den,
            error_budget,
            skip_predicate,
        )

        skipped_den_bound = skipped_den_bound + torch.where(
            skip_pre, den_bound, torch.zeros_like(den_bound)
        )

        compute_row = valid_row_bh & ~skip_pre
        masked_row_blocks += int((~valid_row_bh).sum().item())
        skipped_pre_k_row_blocks += int(skip_pre.sum().item())

        if not bool(compute_row.any()):
            continue

        k_tile = k[:, :, start:end, :]
        scores = torch.einsum("bhsd,bhkd->bhsk", q, k_tile) * scale

        if causal:
            causal_mask = key_pos.view(1, 1, 1, block_len) <= query_pos.view(
                1, 1, seq_q, 1
            )
            scores = scores.masked_fill(~causal_mask, -float("inf"))

        scores = scores.masked_fill(~compute_row[..., None], -float("inf"))

        tile_max = scores.amax(dim=-1)
        if enable_post_qk_gate:
            can_skip_post = (
                compute_row
                & has_state
                & (tile_max <= running_max - post_qk_threshold)
                & (error_budget > 0)
            )
        else:
            can_skip_post = torch.zeros_like(compute_row)
        post_den_bound = torch.zeros_like(acc_den)
        post_den_bound[can_skip_post] = block_len * torch.exp(
            tile_max[can_skip_post] - running_max[can_skip_post]
        )
        skip_post = can_skip_post & _skip_from_den_bound(
            post_den_bound,
            block_value_bound,
            acc_num,
            acc_den,
            error_budget,
            skip_predicate,
        )
        skipped_den_bound = skipped_den_bound + torch.where(
            skip_post,
            post_den_bound,
            torch.zeros_like(post_den_bound),
        )
        skipped_post_qk_row_blocks += int(skip_post.sum().item())

        compute_row = compute_row & ~skip_post
        computed_row_blocks += int(compute_row.sum().item())
        if not bool(compute_row.any()):
            continue

        scores = scores.masked_fill(~compute_row[..., None], -float("inf"))
        v_tile = v[:, :, start:end, :]
        tile_max = scores.amax(dim=-1)
        tile_valid = torch.isfinite(tile_max)
        prev_valid = torch.isfinite(running_max)
        new_valid = prev_valid | tile_valid
        new_max = torch.maximum(running_max, tile_max)
        safe_new_max = torch.where(new_valid, new_max, torch.zeros_like(new_max))

        correction = torch.where(
            prev_valid,
            torch.exp(running_max - safe_new_max),
            torch.zeros_like(acc_den),
        )

        exp_scores = torch.exp(scores - safe_new_max[..., None])
        exp_scores = torch.where(torch.isfinite(scores), exp_scores, torch.zeros_like(exp_scores))

        acc_num = acc_num * correction[..., None] + torch.einsum(
            "bhsk,bhkd->bhsd", exp_scores, v_tile
        )
        acc_den = acc_den * correction + exp_scores.sum(dim=-1)
        skipped_den_bound = skipped_den_bound * correction
        running_max = torch.where(new_valid, new_max, running_max)

    output = _safe_div(acc_num, acc_den[..., None])
    lse = torch.where(
        acc_den > 0,
        running_max + torch.log(acc_den),
        torch.full_like(acc_den, -float("inf")),
    )

    max_value_norm = summaries.max_value_norm.amax(dim=-1)
    max_value_norm = max_value_norm[:, :, None].expand(batch, heads, seq_q)
    skipped_mass_fraction = _safe_div(skipped_den_bound, acc_den + skipped_den_bound)
    row_error_bound_bh = 2.0 * max_value_norm * skipped_mass_fraction

    output_bshd = output.permute(0, 2, 1, 3).contiguous().to(out_dtype)
    row_error_bound = row_error_bound_bh.permute(0, 2, 1).contiguous()
    lse_bsh = lse.permute(0, 2, 1).contiguous()

    if not return_stats:
        return output_bshd

    skipped_row_blocks = skipped_pre_k_row_blocks + skipped_post_qk_row_blocks
    total_row_blocks = batch * heads * seq_q * summaries.num_blocks
    active_row_blocks = skipped_row_blocks + computed_row_blocks
    skip_fraction = (
        skipped_row_blocks / active_row_blocks if active_row_blocks > 0 else 0.0
    )
    stats = CertifiedAttentionStats(
        skipped_row_blocks=skipped_row_blocks,
        skipped_pre_k_row_blocks=skipped_pre_k_row_blocks,
        skipped_post_qk_row_blocks=skipped_post_qk_row_blocks,
        computed_row_blocks=computed_row_blocks,
        masked_row_blocks=masked_row_blocks,
        total_row_blocks=total_row_blocks,
        skip_fraction=float(skip_fraction),
        max_error_bound=float(row_error_bound.max().item()) if row_error_bound.numel() else 0.0,
        mean_error_bound=float(row_error_bound.mean().item()) if row_error_bound.numel() else 0.0,
        row_error_bound=row_error_bound,
        lse=lse_bsh,
    )
    return CertifiedAttentionOutput(output=output_bshd, stats=stats)
