"""Synthetic residual K/V helpers for reduced-work attention research.

The production seed path already computes a normalized seed output and the
seed log-partition.  A compact residual bank can be merged with that state
without revisiting seed tokens.  This module keeps that merge numerically
stable and independent of any particular compactor architecture.
"""

from __future__ import annotations

import math

import torch


def merge_normalized_attention_states(
    left_log_partition: torch.Tensor,
    left_output: torch.Tensor,
    right_log_partition: torch.Tensor,
    right_output: torch.Tensor,
) -> torch.Tensor:
    """Merge two disjoint normalized attention states with online-softmax math."""

    if left_log_partition.shape != right_log_partition.shape:
        raise ValueError("attention-state log partitions must have matching shapes")
    if left_output.shape != right_output.shape:
        raise ValueError("attention-state outputs must have matching shapes")
    if left_output.shape[:-1] != left_log_partition.shape:
        raise ValueError("attention-state output prefix must match log-partition shape")
    common_max = torch.maximum(left_log_partition.float(), right_log_partition.float())
    left_weight = torch.exp(left_log_partition.float() - common_max)
    right_weight = torch.exp(right_log_partition.float() - common_max)
    numerator = left_weight.unsqueeze(-1) * left_output.float()
    numerator = numerator + right_weight.unsqueeze(-1) * right_output.float()
    denominator = left_weight + right_weight
    return numerator / denominator.clamp_min(1.0e-30).unsqueeze(-1)


def _expand_residual_bank(
    bank: torch.Tensor,
    *,
    query_steps: int,
    query_heads: int,
) -> torch.Tensor:
    if bank.dim() == 3:
        kv_heads, residual_tokens, dim = bank.shape
        if query_heads % kv_heads != 0:
            raise ValueError("query head count must be divisible by residual KV heads")
        expanded = bank.repeat_interleave(query_heads // kv_heads, dim=0)
        return expanded.unsqueeze(0).expand(query_steps, query_heads, residual_tokens, dim)
    if bank.dim() == 4:
        if bank.shape[0] != query_steps or bank.shape[1] != query_heads:
            raise ValueError("query-specific residual bank must have shape [steps, q_heads, R, D]")
        return bank
    raise ValueError("residual bank must have shape [kv_heads, R, D] or [steps, q_heads, R, D]")


def merge_seed_with_residual(
    q: torch.Tensor,
    seed_log_partition: torch.Tensor,
    seed_output: torch.Tensor,
    residual_k: torch.Tensor,
    residual_v: torch.Tensor,
) -> torch.Tensor:
    """Merge a normalized seed state with shared or query-specific residual K/V.

    Args:
        q: Post-RoPE queries with shape ``[steps, q_heads, dim]``.
        seed_log_partition: ``log(sum(exp(seed_scores)))`` with shape
            ``[steps, q_heads]``.
        seed_output: Normalized seed attention output, shape
            ``[steps, q_heads, dim]``.
        residual_k/residual_v: Either a true-GQA shared bank with shape
            ``[kv_heads, residual_tokens, dim]`` or a query-specific oracle
            bank with shape ``[steps, q_heads, residual_tokens, dim]``.
    """

    if q.dim() != 3:
        raise ValueError("q must have shape [steps, q_heads, dim]")
    if seed_log_partition.shape != q.shape[:2]:
        raise ValueError("seed_log_partition shape must match q[:2]")
    if seed_output.shape != q.shape:
        raise ValueError("seed_output shape must match q")
    if residual_k.shape != residual_v.shape:
        raise ValueError("residual K/V shapes must match")
    if residual_k.shape[-1] != q.shape[-1]:
        raise ValueError("residual head dim must match q")

    steps, heads, dim = q.shape
    k = _expand_residual_bank(residual_k, query_steps=steps, query_heads=heads)
    v = _expand_residual_bank(residual_v, query_steps=steps, query_heads=heads)
    residual_scores = torch.einsum("thd,thrd->thr", q.float(), k.float()) / math.sqrt(dim)
    residual_log_partition = torch.logsumexp(residual_scores, dim=-1)
    residual_output = torch.einsum(
        "thr,thrd->thd", torch.softmax(residual_scores, dim=-1), v.float()
    )
    return merge_normalized_attention_states(
        seed_log_partition,
        seed_output,
        residual_log_partition,
        residual_output,
    )


def construct_query_exact_residual(
    q: torch.Tensor,
    omitted_log_partition: torch.Tensor,
    omitted_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the one-token-per-query residual representation.

    This is a representation-capacity oracle, not a deployable selector.  It
    proves that one synthetic token can exactly encode an omitted softmax state
    for one query.  A useful compactor must instead produce a bank shared by
    many future queries, which is what the held-out benchmark tests.
    """

    if q.dim() != 3 or omitted_output.shape != q.shape:
        raise ValueError("q and omitted_output must have shape [steps, q_heads, dim]")
    if omitted_log_partition.shape != q.shape[:2]:
        raise ValueError("omitted_log_partition shape must match q[:2]")
    dim = q.shape[-1]
    q32 = q.float()
    norm_sq = (q32 * q32).sum(dim=-1).clamp_min(1.0e-20)
    scale = math.sqrt(dim) * omitted_log_partition.float() / norm_sq
    residual_k = (scale.unsqueeze(-1) * q32).unsqueeze(-2)
    residual_v = omitted_output.float().unsqueeze(-2)
    return residual_k, residual_v
