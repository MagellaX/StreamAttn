import math

import torch
import torch.nn.functional as F

from stream_attention.certified import (
    StreamAttnMetadataCache,
    build_block_summaries,
    certified_attention,
)
from stream_attention.certified.bounds import block_score_upper_bound


def _sdpa_reference(q, k, v, *, causal):
    q_bh = q.permute(0, 2, 1, 3).contiguous()
    k_bh = k.permute(0, 2, 1, 3).contiguous()
    v_bh = v.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(
        q_bh,
        k_bh,
        v_bh,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def test_block_score_upper_bound_contains_all_scores():
    torch.manual_seed(0)
    q = torch.randn(2, 7, 3, 8)
    k = torch.randn(2, 11, 3, 8)
    summaries = build_block_summaries(k, block_size=4)

    q_bh = q.permute(0, 2, 1, 3).contiguous().float()
    k_bh = k.permute(0, 2, 1, 3).contiguous().float()
    scale = 1.0 / math.sqrt(q.shape[-1])

    for block_idx in range(summaries.num_blocks):
        start = block_idx * summaries.block_size
        end = min(start + summaries.block_size, k.shape[1])
        upper = block_score_upper_bound(q_bh, summaries, block_idx, scale=scale)
        scores = torch.einsum("bhsd,bhkd->bhsk", q_bh, k_bh[:, :, start:end, :]) * scale
        assert torch.all(scores <= upper[..., None] + 1e-5)


def test_outlier_split_bound_contains_all_scores():
    torch.manual_seed(4)
    q = torch.randn(1, 5, 2, 8)
    k = torch.randn(1, 9, 2, 8)
    k[:, 3, :, 0] = 20.0
    summaries = build_block_summaries(k, block_size=5, num_outliers=1)

    q_bh = q.permute(0, 2, 1, 3).contiguous().float()
    k_bh = k.permute(0, 2, 1, 3).contiguous().float()
    scale = 1.0 / math.sqrt(q.shape[-1])

    for block_idx in range(summaries.num_blocks):
        start = block_idx * summaries.block_size
        end = min(start + summaries.block_size, k.shape[1])
        upper = block_score_upper_bound(q_bh, summaries, block_idx, scale=scale)
        scores = torch.einsum("bhsd,bhkd->bhsk", q_bh, k_bh[:, :, start:end, :]) * scale
        assert torch.all(scores <= upper[..., None] + 1e-5)


def test_certified_attention_exact_when_budget_zero_noncausal():
    torch.manual_seed(1)
    q = torch.randn(2, 9, 2, 16)
    k = torch.randn(2, 13, 2, 16)
    v = torch.randn(2, 13, 2, 16)

    out = certified_attention(
        q,
        k,
        v,
        causal=False,
        error_budget=0.0,
        block_size=4,
        block_order="reverse",
    )
    ref = _sdpa_reference(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)


def test_certified_attention_exact_when_budget_zero_causal():
    torch.manual_seed(2)
    q = torch.randn(1, 12, 2, 16)
    k = torch.randn(1, 12, 2, 16)
    v = torch.randn(1, 12, 2, 16)

    out = certified_attention(
        q,
        k,
        v,
        causal=True,
        error_budget=0.0,
        block_size=4,
    )
    ref = _sdpa_reference(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)


def test_certified_attention_skips_low_mass_blocks_with_bound():
    q = torch.zeros(1, 4, 1, 4)
    k = torch.zeros(1, 8, 1, 4)
    v = torch.randn(1, 8, 1, 4, generator=torch.Generator().manual_seed(3))

    q[:, :, :, 0] = 8.0
    k[:, :4, :, 0] = 8.0
    k[:, 4:, :, 0] = -8.0

    result = certified_attention(
        q,
        k,
        v,
        causal=False,
        error_budget=1e-3,
        block_size=4,
        return_stats=True,
    )
    ref = _sdpa_reference(q, k, v, causal=False)
    err = torch.linalg.vector_norm(result.output - ref, dim=-1)

    assert result.stats.skipped_row_blocks > 0
    assert result.stats.skipped_pre_k_row_blocks > 0
    assert result.stats.skip_fraction > 0.0
    assert torch.all(err <= result.stats.row_error_bound + 1e-5)
    assert err.max().item() < 1e-4


def test_certified_attention_post_qk_gate_skips_when_summary_gate_disabled():
    q = torch.zeros(1, 4, 1, 4)
    k = torch.zeros(1, 8, 1, 4)
    v = torch.randn(1, 8, 1, 4, generator=torch.Generator().manual_seed(5))

    q[:, :, :, 0] = 8.0
    k[:, :4, :, 0] = 8.0
    k[:, 4:, :, 0] = -8.0

    result = certified_attention(
        q,
        k,
        v,
        causal=False,
        error_budget=1e-3,
        block_size=4,
        enable_summary_gate=False,
        enable_post_qk_gate=True,
        return_stats=True,
    )
    ref = _sdpa_reference(q, k, v, causal=False)
    err = torch.linalg.vector_norm(result.output - ref, dim=-1)

    assert result.stats.skipped_pre_k_row_blocks == 0
    assert result.stats.skipped_post_qk_row_blocks > 0
    assert torch.all(err <= result.stats.row_error_bound + 1e-5)


def test_metadata_cache_builds_value_bounds():
    torch.manual_seed(6)
    v = torch.randn(2, 7, 3, 5)
    cache = StreamAttnMetadataCache.from_value(v, block_size=4)

    assert cache.block_size == 4
    assert cache.seq_len == 7
    assert cache.num_blocks == 2
    assert cache.require_value_norm_bounds().shape == (2, 3, 2)
    cache.validate_for_value(v)

    v_bh = v.permute(0, 2, 1, 3).contiguous()
    first_block_norm = torch.linalg.vector_norm(v_bh[:, :, :4, :], dim=-1).amax(dim=-1)
    torch.testing.assert_close(cache.value_norm_bounds[:, :, 0], first_block_norm)


def test_metadata_cache_incrementally_updates_value_bounds():
    v = torch.zeros(1, 8, 2, 4)
    cache = StreamAttnMetadataCache.from_value(v, block_size=4)
    new_v = torch.zeros(1, 3, 2, 4)
    new_v[:, 0, :, 0] = 3.0
    new_v[:, 1, :, 1] = 4.0
    new_v[:, 2, :, 2] = 5.0

    cache.update_value_bounds_(new_v, start_pos=3)

    assert cache.value_norm_bounds[0, 0, 0].item() == 3.0
    assert cache.value_norm_bounds[0, 0, 1].item() == 5.0
    assert cache.value_norm_bounds[0, 1, 0].item() == 3.0
    assert cache.value_norm_bounds[0, 1, 1].item() == 5.0
