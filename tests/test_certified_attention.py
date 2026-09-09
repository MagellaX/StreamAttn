import math

import pytest
import torch
import torch.nn.functional as F

from stream_attention.certified import (
    StreamAttnMetadataCache,
    build_block_summaries,
    certified_attention,
)
from stream_attention.certified.bounds import block_score_upper_bound
from stream_attention.kernels.metadata_update_triton import (
    TRITON_AVAILABLE as METADATA_UPDATE_TRITON_AVAILABLE,
)


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


@pytest.mark.parametrize("gates", [(True, False), (False, True), (True, True)])
def test_budget_is_cumulative_across_many_small_omissions(gates):
    # Each tail block costs about .002 alone, but dropping all 127 costs .225.
    q = torch.zeros(1, 1, 1, 4)
    k = torch.zeros(1, 512, 1, 4)
    v = torch.zeros_like(k)
    q[..., 0] = 2.0
    k[:, 4:, :, 0] = math.log(1e-3)
    v[:, :4, :, 0] = -1.0
    v[:, 4:, :, 0] = 1.0
    result = certified_attention(
        q, k, v, causal=False, block_size=4, error_budget=0.01,
        enable_summary_gate=gates[0], enable_post_qk_gate=gates[1],
        return_stats=True,
    )
    ref = _sdpa_reference(q, k, v, causal=False)
    error = torch.linalg.vector_norm(result.output - ref, dim=-1)
    assert 0 < result.stats.skipped_row_blocks < 127
    assert result.stats.max_error_bound <= 0.01 + 1e-6
    assert torch.all(error <= result.stats.row_error_bound + 2e-6)


@pytest.mark.parametrize("groups", [1, 4, 8])
@pytest.mark.parametrize("causal", [False, True])
def test_compact_gqa_matches_expanded_reference(groups, causal):
    torch.manual_seed(91)
    q = torch.randn(2, 7, 2 * groups, 16)
    k = torch.randn(2, 19, 2, 16)
    v = torch.randn_like(k)
    result = certified_attention(q, k, v, causal=causal, error_budget=0,
                                 block_size=4, return_stats=True)
    ref = _sdpa_reference(q, k.repeat_interleave(groups, dim=2),
                          v.repeat_interleave(groups, dim=2), causal=causal)
    torch.testing.assert_close(result.output, ref, atol=2e-6, rtol=2e-5)
    assert result.stats.row_error_bound.shape == q.shape[:-1]
    assert result.stats.max_error_bound == 0.0


@pytest.mark.parametrize("order", ["sequential", "reverse", "summary_desc"])
@pytest.mark.parametrize("predicate", ["value_bound", "mass"])
def test_cumulative_bound_survives_rescaling_and_reordering(order, predicate):
    torch.manual_seed(92)
    q = torch.zeros(1, 3, 4, 4)
    q[..., 0] = 2.0
    k = torch.zeros(1, 65, 1, 4)
    k[..., 0] = -6.0
    k[:, :4, :, 0] = 0.0
    k[:, 32:36, :, 0] = 3.0
    v = torch.randn_like(k)
    result = certified_attention(q, k, v, causal=False, block_size=4,
                                 error_budget=0.02, skip_predicate=predicate,
                                 block_order=order, return_stats=True)
    ref = _sdpa_reference(q, k.repeat_interleave(4, dim=2),
                          v.repeat_interleave(4, dim=2), causal=False)
    error = torch.linalg.vector_norm(result.output - ref, dim=-1)
    assert result.stats.skipped_row_blocks > 0
    assert torch.all(error <= result.stats.row_error_bound + 2e-6)
    if predicate == "value_bound":
        assert result.stats.max_error_bound <= 0.02 + 1e-6


@pytest.mark.parametrize("budget", [-1.0, float("nan"), float("inf")])
def test_invalid_omission_budget_rejected(budget):
    x = torch.ones(1, 2, 1, 4)
    with pytest.raises(ValueError, match="finite and non-negative"):
        certified_attention(x, x, x, error_budget=budget)


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


def test_metadata_cache_clone_copies_tensors():
    v = torch.randn(1, 8, 2, 4, generator=torch.Generator().manual_seed(7))
    cache = StreamAttnMetadataCache.from_value(v, block_size=4)
    clone = cache.clone()

    assert clone is not cache
    assert clone.value_norm_bounds is not cache.value_norm_bounds
    torch.testing.assert_close(clone.value_norm_bounds, cache.value_norm_bounds)
    clone.value_norm_bounds.zero_()
    assert torch.any(cache.value_norm_bounds != 0)


def test_metadata_cache_triton_incremental_update_matches_reference():
    if not (torch.cuda.is_available() and METADATA_UPDATE_TRITON_AVAILABLE):
        pytest.skip("CUDA + Triton required for metadata update kernel")

    torch.manual_seed(8)
    v = torch.randn(2, 17, 3, 32, device="cuda", dtype=torch.float16)
    new_v = torch.randn(2, 5, 3, 32, device="cuda", dtype=torch.float16)
    ref = StreamAttnMetadataCache.from_value(v, block_size=4, use_triton=True)
    got = ref.clone()
    ref.update_value_bounds_(new_v, start_pos=3, use_triton=False)
    got.update_value_bounds_(new_v, start_pos=3, use_triton=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(got.value_norm_bounds, ref.value_norm_bounds)
