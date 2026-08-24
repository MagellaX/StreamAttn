import pytest
import torch

from stream_attention.planning import (
    ATTENTION_CACHE_CONTIGUOUS,
    ATTENTION_CACHE_PAGED,
    ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    ATTENTION_GUARANTEE_EXACT,
    ATTENTION_SCHEDULE_ALL,
    ATTENTION_SCHEDULE_SELECTED,
    AttentionProblem,
    AttentionTilePlan,
    fixed_block_tile_ids,
)
from stream_attention.paged import PagedKVCache


def test_exact_contiguous_nhd_problem_and_all_tile_plan():
    q = torch.randn(2, 1, 8, 64)
    k = torch.randn(2, 130, 2, 64)
    v = torch.randn_like(k)

    problem = AttentionProblem.from_contiguous(
        q,
        k,
        v,
        guarantee=ATTENTION_GUARANTEE_EXACT,
    )
    plan = AttentionTilePlan.exact(problem, logical_tile_size=64)

    assert problem.cache_kind == ATTENTION_CACHE_CONTIGUOUS
    assert problem.cache_layout == "NHD"
    assert problem.group_size == 4
    assert problem.kv_lengths == (130, 130)
    assert plan.schedule.kind == ATTENTION_SCHEDULE_ALL
    assert plan.source.logical_tile_counts == (3, 3)
    assert plan.scheduled_tile_counts == (3, 3)
    assert plan.tile_coverage == 1.0


def test_contiguous_hnd_layout_is_inferred_for_native_exact_shape():
    q = torch.randn(1, 1, 16, 128)
    k = torch.randn(1, 2, 32768, 128)
    v = torch.randn_like(k)

    problem = AttentionProblem.from_contiguous(
        q,
        k,
        v,
        guarantee=ATTENTION_GUARANTEE_EXACT,
    )

    assert problem.cache_layout == "HND"
    assert problem.kv_heads == 2
    assert problem.kv_lengths == (32768,)


def test_distribution_verified_selected_plan_tracks_actual_tile_coverage():
    q = torch.randn(2, 1, 8, 64)
    k = torch.randn(2, 128, 2, 64)
    v = torch.randn_like(k)
    problem = AttentionProblem.from_contiguous(
        q,
        k,
        v,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )

    plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=32,
        tile_ids_per_row=((0, 1, 3), (0, 1, 3)),
        policy_id="test-policy",
        reason="calibrated_fixed_blocks",
    )

    assert plan.schedule.kind == ATTENTION_SCHEDULE_SELECTED
    assert plan.scheduled_tile_counts == (3, 3)
    assert plan.tile_coverage == 0.75
    assert plan.policy_id == "test-policy"


def test_fixed_block_builder_preserves_kernel_order_and_deduplicates():
    assert fixed_block_tile_ids(
        kv_len=128,
        tile_size=32,
        sink_tiles=1,
        recent_tiles=1,
        middle_tiles=1,
        tile_order="recent_first",
    ) == (0, 3, 2)
    assert fixed_block_tile_ids(
        kv_len=64,
        tile_size=32,
        sink_tiles=1,
        recent_tiles=1,
        middle_tiles=1,
        tile_order="sequential",
    ) == (0, 1)


def test_exact_guarantee_rejects_selected_tile_schedule():
    q = torch.randn(1, 1, 4, 32)
    k = torch.randn(1, 64, 2, 32)
    v = torch.randn_like(k)
    problem = AttentionProblem.from_contiguous(
        q,
        k,
        v,
        guarantee=ATTENTION_GUARANTEE_EXACT,
    )

    with pytest.raises(ValueError, match="distribution-verified"):
        AttentionTilePlan.selected(
            problem,
            logical_tile_size=32,
            tile_ids_per_row=((0,),),
            policy_id=None,
            reason="invalid",
        )


def test_paged_ragged_problem_describes_fragmented_logical_tiles():
    q = torch.randn(2, 1, 8, 64)
    cache = PagedKVCache(
        key=torch.randn(12, 16, 2, 64),
        value=torch.randn(12, 16, 2, 64),
        page_table=torch.tensor(
            [[0, 1, 2, 3, 4, -1], [5, 6, 7, 8, 9, 10]],
            dtype=torch.int32,
        ),
        sequence_lengths=torch.tensor([65, 96], dtype=torch.int32),
        layout="NHD",
    )
    cache.validate(q)

    problem = AttentionProblem.from_paged(
        q,
        cache,
        guarantee=ATTENTION_GUARANTEE_EXACT,
    )
    plan = AttentionTilePlan.exact(problem, logical_tile_size=64)

    assert problem.cache_kind == ATTENTION_CACHE_PAGED
    assert problem.is_ragged is True
    assert problem.kv_lengths == (65, 96)
    assert plan.source.page_size == 16
    assert plan.source.fragments_per_tile == 4
    assert plan.source.logical_tile_counts == (2, 2)
