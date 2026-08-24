import pytest
import torch

from stream_attention.planning import (
    ATTENTION_CACHE_CONTIGUOUS,
    ATTENTION_CACHE_PAGED,
    ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    ATTENTION_GUARANTEE_EXACT,
    ATTENTION_ROUTE_GRANULARITY_KV_GROUP,
    ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    ATTENTION_SCHEDULE_ALL,
    ATTENTION_SCHEDULE_SELECTED,
    AttentionProblem,
    AttentionRouteCSR,
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


def test_device_route_csr_validates_kv_group_rows_and_ragged_bounds():
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
    problem = AttentionProblem.from_paged(
        q,
        cache,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    rows = ((0, 2), (0,), (1, 2), (0, 1, 2))
    routes = AttentionRouteCSR.from_rows(
        rows,
        granularity=ATTENTION_ROUTE_GRANULARITY_KV_GROUP,
        atom_size=32,
        schedule_epoch=7,
    )
    routes.validate_for(problem)

    plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=32,
        tile_ids_per_row=rows,
        policy_id="kv-group-test",
        reason="per_group_selection",
        route_granularity=ATTENTION_ROUTE_GRANULARITY_KV_GROUP,
        schedule_epoch=7,
        device_routes=routes,
    )

    assert plan.scheduled_tile_counts == (2, 1, 2, 3)
    assert plan.schedule.device_routes is routes
    assert plan.as_dict()["schedule"]["device_routes"]["nnz"] == 8


def test_q_head_route_rows_change_coverage_denominator():
    q = torch.randn(1, 1, 4, 32)
    k = torch.randn(1, 64, 2, 32)
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
        tile_ids_per_row=((0,), (0, 1), (1,), (0, 1)),
        policy_id="head-private-test",
        reason="per_head_selection",
        route_granularity=ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    ).with_device_routes(device="cpu")

    assert plan.scheduled_tile_counts == (1, 2, 1, 2)
    assert plan.tile_coverage == 0.75
    assert plan.schedule.device_routes.row_count == 4


def test_device_route_csr_rejects_out_of_range_ragged_atom():
    q = torch.randn(1, 1, 4, 32)
    cache = PagedKVCache(
        key=torch.randn(4, 16, 2, 32),
        value=torch.randn(4, 16, 2, 32),
        page_table=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        sequence_lengths=torch.tensor([33], dtype=torch.int32),
        layout="NHD",
    )
    problem = AttentionProblem.from_paged(
        q,
        cache,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    routes = AttentionRouteCSR.from_rows(
        ((0, 1, 3),),
        granularity="batch",
        atom_size=16,
    )

    with pytest.raises(ValueError, match="outside the logical cache extent"):
        routes.validate_for(problem)


def test_selected_plan_rejects_host_and_device_route_disagreement():
    q = torch.randn(1, 1, 4, 32)
    k = torch.randn(1, 64, 2, 32)
    v = torch.randn_like(k)
    problem = AttentionProblem.from_contiguous(
        q,
        k,
        v,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    routes = AttentionRouteCSR.from_rows(
        ((1,),),
        granularity="batch",
        atom_size=32,
    )

    with pytest.raises(ValueError, match="does not match selected host rows"):
        AttentionTilePlan.selected(
            problem,
            logical_tile_size=32,
            tile_ids_per_row=((0,),),
            policy_id="mismatch-test",
            reason="invalid",
            device_routes=routes,
        )


def test_selected_device_plan_accepts_csr_only_schedule():
    q = torch.randn(1, 1, 4, 32)
    k = torch.randn(1, 64, 2, 32)
    v = torch.randn_like(k)
    problem = AttentionProblem.from_contiguous(
        q,
        k,
        v,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    routes = AttentionRouteCSR.from_rows(
        ((0,),),
        granularity="batch",
        atom_size=32,
        schedule_epoch=9,
    )

    plan = AttentionTilePlan.selected_device(
        problem,
        logical_tile_size=32,
        device_routes=routes,
        policy_id="device-only-test",
        reason="gpu_selector",
    )

    assert plan.schedule.selected_tile_ids is None
    assert plan.schedule.schedule_epoch == 9
    assert plan.scheduled_tile_counts == (1,)
