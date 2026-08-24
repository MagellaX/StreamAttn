import pytest
import torch

from stream_attention.paged import PagedKVCache
from stream_attention.planning import (
    ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    AttentionProblem,
    AttentionTilePlan,
)
from stream_attention.selected_routes import (
    PACKED_ROUTE_FLAG_ALL_HEADS,
    PACKED_ROUTE_FLAG_STRUCTURALLY_FULL,
    SELECTED_SCHEDULER_STATIC_UNIFORM,
    prepare_paged_routes64,
)


def _paged_problem(
    *,
    batch: int,
    q_heads: int,
    kv_heads: int,
    lengths: list[int],
) -> tuple[torch.Tensor, PagedKVCache, AttentionProblem]:
    dim = 64
    max_pages = max((length + 15) // 16 for length in lengths)
    num_pages = batch * max_pages
    query = torch.randn(batch, 1, q_heads, dim)
    page_table = torch.arange(num_pages, dtype=torch.int32).reshape(batch, max_pages)
    cache = PagedKVCache(
        key=torch.randn(num_pages, 16, kv_heads, dim),
        value=torch.randn(num_pages, 16, kv_heads, dim),
        page_table=page_table,
        sequence_lengths=torch.tensor(lengths, dtype=torch.int32),
        layout="NHD",
    )
    problem = AttentionProblem.from_paged(
        query,
        cache,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    return query, cache, problem


def test_prepare_paged_routes64_resolves_pages_without_copying_kv():
    _query, cache, problem = _paged_problem(
        batch=2,
        q_heads=8,
        kv_heads=2,
        lengths=[65, 96],
    )
    plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=32,
        tile_ids_per_row=((0, 2), (1, 2)),
        policy_id="selected-paged-test",
        reason="fixed_selected_tiles",
        schedule_epoch=3,
    )
    key_ptr = cache.key.data_ptr()
    value_ptr = cache.value.data_ptr()

    packed = prepare_paged_routes64(plan, cache)

    assert packed.row_ptr.tolist() == [0, 1, 2, 3, 4]
    assert packed.logical_atom_origins.tolist() == [
        [0, 16, 64, -1],
        [0, 16, 64, -1],
        [32, 48, 64, 80],
        [32, 48, 64, 80],
    ]
    assert packed.physical_page_ids.tolist() == [
        [0, 1, 4, -1],
        [0, 1, 4, -1],
        [8, 9, 10, 11],
        [8, 9, 10, 11],
    ]
    assert packed.active_head_masks.tolist() == [
        [15, 15, 15, 0],
        [15, 15, 15, 0],
        [15, 15, 15, 15],
        [15, 15, 15, 15],
    ]
    assert packed.token_valid_masks[0].tolist() == [0xFFFF, 0xFFFF, 0x1, 0]
    assert packed.group_route_efficiency == 1.0
    assert packed.scheduler_hint == SELECTED_SCHEDULER_STATIC_UNIFORM
    assert cache.key.data_ptr() == key_ptr
    assert cache.value.data_ptr() == value_ptr
    packed.validate_current(cache, schedule_epoch=3)


def test_prepare_paged_routes64_keeps_per_atom_q_head_masks():
    _query, cache, problem = _paged_problem(
        batch=1,
        q_heads=8,
        kv_heads=2,
        lengths=[128],
    )
    rows = (
        (0, 1),
        (0,),
        (0,),
        (0,),
        (2,),
        (2,),
        (2,),
        (2,),
    )
    plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=16,
        tile_ids_per_row=rows,
        policy_id="head-private-selected-test",
        reason="q_head_dynamic_selection",
        route_granularity=ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    ).with_device_routes(device="cpu")

    packed = prepare_paged_routes64(plan, cache)

    assert packed.row_ptr.tolist() == [0, 1, 2]
    assert packed.logical_atom_origins.tolist() == [
        [0, 16, -1, -1],
        [32, -1, -1, -1],
    ]
    assert packed.active_head_masks.tolist() == [
        [0b1111, 0b0001, 0, 0],
        [0b1111, 0, 0, 0],
    ]
    assert packed.group_route_efficiency == pytest.approx(0.75)
    assert packed.route_flags[0].item() & PACKED_ROUTE_FLAG_ALL_HEADS == 0
    assert packed.route_flags[0].item() & PACKED_ROUTE_FLAG_STRUCTURALLY_FULL == 0


def test_prepared_routes_detect_live_page_table_and_length_mutations():
    _query, cache, problem = _paged_problem(
        batch=1,
        q_heads=8,
        kv_heads=2,
        lengths=[64],
    )
    plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=16,
        tile_ids_per_row=((0, 1, 2, 3),),
        policy_id="stale-route-test",
        reason="fixed_selected_tiles",
        schedule_epoch=4,
    )
    packed = prepare_paged_routes64(plan, cache)

    with pytest.raises(RuntimeError, match="schedule epoch"):
        packed.validate_current(cache, schedule_epoch=5)

    cache.page_table[0, 0], cache.page_table[0, 1] = (
        cache.page_table[0, 1].clone(),
        cache.page_table[0, 0].clone(),
    )
    with pytest.raises(RuntimeError, match="page-table contents"):
        packed.validate_current(cache, schedule_epoch=4)

    refreshed = prepare_paged_routes64(plan, cache)
    cache.sequence_lengths[0] = 63
    with pytest.raises(RuntimeError, match="sequence lengths"):
        refreshed.validate_current(cache, schedule_epoch=4)


def test_prepare_rejects_selected_atom_with_invalid_physical_page():
    _query, cache, problem = _paged_problem(
        batch=1,
        q_heads=8,
        kv_heads=2,
        lengths=[64],
    )
    plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=16,
        tile_ids_per_row=((0, 1),),
        policy_id="invalid-page-test",
        reason="fixed_selected_tiles",
    )
    cache.page_table[0, 1] = -1

    with pytest.raises(ValueError, match="invalid physical page"):
        prepare_paged_routes64(plan, cache)
