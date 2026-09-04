import pytest
import torch

from stream_attention.backends.sm90.micro_prefill import (
    MicroPrefillPlan,
    NaturalMicroPrefillPlan,
    choose_micro_prefill_splits,
    choose_natural_micro_prefill_splits,
    micro_prefill_shape_reasons,
    natural_micro_prefill_query_tiles,
    supports_sm90_micro_prefill,
)
from stream_attention.backends.sm90.transposed_gqa_exact_sources import (
    CPP_SOURCE,
    CUDA_SOURCE,
)


@pytest.mark.parametrize(
    ("query_len", "expected"),
    [(2, 64), (4, 32), (8, 16), (16, 8), (32, 4), (64, 2)],
)
def test_split_rule_tracks_flattened_query_parallelism(query_len, expected):
    assert choose_micro_prefill_splits(
        batch=1,
        query_len=query_len,
        kv_heads=2,
        kv_len=32768,
    ) == expected


def test_split_rule_rejects_shapes_outside_micro_prefill_regime():
    with pytest.raises(ValueError, match="query_len"):
        choose_micro_prefill_splits(
            batch=1, query_len=1, kv_heads=2, kv_len=32768
        )


@pytest.mark.parametrize(
    ("query_len", "group_size", "query_tiles"),
    [(2, 4, 1), (16, 4, 1), (17, 4, 2), (8, 8, 1), (64, 8, 8)],
)
def test_natural_family_packs_query_positions_and_gqa_heads(
    query_len, group_size, query_tiles
):
    assert natural_micro_prefill_query_tiles(
        query_len=query_len, group_size=group_size
    ) == query_tiles


def test_natural_split_rule_restores_parallelism_after_kv_reuse():
    assert choose_natural_micro_prefill_splits(
        batch=1,
        query_len=16,
        kv_heads=4,
        group_size=4,
        kv_len=32768,
    ) == 64
    assert choose_natural_micro_prefill_splits(
        batch=1,
        query_len=64,
        kv_heads=2,
        group_size=8,
        kv_len=32768,
    ) == 16
    with pytest.raises(ValueError, match="divisible by 64"):
        choose_micro_prefill_splits(
            batch=1, query_len=8, kv_heads=2, kv_len=32767
        )


def test_shape_contract_spans_m2_to_m64_without_cell_registration():
    for query_len in (2, 4, 8, 16, 32, 64):
        q = torch.empty(
            1, query_len, 16, 128, dtype=torch.bfloat16, device="meta"
        )
        kv = torch.empty(
            1, 2, 32768, 128, dtype=torch.bfloat16, device="meta"
        )
        assert micro_prefill_shape_reasons(q, kv, kv) == []
        assert not supports_sm90_micro_prefill(q, kv, kv)


def test_shape_contract_reports_semantic_boundaries():
    q = torch.empty(1, 1, 12, 96, dtype=torch.float16, device="meta")
    kv = torch.empty(1, 2, 4097, 96, dtype=torch.float16, device="meta")
    reasons = micro_prefill_shape_reasons(q, kv, kv)
    assert set(reasons) == {"query_len", "gqa", "head_dim", "kv_len", "dtype"}


def test_source_exposes_exact_flattened_query_to_kv_mapping():
    assert 'm.def("micro_prefill_out"' in CPP_SOURCE
    assert "query_positions_per_batch" in CUDA_SOURCE
    assert "const int cache_group" in CUDA_SOURCE
    assert "group / (query_positions_per_batch * kv_heads)" in CUDA_SOURCE
    assert "streamattn_transposed_wgmma_exact_merge_warp_kernel" in CUDA_SOURCE
    assert 'm.def("natural_micro_prefill_out"' in CPP_SOURCE
    assert "streamattn_natural_wgmma_micro_prefill_partial_kernel" in CUDA_SOURCE
    assert "query_positions_per_tile" in CUDA_SOURCE


def test_plan_run_uses_one_combined_extension_dispatch():
    calls: list[tuple[object, ...]] = []

    def launch(*args) -> None:
        calls.append(args)

    tensor = torch.empty(1)
    plan = MicroPrefillPlan(
        query=tensor,
        key_cache=tensor,
        value_cache=tensor,
        output=tensor,
        query_group=tensor,
        output_group=tensor,
        partial_output=tensor,
        partial_lse=tensor,
        num_splits=7,
        extension=None,
        launch=launch,
    )

    assert plan.run() is tensor
    assert len(calls) == 1
    assert calls[0][-1] == 7

    calls.clear()
    natural_plan = NaturalMicroPrefillPlan(
        query=tensor,
        key_cache=tensor,
        value_cache=tensor,
        output=tensor,
        partial_output=tensor,
        partial_lse=tensor,
        num_splits=5,
        query_tiles=2,
        extension=None,
        launch=launch,
    )
    assert natural_plan.run() is tensor
    assert len(calls) == 1
    assert calls[0][-1] == 5
