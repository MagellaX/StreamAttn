import pytest

from benchmarks.estimate_splitk_inline_projection import _split_middle_counts, estimate_row


def _row():
    return {
        "prompt_type": "code",
        "kv_len": 16384,
        "dense_all_ms": 0.12,
        "inline_sparse_group_ms": 0.16,
        "sparse_heads": [2, 3],
        "sink_blocks": 2,
        "recent_blocks": 2,
        "middle_seed_blocks": 8,
        "stats": {
            "middle_blocks": 100,
            "projection_skipped_blocks": 70,
            "projection_computed_blocks": 30,
            "gate1_post_qk_skipped_blocks": 10,
            "pv_executed_blocks": 12,
            "seed_computed_blocks": 24,
            "projection_skip_fraction": 0.7,
        },
    }


def test_split_middle_counts_remove_seed_from_chunk_qk():
    counts = _split_middle_counts(_row())

    assert counts["head_count"] == 2
    assert counts["non_middle_seed_blocks"] == 8
    assert counts["middle_seed_blocks"] == 16
    assert counts["chunk_middle_blocks"] == 84
    assert counts["chunk_qk_blocks"] == 14
    assert counts["chunk_pv_blocks"] == 4
    assert counts["chunk_projection_only_blocks"] == 70


def test_splitk_estimate_improves_with_more_uniform_chunks():
    row2 = estimate_row(
        _row(),
        chunks=2,
        projection_weight=1.0,
        qk_weight=8.0,
        pv_weight=4.0,
        merge_base_ms=0.0,
        merge_per_head_chunk_ms=0.0,
    )
    row4 = estimate_row(
        _row(),
        chunks=4,
        projection_weight=1.0,
        qk_weight=8.0,
        pv_weight=4.0,
        merge_base_ms=0.0,
        merge_per_head_chunk_ms=0.0,
    )

    assert row4["uniform_estimated_ms"] < row2["uniform_estimated_ms"]
    assert row4["clustered_estimated_ms"] < row2["clustered_estimated_ms"]
    assert row4["uniform_speedup_vs_current_inline"] > 1.0
    assert row4["per_chunk_active_work"]["uniform_max_pv_blocks"] == 1


def test_splitk_estimate_includes_merge_overhead():
    without_merge = estimate_row(
        _row(),
        chunks=4,
        projection_weight=1.0,
        qk_weight=8.0,
        pv_weight=4.0,
        merge_base_ms=0.0,
        merge_per_head_chunk_ms=0.0,
    )
    with_merge = estimate_row(
        _row(),
        chunks=4,
        projection_weight=1.0,
        qk_weight=8.0,
        pv_weight=4.0,
        merge_base_ms=0.01,
        merge_per_head_chunk_ms=0.001,
    )

    assert with_merge["uniform_estimated_ms"] == pytest.approx(
        without_merge["uniform_estimated_ms"] + 0.018
    )
