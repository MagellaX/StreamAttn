from benchmarks.model_head_mode_kv_group_work import model_kv_group_work


def test_qwen_moderate_policy_saves_rows_but_not_kv_loads() -> None:
    result = model_kv_group_work(
        q_heads=14,
        kv_heads=2,
        kv_len=32768,
        tile_size=32,
        seed_heads=[2, 3, 4, 6, 7],
        sink_blocks=2,
        recent_blocks=2,
        middle_seed_blocks=2,
        block_order="recent_first",
        padded_group_rows=8,
    )

    totals = result["totals"]
    assert result["policy"]["seed_tile_count"] == 6
    assert totals["dense_row_tile_work"] == 14336
    assert totals["hybrid_row_tile_work"] == 9246
    assert totals["row_work_reduction"] > 0.35
    assert totals["kv_tile_load_reduction"] == 0.0
    assert totals["padded_row_work_reduction"] == 0.0

    kv0, kv1 = result["per_kv_group"]
    assert kv0["seed_heads"] == [2, 3, 4, 6]
    assert kv0["exact_heads"] == [0, 1, 5]
    assert kv0["row_work_reduction"] > 0.56
    assert kv1["seed_heads"] == [7]
    assert kv1["row_work_reduction"] < 0.15
    assert "row work" in result["interpretation"]


def test_whole_seed_kv_group_can_skip_kv_tiles() -> None:
    result = model_kv_group_work(
        q_heads=14,
        kv_heads=2,
        kv_len=32768,
        tile_size=32,
        seed_heads=list(range(7)),
        sink_blocks=2,
        recent_blocks=2,
        middle_seed_blocks=2,
        block_order="recent_first",
        padded_group_rows=8,
    )

    kv0, kv1 = result["per_kv_group"]
    assert kv0["seed_only_whole_group"] is True
    assert kv0["hybrid_kv_tile_loads"] == 6
    assert kv0["kv_tile_load_reduction"] > 0.99
    assert kv1["seed_only_whole_group"] is False
    assert result["totals"]["kv_tile_load_reduction"] > 0.49
    assert "K/V tile loads" in result["interpretation"]


def test_rejects_invalid_true_gqa_shapes_and_heads() -> None:
    try:
        model_kv_group_work(
            q_heads=13,
            kv_heads=2,
            kv_len=1024,
            tile_size=32,
            seed_heads=[],
            sink_blocks=1,
            recent_blocks=1,
            middle_seed_blocks=0,
            block_order="recent_first",
        )
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("expected invalid true-GQA shape to fail")

    try:
        model_kv_group_work(
            q_heads=14,
            kv_heads=2,
            kv_len=1024,
            tile_size=32,
            seed_heads=[99],
            sink_blocks=1,
            recent_blocks=1,
            middle_seed_blocks=0,
            block_order="recent_first",
        )
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("expected invalid seed head to fail")
