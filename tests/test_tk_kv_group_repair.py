from benchmarks.profile_tk_kv_group_repair import repair_work_model


def test_repair_work_model_all_seed_skips_nonseed_tiles() -> None:
    model = repair_work_model(
        q_heads=7,
        kv_len=1024,
        num_chunks=16,
        seed_heads=list(range(7)),
        block_size=32,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        block_order="recent_first",
    )

    assert model["repair_rows"] == []
    assert model["trusted_seed_rows"] == list(range(7))
    assert model["kv_tiles_loaded"] < model["dense_kv_tiles"]
    assert model["kv_tile_load_reduction"] > 0.0
    assert model["row_work_reduction"] > 0.0


def test_repair_work_model_one_repair_row_forces_all_kv_tiles() -> None:
    model = repair_work_model(
        q_heads=7,
        kv_len=1024,
        num_chunks=16,
        seed_heads=[1, 2, 3, 4, 5, 6],
        block_size=32,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        block_order="recent_first",
    )

    assert model["repair_rows"] == [0]
    assert model["kv_tiles_loaded"] == model["dense_kv_tiles"]
    assert model["kv_tile_load_reduction"] == 0.0
    assert 0.0 < model["row_work_reduction"] < 1.0


def test_repair_work_model_all_repair_matches_dense_work() -> None:
    model = repair_work_model(
        q_heads=7,
        kv_len=1024,
        num_chunks=16,
        seed_heads=[],
        block_size=32,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        block_order="recent_first",
    )

    assert model["repair_rows"] == list(range(7))
    assert model["trusted_seed_rows"] == []
    assert model["kv_tile_load_reduction"] == 0.0
    assert model["row_work_reduction"] == 0.0
