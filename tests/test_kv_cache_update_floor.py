import pytest

from benchmarks.profile_kv_cache_update_floor import cache_bytes, _summarize_method


def test_cache_bytes_counts_k_and_v_storage():
    assert (
        cache_bytes(
            layer_count=7,
            batch_size=8,
            kv_heads=2,
            max_len=32768,
            head_dim=128,
            dtype_bytes=2,
        )
        == 2 * 7 * 8 * 2 * 32768 * 128 * 2
    )


def test_summarize_method_reports_decode_and_layer_step_costs():
    summary = _summarize_method(
        "direct",
        [8.0, 4.0, 6.0],
        update_steps=2,
        layer_count=3,
    )

    assert summary["median_ms"] == 6.0
    assert summary["median_ms_per_decode_step"] == 3.0
    assert summary["median_us_per_layer_step"] == pytest.approx(1000.0)
