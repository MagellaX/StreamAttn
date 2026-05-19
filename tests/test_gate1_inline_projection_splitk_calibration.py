import json

import pytest

from benchmarks.summarize_gate1_inline_projection_splitk import (
    _calibrate_heads,
    _grouped_profiles,
    _layer_oracle,
    _load_rows,
)


def test_splitk_calibration_selects_fastest_safe_row(tmp_path):
    payload = {
        "results": [
            {
                "model_id": "m",
                "prompt_type": "code",
                "layer_id": 8,
                "head_index": 3,
                "kv_len": 16384,
                "projection_dim": 8,
                "projection_seed": 1,
                "filter_margin": 64.0,
                "splitk_total_ms": 0.12,
                "dense_ms": 0.03,
                "splitk_stats": {"projection_skip_fraction": 1.0, "pv_executed_fraction": 0.0},
                "splitk_error_vs_dense": {"max_abs_error": 0.0, "mean_abs_error": 0.0},
            },
            {
                "model_id": "m",
                "prompt_type": "code",
                "layer_id": 8,
                "head_index": 3,
                "kv_len": 16384,
                "projection_dim": 16,
                "projection_seed": 2,
                "filter_margin": 80.0,
                "splitk_total_ms": 0.10,
                "dense_ms": 0.03,
                "splitk_stats": {"projection_skip_fraction": 0.9, "pv_executed_fraction": 0.0},
                "splitk_error_vs_dense": {"max_abs_error": 0.0, "mean_abs_error": 0.0},
            },
        ]
    }
    path = tmp_path / "splitk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows = _load_rows([str(path)])
    heads = _calibrate_heads(rows, max_error=1e-3, max_mean_error=1e-4, min_skip_fraction=0.25)

    assert len(heads) == 1
    assert heads[0]["safe"] is True
    assert heads[0]["projection_dim"] == 16
    assert heads[0]["projection_seed"] == 2
    assert heads[0]["splitk_total_ms"] == 0.10


def test_splitk_layer_oracle_uses_dense_for_unsafe_head():
    heads = [
        {
            "model_id": "m",
            "prompt_type": "code",
            "layer_id": 8,
            "kv_len": 16384,
            "head": 0,
            "safe": True,
            "dense_ms": 0.03,
            "splitk_total_ms": 0.02,
        },
        {
            "model_id": "m",
            "prompt_type": "code",
            "layer_id": 8,
            "kv_len": 16384,
            "head": 1,
            "safe": False,
            "dense_ms": 0.03,
            "selected_ms": 0.03,
        },
    ]

    layers = _layer_oracle(heads)

    assert len(layers) == 1
    assert layers[0]["safe_sparse_heads"] == [0]
    assert layers[0]["unsafe_heads"] == [1]
    assert layers[0]["selective_serial_sum_ms"] == pytest.approx(0.05)
    assert layers[0]["selective_serial_speedup_vs_dense_sum"] == pytest.approx(1.2)
    assert layers[0]["selective_group_max_speedup_vs_dense_sum"] == pytest.approx(2.0)
    assert layers[0]["dense_all_max_ms"] == pytest.approx(0.03)
    assert layers[0]["splitk_safe_max_ms"] == pytest.approx(0.02)
    assert layers[0]["dense_unsafe_max_ms"] == pytest.approx(0.03)
    assert layers[0]["selective_head_parallel_lower_bound_ms"] == pytest.approx(0.03)
    assert layers[0]["selective_head_parallel_speedup_vs_dense_max"] == pytest.approx(1.0)


def test_splitk_grouped_profiles_report_selected_head_speedup():
    rows = [
        {
            "model_id": "m",
            "prompt_type": "code",
            "layer_id": 8,
            "kv_len": 16384,
            "head_indices": [2, 3, 4],
            "selected_head_count": 3,
            "num_chunks": 8,
            "filter_margin": 64.0,
            "splitk_total_ms": 0.12,
            "dense_ms": 0.09,
            "splitk_stats": {"projection_skip_fraction": 0.9, "pv_executed_fraction": 0.0},
            "splitk_error_vs_dense": {"max_abs_error": 0.004, "mean_abs_error": 0.0005},
        }
    ]

    groups = _grouped_profiles(rows)

    assert len(groups) == 1
    assert groups[0]["selected_head_count"] == 3
    assert groups[0]["splitk_speedup_vs_dense"] == pytest.approx(0.75)
