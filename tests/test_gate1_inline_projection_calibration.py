import json

import pytest

from benchmarks.summarize_gate1_inline_projection_sweep import _calibrate_heads, _layer_oracle, _load_rows


def test_inline_projection_calibration_selects_best_safe_row(tmp_path):
    payload = {
        "results": [
            {
                "model_id": "m",
                "prompt_type": "needle",
                "layer_id": 8,
                "head_index": 3,
                "block_size": 16,
                "middle_seed_blocks": 16,
                "filter_margin": 32.0,
                "block_order": "recent_first",
                "qproj_mode": "precomputed",
                "dense_ms": 0.03,
                "gate1_mass_ms": 0.12,
                "inline_projection_ms": 0.07,
                "q_projection_ms": 0.10,
                "inline_total_ms": 0.17,
                "projection_skip_fraction": 0.9,
                "pv_executed_fraction": 0.1,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
            },
            {
                "model_id": "m",
                "prompt_type": "needle",
                "layer_id": 8,
                "head_index": 3,
                "block_size": 32,
                "middle_seed_blocks": 8,
                "filter_margin": 32.0,
                "block_order": "recent_first",
                "qproj_mode": "fused",
                "dense_ms": 0.03,
                "gate1_mass_ms": 0.12,
                "inline_projection_ms": 0.08,
                "inline_total_ms": 0.08,
                "projection_skip_fraction": 0.85,
                "pv_executed_fraction": 0.12,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
            },
        ]
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows = _load_rows([str(path)])
    heads = _calibrate_heads(rows, max_error=1e-3, max_mean_error=1e-4, min_skip_fraction=0.25)

    assert len(heads) == 1
    assert heads[0]["safe"] is True
    assert heads[0]["qproj_mode"] == "fused"
    assert heads[0]["best_block_size"] == 32
    assert heads[0]["inline_total_ms"] == 0.08


def test_inline_projection_selective_oracle_uses_dense_for_unsafe_head():
    heads = [
        {
            "model_id": "m",
            "prompt_type": "needle",
            "layer_id": 8,
            "head": 0,
            "safe": True,
            "dense_ms": 0.03,
            "gate1_mass_ms": 0.12,
            "selected_ms": 0.02,
        },
        {
            "model_id": "m",
            "prompt_type": "needle",
            "layer_id": 8,
            "head": 1,
            "safe": False,
            "dense_ms": 0.03,
            "gate1_mass_ms": 0.12,
            "selected_ms": 0.03,
        },
    ]

    layers = _layer_oracle(heads)

    assert len(layers) == 1
    assert layers[0]["safe_sparse_heads"] == [0]
    assert layers[0]["unsafe_heads"] == [1]
    assert layers[0]["selective_oracle_sum_ms"] == 0.05
    assert layers[0]["selective_speedup_vs_dense_sum"] == pytest.approx(1.2)
