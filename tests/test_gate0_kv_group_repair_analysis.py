from benchmarks.analyze_gate0_kv_group_repair import analyze_row, parse_budgets


def _row():
    return {
        "_capture": {"model_id": "qwen", "prompt_type": "needle"},
        "capture": {"layer_id": 8, "logical_num_kv_heads": 2},
        "policy": {"seed_heads": [0, 1, 2, 3, 4, 5, 6]},
        "shape": {"q_heads": 14, "true_kv_heads": 2, "kv_len": 32768},
        "quality": {
            "hybrid_seed_error_vs_true_dense_per_head": {
                "per_head": [
                    {"head": 0, "max_abs_error": 0.03, "mean_abs_error": 0.002},
                    {"head": 1, "max_abs_error": 0.011, "mean_abs_error": 0.001},
                    {"head": 2, "max_abs_error": 0.002, "mean_abs_error": 0.0001},
                ]
            }
        },
        "timing": {
            "exact_remaining_flashinfer_group_parallel_oracle_ms": 0.04,
            "seed_only_group_parallel_oracle_ms": 0.03,
            "reference_exact_ms": 0.05,
            "true_gqa_dense_all_ms": 0.06,
        },
    }


def test_kv_group_repair_analysis_identifies_repair_heads_by_budget():
    rows = analyze_row(_row(), parse_budgets("strict:1e-2,moderate:1.5e-2"))

    strict, moderate = rows
    assert strict["repair_heads"] == [0, 1]
    assert strict["corrected_max_abs_error"] == 0.002
    assert moderate["repair_heads"] == [0]
    assert moderate["corrected_max_abs_error"] == 0.011
    assert moderate["speculative_seed_kv_groups"] == [0]
    assert moderate["optimistic_speedup_vs_reference_exact"] == 1.25
