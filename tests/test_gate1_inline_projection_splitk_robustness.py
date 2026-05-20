from benchmarks.summarize_gate1_inline_projection_splitk_robustness import (
    collect_rows_from_payload,
    parse_budgets,
    summarize_rows,
)


def _row(*, prompt_type: str, speedup: float, max_error: float, mean_error: float = 0.0):
    splitk_ms = 1.0
    dense_ms = splitk_ms * speedup
    return {
        "model_id": "qwen",
        "prompt_type": prompt_type,
        "layer_id": 8,
        "kv_len": 16384,
        "head_indices": [2, 3],
        "selected_head_count": 2,
        "filter_margin": 64.0,
        "num_chunks": 8,
        "projection_dim": 8,
        "projection_seed": 1,
        "splitk_workspace": "reuse",
        "splitk_total_ms": splitk_ms,
        "dense_ms": dense_ms,
        "splitk_stats": {"projection_skip_fraction": 0.9, "pv_executed_fraction": 0.1},
        "splitk_error_vs_dense": {"max_abs_error": max_error, "mean_abs_error": mean_error},
    }


def test_parse_robustness_budgets():
    budgets = parse_budgets("strict:1e-3:1e-4:1.0,research:5e-2:5e-3:1.25")

    assert budgets[0]["name"] == "strict"
    assert budgets[0]["max_error"] == 1e-3
    assert budgets[1]["min_speedup"] == 1.25


def test_collect_rows_from_robustness_payload():
    payload = {"runs": [{"capture": {"prompt_type": "needle"}, "results": [_row(prompt_type="needle", speedup=1.3, max_error=0.01)]}]}

    rows = collect_rows_from_payload(payload)

    assert len(rows) == 1
    assert rows[0]["prompt_type"] == "needle"


def test_summarize_rows_reports_budget_passes_and_configs():
    rows = [
        _row(prompt_type="needle", speedup=1.3, max_error=0.02, mean_error=0.001),
        _row(prompt_type="code", speedup=1.1, max_error=0.02, mean_error=0.001),
        _row(prompt_type="long_doc", speedup=1.4, max_error=0.2, mean_error=0.01),
    ]
    budgets = parse_budgets("research:5e-2:5e-3:1.25")

    payload = summarize_rows(rows, budgets)

    assert payload["budget_summaries"][0]["passed_rows"] == 1
    assert payload["budget_summaries"][0]["prompt_types_passed"] == ["needle"]
    assert payload["robust_configs"][0]["passed_rows"] == 1
    assert payload["robust_configs"][0]["rows"] == 3
