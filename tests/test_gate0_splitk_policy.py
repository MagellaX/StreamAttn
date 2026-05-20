from benchmarks.build_gate0_splitk_policy import build_policy
from benchmarks.summarize_gate1_inline_projection_splitk_robustness import parse_budgets


def _row(
    *,
    prompt_type: str = "code",
    heads=None,
    speedup: float = 1.2,
    max_error: float = 0.005,
    mean_error: float = 0.0005,
    skip: float = 0.7,
    margin: float = 64.0,
    chunks: int = 32,
):
    heads = [2, 3, 4] if heads is None else heads
    splitk_ms = 1.0
    dense_ms = splitk_ms * speedup
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt_type": prompt_type,
        "layer_id": 8,
        "kv_len": 32768,
        "head_indices": heads,
        "selected_head_count": len(heads),
        "block_size": 32,
        "sink_blocks": 2,
        "recent_blocks": 2,
        "middle_seed_blocks": 8,
        "chunk_anchor_blocks": 0,
        "block_order": "sink_recent_first",
        "num_chunks": chunks,
        "seed_strategy": "recompute_seed",
        "filter_margin": margin,
        "error_budget": 0.001,
        "projection_kind": "random",
        "projection_dim": 8,
        "projection_seed": 1,
        "projection_metadata_dtype": "fp16",
        "qproj_mode": "fused",
        "splitk_workspace": "reuse",
        "splitk_total_ms": splitk_ms,
        "dense_ms": dense_ms,
        "splitk_stats": {"projection_skip_fraction": skip, "pv_executed_fraction": 1.0 - skip},
        "splitk_error_vs_dense": {"max_abs_error": max_error, "mean_abs_error": mean_error},
    }


def test_build_policy_promotes_fastest_passing_config():
    rows = [
        _row(speedup=1.16, margin=64.0),
        _row(speedup=1.24, margin=80.0),
        _row(prompt_type="needle", speedup=1.1),
    ]
    budgets = parse_budgets("moderate:1e-2:1e-3:1.15")

    payload = build_policy(rows, budgets)

    assert payload["summary"]["entries"] == 1
    entry = payload["entries"][0]
    assert entry["prompt_type"] == "code"
    assert entry["runtime"]["filter_margin"] == 80.0
    assert entry["quality"]["speedup_vs_dense"] == 1.24
    assert entry["fallback"] == "dense"


def test_build_policy_reports_frontier_and_follow_up_experiments():
    rows = [
        _row(
            prompt_type="needle",
            speedup=1.18,
            max_error=0.012,
            mean_error=0.0012,
            heads=[2, 3, 4, 7],
            margin=64.0,
        )
    ]
    budgets = parse_budgets("moderate:1e-2:1e-3:1.15")

    payload = build_policy(
        rows,
        budgets,
        frontier_error_multiplier=1.5,
        frontier_mean_error_multiplier=1.5,
    )

    assert payload["summary"]["entries"] == 0
    assert payload["summary"]["frontier_candidates"] == 1
    frontier = payload["frontier"][0]
    assert frontier["failed_constraints"] == ["max_error", "mean_error"]
    assert any(item["kind"] == "tighten_margin" for item in frontier["experiments"])
    assert any(item["kind"] == "leave_one_out_group" for item in frontier["experiments"])
