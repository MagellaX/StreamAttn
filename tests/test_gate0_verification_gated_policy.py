from benchmarks.build_gate0_verification_gated_policy import build_gated_policy
from benchmarks.summarize_gate1_inline_projection_splitk_robustness import parse_budgets


def _row():
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt_type": "needle",
        "layer_id": 8,
        "kv_len": 32768,
        "head_indices": [2, 3, 4, 6],
        "selected_head_count": 4,
        "block_size": 32,
        "sink_blocks": 2,
        "recent_blocks": 2,
        "middle_seed_blocks": 1,
        "chunk_anchor_blocks": 0,
        "block_order": "sink_recent_first",
        "num_chunks": 32,
        "seed_strategy": "recompute_seed",
        "filter_margin": 64.0,
        "error_budget": 0.001,
        "projection_kind": "random",
        "projection_dim": 8,
        "projection_seed": 1,
        "projection_metadata_dtype": "fp16",
        "qproj_mode": "fused",
        "splitk_workspace": "reuse",
        "splitk_total_ms": 1.0,
        "dense_ms": 1.4,
        "splitk_stats": {"projection_skip_fraction": 0.75, "pv_executed_fraction": 0.25},
        "splitk_error_vs_dense": {"max_abs_error": 0.04, "mean_abs_error": 0.004},
        "splitk_per_head_stats": {
            "per_head": [
                {"head": 0, "projection_skip_fraction": 0.6, "pv_executed_fraction": 0.4},
                {"head": 1, "projection_skip_fraction": 0.8, "pv_executed_fraction": 0.2},
                {"head": 2, "projection_skip_fraction": 0.7, "pv_executed_fraction": 0.3},
                {"head": 3, "projection_skip_fraction": 0.9, "pv_executed_fraction": 0.1},
            ]
        },
        "splitk_error_vs_dense_per_head": {
            "per_head": [
                {"head": 0, "max_abs_error": 0.004, "mean_abs_error": 0.0012},
                {"head": 1, "max_abs_error": 0.0, "mean_abs_error": 0.0},
                {"head": 2, "max_abs_error": 0.03, "mean_abs_error": 0.002},
                {"head": 3, "max_abs_error": 0.04, "mean_abs_error": 0.004},
            ],
            "worst_head": 3,
        },
    }


def test_gated_policy_splits_fast_unsafe_union_by_per_head_error():
    payload = build_gated_policy(
        [_row()],
        parse_budgets("moderate:1e-2:1e-3:1.15"),
        min_head_skip_fraction=0.25,
    )

    assert payload["summary"]["candidates"] == 1
    candidate = payload["candidates"][0]
    assert candidate["runtime"]["aggressive_union"]["head_indices"] == [2, 3, 4, 6]
    assert candidate["runtime"]["sparse_candidate"]["head_indices"] == [2, 3]
    assert candidate["runtime"]["fallback_candidate"]["head_indices"] == [4, 6]
    assert candidate["verification"]["required"] is True


def test_gated_policy_can_enforce_per_head_mean_error():
    payload = build_gated_policy(
        [_row()],
        parse_budgets("moderate:1e-2:1e-3:1.15"),
        min_head_skip_fraction=0.25,
        enforce_mean=True,
    )

    assert payload["summary"]["candidates"] == 1
    candidate = payload["candidates"][0]
    assert candidate["runtime"]["sparse_candidate"]["head_indices"] == [3]
    assert candidate["runtime"]["fallback_candidate"]["head_indices"] == [2, 4, 6]
    assert candidate["verification"]["verify_heads"] == [2, 4, 6]
    assert candidate["quality"]["safe_head_count"] == 1


def test_gated_policy_requires_fast_union_before_splitting():
    row = _row()
    row["dense_ms"] = 1.05

    payload = build_gated_policy(
        [row],
        parse_budgets("moderate:1e-2:1e-3:1.15"),
        min_head_skip_fraction=0.25,
    )

    assert payload["summary"]["candidates"] == 0
