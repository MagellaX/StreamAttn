from benchmarks.plan_gate0_splitk_frontiers import plan_frontiers


def _entry(prompt_type: str, heads, *, speedup: float = 1.2, margin: float = 64.0):
    return {
        "model_id": "qwen",
        "prompt_type": prompt_type,
        "layer_id": 8,
        "kv_len_bucket": 32768,
        "safety_budget": {"name": "moderate", "max_abs_error": 1e-2},
        "runtime": {
            "head_indices": heads,
            "head_group": ",".join(str(head) for head in heads),
            "selected_head_count": len(heads),
            "num_chunks": 32,
            "filter_margin": margin,
            "projection_dim": 8,
            "projection_seed": 1,
        },
        "quality": {
            "speedup_vs_dense": speedup,
            "max_abs_error": 0.005,
            "mean_abs_error": 0.0005,
            "projection_skip_fraction": 0.7,
        },
    }


def test_frontier_plan_adds_prompt_intersection_and_union():
    policy = {
        "entries": [
            _entry("code", [2, 3, 4, 6, 7, 8, 11]),
            _entry("needle", [2, 3, 4, 7, 8, 9]),
        ],
        "budgets": [{"name": "moderate"}],
    }

    payload = plan_frontiers(policy)

    pathways = {row["name"]: row for row in payload["pathways"]}
    intersection = pathways["prompt_agnostic_head_intersection"]["experiments"][0]
    union = pathways["aggressive_union_with_online_verification"]["experiments"][0]

    assert intersection["runtime"]["head_group"] == "2,3,4,7,8"
    assert union["runtime"]["head_group"] == "2,3,4,6,7,8,9,11"
    assert union["runtime"]["verification"] == "sample_skipped_blocks"


def test_frontier_plan_proposes_strict_margin_recovery_for_moderate_entries():
    policy = {
        "entries": [_entry("code", [2, 3, 4])],
        "budgets": [{"name": "strict"}, {"name": "moderate"}],
    }

    payload = plan_frontiers(policy)

    strict = next(row for row in payload["pathways"] if row["name"] == "strict_mode_recovery")
    margins = {item["runtime"]["filter_margin"] for item in strict["experiments"]}
    assert margins == {48.0, 56.0}
