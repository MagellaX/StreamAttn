import argparse

from benchmarks.summarize_real_llm_gate1_heads import summarize


def _args():
    return argparse.Namespace(
        dense_ms=1.0,
        qk_only_ms=0.5,
        predicate_overhead_ms=0.0,
        safety_margin=1.05,
        aggregate_threshold=0.30,
        head_index_tax_ms=0.0,
        disagreement_threshold=0.05,
        grouped_speedup_threshold=1.15,
        kv_spread_threshold=0.25,
        value_bound_disagreement_rate=0.20,
    )


def test_summary_recommends_q_head_index_when_kv_group_is_mixed():
    payload = {
        "model": "test",
        "prompts": 1,
        "block_size": 64,
        "tile_size_q": 64,
        "error_budget": 1e-3,
        "results": [
            {
                "skipped": False,
                "layer_id": 0,
                "module_name": "layer0.attn",
                "per_head": [
                    {
                        "layer_id": 0,
                        "module_name": "layer0.attn",
                        "head_id": 0,
                        "kv_head_id": 0,
                        "q_group_id": 0,
                        "active_mass": 0.05,
                        "active_value_bound": 0.06,
                    },
                    {
                        "layer_id": 0,
                        "module_name": "layer0.attn",
                        "head_id": 1,
                        "kv_head_id": 0,
                        "q_group_id": 1,
                        "active_mass": 0.90,
                        "active_value_bound": 0.91,
                    },
                    {
                        "layer_id": 0,
                        "module_name": "layer0.attn",
                        "head_id": 2,
                        "kv_head_id": 1,
                        "q_group_id": 0,
                        "active_mass": 0.08,
                        "active_value_bound": 0.09,
                    },
                    {
                        "layer_id": 0,
                        "module_name": "layer0.attn",
                        "head_id": 3,
                        "kv_head_id": 1,
                        "q_group_id": 1,
                        "active_mass": 0.92,
                        "active_value_bound": 0.93,
                    },
                ],
            }
        ],
    }

    summary = summarize(payload, _args())
    layer = summary["layers"][0]

    assert layer["heads_mass_active_lt_0_25"] == 2
    assert layer["max_within_kv_active_spread"] > 0.8
    assert layer["recommendation"] == "build_q_head_index_grouped_gate1"


def test_summary_flags_value_bound_disagreement():
    payload = {
        "model": "test",
        "results": [
            {
                "skipped": False,
                "layer_id": 0,
                "module_name": "layer0.attn",
                "per_head": [
                    {
                        "layer_id": 0,
                        "module_name": "layer0.attn",
                        "head_id": 0,
                        "kv_head_id": 0,
                        "q_group_id": 0,
                        "active_mass": 0.05,
                        "active_value_bound": 0.40,
                    }
                ],
            }
        ],
    }

    args = _args()
    args.value_bound_disagreement_rate = 0.01
    summary = summarize(payload, args)

    assert summary["layers"][0]["recommendation"] == "calibrate_value_bound_or_lower_mass_budget"
