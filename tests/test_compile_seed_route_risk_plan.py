import pytest

from benchmarks.compile_seed_route_risk_plan import (
    compile_artifact,
    format_step_row_plan,
    row_failure_reasons,
)


def test_row_failure_reasons_matches_strict_gate():
    gate = {"max_kl": 1.0e-4, "min_topk_overlap": 4, "max_logprob_delta": 2.0e-3}

    reasons = row_failure_reasons(
        {
            "kl_ref_to_candidate": 2.0e-4,
            "topk_overlap": 3,
            "top1_changed": False,
            "sample_token_changed": True,
            "reference_top1_logprob_delta": 3.0e-3,
        },
        gate,
    )

    assert reasons == [
        "sample_token_changed",
        "kl_over",
        "topk_under",
        "logprob_delta_over",
    ]


def test_compile_artifact_extracts_failed_step_rows():
    payload = {
        "decision": {
            "passed": False,
            "gates": {"max_kl": 1.0e-4, "min_topk_overlap": 4, "max_logprob_delta": 2.0e-3},
        },
        "timing": {"speedup_vs_dense_decode": 1.12},
        "steps": [
            {
                "step": 38,
                "rows": [
                    {"row": 2, "prompt_bucket": "needle", "kl_ref_to_candidate": 2.0e-4, "topk_overlap": 5},
                    {"row": 3, "prompt_bucket": "code", "kl_ref_to_candidate": 1.0e-6, "topk_overlap": 5},
                ],
            },
            {
                "step": 57,
                "rows": [
                    {"row": 6, "prompt_bucket": "needle", "kl_ref_to_candidate": 1.0e-6, "topk_overlap": 3},
                ],
            },
        ],
    }

    plan = compile_artifact(payload)

    assert plan["failure_row_source"] == "steps"
    assert plan["compiled_step_row_plan"] == {"38": [2], "57": [6]}
    assert plan["compiled_step_row_plan_text"] == "38:2;57:6"
    assert [row["reasons"] for row in plan["failure_rows"]] == [["topk_under"], ["kl_over"]]
    assert plan["recommendation"]["status"] == "needs_risk_repair"


def test_compile_artifact_uses_margin_worst_rows_without_steps():
    payload = {
        "decision": {
            "passed": False,
            "gates": {"max_kl": 1.0e-4, "min_topk_overlap": 4, "max_logprob_delta": 2.0e-3},
        },
        "margin_forensics": {
            "failure_count": 2,
            "worst_rows": [
                {"step": 91, "row": 7, "prompt_bucket": "chat_doc", "kl_ref_to_candidate": 5.0e-5, "topk_overlap": 3},
                {"step": 38, "row": 2, "prompt_bucket": "needle", "kl_ref_to_candidate": 2.0e-4, "topk_overlap": 5},
            ],
        },
    }

    plan = compile_artifact(payload)

    assert plan["failure_row_source"] == "margin_forensics.worst_rows"
    assert plan["compiled_step_row_plan_text"] == "38:2;91:7"
    assert {tuple(row["reasons"]) for row in plan["failure_rows"]} == {("topk_under",), ("kl_over",)}


def test_compile_artifact_extracts_stress_safety_bucket_rows():
    payload = {
        "decision": {
            "passed": False,
            "gates": {"max_kl": 1.0e-4, "min_topk_overlap": 4, "max_logprob_delta": 2.0e-3},
        },
        "safety": {
            "by_prompt_bucket": {
                "needle_rag": {
                    "first_divergence": {"rows": [4], "step": 21},
                    "first_sample_divergence": {"rows": [4], "step": 5},
                    "worst_case_by_kl": {
                        "row": 4,
                        "prompt_bucket": "needle_rag",
                        "kl_ref_to_candidate": 0.05,
                        "topk_overlap": 5,
                    },
                },
                "json_tool": {
                    "first_sample_divergence": {"rows": [6], "step": 5},
                    "worst_case_by_kl": {
                        "row": 6,
                        "prompt_bucket": "json_tool",
                        "kl_ref_to_candidate": 0.2,
                        "topk_overlap": 3,
                    },
                },
            }
        },
    }

    plan = compile_artifact(payload)

    assert plan["failure_row_source"] == "safety.by_prompt_bucket"
    assert plan["compiled_step_row_plan_text"] == "0:4,6;5:4,6;21:4"
    reasons_by_step_row = {(row["step"], row["row"]): row["reasons"] for row in plan["failure_rows"]}
    assert reasons_by_step_row[(21, 4)] == ["top1_changed"]
    assert reasons_by_step_row[(5, 4)] == ["sample_token_changed"]
    assert set(reasons_by_step_row[(0, 6)]) == {"kl_over", "topk_under"}


def test_compile_artifact_marks_speed_negative_exact_refresh_as_research_only():
    payload = {
        "decision": {
            "passed": True,
            "gates": {"max_kl": 1.0e-4, "min_topk_overlap": 4, "max_logprob_delta": 2.0e-3},
        },
        "timing": {"speedup_vs_dense_decode": 0.74},
        "safety": {
            "kl_max": 8.0e-5,
            "kl_p99": 7.0e-5,
            "topk_overlap_min": 4,
            "top1_changed_count": 0,
            "sample_token_changed_count": 0,
        },
        "route_bundle": {
            "exact_refresh_backend": "triton_splitk",
            "exact_refresh_plan": {"91": "all"},
            "exact_refresh_row_plan": {"38": [2, 6], "91": [3, 7]},
        },
        "margin_forensics": {"failure_count": 0, "worst_rows": []},
    }

    plan = compile_artifact(payload)

    assert plan["recommendation"]["status"] == "strict_pass_speed_negative"
    assert any("do not promote exact-refresh backend" in action for action in plan["recommendation"]["actions"])
    assert plan["route_bundle"]["exact_refresh_row_plan_text"] == "38:2,6;91:3,7"


def test_format_step_row_plan_sorts_steps_and_rows():
    assert format_step_row_plan({91: {7, 3}, 38: {6, 2}}) == "38:2,6;91:3,7"


def test_compile_artifact_flags_near_gate_bucket():
    payload = {
        "decision": {"passed": True},
        "margin_forensics": {
            "by_bucket": {
                "needle": {"kl_max": 8.5e-5, "kl_p99": 6.0e-5, "topk_overlap_min": 5},
                "code": {"kl_max": 1.0e-6, "kl_p99": 1.0e-6, "topk_overlap_min": 5},
            },
            "worst_rows": [],
        },
    }

    plan = compile_artifact(payload, near_kl_ratio=0.8)

    assert len(plan["bucket_risks"]) == 1
    row = plan["bucket_risks"][0]
    assert row["bucket"] == "needle"
    assert row["level"] == "near_gate"
    assert row["kl_max"] == pytest.approx(8.5e-5)
    assert row["kl_p99"] == pytest.approx(6.0e-5)
    assert row["topk_overlap_min"] == 5
