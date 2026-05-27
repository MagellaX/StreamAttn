import pytest

from benchmarks.profile_seed_only_model_decode_marginals import build_marginal_cases, score_cases


def test_build_marginal_cases_adds_base_add_and_leave_one_out():
    cases = build_marginal_cases(
        base_layers=[0, 14, 16],
        candidate_layers=[0, 2, 14, 16, 18],
        modes=["base", "add", "leave_one_out"],
    )

    by_name = {case.name: case for case in cases}
    assert by_name["base"].layers == [0, 14, 16]
    assert by_name["base_plus_l2"].layers == [0, 2, 14, 16]
    assert by_name["base_plus_l18"].layers == [0, 14, 16, 18]
    assert by_name["base_minus_l14"].layers == [0, 16]
    assert "base_plus_l0" not in by_name


def test_score_cases_rejects_safety_before_runtime():
    cases = [
        {
            "name": "base",
            "total_ms": 100.0,
            "saved_ms_total": 10.0,
            "decision": {"passed": True},
            "safety": {"kl_max": 0.00005},
        },
        {
            "name": "base_plus_l29",
            "total_ms": 98.0,
            "saved_ms_total": 12.0,
            "decision": {"passed": False},
            "safety": {"kl_max": 0.00017},
        },
        {
            "name": "base_plus_l2",
            "total_ms": 99.0,
            "saved_ms_total": 11.0,
            "decision": {"passed": True},
            "safety": {"kl_max": 0.00006},
        },
        {
            "name": "base_plus_l18",
            "total_ms": 101.0,
            "saved_ms_total": 9.0,
            "decision": {"passed": True},
            "safety": {"kl_max": 0.00004},
        },
    ]

    scored = {case["name"]: case for case in score_cases(cases)}
    assert scored["base"]["recommendation"] == "keep"
    assert scored["base_plus_l29"]["recommendation"] == "reject_safety"
    assert scored["base_plus_l2"]["recommendation"] == "candidate_add"
    assert scored["base_plus_l18"]["recommendation"] == "reject_runtime"
    assert scored["base_plus_l2"]["marginal_vs_base_ms_total"] == 1.0
    assert scored["base_plus_l2"]["marginal_vs_base_kl"] == pytest.approx(0.00001)


def test_score_cases_uses_leave_one_out_recommendations():
    cases = [
        {
            "name": "base",
            "kind": "base",
            "total_ms": 100.0,
            "saved_ms_total": 10.0,
            "decision": {"passed": True},
            "safety": {"kl_max": 0.00005},
        },
        {
            "name": "base_minus_l14",
            "kind": "leave_one_out",
            "total_ms": 98.0,
            "saved_ms_total": 12.0,
            "decision": {"passed": True},
            "safety": {"kl_max": 0.00006},
        },
        {
            "name": "base_minus_l26",
            "kind": "leave_one_out",
            "total_ms": 101.0,
            "saved_ms_total": 9.0,
            "decision": {"passed": True},
            "safety": {"kl_max": 0.00004},
        },
    ]

    scored = {case["name"]: case for case in score_cases(cases)}
    assert scored["base_minus_l14"]["recommendation"] == "candidate_remove"
    assert scored["base_minus_l26"]["recommendation"] == "keep_layer_runtime"
