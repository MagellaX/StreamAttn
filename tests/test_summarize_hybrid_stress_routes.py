import json
from pathlib import Path

from benchmarks.plan_hybrid_stress_routes import build_plan
from benchmarks.summarize_hybrid_stress_routes import summarize_plan


def _artifact_path(route):
    command = route["command"]
    return Path(command[command.index("--output-json") + 1])


def _write_artifact(path: Path, *, passed: bool, speedup: float = 1.05, kl: float = 5e-5):
    path.parent.mkdir(parents=True, exist_ok=True)
    topk = 4 if passed else 3
    payload = {
        "decision": {"passed": passed},
        "timing": {
            "dense_decode_ms_per_token": 10.0,
            "streamattn_decode_ms_per_token": 10.0 / speedup,
            "speedup_vs_dense_decode": speedup,
        },
        "safety": {
            "case_count": 32,
            "kl_max": kl,
            "kl_p99": kl,
            "top1_changed_count": 0,
            "sample_token_changed_count": 0,
            "topk_overlap_min": topk,
            "reference_top1_logprob_delta_max_abs": 1e-4,
            "by_prompt_bucket": {
                "json_tool": {
                    "kl_max": kl,
                    "kl_p99": kl,
                    "topk_overlap_min": topk,
                    "reference_top1_logprob_delta_max_abs": 1e-4,
                }
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_plan(tmp_path: Path):
    plan = build_plan(output_dir=str(tmp_path / "routes"), steps=32)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, {route["name"]: route for route in plan["routes"]}


def test_missing_hybrid_artifacts_waits_for_results(tmp_path):
    plan_path, _ = _write_plan(tmp_path)

    summary = summarize_plan(plan_path)

    assert summary["promotion"]["decision"] == "await_results"
    assert len(summary["promotion"]["missing_routes"]) == 5


def test_l27_exact_l26_seed_is_first_promotion(tmp_path):
    plan_path, routes = _write_plan(tmp_path)
    _write_artifact(_artifact_path(routes["stress_l27_exact_l26_seed"]), passed=True)

    summary = summarize_plan(plan_path)

    assert summary["promotion"]["decision"] == "promote_l27_exact_l26_seed"
    assert summary["promotion"]["route"] == "stress_l27_exact_l26_seed"


def test_dynamic_route_wins_when_seed_route_fails(tmp_path):
    plan_path, routes = _write_plan(tmp_path)
    _write_artifact(_artifact_path(routes["stress_l27_exact_l26_seed"]), passed=False)
    _write_artifact(
        _artifact_path(routes["stress_l27_exact_l26_dynamic_extreme4"]),
        passed=True,
        speedup=1.04,
    )
    _write_artifact(
        _artifact_path(routes["stress_l27_exact_l26_dynamic_qk"]),
        passed=True,
        speedup=1.06,
    )

    summary = summarize_plan(plan_path)

    assert summary["promotion"]["decision"] == "promote_l27_exact_l26_dynamic"
    assert summary["promotion"]["route"] == "stress_l27_exact_l26_dynamic_qk"


def test_l26_l27_exact_promotion_after_seed_and_dynamic_fail(tmp_path):
    plan_path, routes = _write_plan(tmp_path)
    for name in [
        "stress_l27_exact_l26_seed",
        "stress_l27_exact_l26_dynamic_extreme4",
        "stress_l27_exact_l26_dynamic_qk",
    ]:
        _write_artifact(_artifact_path(routes[name]), passed=False)
    _write_artifact(_artifact_path(routes["stress_l26_l27_exact"]), passed=True)

    summary = summarize_plan(plan_path)

    assert summary["promotion"]["decision"] == "promote_l26_l27_exact"
    assert summary["promotion"]["route"] == "stress_l26_l27_exact"
