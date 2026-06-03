import json
from pathlib import Path

from benchmarks.build_stress_admissibility_dataset import build_dataset


def _bucket_row(*, passed: bool):
    return {
        "case_count": 32,
        "kl_max": 5e-5 if passed else 0.1,
        "kl_p99": 4e-5 if passed else 0.08,
        "top1_changed_count": 0 if passed else 1,
        "sample_token_changed_count": 0 if passed else 1,
        "topk_overlap_min": 4 if passed else 3,
        "reference_top1_logprob_delta_max_abs": 1e-4 if passed else 0.2,
    }


def _write_route(tmp_path: Path, name: str, buckets):
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps({"safety": {"by_prompt_bucket": buckets}}),
        encoding="utf-8",
    )
    return {
        "name": name,
        "artifact": str(path),
        "status": "complete",
        "seed_layers": [0, 14],
        "exact_layers": [27],
        "dynamic_layers": [26] if "dynamic" in name else [],
        "dynamic_profile": "qk_block_max" if "dynamic" in name else "",
        "speedup_vs_dense_decode": 1.05,
    }


def _write_selector_coverage(tmp_path: Path):
    path = tmp_path / "selector.json"
    rows = []
    for bucket in ["code", "json_tool", "multilingual"]:
        rows.append(
            {
                "condition": "route_conditioned",
                "bucket": bucket,
                "layer": 26,
                "selector": "fixed_policy",
                "mass_omitted": 0.5,
                "support_out_seed": 0.2,
                "delta_collapse": 0.1,
                "value_residual_ratio": 0.6,
                "dense_vs_route_attention_js": 0.05,
                "selector_estimated_dot_token_ratio": 0.01,
            }
        )
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def test_dataset_labels_least_conservative_passing_route(tmp_path):
    routes = [
        _write_route(
            tmp_path,
            "stress_l27_exact_l26_seed",
            {
                "code": _bucket_row(passed=True),
                "json_tool": _bucket_row(passed=False),
                "multilingual": _bucket_row(passed=False),
            },
        ),
        _write_route(
            tmp_path,
            "stress_l27_exact_l26_dynamic_qk",
            {
                "code": _bucket_row(passed=True),
                "json_tool": _bucket_row(passed=True),
                "multilingual": _bucket_row(passed=False),
            },
        ),
    ]
    hybrid_summary = tmp_path / "hybrid_summary.json"
    hybrid_summary.write_text(json.dumps({"routes": routes}), encoding="utf-8")

    dataset = build_dataset(hybrid_summary, _write_selector_coverage(tmp_path))
    decisions = {row["bucket"]: row for row in dataset["bucket_decisions"]}

    assert decisions["code"]["recommended_mode"] == "seed_ok"
    assert decisions["code"]["recommended_route"] == "stress_l27_exact_l26_seed"
    assert decisions["json_tool"]["recommended_mode"] == "dynamic_ok"
    assert decisions["json_tool"]["recommended_route"] == "stress_l27_exact_l26_dynamic_qk"
    assert decisions["multilingual"]["recommended_mode"] == "exact_required"
    assert len(dataset["admissibility_rows"]) == 3


def test_dataset_defaults_to_exact_when_no_hybrid_route_passes(tmp_path):
    route = _write_route(
        tmp_path,
        "stress_l24_l26_l27_exact",
        {"multilingual": _bucket_row(passed=False)},
    )
    hybrid_summary = tmp_path / "hybrid_summary.json"
    hybrid_summary.write_text(json.dumps({"routes": [route]}), encoding="utf-8")

    dataset = build_dataset(hybrid_summary, _write_selector_coverage(tmp_path))
    decisions = {row["bucket"]: row for row in dataset["bucket_decisions"]}

    assert decisions["multilingual"]["recommended_mode"] == "exact_required"
    assert decisions["multilingual"]["best_observed_route"] == "stress_l24_l26_l27_exact"
