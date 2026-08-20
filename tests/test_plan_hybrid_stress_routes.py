from pathlib import Path

from benchmarks.plan_hybrid_stress_routes import build_hybrid_specs, build_plan


def test_build_hybrid_specs_keeps_l27_exact_in_first_three_routes():
    specs = build_hybrid_specs()

    first_three = specs[:3]
    assert [spec.name for spec in first_three] == [
        "stress_l27_exact_l26_seed",
        "stress_l27_exact_l26_dynamic_extreme4",
        "stress_l27_exact_l26_dynamic_qk",
    ]
    for spec in first_three:
        assert 27 not in spec.seed_layers
        assert 26 in spec.seed_layers


def test_build_hybrid_plan_emits_modal_commands_with_secret_flag():
    plan = build_plan(output_dir="out", steps=32, use_hf_token_secret=True)
    route = {row["name"]: row for row in plan["routes"]}["stress_l27_exact_l26_dynamic_extreme4"]

    assert route["seed_layers"] == [0, 14, 16, 24, 26, 35]
    assert route["exact_layers"] == [27]
    assert route["dynamic_layers"] == [26]
    assert route["dynamic_profile"] == "support_extreme4_mean_refine32"
    command = route["command"]
    assert command[:3] == ["modal", "run", "benchmarks\\modal_seed_only_route_bundle_decode.py"]
    assert "--dynamic-selector-layers" in command
    assert "26" in command
    assert "--dynamic-selector-profile" in command
    assert "support_extreme4_mean_refine32" in command
    assert "--use-hf-token-secret" in command
    output_path = command[command.index("--output-json") + 1]
    assert Path(output_path) == Path(
        "out/stress_l27_exact_l26_dynamic_extreme4_32step_h100.json"
    )


def test_build_hybrid_plan_marks_l26_l27_exact_route():
    plan = build_plan(output_dir="out", steps=128)
    route = {row["name"]: row for row in plan["routes"]}["stress_l26_l27_exact"]

    assert route["seed_layers"] == [0, 14, 16, 24, 35]
    assert route["exact_layers"] == [26, 27]
    assert route["dynamic_layers"] == []
