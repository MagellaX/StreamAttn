from pathlib import Path

from benchmarks.compile_streamattn_seed_policy import compile_seed_policy_cells


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_seed_policy_compiler_reproduces_qwen_32k_green_cells():
    compiled = compile_seed_policy_cells(
        sweep_json=REPO_ROOT
        / "artifacts"
        / "gate0"
        / "seed_only_layer_sweep_l0_l23_b4_h100.json",
        closed_loop_dir=REPO_ROOT / "artifacts" / "gate0",
        policy_dir=REPO_ROOT / "stream_attention" / "policies",
        registry_json=REPO_ROOT / "stream_attention" / "policies" / "registry.json",
    )

    assert compiled["green_layers"] == [1, 2, 5, 6, 8, 18]
    assert [entry["layer_id"] for entry in compiled["registry"]["policies"]] == [
        1,
        2,
        5,
        6,
        8,
        18,
    ]
    assert compiled["registry"]["default"] == "qwen25_05b_l8_32k_seed_only_batched"
    assert compiled["policies"][0]["max_logprob_delta_budget"] == 0.002
    assert compiled["policies"][4]["policy_id"] == "qwen25_05b_l8_32k_fp16_b4_seed_only_v2"


def test_seed_policy_compiler_rejects_layers_without_rollout_or_sweep_gate():
    compiled = compile_seed_policy_cells(
        sweep_json=REPO_ROOT
        / "artifacts"
        / "gate0"
        / "seed_only_layer_sweep_l0_l23_b4_h100.json",
        closed_loop_dir=REPO_ROOT / "artifacts" / "gate0",
        policy_dir=REPO_ROOT / "stream_attention" / "policies",
        registry_json=REPO_ROOT / "stream_attention" / "policies" / "registry.json",
    )

    rejected = {entry["layer_id"]: entry["reasons"] for entry in compiled["rejected_layers"]}

    assert "sweep_distribution_gate_failed" in rejected[0]
    assert "sweep_summary_failed" in rejected[0]
    assert "missing_closed_loop_artifact" in rejected[3]
    assert 1 not in rejected
