import json
from pathlib import Path

from benchmarks.compile_streamattn_seed_policy import (
    _write_compiled_outputs,
    compile_seed_policy_cells,
)


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


def test_seed_policy_compiler_write_merges_registry(tmp_path):
    registry_dir = tmp_path / "stream_attention" / "policies"
    registry_dir.mkdir(parents=True)
    registry_path = registry_dir / "registry.json"
    registry_path.write_text(
        """
{
  "schema": "streamattn.policy_registry.v1",
  "default": "dummy_policy",
  "policies": [
    {
      "name": "dummy_policy",
      "path": "policies/dummy_policy.json",
      "policy_id": "dummy_policy_v1",
      "aliases": [],
      "status": "green",
      "model_id": "Dummy/Model",
      "layer_id": 0,
      "mode": "all_seed_only",
      "tensor_space": "post_rope",
      "dtype": "fp16",
      "kv_len_bucket": 32768,
      "min_batch": 4,
      "q_heads": 14,
      "kv_heads": 2,
      "head_dim": 64,
      "attention_type": "true_gqa",
      "kernel_modes": {
        "batch_ge_4": "head_private_direct_seed",
        "batch_lt_4": "exact_native"
      }
    }
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    compiled = compile_seed_policy_cells(
        sweep_json=REPO_ROOT
        / "artifacts"
        / "gate0"
        / "seed_only_layer_sweep_l0_l23_b4_h100.json",
        closed_loop_dir=REPO_ROOT / "artifacts" / "gate0",
        policy_dir=registry_dir,
        registry_json=registry_path,
    )

    _write_compiled_outputs(
        compiled,
        policy_dir=registry_dir,
        registry_json=registry_path,
        write_existing=True,
    )

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    names = {entry["name"] for entry in payload["policies"]}

    assert payload["default"] == "dummy_policy"
    assert "dummy_policy" in names
    assert "qwen25_05b_l1_32k_seed_only_batched" in names
    assert "qwen25_05b_l18_32k_seed_only_batched" in names
