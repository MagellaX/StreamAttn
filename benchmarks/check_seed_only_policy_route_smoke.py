"""CI-safe smoke check for the packaged seed-only StreamAttn route.

The default mode validates the committed policy artifact without requiring
PyTorch or a GPU.  When PyTorch is available, the same script also exercises the
planner's fail-closed decisions on synthetic tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    REPO_ROOT
    / "stream_attention"
    / "policies"
    / "qwen25_05b_l8_32k_seed_only_batched.json"
)
REGISTRY_PATH = REPO_ROOT / "stream_attention" / "policies" / "registry.json"


def _load_policy_json() -> Dict[str, Any]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _load_registry_json() -> Dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _check_policy_registry(payload: Dict[str, Any]) -> Dict[str, Any]:
    failures = []
    policies = payload.get("policies") or []
    default = payload.get("default")
    names = {entry.get("name") for entry in policies}
    expected_names = {
        "qwen25_05b_l1_32k_seed_only_batched",
        "qwen25_05b_l2_32k_seed_only_batched",
        "qwen25_05b_l5_32k_seed_only_batched",
        "qwen25_05b_l6_32k_seed_only_batched",
        "qwen25_05b_l8_32k_seed_only_batched",
        "qwen25_05b_l18_32k_seed_only_batched",
        "qwen25_15b_l0_32k_seed_only_batched",
        "qwen25_15b_l3_32k_seed_only_batched",
        "qwen25_3b_l0_32k_seed_only_batched",
        "qwen25_3b_l14_32k_seed_only_batched",
        "qwen25_3b_l16_32k_seed_only_batched",
        "qwen25_3b_l24_32k_seed_only_batched",
        "qwen25_3b_l26_32k_seed_only_batched",
        "qwen25_3b_l27_32k_seed_only_batched",
        "qwen25_3b_l29_32k_seed_only_batched",
        "qwen25_3b_l35_32k_seed_only_batched",
    }
    green = [entry for entry in policies if entry.get("status") == "green"]

    if payload.get("schema") != "streamattn.policy_registry.v1":
        failures.append("registry_schema_mismatch")
    if default != "qwen25_05b_l8_32k_seed_only_batched":
        failures.append("registry_default_mismatch")
    if default not in names:
        failures.append("registry_default_missing")
    if len(green) < len(expected_names):
        failures.append("registry_green_policy_count_mismatch")
    for entry in green:
        if entry.get("min_batch") != 4:
            failures.append("registry_green_min_batch_mismatch")
        if entry.get("kernel_modes", {}).get("batch_ge_4") != "head_private_direct_seed":
            failures.append("registry_green_kernel_mode_mismatch")
    missing_names = sorted(expected_names - names)
    for name in missing_names:
        failures.append(f"registry_missing_green_cell:{name}")

    return {
        "registry_failures": failures,
        "registry_passed": not failures,
        "registry_policy_count": len(policies),
    }


def _check_policy_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
    safety = payload.get("safety") or {}
    timing = payload.get("timing") or {}
    kernel_modes = payload.get("kernel_modes") or {}
    shape = payload.get("shape") or {}
    seed = payload.get("seed_config") or {}
    failures = []

    expected = {
        "mode": "all_seed_only",
        "layer_id": 8,
        "tensor_space": "post_rope",
        "dtype": "fp16",
        "kv_len_bucket": 32768,
        "min_batch": 4,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            failures.append(f"{key}_mismatch")

    shape_expected = {"q_heads": 14, "true_kv_heads": 2, "dim": 64, "kv_len": 32768}
    for key, value in shape_expected.items():
        if shape.get(key) != value:
            failures.append(f"shape_{key}_mismatch")

    if seed.get("block_size") != 32:
        failures.append("seed_block_size_mismatch")
    if int(seed.get("sink_blocks", -1)) + int(seed.get("recent_blocks", -1)) + int(
        seed.get("middle_seed_blocks", -1)
    ) <= 0:
        failures.append("seed_blocks_invalid")

    if not safety.get("top1_must_match", False):
        failures.append("top1_gate_disabled")
    if int(safety.get("min_top5_overlap", 0)) < 4:
        failures.append("top5_gate_too_weak")
    if float(safety.get("max_kl", 1.0)) > 1.0e-4:
        failures.append("kl_gate_too_weak")
    if not safety.get("teacher_forced_last32"):
        failures.append("missing_teacher_forced_evidence")
    if not safety.get("greedy_closed_loop_32"):
        failures.append("missing_greedy_evidence")
    if not safety.get("coupled_top_p_sampling_32"):
        failures.append("missing_sampling_evidence")

    if float(timing.get("h100_product_wrapper_b8_speedup", 0.0)) < 1.10:
        failures.append("h100_b8_speedup_below_gate")
    if float(timing.get("h100_service_b4_speedup", 0.0)) < 1.10:
        failures.append("h100_b4_service_speedup_below_gate")
    if float(timing.get("h100_planned_direct_b4_speedup", 0.0)) < 1.10:
        failures.append("h100_b4_planned_direct_speedup_below_gate")
    if float(timing.get("a100_product_wrapper_b8_speedup", 0.0)) < 1.10:
        failures.append("a100_b8_speedup_below_gate")
    if kernel_modes.get("batch_ge_4") != "head_private_direct_seed":
        failures.append("kernel_mode_batch_ge_4_not_direct_seed")
    if kernel_modes.get("batch_lt_4") != "exact_native":
        failures.append("kernel_mode_batch_lt_4_not_exact_native")
    if kernel_modes.get("two_kernel_split_seed_status") != "diagnostic_not_product_profitable":
        failures.append("split_seed_status_not_diagnostic_no_go")

    return {
        "artifact_policy_id": payload.get("policy_id"),
        "artifact_failures": failures,
        "artifact_passed": not failures,
    }


def _check_route_with_torch() -> Dict[str, Any]:
    import torch

    from stream_attention import (
        StreamAttnDecodePolicy,
        load_packaged_gate0_seed_only_batched_policy,
        stream_attn_decode_plan,
    )

    policy = load_packaged_gate0_seed_only_batched_policy()
    q = torch.empty((4, 1, 14, 64), dtype=torch.float16)
    k = torch.empty((4, 32768, 2, 64), dtype=torch.float16)
    plan = stream_attn_decode_plan(
        q,
        k,
        gate0_seed_only_batched_policy=policy,
        policy=StreamAttnDecodePolicy(collect_telemetry_every=0),
        active_fraction_hint=1.0,
        block_size=policy.block_size,
        num_warps=policy.num_warps,
        num_stages=policy.num_stages,
    )

    q_small = torch.empty((2, 1, 14, 64), dtype=torch.float16)
    k_small = torch.empty((2, 32768, 2, 64), dtype=torch.float16)
    small_plan = stream_attn_decode_plan(
        q_small,
        k_small,
        gate0_seed_only_batched_policy=policy,
        policy=StreamAttnDecodePolicy(collect_telemetry_every=0),
        active_fraction_hint=1.0,
        block_size=policy.block_size,
        num_warps=policy.num_warps,
        num_stages=policy.num_stages,
    )

    q_wrong_dtype = torch.empty((8, 1, 14, 64), dtype=torch.float32)
    k_wrong_dtype = torch.empty((8, 32768, 2, 64), dtype=torch.float32)
    wrong_dtype_reasons = policy.mismatch_reasons(q_wrong_dtype, k_wrong_dtype)

    failures = []
    if plan.backend != "gate0_seed_only_batched":
        failures.append("matching_route_not_seed_only")
    if small_plan.backend != "dense" or small_plan.fallback_reason != "batch_below_min":
        failures.append("small_batch_not_fail_closed")
    if "dtype_mismatch" not in wrong_dtype_reasons:
        failures.append("dtype_mismatch_not_reported")

    return {
        "route_backend": plan.backend,
        "small_batch_backend": small_plan.backend,
        "small_batch_fallback_reason": small_plan.fallback_reason,
        "wrong_dtype_reasons": wrong_dtype_reasons,
        "route_failures": failures,
        "route_passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-no-torch", action="store_true")
    args = parser.parse_args()

    payload = _load_policy_json()
    registry = _load_registry_json()
    result: Dict[str, Any] = {
        "schema": "streamattn.gate0.seed_only_policy_route_smoke.v1",
        "policy_path": str(POLICY_PATH.relative_to(REPO_ROOT)),
        "registry_path": str(REGISTRY_PATH.relative_to(REPO_ROOT)),
        "registry": _check_policy_registry(registry),
        "artifact": _check_policy_artifact(payload),
    }
    try:
        result["route"] = _check_route_with_torch()
    except ModuleNotFoundError as exc:
        if not args.allow_no_torch:
            raise
        result["route"] = {
            "route_skipped": True,
            "skip_reason": f"missing_module:{exc.name}",
        }

    passed = (
        bool(result["registry"]["registry_passed"])
        and bool(result["artifact"]["artifact_passed"])
        and bool(
            result["route"].get("route_passed", result["route"].get("route_skipped", False))
        )
    )
    result["passed"] = passed
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
