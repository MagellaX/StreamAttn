from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.profile_gate0_seed_only_multi_layer_rollout import (
    _parse_layer_ids,
    _route_bundle_from_args,
    _timing_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(**updates):
    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "layers": "0,14,16,24,26,27,29,35",
        "policy_names": "",
        "use_packaged_policies": True,
        "dtype": "fp16",
        "max_seq": 32768,
        "batch_size": 4,
        "q_heads": 16,
        "kv_heads": 2,
        "head_dim": 128,
        "block_size": 32,
        "sink_blocks": 2,
        "recent_blocks": 2,
        "middle_seed_blocks": 8,
        "block_order": "recent_first",
        "num_warps": 4,
        "num_stages": 2,
        "max_kl": 1.0e-4,
        "min_topk_overlap": 4,
        "max_logprob_delta": 2.0e-3,
    }
    payload.update(updates)
    return argparse.Namespace(**payload)


def test_parse_layer_ids_sorts_and_deduplicates():
    assert _parse_layer_ids("35,0,14,0") == [0, 14, 35]


def test_parse_layer_ids_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        _parse_layer_ids("")


def test_qwen3_packaged_multi_layer_bundle_loads_green_cells():
    bundle = _route_bundle_from_args(_args())

    assert bundle.layer_ids == [0, 14, 16, 24, 26, 27, 29, 35]
    assert len(bundle.policy_names) == 8
    assert all(policy.model_id == "Qwen/Qwen2.5-3B-Instruct" for policy in bundle.policies)
    assert all(policy.min_batch == 4 for policy in bundle.policies)
    assert all(policy.heads == 16 and policy.kv_heads == 2 and policy.dim == 128 for policy in bundle.policies)


def test_qwen3_packaged_multi_layer_bundle_rejects_below_min_batch():
    with pytest.raises(ValueError, match="no packaged policies matched"):
        _route_bundle_from_args(_args(batch_size=2))


def test_qwen3_timing_bundle_reports_selected_layer_speedup():
    bundle = _route_bundle_from_args(_args())
    timing = _timing_bundle(bundle.policies, bundle.artifacts, batch_size=4)

    assert timing["complete"] is True
    assert timing["layer_count"] == 8
    assert timing["service_speedup_vs_flashinfer_selected_layers"] > 1.5
    assert timing["planned_direct_speedup_vs_flashinfer_selected_layers"] > 1.7


def test_qwen3_strict_multi_layer_bundle_artifact():
    path = REPO_ROOT / "stream_attention" / "policies" / "qwen25_3b_32k_b4_seed_only_bundle.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == "streamattn.seed_only_route_bundle.v1"
    assert payload["bundle_id"] == "qwen25_3b_32k_fp16_b4_seed_only_7layer_v1"
    assert payload["seed_only_layers"] == [0, 14, 16, 24, 26, 27, 35]
    assert payload["safety"]["passed"] is True
    assert payload["safety"]["max_kl_observed"] < 1.0e-4
    assert payload["safety"]["top1_changes"] == 0
    assert payload["safety"]["sample_token_changed_count"] == 0
    assert payload["timing"]["service_speedup_vs_flashinfer_selected_layers"] > 1.5
    assert payload["diagnostics"]["all_8_green_layers"]["passed_strict_gate"] is False
