from __future__ import annotations

import pytest

from stream_attention.inference_workload import (
    AttentionBatchV2,
    RequestPhase,
    load_universal_inference_manifest,
)


def _request(request_id: str, phase: str, query_len: int, kv_len: int) -> dict:
    return {
        "request_id": request_id,
        "phase": phase,
        "query_len": query_len,
        "kv_len": kv_len,
    }


def _batch(**overrides: object) -> dict:
    raw: dict[str, object] = {
        "batch_id": "batch-1",
        "architecture": "sm90",
        "phase": "verify",
        "requests": [_request("r0", "verify", 4, 32768)],
        "attention_kind": "gqa",
        "q_heads": 16,
        "kv_heads": 2,
        "d_qk": 128,
        "d_v": 128,
        "q_dtype": "bf16",
        "kv_dtype": "bf16",
        "accumulator_dtype": "fp32",
        "output_dtype": "bf16",
        "scale_format": "scalar_fp32",
        "cache_kind": "contiguous",
        "cache_layout": "bshd",
        "mask_kind": "noncausal",
        "execution_mode": "cuda_graph",
        "fixed_workspace_bytes": 1 << 20,
        "maximum_captured_batch": 8,
        "objective": "latency",
    }
    raw.update(overrides)
    return raw


def test_manifest_is_distribution_based_and_covers_all_query_regimes():
    manifest = load_universal_inference_manifest()

    assert manifest.manifest_id == "universal_inference_v2"
    assert {source.kind.value for source in manifest.sources} == {
        "trace",
        "stratified",
        "boundary",
    }
    assert manifest.regime_for(1).name == "scalar_decode"
    assert manifest.regime_for(8).name == "speculative_verify"
    assert manifest.regime_for(64).name == "micro_prefill"
    assert manifest.regime_for(4096).name == "prefill"
    assert manifest.acceptance.holdout_p90_routing_regret == 0.05


def test_batch_round_trip_preserves_fingerprint():
    batch = AttentionBatchV2.from_dict(_batch())
    restored = AttentionBatchV2.from_dict(batch.as_dict())

    assert batch.query_lengths == (4,)
    assert batch.kv_lengths == (32768,)
    assert batch.group_size == 8
    assert batch.fingerprint == restored.fingerprint


def test_mixed_ragged_batch_is_a_first_class_workload():
    batch = AttentionBatchV2.from_dict(
        _batch(
            phase="mixed",
            requests=[
                _request("decode", "decode", 1, 32768),
                _request("verify", "verify", 8, 8192),
                _request("prefill", "micro_prefill", 32, 4096),
            ],
        )
    )

    assert batch.phase is RequestPhase.MIXED
    assert batch.is_ragged
    assert batch.batch_size == 3


def test_paged_metadata_must_exactly_describe_each_kv_extent():
    request = _request("r0", "decode", 1, 19)
    request.update(cache_page_ids=[7, 2], last_page_len=3)
    batch = AttentionBatchV2.from_dict(
        _batch(
            phase="decode",
            requests=[request],
            cache_kind="paged",
            cache_layout="nhd",
            page_size=16,
        )
    )
    assert batch.kv_lengths == (19,)

    request["kv_len"] = 20
    with pytest.raises(ValueError, match="exactly describe kv_len"):
        AttentionBatchV2.from_dict(
            _batch(
                phase="decode",
                requests=[request],
                cache_kind="paged",
                cache_layout="nhd",
                page_size=16,
            )
        )


def test_speculative_tree_accepts_root_sentinel_and_rejects_forward_parent():
    request = _request("r0", "verify", 4, 1024)
    request["speculative_tree_parents"] = [-1, 0, 0, 2]
    batch = AttentionBatchV2.from_dict(_batch(requests=[request]))
    assert batch.requests[0].speculative_tree_parents == (-1, 0, 0, 2)

    request["speculative_tree_parents"] = [-1, 2, 0, 2]
    with pytest.raises(ValueError, match="parents must precede"):
        AttentionBatchV2.from_dict(_batch(requests=[request]))


def test_graph_batch_rejects_workspace_or_capacity_mismatch():
    with pytest.raises(ValueError, match="maximum_captured_batch"):
        AttentionBatchV2.from_dict(_batch(maximum_captured_batch=None))
    with pytest.raises(ValueError, match="smaller than the live batch"):
        AttentionBatchV2.from_dict(
            _batch(
                requests=[
                    _request("r0", "verify", 4, 1024),
                    _request("r1", "verify", 4, 1024),
                ],
                maximum_captured_batch=1,
            )
        )
