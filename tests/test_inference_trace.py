from __future__ import annotations

import copy

import pytest

from stream_attention.inference_trace import freeze_trace_records, frozen_split_for


def _raw(record_id: str, kv_len: int = 1024) -> dict:
    return {
        "record_id": record_id,
        "source": "serving-replay",
        "source_trace_id": f"upstream-{record_id}",
        "workload": {
            "batch_id": f"batch-{record_id}",
            "architecture": "sm90",
            "phase": "decode",
            "requests": [
                {
                    "request_id": "r0",
                    "phase": "decode",
                    "query_len": 1,
                    "kv_len": kv_len,
                }
            ],
            "attention_kind": "gqa",
            "q_heads": 16,
            "kv_heads": 2,
            "d_qk": 128,
            "d_v": 128,
            "q_dtype": "bf16",
            "kv_dtype": "bf16",
            "output_dtype": "bf16",
            "cache_kind": "contiguous",
            "cache_layout": "bshd",
            "mask_kind": "noncausal",
            "execution_mode": "eager",
            "objective": "latency",
        },
    }


def test_frozen_split_is_order_independent():
    ids = [f"row-{index}" for index in range(40)]
    forward = {record.record_id: record.split for record in freeze_trace_records(map(_raw, ids))}
    reverse = {
        record.record_id: record.split
        for record in freeze_trace_records(map(_raw, reversed(ids)))
    }
    assert forward == reverse
    assert frozen_split_for("row-17") == frozen_split_for("row-17")


def test_reimport_is_idempotent_but_payload_drift_is_rejected():
    prior = freeze_trace_records([_raw("stable")])
    replay = freeze_trace_records([_raw("stable")], existing=prior)
    assert replay == prior

    changed = _raw("stable", kv_len=2048)
    with pytest.raises(ValueError, match="frozen trace record changed"):
        freeze_trace_records([changed], existing=prior)


def test_duplicate_record_ids_are_rejected():
    row = _raw("duplicate")
    with pytest.raises(ValueError, match="duplicate trace record_id"):
        freeze_trace_records([row, copy.deepcopy(row)])
