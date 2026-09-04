from __future__ import annotations

from stream_attention.baseline_resolver import (
    ExactBaselineMeasurement,
    fastest_measured_exact_baseline,
    load_exact_baseline_descriptors,
    resolve_direct_exact_baselines,
)
from stream_attention.inference_workload import AttentionBatchV2


def _batch(cache_kind: str = "contiguous", cache_layout: str = "bshd") -> AttentionBatchV2:
    request = {
        "request_id": "r0",
        "phase": "micro_prefill",
        "query_len": 16,
        "kv_len": 4096,
    }
    raw = {
        "batch_id": "baseline-cell",
        "architecture": "sm90",
        "phase": "micro_prefill",
        "requests": [request],
        "attention_kind": "gqa",
        "q_heads": 16,
        "kv_heads": 2,
        "d_qk": 128,
        "d_v": 128,
        "q_dtype": "bf16",
        "kv_dtype": "bf16",
        "output_dtype": "bf16",
        "cache_kind": cache_kind,
        "cache_layout": cache_layout,
        "mask_kind": "noncausal",
        "execution_mode": "cuda_graph",
        "fixed_workspace_bytes": 1 << 20,
        "maximum_captured_batch": 8,
        "objective": "latency",
    }
    if cache_kind == "paged":
        request.update(cache_page_ids=list(range(256)), last_page_len=16)
        raw["page_size"] = 16
    return AttentionBatchV2.from_dict(raw)


def test_resolver_reports_semantic_incompatibility_instead_of_guessing():
    descriptors = load_exact_baseline_descriptors()
    rows = {
        row.baseline_id: row
        for row in resolve_direct_exact_baselines(_batch("paged", "nhd"), descriptors)
    }

    assert not rows["pytorch_sdpa_2_7_1"].eligible
    assert "cache_kind" in rows["pytorch_sdpa_2_7_1"].reasons
    assert rows["flashinfer_batch_attention_0_6_17"].eligible
    assert not rows["flashattention_3_beta"].eligible


def test_fastest_selection_requires_eligible_correct_matching_revision():
    batch = _batch()
    descriptors = load_exact_baseline_descriptors()
    environment = "e" * 64
    measurements = (
        ExactBaselineMeasurement(
            baseline_id="pytorch_sdpa_2_7_1",
            backend_revision="2.7.1",
            workload_sha256=batch.fingerprint,
            environment_sha256=environment,
            latency_us=20.0,
            correctness_passed=True,
            graph_replay=True,
        ),
        ExactBaselineMeasurement(
            baseline_id="flashattention_3_beta",
            backend_revision="wrong-revision",
            workload_sha256=batch.fingerprint,
            environment_sha256=environment,
            latency_us=5.0,
            correctness_passed=True,
            graph_replay=True,
        ),
    )

    winner = fastest_measured_exact_baseline(batch, descriptors, measurements)
    assert winner is not None
    assert winner.baseline_id == "pytorch_sdpa_2_7_1"


def test_graph_workload_rejects_eager_only_measurement():
    batch = _batch()
    descriptors = load_exact_baseline_descriptors()
    measurement = ExactBaselineMeasurement(
        baseline_id="pytorch_sdpa_2_7_1",
        backend_revision="2.7.1",
        workload_sha256=batch.fingerprint,
        environment_sha256="f" * 64,
        latency_us=10.0,
        correctness_passed=True,
        graph_replay=False,
    )
    assert fastest_measured_exact_baseline(batch, descriptors, [measurement]) is None
