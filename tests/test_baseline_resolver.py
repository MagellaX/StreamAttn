from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from stream_attention.baseline_resolver import (
    DEFAULT_BASELINE_MANIFEST,
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


def _measurement(batch: AttentionBatchV2) -> ExactBaselineMeasurement:
    return ExactBaselineMeasurement(
        baseline_id="pytorch_sdpa_2_7_1",
        backend_revision="2.7.1",
        workload_sha256=batch.fingerprint,
        environment_sha256="e" * 64,
        latency_us=20.0,
        correctness_passed=True,
        graph_replay=True,
    )


@pytest.mark.parametrize("latency", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_measurement_rejects_nonfinite_or_nonpositive_latency(latency):
    with pytest.raises(ValueError, match="latency must be finite and positive"):
        replace(_measurement(_batch()), latency_us=latency)


@pytest.mark.parametrize("latency", [1e-12, 1.0, 1e12])
def test_measurement_accepts_finite_positive_latency(latency):
    assert replace(_measurement(_batch()), latency_us=latency).latency_us == latency


@pytest.mark.parametrize("field", ["workload_sha256", "environment_sha256"])
@pytest.mark.parametrize(
    "digest", [None, 1, "", "a" * 63, "a" * 65, "g" * 64, " " * 64, "a" * 63 + "\n"]
)
def test_measurement_rejects_invalid_digests(field, digest):
    with pytest.raises(ValueError, match=f"{field} must be a 64-character hexadecimal digest"):
        replace(_measurement(_batch()), **{field: digest})


@pytest.mark.parametrize("field", ["baseline_id", "backend_revision"])
def test_measurement_rejects_empty_identity(field):
    with pytest.raises(ValueError, match=f"{field} must be non-empty"):
        replace(_measurement(_batch()), **{field: ""})


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


@pytest.mark.parametrize("reverse", [False, True])
def test_fastest_selection_rejects_mixed_environments_without_expected_digest(reverse):
    batch = _batch()
    descriptors = load_exact_baseline_descriptors()
    measurement = _measurement(batch)
    measurements = [
        measurement,
        replace(
            measurement,
            baseline_id="flashattention_3_beta",
            backend_revision="3.0.0b1",
            environment_sha256="f" * 64,
            latency_us=5.0,
        ),
    ]
    if reverse:
        measurements.reverse()

    with pytest.raises(ValueError, match="multiple environments.*expected_environment_sha256"):
        fastest_measured_exact_baseline(batch, descriptors, iter(measurements))


def test_fastest_selection_filters_expected_environment():
    batch = _batch()
    descriptors = load_exact_baseline_descriptors()
    measurement = _measurement(batch)
    measurements = [
        replace(measurement, environment_sha256="f" * 64, latency_us=1.0),
        measurement,
        replace(measurement, latency_us=10.0),
    ]

    assert fastest_measured_exact_baseline(
        batch,
        iter(descriptors),
        iter(measurements),
        expected_environment_sha256=measurement.environment_sha256,
    ) is measurements[2]
    assert fastest_measured_exact_baseline(
        batch, descriptors, measurements, expected_environment_sha256="a" * 64
    ) is None


@pytest.mark.parametrize("digest", ["", "a" * 63, "g" * 64, 1])
def test_fastest_selection_rejects_invalid_expected_environment_even_without_measurements(digest):
    with pytest.raises(ValueError, match="expected_environment_sha256.*hexadecimal digest"):
        fastest_measured_exact_baseline(
            _batch(), [], [], expected_environment_sha256=digest
        )


@pytest.mark.parametrize("expected_environment", [None, "E" * 64])
def test_fastest_selection_compares_hex_digests_case_insensitively(expected_environment):
    batch = _batch()
    measurement = _measurement(batch)
    faster = replace(
        measurement,
        workload_sha256=measurement.workload_sha256.upper(),
        environment_sha256=measurement.environment_sha256.upper(),
        latency_us=10.0,
    )

    assert fastest_measured_exact_baseline(
        batch,
        load_exact_baseline_descriptors(),
        [measurement, faster],
        expected_environment_sha256=expected_environment,
    ) is faster


@pytest.mark.parametrize(
    "changes",
    [
        {"workload_sha256": "0" * 64},
        {"correctness_passed": False},
        {"baseline_id": "unknown"},
        {"backend_revision": "wrong-revision"},
        {"graph_replay": False},
        {"baseline_id": "flashinfer_batch_attention_0_6_17", "backend_revision": "0.6.17"},
    ],
)
def test_ineligible_measurements_do_not_create_environment_ambiguity(changes):
    batch = _batch()
    measurement = _measurement(batch)
    rejected = replace(
        measurement, environment_sha256="f" * 64, latency_us=1.0, **changes
    )

    assert fastest_measured_exact_baseline(
        batch, load_exact_baseline_descriptors(), [rejected, measurement]
    ) is measurement
    assert fastest_measured_exact_baseline(
        batch, load_exact_baseline_descriptors(), [rejected]
    ) is None


def test_fastest_selection_accepts_repeated_measurements_in_one_environment():
    batch = _batch()
    measurement = _measurement(batch)
    faster = replace(measurement, latency_us=10.0)

    assert fastest_measured_exact_baseline(
        batch, load_exact_baseline_descriptors(), iter([measurement, faster, measurement])
    ) is faster


def test_fastest_selection_with_no_candidates_returns_none():
    batch = _batch()
    assert fastest_measured_exact_baseline(batch, [], [_measurement(batch)]) is None
    assert fastest_measured_exact_baseline(batch, load_exact_baseline_descriptors(), []) is None


@pytest.mark.parametrize("changes", [{}, {"revision": "another-revision"}, {"direct_layout": False}])
@pytest.mark.parametrize("reverse", [False, True])
def test_resolvers_reject_duplicate_descriptor_ids(changes, reverse):
    batch = _batch()
    descriptor = load_exact_baseline_descriptors()[0]
    descriptors = [descriptor, replace(descriptor, **changes)]
    if reverse:
        descriptors.reverse()

    with pytest.raises(ValueError, match="descriptor IDs must be unique"):
        resolve_direct_exact_baselines(batch, iter(descriptors))
    with pytest.raises(ValueError, match="descriptor IDs must be unique"):
        fastest_measured_exact_baseline(batch, iter(descriptors), [_measurement(batch)])


@pytest.mark.parametrize("revision", ["2.7.1", "another-revision"])
def test_manifest_rejects_duplicate_descriptor_ids(tmp_path: Path, revision):
    raw = yaml.safe_load(DEFAULT_BASELINE_MANIFEST.read_text(encoding="utf-8"))
    raw["baselines"].append({**raw["baselines"][0], "revision": revision})
    path = tmp_path / "duplicate_baselines.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="descriptor IDs must be unique"):
        load_exact_baseline_descriptors(path)
