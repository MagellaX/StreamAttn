from __future__ import annotations

import pytest

from stream_attention.benchmark_evidence import (
    measured_evidence,
    outcome_evidence,
    timing_evidence,
)
from stream_attention.phase_database import (
    EnvironmentFingerprint,
    MeasurementStatus,
)


def _environment() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        architecture="sm90",
        device_name="H100",
        device_uuid="GPU-test",
        driver_version="600.0",
        cuda_version="13.0",
        torch_version="2.9.0",
        library_versions={"flashinfer": "0.6.17"},
        compiler_versions={"nvcc": "13.0"},
    )


def test_timing_evidence_uses_raw_quantiles_and_retains_allocation_count():
    timing = timing_evidence(
        [4.0, 1.0, 3.0, 2.0, 5.0],
        timed_allocation_count=1,
    )

    assert timing.p10_ms == pytest.approx(1.4)
    assert timing.p50_ms == 3.0
    assert timing.p90_ms == pytest.approx(4.6)
    assert timing.sample_count == 5
    assert timing.timed_allocation_count == 1
    assert timing.confidence == 0.80


def test_measured_evidence_marks_allocating_external_path_unusable():
    evidence = measured_evidence(
        evidence_id="cell:external:pytorch:eager",
        cell_id="cell",
        provider="external",
        requested_backend="pytorch_sdpa",
        resolved_backend="torch_sdpa_flash:eager",
        environment=_environment(),
        samples_ms=[1.0, 1.1, 0.9],
        correctness_reference="fp32",
        checked_cases=16,
        max_abs_error=1.0e-3,
        max_relative_error=2.0e-3,
        workspace_bytes=0,
        supported_range={"cell_ids": ["cell"]},
        native=False,
        timed_allocation_count=1,
    )

    assert evidence.status is MeasurementStatus.MEASURED
    assert evidence.correctness is not None and evidence.correctness.passed
    assert evidence.is_routable
    assert not evidence.is_usable


def test_nonmeasured_outcome_requires_an_explicit_reason():
    evidence = outcome_evidence(
        evidence_id="cell:external:cudnn:unsupported",
        cell_id="cell",
        provider="external",
        requested_backend="cudnn_sdpa",
        environment=_environment(),
        status=MeasurementStatus.UNSUPPORTED,
        detail="paged KV is unsupported",
    )

    assert evidence.status is MeasurementStatus.UNSUPPORTED
    assert evidence.detail == "paged KV is unsupported"
