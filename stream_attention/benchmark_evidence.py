"""GPU benchmark helpers for strict universal-exact evidence artifacts."""

from __future__ import annotations

import importlib.metadata
import math
import statistics
import subprocess
from typing import Iterable, Mapping, Optional, Sequence

import torch

from .phase_database import (
    BackendEvidence,
    CorrectnessEvidence,
    EnvironmentFingerprint,
    MeasurementStatus,
    TimingEvidence,
)


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _nvidia_smi_field(field: str, device_index: int) -> Optional[str]:
    value = _command_output(
        (
            "nvidia-smi",
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
            f"--id={device_index}",
        )
    )
    if value is None:
        return None
    return value.splitlines()[0].strip()


def _architecture_for_capability(capability: tuple[int, int]) -> str:
    if capability[0] == 8:
        return "sm80"
    if capability[0] == 9:
        return "sm90"
    if capability[0] == 10:
        return "sm100"
    raise ValueError(f"unsupported CUDA capability for v1 evidence: {capability}")


def cuda_environment_fingerprint(
    *,
    library_names: Iterable[str] = (),
    compiler_versions: Optional[Mapping[str, str]] = None,
) -> EnvironmentFingerprint:
    """Capture the device and software identity used by one evidence process."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to capture a GPU environment")
    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    libraries: dict[str, str] = {}
    for name in library_names:
        version = _package_version(name)
        libraries[name] = version or "not-installed"
    cudnn_version = torch.backends.cudnn.version()
    libraries["cudnn"] = "unknown" if cudnn_version is None else str(cudnn_version)

    compilers = dict(compiler_versions or {})
    nvcc = _command_output(("nvcc", "--version"))
    if nvcc is not None:
        compilers.setdefault("nvcc", nvcc.splitlines()[-1].strip())
    triton_version = _package_version("triton")
    if triton_version is not None:
        compilers.setdefault("triton", triton_version)
    if not compilers:
        compilers["runtime"] = "precompiled-only"

    device_uuid = _nvidia_smi_field("uuid", device_index)
    driver_version = _nvidia_smi_field("driver_version", device_index)
    return EnvironmentFingerprint(
        architecture=_architecture_for_capability(capability),
        device_name=torch.cuda.get_device_name(device_index),
        device_uuid=device_uuid or f"cuda-device-{device_index}",
        driver_version=driver_version or "unknown-driver",
        cuda_version=torch.version.cuda or "unknown-cuda",
        torch_version=torch.__version__,
        library_versions=libraries,
        compiler_versions=compilers,
    )


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in samples)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_evidence(
    samples_ms: Sequence[float],
    *,
    cold_ms: Optional[float] = None,
    process_count: int = 1,
    timed_allocation_count: int = 0,
    confidence: Optional[float] = None,
) -> TimingEvidence:
    """Convert raw per-call samples into the immutable timing contract."""

    samples = tuple(float(value) for value in samples_ms)
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError("timing samples must be finite and positive")
    if confidence is None:
        confidence = 0.95 if len(samples) >= 9 else 0.80
    return TimingEvidence(
        cold_ms=float(samples[0] if cold_ms is None else cold_ms),
        p10_ms=_percentile(samples, 0.10),
        p50_ms=float(statistics.median(samples)),
        p90_ms=_percentile(samples, 0.90),
        variance_ms2=float(statistics.pvariance(samples)),
        process_count=process_count,
        sample_count=len(samples),
        timed_allocation_count=timed_allocation_count,
        confidence=confidence,
    )


def measured_evidence(
    *,
    evidence_id: str,
    cell_id: str,
    provider: str,
    requested_backend: str,
    resolved_backend: str,
    environment: EnvironmentFingerprint,
    samples_ms: Sequence[float],
    correctness_reference: str,
    checked_cases: int,
    max_abs_error: float,
    max_relative_error: float,
    workspace_bytes: int,
    supported_range: Mapping[str, object],
    native: bool,
    correctness_passed: bool = True,
    failure_reason: Optional[str] = None,
    family_id: Optional[str] = None,
    kernel_key: Optional[str] = None,
    timed_allocation_count: int = 0,
    detail: Optional[str] = None,
) -> BackendEvidence:
    return BackendEvidence(
        evidence_id=evidence_id,
        cell_id=cell_id,
        provider=provider,
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        status=MeasurementStatus.MEASURED,
        native=native,
        environment=environment,
        family_id=family_id,
        kernel_key=kernel_key,
        workspace_bytes=workspace_bytes,
        supported_range=dict(supported_range),
        timing=timing_evidence(
            samples_ms,
            timed_allocation_count=timed_allocation_count,
        ),
        correctness=CorrectnessEvidence(
            passed=correctness_passed,
            reference=correctness_reference,
            checked_cases=checked_cases,
            max_abs_error=max_abs_error,
            max_relative_error=max_relative_error,
            failure_reason=failure_reason,
        ),
        detail=detail,
    )


def outcome_evidence(
    *,
    evidence_id: str,
    cell_id: str,
    provider: str,
    requested_backend: str,
    environment: EnvironmentFingerprint,
    status: MeasurementStatus,
    detail: str,
    native: bool = False,
    family_id: Optional[str] = None,
) -> BackendEvidence:
    if status is MeasurementStatus.MEASURED:
        raise ValueError("outcome_evidence is only for non-measured outcomes")
    return BackendEvidence(
        evidence_id=evidence_id,
        cell_id=cell_id,
        provider=provider,
        requested_backend=requested_backend,
        resolved_backend=None,
        status=status,
        native=native,
        environment=environment,
        family_id=family_id,
        detail=detail,
    )


__all__ = [
    "cuda_environment_fingerprint",
    "measured_evidence",
    "outcome_evidence",
    "timing_evidence",
]
