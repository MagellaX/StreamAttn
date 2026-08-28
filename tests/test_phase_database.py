from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from stream_attention.exact_compiler import (
    ExactWorkloadCell,
    load_universal_exact_manifest,
    matching_kernel_families,
    registered_exact_kernel_families,
)
from stream_attention.phase_database import (
    BackendEvidence,
    CorrectnessEvidence,
    EnvironmentFingerprint,
    MeasurementStatus,
    PhaseDatabase,
    PhaseEntryStatus,
    TimingEvidence,
    compile_phase_database,
    load_backend_evidence,
    load_phase_database,
    resolve_fastest_correct_baseline,
    write_backend_evidence,
)


def _environment(architecture: str) -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        architecture=architecture,
        device_name={"sm80": "A100", "sm90": "H100", "sm100": "B200"}[
            architecture
        ],
        device_uuid=f"GPU-{architecture}",
        driver_version="600.1",
        cuda_version="13.0",
        torch_version="2.9.0",
        library_versions={"flashinfer": "0.6.17", "cudnn": "9.14"},
        compiler_versions={"nvcc": "13.0", "triton": "3.4"},
    )


def test_environment_compatibility_ignores_physical_gpu_uuid():
    first = _environment("sm80")
    second = EnvironmentFingerprint(
        **{**first.as_dict(), "device_uuid": "GPU-second-worker"}
    )

    assert first.fingerprint_id != second.fingerprint_id
    assert first.compatibility_id == second.compatibility_id
    assert first.compatible_with(second)


def test_environment_compatibility_allows_unqueried_optional_library():
    first = _environment("sm80")
    second = EnvironmentFingerprint(
        **{
            **first.as_dict(),
            "device_uuid": "GPU-second-worker",
            "library_versions": {"cudnn": "9.14"},
        }
    )

    assert first.compatibility_id == second.compatibility_id
    assert first.compatible_with(second)


def test_environment_compatibility_rejects_conflicting_recorded_library():
    first = _environment("sm80")
    second = EnvironmentFingerprint(
        **{
            **first.as_dict(),
            "device_uuid": "GPU-second-worker",
            "library_versions": {"cudnn": "9.15"},
        }
    )

    assert not first.compatible_with(second)


def _timing(latency_ms: float, *, allocations: int = 0) -> TimingEvidence:
    return TimingEvidence(
        cold_ms=latency_ms * 1.2,
        p10_ms=latency_ms * 0.98,
        p50_ms=latency_ms,
        p90_ms=latency_ms * 1.02,
        variance_ms2=latency_ms * latency_ms * 1.0e-4,
        process_count=3,
        sample_count=27,
        timed_allocation_count=allocations,
        confidence=0.97,
    )


def _correctness(*, passed: bool = True) -> CorrectnessEvidence:
    return CorrectnessEvidence(
        passed=passed,
        reference="fp32_dense_reference",
        checked_cases=4,
        max_abs_error=2.0e-4 if passed else 1.0,
        max_relative_error=3.0e-4 if passed else 1.0,
        failure_reason=None if passed else "numerical_mismatch",
    )


def _external(
    cell: ExactWorkloadCell,
    backend: str,
    latency_ms: float,
    *,
    suffix: str = "ok",
    passed: bool = True,
    allocations: int = 0,
) -> BackendEvidence:
    return BackendEvidence(
        evidence_id=f"{cell.cell_id}:external:{backend}:{suffix}",
        cell_id=cell.cell_id,
        provider="external",
        requested_backend=backend,
        resolved_backend=f"resolved_{backend}",
        status=MeasurementStatus.MEASURED,
        native=False,
        environment=_environment(cell.architecture),
        workspace_bytes=4096,
        supported_range={"cell_ids": [cell.cell_id]},
        timing=_timing(latency_ms, allocations=allocations),
        correctness=_correctness(passed=passed),
    )


def _unsupported(cell: ExactWorkloadCell, backend: str) -> BackendEvidence:
    return BackendEvidence(
        evidence_id=f"{cell.cell_id}:external:{backend}:unsupported",
        cell_id=cell.cell_id,
        provider="external",
        requested_backend=backend,
        resolved_backend=None,
        status=MeasurementStatus.UNSUPPORTED,
        native=False,
        environment=_environment(cell.architecture),
        detail="backend rejected the semantic cell",
    )


def _streamattn(
    cell: ExactWorkloadCell,
    family_id: str,
    latency_ms: float,
    *,
    suffix: str = "default",
    allocations: int = 0,
) -> BackendEvidence:
    return BackendEvidence(
        evidence_id=f"{cell.cell_id}:streamattn:{family_id}:{suffix}",
        cell_id=cell.cell_id,
        provider="streamattn",
        requested_backend=family_id,
        resolved_backend=family_id,
        status=MeasurementStatus.MEASURED,
        native=True,
        environment=_environment(cell.architecture),
        family_id=family_id,
        kernel_key=f"{cell.architecture}:{family_id}:{suffix}",
        workspace_bytes=8192,
        supported_range={"cell_ids": [cell.cell_id]},
        timing=_timing(latency_ms, allocations=allocations),
        correctness=_correctness(),
    )


def _complete_evidence() -> list[BackendEvidence]:
    manifest = load_universal_exact_manifest()
    families = registered_exact_kernel_families()
    rows: list[BackendEvidence] = []
    for cell in manifest.cells:
        for index, backend in enumerate(cell.baseline_candidates):
            if index == len(cell.baseline_candidates) - 1 and cell.surface == "boundary":
                rows.append(_unsupported(cell, backend))
            else:
                rows.append(_external(cell, backend, 1.0 + 0.1 * index))
        native = matching_kernel_families(cell, families, native_only=True)
        if native:
            rows.append(_streamattn(cell, native[0].family_id, 0.9))
    return rows


def test_fastest_baseline_uses_resolved_correct_backend_and_ignores_invalid_speed():
    cell = load_universal_exact_manifest().cells[0]
    evidence = [
        _external(cell, cell.baseline_candidates[0], 0.2, suffix="wrong", passed=False),
        _external(cell, cell.baseline_candidates[0], 0.5, suffix="alloc", allocations=1),
        _external(cell, cell.baseline_candidates[1], 0.8),
        _external(cell, cell.baseline_candidates[2], 1.0),
    ]

    winner = resolve_fastest_correct_baseline(cell, evidence)

    assert winner is not None
    assert winner.requested_backend == cell.baseline_candidates[1]
    assert winner.resolved_backend == f"resolved_{cell.baseline_candidates[1]}"


def test_allocating_baseline_is_routable_when_no_fixed_buffer_candidate_exists():
    cell = load_universal_exact_manifest().cells[0]
    allocating = _external(
        cell,
        cell.baseline_candidates[0],
        0.5,
        suffix="alloc-only",
        allocations=1,
    )

    winner = resolve_fastest_correct_baseline(cell, [allocating])

    assert winner is allocating
    assert winner.is_routable
    assert not winner.is_usable


def test_equal_latency_baselines_resolve_deterministically_by_evidence_id():
    cell = load_universal_exact_manifest().cells[0]
    rows = [
        _external(cell, cell.baseline_candidates[1], 0.8, suffix="z"),
        _external(cell, cell.baseline_candidates[0], 0.8, suffix="a"),
    ]

    winner = resolve_fastest_correct_baseline(cell, tuple(reversed(rows)))

    assert winner is not None
    assert winner.evidence_id == min(row.evidence_id for row in rows)


def test_phase_database_requires_an_outcome_for_every_eligible_baseline():
    manifest = load_universal_exact_manifest()
    cell = next(cell for cell in manifest.cells if cell.architecture == "sm80")
    evidence = [
        _external(cell, cell.baseline_candidates[0], 1.0),
        _streamattn(
            cell,
            matching_kernel_families(
                cell, registered_exact_kernel_families(), native_only=True
            )[0].family_id,
            0.9,
        ),
    ]
    for other in (row for row in _complete_evidence() if row.cell_id != cell.cell_id):
        evidence.append(other)

    database = compile_phase_database(
        manifest,
        evidence,
        architecture="sm80",
        source_commit="a" * 40,
    )
    entry = next(row for row in database.entries if row.cell_id == cell.cell_id)

    assert entry.status is PhaseEntryStatus.INCOMPLETE_BASELINES
    assert entry.missing_baselines == cell.baseline_candidates[1:]
    assert database.acceptance["telemetry_coverage"] < 1.0
    assert database.acceptance["compiler_v1_pass"] is False


def test_phase_database_retains_losses_and_quantifies_explicit_routing_regret():
    manifest = load_universal_exact_manifest()
    evidence = _complete_evidence()
    cell = next(
        cell
        for cell in manifest.cells
        if cell.architecture == "sm90"
        and matching_kernel_families(
            cell, registered_exact_kernel_families(), native_only=True
        )
    )
    native_family = matching_kernel_families(
        cell, registered_exact_kernel_families(), native_only=True
    )[0].family_id
    slower = _streamattn(cell, native_family, 1.2, suffix="slower")
    evidence.append(slower)
    selected = {cell.cell_id: slower.evidence_id}

    database = compile_phase_database(
        manifest,
        evidence,
        architecture="sm90",
        source_commit="b" * 40,
        selected_evidence_ids=selected,
    )
    entry = next(row for row in database.entries if row.cell_id == cell.cell_id)

    assert entry.status is PhaseEntryStatus.NATIVE
    assert entry.routing_regret == pytest.approx(1.2 / 0.9 - 1.0)
    assert entry.speedup_vs_baseline == pytest.approx(1.0 / 1.2)
    assert cell.cell_id in database.acceptance["negative_cells"]
    assert slower.evidence_id in entry.evidence_ids


def test_explicit_selection_can_pin_a_conservative_external_fallback():
    manifest = load_universal_exact_manifest()
    evidence = _complete_evidence()
    cell = next(cell for cell in manifest.cells if cell.architecture == "sm80")
    baseline = next(
        row
        for row in evidence
        if row.cell_id == cell.cell_id
        and row.provider == "external"
        and row.is_routable
    )

    database = compile_phase_database(
        manifest,
        evidence,
        architecture="sm80",
        source_commit="e" * 40,
        selected_evidence_ids={cell.cell_id: baseline.evidence_id},
    )
    entry = database.entry_for(cell.cell_id)

    assert entry.status is PhaseEntryStatus.EXTERNAL_FALLBACK
    assert entry.selected_evidence_id == baseline.evidence_id
    assert entry.selected_family_id is None
    assert entry.kernel_key is None


def test_explicit_selection_rejects_non_routable_evidence():
    manifest = load_universal_exact_manifest()
    evidence = _complete_evidence()
    cell = next(cell for cell in manifest.cells if cell.architecture == "sm80")

    with pytest.raises(ValueError, match="is not routable"):
        compile_phase_database(
            manifest,
            evidence,
            architecture="sm80",
            source_commit="f" * 40,
            selected_evidence_ids={cell.cell_id: "missing-evidence"},
        )


def test_default_route_falls_back_when_fastest_native_loses():
    manifest = load_universal_exact_manifest()
    evidence = _complete_evidence()
    cell = next(
        cell
        for cell in manifest.cells
        if cell.architecture == "sm90"
        and matching_kernel_families(
            cell, registered_exact_kernel_families(), native_only=True
        )
    )
    evidence = [
        row
        for row in evidence
        if not (row.cell_id == cell.cell_id and row.provider == "streamattn")
    ]
    family = matching_kernel_families(
        cell, registered_exact_kernel_families(), native_only=True
    )[0]
    losing_native = _streamattn(cell, family.family_id, 1.2, suffix="losing")
    evidence.append(losing_native)

    database = compile_phase_database(
        manifest,
        evidence,
        architecture="sm90",
        source_commit="2" * 40,
    )
    entry = database.entry_for(cell.cell_id)

    assert entry.status is PhaseEntryStatus.EXTERNAL_FALLBACK
    assert entry.selected_evidence_id == entry.baseline_evidence_id
    assert entry.speedup_vs_baseline == 1.0
    assert entry.routing_regret == 0.0
    assert cell.cell_id in database.acceptance["negative_cells"]
    assert losing_native.evidence_id in entry.evidence_ids


def test_native_capability_gap_is_an_explicit_external_fallback():
    manifest = load_universal_exact_manifest()
    database = compile_phase_database(
        manifest,
        _complete_evidence(),
        architecture="sm80",
        source_commit="c" * 40,
    )
    entry = database.entry_for("sm80_train_dropout_mha_d128")

    assert entry.status is PhaseEntryStatus.EXTERNAL_FALLBACK
    assert entry.selected_evidence_id == entry.baseline_evidence_id
    assert entry.kernel_key is None
    assert database.acceptance["native_coverage"] < 1.0


def test_complete_phase_database_round_trips_without_dropping_negative_evidence(tmp_path):
    manifest = load_universal_exact_manifest()
    evidence = _complete_evidence()
    cell = next(
        cell
        for cell in manifest.cells
        if cell.architecture == "sm100"
        and matching_kernel_families(
            cell, registered_exact_kernel_families(), native_only=True
        )
    )
    family = matching_kernel_families(
        cell, registered_exact_kernel_families(), native_only=True
    )[0]
    discarded = _streamattn(
        cell,
        family.family_id,
        0.1,
        suffix="allocating_discarded",
        allocations=1,
    )
    evidence.append(discarded)
    database = compile_phase_database(
        manifest,
        evidence,
        architecture="sm100",
        source_commit="d" * 40,
    )
    evidence_path = tmp_path / "evidence.json"
    database_path = tmp_path / "sm100.json"

    write_backend_evidence(evidence, evidence_path)
    database.write_json(database_path)
    loaded_evidence = load_backend_evidence(evidence_path)
    loaded_database = load_phase_database(database_path)

    assert len(loaded_evidence) == len(evidence)
    assert loaded_database.as_dict() == database.as_dict()
    assert loaded_database.acceptance["compiler_v1_pass"] is True
    assert len(loaded_database.entries) == 12
    assert discarded.evidence_id in {
        row.evidence_id for row in loaded_database.evidence
    }


def test_phase_database_rejects_incompatible_environments():
    manifest = load_universal_exact_manifest()
    evidence = _complete_evidence()
    row_index = next(
        index
        for index, row in enumerate(evidence)
        if row.environment.architecture == "sm90"
    )
    evidence[row_index] = replace(
        evidence[row_index],
        environment=replace(evidence[row_index].environment, driver_version="other"),
    )

    with pytest.raises(ValueError, match="mix incompatible environments"):
        compile_phase_database(
            manifest,
            evidence,
            architecture="sm90",
            source_commit="e" * 40,
        )


def test_phase_database_loader_rejects_missing_referenced_evidence():
    manifest = load_universal_exact_manifest()
    database = compile_phase_database(
        manifest,
        _complete_evidence(),
        architecture="sm100",
        source_commit="f" * 40,
    )
    payload = database.as_dict()
    selected_id = payload["entries"][0]["selected_evidence_id"]
    payload["evidence"] = [
        row for row in payload["evidence"] if row["evidence_id"] != selected_id
    ]

    with pytest.raises(ValueError, match="missing evidence"):
        PhaseDatabase.from_dict(payload)


def test_phase_database_cli_emits_all_architectures_and_integrity_index(tmp_path):
    root = Path(__file__).resolve().parents[1]
    evidence_path = tmp_path / "evidence.json"
    output_dir = tmp_path / "phase_db"
    write_backend_evidence(_complete_evidence(), evidence_path)

    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "compile_universal_exact_phase_db.py"),
            str(evidence_path),
            "--output-dir",
            str(output_dir),
            "--source-commit",
            "0" * 40,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert {path.name for path in output_dir.iterdir()} == {
        "index.json",
        "sm80.json",
        "sm90.json",
        "sm100.json",
    }
    for architecture in ("sm80", "sm90", "sm100"):
        database = load_phase_database(output_dir / f"{architecture}.json")
        assert database.architecture == architecture
        assert database.acceptance["compiler_v1_pass"] is True


def test_phase_database_cli_can_compile_a_partial_architecture_campaign(tmp_path):
    root = Path(__file__).resolve().parents[1]
    evidence_path = tmp_path / "evidence.json"
    output_dir = tmp_path / "phase_db"
    write_backend_evidence(_complete_evidence(), evidence_path)

    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "compile_universal_exact_phase_db.py"),
            str(evidence_path),
            "--architectures",
            "sm90",
            "sm100",
            "--output-dir",
            str(output_dir),
            "--source-commit",
            "1" * 40,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert {path.name for path in output_dir.iterdir()} == {
        "index.json",
        "sm90.json",
        "sm100.json",
    }
    index = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    assert [row["architecture"] for row in index["databases"]] == ["sm90", "sm100"]
