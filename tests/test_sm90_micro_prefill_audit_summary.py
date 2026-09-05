from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

from benchmarks.micro_prefill_baselines import BASELINE_IDS, resolve_measurements
from benchmarks.summarize_sm90_micro_prefill_audit import summarize


def _metrics(passed=True, *, mutation=False):
    metrics = dict(
        finite=True,
        max_abs=0.001 if passed else 0.1,
        relative_l2=0.001 if passed else 0.1,
        passed=passed,
    )
    if mutation:
        metrics["allocated_bytes_delta"] = 0
    return metrics


def _provenance():
    packages = {
        name: dict(
            version="1.0", record_sha256="a" * 64, interfaces={"api.py": "b" * 64}
        )
        for name in (
            "torch",
            "flashinfer-python",
            "flashinfer-cubin",
            "flash_attn_3",
            "xformers",
        )
    }
    dependencies = {
        "torch_flash": ("torch",),
        "cudnn": ("torch",),
        "flashinfer_fa2": ("flashinfer-python", "flashinfer-cubin"),
        "flashinfer_fa3": ("flashinfer-python", "flashinfer-cubin"),
        "flashattention3": ("flash_attn_3",),
        "cutlass_xformers": ("xformers", "torch"),
    }
    versions = {}
    for name, deps in dependencies.items():
        digest = hashlib.sha256(
            json.dumps({p: packages[p] for p in deps}, sort_keys=True).encode()
        ).hexdigest()
        versions[name] = f"1.0;installation={digest}"
    return dict(
        packages=packages,
        package_versions=dict.fromkeys(BASELINE_IDS, "1.0"),
        versions=versions,
        adapter_sha256="c" * 64,
    )


def _environment(result):
    result["environment_sha256"] = hashlib.sha256(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "torch",
                    "cuda",
                    "device",
                    "gpu_inventory",
                    "baseline_provenance",
                    "protocol_sha256",
                    "source_sha256",
                    "timing",
                )
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _resolve(result):
    _environment(result)
    for row in result["rows"]:
        row["case_sha256"] = hashlib.sha256(
            json.dumps(row["case"], sort_keys=True).encode()
        ).hexdigest()
        row["median_ms"] = {
            name: statistics.median(t[name] for t in row["trials_ms"])
            for name in row["trials_ms"][0]
        }
        resolution = resolve_measurements(
            row["case"],
            result["baseline_provenance"]["versions"],
            result["environment_sha256"],
            row["median_ms"],
            row["accuracy"],
            row["mutation"],
            requested=result["requested_baselines"],
            loaded_binary_provenance=row.get("loaded_binary_provenance"),
        )
        row["baseline_resolution"] = resolution
        row["baseline_set_complete"] = resolution["complete"]
        winner = resolution["winner"]
        row["fastest_measured_baseline"] = winner["baseline_id"] if winner else None
        natives = [n for n in ("natural", "transposed") if n in row["median_ms"]]
        row["oracle_native_winner"] = (
            min(natives, key=row["median_ms"].get) if natives else None
        )
        row["oracle_speedup"] = (
            row["median_ms"][winner["baseline_id"]]
            / row["median_ms"][row["oracle_native_winner"]]
            if row["exact_pass"] and winner and natives
            else None
        )
    return result


def fixture_result():
    measured = ("torch_flash", "flashinfer_fa3", "natural", "transposed")
    result = dict(
        schema="streamattn.sm90_micro_prefill_audit.v2",
        provider="test",
        cohort="smoke",
        torch="1.0",
        cuda="12.8",
        device="test H100",
        gpu_inventory="test inventory",
        source_sha256="d" * 64,
        protocol_sha256="e" * 64,
        timing=dict(warmup=4, iterations=40, repeats=3),
        requested_baselines=list(BASELINE_IDS),
        baseline_provenance=_provenance(),
        complete=True,
        rows=[
            dict(
                case=dict(
                    batch=1,
                    m=64,
                    n=4096,
                    hq=16,
                    g=8,
                    d=128,
                    splits=None,
                    purpose="boundary",
                ),
                exact_pass=True,
                trials_ms=[
                    dict(
                        torch_flash=t,
                        flashinfer_fa3=t + 2,
                        natural=8.0,
                        transposed=9.0,
                        natural_producer=7.2,
                        natural_merge=0.8,
                    )
                    for t in (10.0, 7.0, 13.0)
                ],
                natural_producer_fraction=0.9,
                unavailable_baselines={
                    name: "not installed"
                    for name in BASELINE_IDS
                    if name not in measured
                },
                accuracy={name: _metrics() for name in measured},
                mutation={name: _metrics(mutation=True) for name in measured},
                natural_lse_max_abs=0.001,
                composed_correct=_metrics(),
            )
        ],
    )
    return json.loads(json.dumps(_resolve(result)))


def test_oracle_average_is_not_a_paired_win_or_promotion():
    result = summarize(fixture_result())
    row = result["groups"][0]
    assert row["oracle_geomean"] == pytest.approx(1.25)
    assert row["paired_wins"] == 0
    assert row["worst_paired_ratio"] == pytest.approx(0.875)
    assert row["complete_baseline_cells"] == 0
    assert result["baseline_failures"] == {
        name: {"not installed": 1}
        for name in BASELINE_IDS
        if name not in ("torch_flash", "flashinfer_fa3")
    }
    assert result["source_sha256"] == "d" * 64
    assert result["protocol_sha256"] == "e" * 64
    assert (
        result["evidence_kind"] == "calibration_oracle_not_holdout_or_public_promotion"
    )


@pytest.mark.parametrize("failure", ["exact", "baseline"])
def test_unresolved_or_incorrect_rows_are_not_speedup_evidence(failure):
    result = fixture_result()
    row = result["rows"][0]
    if failure == "exact":
        row["mutation"]["natural"] = _metrics(False, mutation=True)
        row["exact_pass"] = False
    else:
        for name in ("torch_flash", "flashinfer_fa3"):
            row["mutation"][name] = _metrics(False, mutation=True)
    _resolve(result)
    summary = summarize(result)["groups"][0]
    assert summary["cells"] == 1
    assert summary["resolved_cells"] == 0
    assert summary["oracle_geomean"] is None
    assert summary["worst_paired_ratio"] is None


def test_prepared_layout_v1_cannot_be_silently_combined():
    result = fixture_result()
    result["schema"] = "streamattn.sm90_micro_prefill_audit.v1"
    with pytest.raises(ValueError, match="direct-layout"):
        summarize(result)


@pytest.mark.parametrize(
    "field", ["environment_sha256", "source_sha256", "protocol_sha256"]
)
@pytest.mark.parametrize("value", [None, "g" * 64, "a" * 63, "a" * 64])
def test_artifact_identities_are_valid_and_bound_to_environment(field, value):
    result = fixture_result()
    result[field] = value
    with pytest.raises(ValueError, match="digest|environment_sha256"):
        summarize(result)


@pytest.mark.parametrize("location", ["row", "resolution"])
@pytest.mark.parametrize(
    "field", ["environment_sha256", "source_sha256", "protocol_sha256"]
)
def test_nested_identity_cannot_disagree_with_artifact(location, field):
    result = fixture_result()
    record = result["rows"][0]
    if location == "resolution":
        record = record["baseline_resolution"]
    record[field] = "f" * 64
    with pytest.raises(ValueError, match="does not match the artifact"):
        summarize(result)


@pytest.mark.parametrize("field", ["warmup", "iterations", "repeats"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_timing_settings_are_rejected(field, value):
    result = fixture_result()
    result["timing"][field] = value
    _environment(result)
    with pytest.raises(ValueError, match=f"timing.{field}"):
        summarize(result)


def test_trial_count_must_match_recorded_repeats():
    result = fixture_result()
    result["rows"][0]["trials_ms"].append(deepcopy(result["rows"][0]["trials_ms"][0]))
    with pytest.raises(ValueError, match="trial count"):
        summarize(result)


@pytest.mark.parametrize(
    "field", ["case", "case_sha256", "workload", "workload_sha256", "schema"]
)
def test_workload_identity_must_match_case_and_schema(field):
    result = fixture_result()
    row = result["rows"][0]
    if field == "case":
        row["case"]["m"] = 32
    elif field == "case_sha256":
        row[field] = "a" * 64
    elif field == "workload":
        row["baseline_resolution"][field]["cache_layout"] = "bshd"
    elif field == "schema":
        row["baseline_resolution"]["workload"]["schema_version"] = 1
    else:
        row["baseline_resolution"][field] = "a" * 64
    with pytest.raises(ValueError, match="case|workload"):
        summarize(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment_sha256", "f" * 64),
        ("workload_sha256", "f" * 64),
        ("backend_revision", "wrong-revision"),
        ("baseline_id", "unknown"),
        ("latency_us", 1.0),
        ("correctness_passed", False),
        ("graph_replay", False),
        ("correctness_passed", 1),
        ("graph_replay", "true"),
    ],
)
@pytest.mark.parametrize(
    "location", ["winner", "winning_measurement", "other_measurement"]
)
def test_every_full_measurement_must_match_recorded_evidence(field, value, location):
    result = fixture_result()
    resolution = result["rows"][0]["baseline_resolution"]
    record = (
        resolution["winner"]
        if location == "winner"
        else resolution["measurements"][location == "other_measurement"]
    )
    record[field] = value
    with pytest.raises(ValueError):
        summarize(result)


@pytest.mark.parametrize(
    "damage", ["duplicate", "missing", "missing_field", "extra_field"]
)
def test_measurement_schema_and_coverage_are_exact(damage):
    result = fixture_result()
    measurements = result["rows"][0]["baseline_resolution"]["measurements"]
    if damage == "duplicate":
        measurements.append(deepcopy(measurements[0]))
    elif damage == "missing":
        measurements.pop()
    elif damage == "missing_field":
        del measurements[0]["backend_revision"]
    else:
        measurements[0]["unexpected"] = True
    with pytest.raises(ValueError):
        summarize(result)


@pytest.mark.parametrize(
    "location", ["trial", "median", "measurement", "speedup", "fraction"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_numeric_evidence_is_rejected(location, value):
    result = fixture_result()
    row = result["rows"][0]
    if location == "trial":
        row["trials_ms"][0]["torch_flash"] = value
    elif location == "median":
        row["median_ms"]["torch_flash"] = value
    elif location == "measurement":
        row["baseline_resolution"]["measurements"][0]["latency_us"] = value
    else:
        row[
            "oracle_speedup" if location == "speedup" else "natural_producer_fraction"
        ] = value
    with pytest.raises(ValueError):
        summarize(result)


@pytest.mark.parametrize(
    "damage", ["empty", "missing_backend", "extra_backend", "median"]
)
def test_trials_and_medians_must_agree(damage):
    result = fixture_result()
    row = result["rows"][0]
    if damage == "empty":
        row["trials_ms"] = []
    elif damage == "missing_backend":
        del row["trials_ms"][0]["torch_flash"]
    elif damage == "extra_backend":
        row["trials_ms"][0]["unknown"] = 1.0
    else:
        row["median_ms"]["torch_flash"] = 100.0
    with pytest.raises(ValueError, match="trial|median"):
        summarize(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fastest_measured_baseline", "flashinfer_fa3"),
        ("oracle_native_winner", "transposed"),
        ("oracle_speedup", 2.0),
        ("oracle_speedup", None),
        ("natural_producer_fraction", 0.5),
        ("baseline_set_complete", True),
        ("exact_pass", False),
    ],
)
def test_winners_ratios_and_flags_cannot_be_overridden(field, value):
    result = fixture_result()
    result["rows"][0][field] = value
    with pytest.raises(ValueError, match=field):
        summarize(result)


@pytest.mark.parametrize("damage", ["none", "slower", "eligibility", "complete"])
def test_resolver_claims_are_recomputed(damage):
    result = fixture_result()
    resolution = result["rows"][0]["baseline_resolution"]
    if damage == "none":
        resolution["winner"] = None
    elif damage == "slower":
        resolution["winner"] = deepcopy(resolution["measurements"][1])
    elif damage == "eligibility":
        resolution["eligibility"][0]["eligible"] = False
    else:
        resolution["complete"] = True
    with pytest.raises(ValueError, match="resolver|eligibility|complete"):
        summarize(result)


@pytest.mark.parametrize(
    "backend", ["torch_flash", "flashinfer_fa3", "natural", "transposed"]
)
@pytest.mark.parametrize(
    "damage", ["missing", "failed", "lying_flag", "nonfinite", "allocation_type"]
)
def test_mutation_evidence_cannot_be_missing_or_forged(backend, damage):
    result = fixture_result()
    mutation = result["rows"][0]["mutation"]
    if damage == "missing":
        del mutation[backend]
    elif damage == "failed":
        mutation[backend] = _metrics(False, mutation=True)
    elif damage == "lying_flag":
        mutation[backend]["max_abs"] = 0.1
    elif damage == "nonfinite":
        mutation[backend]["relative_l2"] = float("nan")
    else:
        mutation[backend]["allocated_bytes_delta"] = True
    with pytest.raises(ValueError):
        summarize(result)


@pytest.mark.parametrize("damage", ["accuracy", "allocation", "lse", "composed"])
def test_native_exactness_requires_all_evidence(damage):
    result = fixture_result()
    row = result["rows"][0]
    if damage == "accuracy":
        row["accuracy"]["natural"] = _metrics(False)
    elif damage == "allocation":
        row["mutation"]["natural"]["allocated_bytes_delta"] = 32
    elif damage == "lse":
        row["natural_lse_max_abs"] = 0.1
    else:
        row["composed_correct"] = _metrics(False)
    with pytest.raises(ValueError):
        summarize(result)


def test_mutation_failed_fastest_baseline_is_excluded_when_evidence_is_consistent():
    result = fixture_result()
    result["rows"][0]["mutation"]["torch_flash"] = _metrics(False, mutation=True)
    _resolve(result)
    summary = summarize(result)
    assert summary["groups"][0]["oracle_geomean"] == pytest.approx(1.5)
    assert summary["baseline_failures"]["torch_flash"] == {"mutation_failed": 1}


def test_unresolved_installed_revision_can_have_timings_but_cannot_win():
    result = fixture_result()
    provenance = result["baseline_provenance"]
    provenance["packages"]["flashinfer-cubin"]["record_sha256"] = None
    for name in ("flashinfer_fa2", "flashinfer_fa3"):
        provenance["versions"][name] = "unresolved"
    _resolve(result)
    assert summarize(result)["groups"][0]["resolved_cells"] == 1
    assert result["rows"][0]["fastest_measured_baseline"] == "torch_flash"
    assert len(result["rows"][0]["baseline_resolution"]["measurements"]) == 1


def test_unknown_revision_is_not_relabelled_as_measured_provenance():
    result = fixture_result()
    result["baseline_provenance"]["versions"]["torch_flash"] = "v1"
    _environment(result)
    with pytest.raises(ValueError, match="installation identity"):
        summarize(result)


@pytest.mark.parametrize(
    "damage",
    ["duplicate_case", "duplicate_baseline", "missing_resolution", "empty_rows"],
)
def test_missing_or_duplicate_evidence_is_rejected(damage):
    result = fixture_result()
    if damage == "duplicate_case":
        result["rows"].append(deepcopy(result["rows"][0]))
    elif damage == "duplicate_baseline":
        result["requested_baselines"].append(result["requested_baselines"][0])
    elif damage == "missing_resolution":
        del result["rows"][0]["baseline_resolution"]
    else:
        result["rows"] = []
    with pytest.raises(ValueError):
        summarize(result)


def test_summary_does_not_modify_input_and_accepts_partial_runs():
    result = fixture_result()
    result["complete"] = False
    before = deepcopy(result)
    assert not summarize(result)["complete"]
    assert result == before


def test_complete_baseline_set_is_derived_from_all_six_measurements():
    result = fixture_result()
    row = result["rows"][0]
    for name in row["unavailable_baselines"]:
        for trial in row["trials_ms"]:
            trial[name] = 20.0
        row["accuracy"][name] = _metrics()
        row["mutation"][name] = _metrics(mutation=True)
    row["unavailable_baselines"] = {}
    _resolve(result)
    assert summarize(result)["groups"][0]["complete_baseline_cells"] == 1


def test_reported_nonfinite_output_remains_valid_failure_evidence():
    result = fixture_result()
    row = result["rows"][0]
    row["mutation"]["natural"] = dict(
        finite=False,
        passed=False,
        max_abs=None,
        relative_l2=None,
        allocated_bytes_delta=0,
    )
    row["exact_pass"] = False
    row["oracle_speedup"] = None
    summary = summarize(result)
    assert summary["groups"][0]["resolved_cells"] == 0
    assert summary["baseline_failures"]["natural"] == {"mutation_failed": 1}


def test_incorrect_uncaptured_native_remains_valid_diagnostic_row():
    result = fixture_result()
    row = result["rows"][0]
    row["accuracy"]["natural"] = _metrics(False)
    del row["mutation"]["natural"]
    for trial in row["trials_ms"]:
        del trial["natural"]
    row["unavailable_baselines"]["natural"] = "correctness_failed"
    row["exact_pass"] = False
    row["natural_producer_fraction"] = 0.0
    _resolve(result)
    assert summarize(result)["groups"][0]["resolved_cells"] == 0
    assert row["oracle_native_winner"] == "transposed"


def test_paired_ratio_must_remain_finite_even_with_finite_trials():
    result = fixture_result()
    row = result["rows"][0]
    row["trials_ms"][0]["torch_flash"] = 1e300
    row["trials_ms"][2]["torch_flash"] = 10.0
    row["trials_ms"][0]["natural"] = 1e-300
    _resolve(result)
    with pytest.raises(ValueError, match="paired ratio"):
        summarize(result)


@pytest.mark.parametrize(
    "field", ["accuracy", "mutation", "unavailable_baselines", "baseline_resolution"]
)
def test_malformed_evidence_mappings_raise_validation_errors(field):
    result = fixture_result()
    result["rows"][0][field] = []
    with pytest.raises(ValueError, match="mapping"):
        summarize(result)


def test_equal_measurement_latencies_use_resolver_tie_order():
    result = fixture_result()
    row = result["rows"][0]
    for trial in row["trials_ms"]:
        trial["flashinfer_fa3"] = trial["torch_flash"]
        trial["transposed"] = trial["natural"]
    _resolve(result)
    assert row["fastest_measured_baseline"] == "torch_flash"
    assert row["oracle_native_winner"] == "natural"
    assert summarize(result)["groups"][0]["resolved_cells"] == 1


def test_cli_validates_json_artifact_from_outside_repository(tmp_path: Path):
    artifact = tmp_path / "audit.json"
    artifact.write_text(json.dumps(fixture_result()), encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/summarize_sm90_micro_prefill_audit.py"
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(script), str(artifact)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "64 128 1 1 1 0 0 1.25" in completed.stdout
