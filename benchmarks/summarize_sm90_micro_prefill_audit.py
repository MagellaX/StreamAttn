"""Summarize direct exact baseline resolution without treating canaries as promotion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.micro_prefill_baselines import (  # noqa: E402
    BASELINE_IDS,
    requested_baselines,
    resolve_measurements,
)
from stream_attention.baseline_resolver import ExactBaselineMeasurement  # noqa: E402


IDENTITIES = ("environment_sha256", "source_sha256", "protocol_sha256")
NATIVE_IDS = ("natural", "transposed")
COMPONENT_IDS = ("natural_producer", "natural_merge")
ENVIRONMENT_FIELDS = (
    "torch",
    "cuda",
    "device",
    "gpu_inventory",
    "baseline_provenance",
    "protocol_sha256",
    "source_sha256",
    "timing",
)
PACKAGE_DEPENDENCIES = {
    "torch_flash": ("torch",),
    "cudnn": ("torch",),
    "flashinfer_fa2": ("flashinfer-python", "flashinfer-cubin"),
    "flashinfer_fa3": ("flashinfer-python", "flashinfer-cubin"),
    "flashattention3": ("flash_attn_3",),
    "cutlass_xformers": ("xformers", "torch"),
}


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{label} must be a SHA-256 hexadecimal digest")
    return value.lower()


def _number(value, label, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be finite and {bound}")
    return value


def _boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _equal_number(actual, expected, label, *, positive=True):
    _number(actual, label, positive=positive)
    _number(expected, f"computed {label}", positive=positive)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(f"{label} disagrees with recorded evidence")


def _matching_identities(record, identities, label):
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a mapping")
    for name, expected in identities.items():
        if name in record and _digest(record[name], f"{label}.{name}") != expected:
            raise ValueError(f"{label}.{name} does not match the artifact")


def _error_metrics(metrics, label, *, mutation=False):
    if not isinstance(metrics, dict):
        raise ValueError(f"{label} must be a mapping")
    finite = _boolean(metrics["finite"], f"{label}.finite")
    passed = _boolean(metrics["passed"], f"{label}.passed")
    if finite:
        maximum = _number(metrics["max_abs"], f"{label}.max_abs")
        relative = _number(metrics["relative_l2"], f"{label}.relative_l2")
        expected = maximum <= 0.04 and relative <= 0.02
    else:
        if metrics["max_abs"] is not None or metrics["relative_l2"] is not None:
            raise ValueError(f"{label}: nonfinite output must have null error metrics")
        expected = False
    if passed != expected:
        raise ValueError(f"{label}.passed disagrees with error metrics")
    if mutation and type(metrics["allocated_bytes_delta"]) is not int:
        raise ValueError(f"{label}.allocated_bytes_delta must be an integer")


def _measurement(raw, label):
    measurement = ExactBaselineMeasurement(**raw)
    _number(measurement.latency_us, f"{label}.latency_us", positive=True)
    _boolean(measurement.correctness_passed, f"{label}.correctness_passed")
    _boolean(measurement.graph_replay, f"{label}.graph_replay")
    normalized = asdict(measurement)
    for name in ("environment_sha256", "workload_sha256"):
        normalized[name] = _digest(normalized[name], f"{label}.{name}")
    return normalized


def _validate_file_identity(entry):
    if not isinstance(entry, dict) or not isinstance(entry["path"], str) or not entry["path"]:
        raise ValueError("file identity needs an actual path")
    # Artifacts can come from either Linux workers or Windows tests.
    if not (entry["path"].startswith("/") or re.match(r"^[A-Za-z]:[/\\]", entry["path"])):
        raise ValueError("file identity path must be absolute")
    _digest(entry["sha256"], "file.sha256")
    if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
        raise ValueError("file size_bytes must be a nonnegative integer")


def _validate_loaded_provenance(evidence, requested):
    if not isinstance(evidence, dict) or set(evidence) != set(requested) | set(NATIVE_IDS):
        raise ValueError("loaded binary provenance must cover requested and native backends")
    for name, record in evidence.items():
        modules, binaries, errors = record["modules"], record["binaries"], record["errors"]
        if not isinstance(modules, dict) or not isinstance(binaries, list) or not isinstance(errors, list):
            raise ValueError("invalid loaded binary provenance collections")
        if any(not isinstance(error, str) or not error for error in errors):
            raise ValueError("invalid loaded binary provenance error")
        for entry in list(modules.values()) + binaries:
            _validate_file_identity(entry)
        paths = [entry["path"] for entry in binaries]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate loaded binary path")
        for entry in binaries:
            if not any(s in entry["path"].lower() for s in (".so", ".pyd", ".dll", ".cubin")):
                raise ValueError("loaded binary path is not a binary")
        if _boolean(record["resolved"], f"{name}.resolved") != bool(modules and binaries and not errors):
            raise ValueError("loaded binary resolved flag disagrees with evidence")


def _validate_row(row, versions, identities, requested=BASELINE_IDS, *, require_loaded=False):
    if not isinstance(row, dict):
        raise ValueError("row must be a mapping")
    _matching_identities(row, identities, "row")
    case_sha256 = hashlib.sha256(
        json.dumps(row["case"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if _digest(row["case_sha256"], "case_sha256") != case_sha256:
        raise ValueError("case_sha256 does not match case")

    medians, trials = row["median_ms"], row["trials_ms"]
    if not isinstance(medians, dict) or not medians:
        raise ValueError("median_ms must be a nonempty mapping")
    if not isinstance(trials, list) or not trials:
        raise ValueError("trials_ms must contain at least one trial")
    if set(medians) - set(tuple(requested) + NATIVE_IDS + COMPONENT_IDS):
        raise ValueError("median_ms contains an unknown backend")
    if not set(COMPONENT_IDS) <= set(medians):
        raise ValueError("component timings are missing")
    for trial in trials:
        if not isinstance(trial, dict) or set(trial) != set(medians):
            raise ValueError("trial backends do not match median_ms")
        for name, latency in trial.items():
            _number(latency, f"trials_ms.{name}", positive=True)
    for name, latency in medians.items():
        _equal_number(
            latency, statistics.median(t[name] for t in trials), f"median_ms.{name}"
        )

    accuracy, mutation = row["accuracy"], row["mutation"]
    if not isinstance(accuracy, dict) or not isinstance(mutation, dict):
        raise ValueError("accuracy and mutation must be mappings")
    runners = set(tuple(requested) + NATIVE_IDS)
    if set(accuracy) - runners or set(mutation) - (runners & set(medians)):
        raise ValueError("accuracy or mutation names do not match measured runners")
    for name, metrics in accuracy.items():
        _error_metrics(metrics, f"accuracy.{name}")
    for name, metrics in mutation.items():
        _error_metrics(metrics, f"mutation.{name}", mutation=True)
    unavailable = row["unavailable_baselines"]
    if not isinstance(unavailable, dict):
        raise ValueError("unavailable_baselines must be a mapping")
    if set(unavailable) - runners:
        raise ValueError("unavailable_baselines contains an unknown backend")
    for name, reason in unavailable.items():
        if not isinstance(reason, str) or not reason.strip() or name in medians:
            raise ValueError(
                f"unavailable_baselines.{name} contradicts measured evidence"
            )
    for name in runners & set(medians):
        if not accuracy.get(name, {}).get("passed", False) or name not in mutation:
            raise ValueError(
                f"{name} timing lacks passing accuracy or mutation evidence"
            )
    if set(requested) - set(medians) - set(unavailable):
        raise ValueError("missing baseline measurements must have unavailable reasons")

    loaded = row.get("loaded_binary_provenance")
    if require_loaded or loaded is not None:
        _validate_loaded_provenance(loaded, requested)

    # Reconstruct the producer's resolver result, not just its selected name.
    expected = resolve_measurements(
        row["case"],
        versions,
        identities["environment_sha256"],
        medians,
        accuracy,
        mutation,
        requested=requested,
        loaded_binary_provenance=loaded,
    )
    resolution = row["baseline_resolution"]
    _matching_identities(resolution, identities, "baseline_resolution")
    workload_sha256 = _digest(resolution["workload_sha256"], "workload_sha256")
    if workload_sha256 != expected["workload_sha256"]:
        raise ValueError("workload_sha256 does not match case")
    _matching_identities(row, {"workload_sha256": workload_sha256}, "row")
    if _json(resolution["workload"]) != _json(expected["workload"]):
        raise ValueError("resolver workload does not match case or workload schema")
    eligibility = resolution["eligibility"]
    for item in eligibility:
        _boolean(item["eligible"], "eligibility.eligible")
    if _json(eligibility) != _json(expected["eligibility"]):
        raise ValueError("resolver eligibility does not match workload and revisions")
    if not isinstance(resolution["measurements"], list):
        raise ValueError("resolver measurements must be a list")
    actual_measurements = {}
    for raw in resolution["measurements"]:
        measurement = _measurement(raw, "measurement")
        name = measurement["baseline_id"]
        if name in actual_measurements:
            raise ValueError("duplicate baseline measurement ID")
        actual_measurements[name] = measurement
    expected_measurements = {m["baseline_id"]: m for m in expected["measurements"]}
    if actual_measurements != expected_measurements:
        raise ValueError(
            "resolver measurements disagree with timings, identities or correctness"
        )
    winner = resolution["winner"]
    if winner is not None:
        winner = _measurement(winner, "winner")
    if winner != expected["winner"]:
        raise ValueError("resolver winner does not match eligible measurements")
    baseline = winner["baseline_id"] if winner else None
    if row["fastest_measured_baseline"] != baseline:
        raise ValueError("fastest_measured_baseline does not match resolver winner")
    for label, value in (
        ("baseline_set_complete", row["baseline_set_complete"]),
        ("baseline_resolution.complete", resolution["complete"]),
    ):
        if _boolean(value, label) != expected["complete"]:
            raise ValueError(f"{label} disagrees with resolver measurements")
    if "requested_baseline_set_complete" in row:
        complete = all(any(m["baseline_id"] == name and m["correctness_passed"]
                           for m in expected["measurements"]) for name in requested)
        if _boolean(row["requested_baseline_set_complete"], "requested_baseline_set_complete") != complete:
            raise ValueError("requested_baseline_set_complete disagrees with measurements")

    _error_metrics(row["composed_correct"], "composed_correct")
    lse = row["natural_lse_max_abs"]
    if lse is not None:
        _number(lse, "natural_lse_max_abs")
    exact = (
        all(accuracy.get(n, {}).get("passed", False) for n in NATIVE_IDS)
        and lse is not None
        and lse <= 0.02
        and row["composed_correct"]["passed"]
        and all(mutation.get(n, {}).get("passed", False) for n in NATIVE_IDS)
        and all(
            mutation.get(n, {}).get("allocated_bytes_delta") == 0 for n in NATIVE_IDS
        )
    )
    if _boolean(row["exact_pass"], "exact_pass") != exact:
        raise ValueError(
            "exact_pass disagrees with native correctness and mutation evidence"
        )
    native_names = [name for name in NATIVE_IDS if name in medians]
    native = min(native_names, key=medians.get) if native_names else None
    if row["oracle_native_winner"] != native:
        raise ValueError(
            "oracle_native_winner is not the fastest measured native backend"
        )
    if exact and baseline and native:
        _equal_number(
            row["oracle_speedup"], medians[baseline] / medians[native], "oracle_speedup"
        )
        for trial in trials:
            _number(trial[baseline] / trial[native], "paired ratio", positive=True)
    elif row["oracle_speedup"] is not None:
        raise ValueError("oracle_speedup must be null without exact, resolved evidence")
    fraction = (
        medians["natural_producer"] / medians["natural"]
        if "natural" in medians
        else 0.0
    )
    _equal_number(
        row["natural_producer_fraction"],
        fraction,
        "natural_producer_fraction",
        positive=False,
    )
    return case_sha256


def _validate_provenance(provenance):
    if not isinstance(provenance, dict):
        raise ValueError("baseline_provenance must be a mapping")
    versions, packages = provenance["versions"], provenance["packages"]
    package_versions = provenance["package_versions"]
    if not all(
        isinstance(value, dict) for value in (versions, packages, package_versions)
    ):
        raise ValueError("provenance versions and packages must be mappings")
    _digest(provenance["adapter_sha256"], "adapter_sha256")
    for name, package in packages.items():
        if not isinstance(package, dict) or not isinstance(package["interfaces"], dict):
            raise ValueError(f"invalid package identity: {name}")
        if not isinstance(package["version"], str) or not package["version"].strip():
            raise ValueError(f"invalid package version: {name}")
        if package["record_sha256"] is not None:
            _digest(package["record_sha256"], f"{name}.record_sha256")
        for path, digest in package["interfaces"].items():
            _digest(digest, f"{name}.interfaces.{path}")
    for name, dependencies in PACKAGE_DEPENDENCIES.items():
        version = package_versions[name]
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"invalid package_versions.{name}")
        if version == "not_installed" or any(
            not packages[p]["record_sha256"] for p in dependencies
        ):
            expected = "unresolved"
        else:
            digest = hashlib.sha256(
                _json({p: packages[p] for p in dependencies}).encode()
            ).hexdigest()
            expected = f"{version};installation={digest}"
        if versions[name] != expected:
            raise ValueError(
                f"backend revision does not match installation identity: {name}"
            )
    return versions


def _validate_result(result, *, allow_empty=False):
    if not isinstance(result, dict):
        raise ValueError("audit artifact must be a mapping")
    if result.get("schema") != "streamattn.sm90_micro_prefill_audit.v2":
        raise ValueError("expected direct-layout v2 audit; v1 excluded KV conversion")
    identities = {name: _digest(result[name], name) for name in IDENTITIES}
    if not isinstance(result["provider"], str) or not result["provider"].strip():
        raise ValueError("provider must be nonempty")
    _boolean(result.get("complete", False), "complete")
    requested = result["requested_baselines"]
    if (
        not isinstance(requested, list)
        or len(requested) != len(set(requested))
        or not requested
        or set(requested) - set(BASELINE_IDS)
    ):
        raise ValueError("requested_baselines must contain a unique nonempty audit subset")
    require_loaded = _boolean(result.get("loaded_binary_provenance_required", False),
                              "loaded_binary_provenance_required")
    versions = _validate_provenance(result["baseline_provenance"])
    timing = result["timing"]
    for name in ("warmup", "iterations", "repeats"):
        if type(timing[name]) is not int or timing[name] <= 0:
            raise ValueError(f"timing.{name} must be a positive integer")
    environment = hashlib.sha256(
        _json({name: result[name] for name in ENVIRONMENT_FIELDS}).encode()
    ).hexdigest()
    if environment != identities["environment_sha256"]:
        raise ValueError(
            "environment_sha256 does not match recorded provenance and timing"
        )
    if not isinstance(result["rows"], list) or (not result["rows"] and not allow_empty):
        raise ValueError("rows must be a nonempty list")
    seen = set()
    for index, row in enumerate(result["rows"]):
        try:
            case_sha256 = _validate_row(row, versions, identities, requested, require_loaded=require_loaded)
            if len(row["trials_ms"]) != timing["repeats"]:
                raise ValueError("trial count does not match timing.repeats")
            if case_sha256 in seen:
                raise ValueError("duplicate case identity")
            seen.add(case_sha256)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid audit row {index}: {exc}") from exc


def summarize(result):
    if isinstance(result, dict) and result.get("schema") == ISOLATED_SCHEMA:
        return summarize_isolated(result)
    try:
        _validate_result(result)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid audit artifact: {exc}") from exc
    rows = result["rows"]
    grouped = {}
    for row in rows:
        key = (row["case"]["m"], row["case"]["d"])
        grouped.setdefault(key, []).append(row)
    groups = []
    for (m, d), cells in sorted(grouped.items()):
        eligible = [
            r for r in cells if r["exact_pass"] and r["oracle_speedup"] is not None
        ]
        ratios = [r["oracle_speedup"] for r in eligible]
        pairs = []
        for row in eligible:
            base, native = row["fastest_measured_baseline"], row["oracle_native_winner"]
            pairs.append(min(t[base] / t[native] for t in row["trials_ms"]))
        groups.append(
            dict(
                m=m,
                d=d,
                cells=len(cells),
                exact_cells=sum(r["exact_pass"] for r in cells),
                resolved_cells=len(eligible),
                complete_baseline_cells=sum(r["baseline_set_complete"] for r in cells),
                oracle_geomean=(
                    math.exp(statistics.mean(map(math.log, ratios))) if ratios else None
                ),
                paired_wins=sum(r > 1.0 for r in pairs),
                worst_paired_ratio=min(pairs) if pairs else None,
                natural_producer_fraction_median=statistics.median(
                    r["natural_producer_fraction"] for r in cells
                ),
            )
        )
    failures = {}
    for row in rows:
        for name, reason in row["unavailable_baselines"].items():
            failures.setdefault(name, {})
            failures[name][reason] = failures[name].get(reason, 0) + 1
        for name, metrics in row.get("mutation", {}).items():
            if not metrics["passed"]:
                failures.setdefault(name, {})
                failures[name]["mutation_failed"] = (
                    failures[name].get("mutation_failed", 0) + 1
                )
    return dict(
        schema="streamattn.sm90_micro_prefill_audit_summary.v2",
        evidence_kind="calibration_oracle_not_holdout_or_public_promotion",
        provider=result["provider"],
        complete=result.get("complete", False),
        requested_baselines=result["requested_baselines"],
        environment_sha256=result["environment_sha256"],
        source_sha256=result["source_sha256"],
        protocol_sha256=result["protocol_sha256"],
        cells=len(rows),
        groups=groups,
        baseline_failures=failures,
        caveat="Per-cell family oracle; isolated stage ratios are diagnostics, not overlap bounds.",
    )


ISOLATED_SCHEMA = "streamattn.sm90_micro_prefill_isolated_audit.v1"


def validate_isolated_worker(worker, *, provider, cohort):
    """Validate retained v2 evidence, including partial runs, without comparing clocks."""
    from benchmarks.profile_sm90_micro_prefill_audit import cases

    baseline = worker["baseline_id"]
    requested_baselines([baseline])
    result = worker["result"]
    if result is None:
        return None
    if result.get("schema") != "streamattn.sm90_micro_prefill_audit.v2":
        raise ValueError("isolated workers must retain v2 artifacts")
    if (result["provider"] != provider or result["cohort"] != cohort
            or result["requested_baselines"] != [baseline]):
        raise ValueError("worker provider/cohort/requested subset mismatch")
    if result.get("loaded_binary_provenance_required") is not True:
        raise ValueError("isolated workers require loaded binary provenance")
    if not result.get("rows"):
        if result.get("complete") is not False:
            raise ValueError("empty worker cannot be complete")
        _validate_result(result, allow_empty=True)
        return None
    summary = summarize(result)
    expected = cases(cohort)
    actual = [r["case"] for r in result["rows"]]
    if actual != expected[:len(actual)] or (result["complete"] and actual != expected):
        raise ValueError("worker case coverage does not match cohort prefix")
    return summary


def summarize_isolated(result):
    try:
        requested = requested_baselines(result["requested_baselines"])
        workers = result["workers"]
        if not isinstance(workers, list):
            raise ValueError("workers must be a list")
        if [w["baseline_id"] for w in workers] != list(requested[:len(workers)]):
            raise ValueError("worker IDs must be a unique prefix of requested baselines")
        finished = _boolean(result["finished"], "finished")
        if finished and len(workers) != len(requested):
            raise ValueError("finished isolated audit is missing workers")
        summaries = []
        for worker in workers:
            for label in ("stdout", "stderr"):
                _validate_file_identity(worker[label])
            if worker["result_json"] is not None:
                _validate_file_identity(worker["result_json"])
            rc = worker["returncode"]
            if rc is not None and type(rc) is not int:
                raise ValueError("worker returncode must be an integer or null")
            timed_out = _boolean(worker["timed_out"], "timed_out")
            nested = worker["result"]
            if nested is not None:
                digest = hashlib.sha256(_json(nested).encode()).hexdigest()
                if _digest(worker["result_sha256"], "result_sha256") != digest:
                    raise ValueError("nested worker artifact digest mismatch")
            elif worker["result_sha256"] is not None:
                raise ValueError("absent worker artifact cannot have a digest")
            validation_error = None
            try:
                summary = validate_isolated_worker(worker, provider=result["provider"], cohort=result["cohort"])
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                validation_error = str(exc)
                summary = None
            if validation_error != worker["validation_error"]:
                raise ValueError("worker validation_error disagrees with nested evidence")
            valid_rows = summary is not None
            state = ("invalid" if validation_error else
                     "complete" if valid_rows and nested["complete"] else
                     "partial" if valid_rows else "empty" if nested is not None else "missing")
            status = ("complete" if state == "complete" and rc == 0 and not timed_out else
                      "partial" if valid_rows else "failed")
            if worker["artifact_state"] != state or worker["status"] != status:
                raise ValueError("worker status disagrees with exit and artifact evidence")
            paired = []
            if valid_rows:
                for row in nested["rows"]:
                    if row["exact_pass"] and row["fastest_measured_baseline"] == worker["baseline_id"]:
                        paired.append(dict(
                            case=row["case"], case_sha256=row["case_sha256"],
                            native=row["oracle_native_winner"],
                            baseline_to_native_ratio=row["oracle_speedup"],
                            trial_ratios=[t[worker["baseline_id"]] / t[row["oracle_native_winner"]]
                                          for t in row["trials_ms"]],
                        ))
            summaries.append(dict(
                baseline_id=worker["baseline_id"], status=status, artifact_state=state,
                returncode=rc, timed_out=timed_out, validation_error=validation_error,
                environment_sha256=nested.get("environment_sha256") if nested else None,
                source_sha256=nested.get("source_sha256") if nested else None,
                protocol_sha256=nested.get("protocol_sha256") if nested else None,
                paired_native_ratios=paired, v2_summary=summary,
            ))
        complete = bool(finished and workers and all(w["status"] == "complete" for w in workers))
        if _boolean(result["complete"], "complete") != complete:
            raise ValueError("isolated complete flag disagrees with worker outcomes")
        return dict(schema="streamattn.sm90_micro_prefill_isolated_audit_summary.v1",
                    provider=result["provider"], cohort=result["cohort"],
                    requested_baselines=list(requested), finished=finished, complete=complete,
                    evidence_kind="isolated_calibration_not_public_promotion",
                    comparison_scope="within_worker_paired_native_only",
                    global_promotion=False, workers=summaries)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid isolated audit artifact: {exc}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    summary = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    if "workers" in summary:
        print(json.dumps(summary, indent=2, allow_nan=False))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with args.output_json.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return
    print(
        "M D cells exact resolved complete_baselines paired_wins oracle_geomean producer_fraction"
    )
    for row in summary["groups"]:
        print(
            " ".join(
                str(row[k])
                for k in (
                    "m",
                    "d",
                    "cells",
                    "exact_cells",
                    "resolved_cells",
                    "complete_baseline_cells",
                    "paired_wins",
                    "oracle_geomean",
                    "natural_producer_fraction_median",
                )
            )
        )
    print(json.dumps(summary["baseline_failures"], indent=2))
    if args.output_json:
        if args.output_json.exists():
            raise FileExistsError("preserve existing evidence; choose a new output")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
