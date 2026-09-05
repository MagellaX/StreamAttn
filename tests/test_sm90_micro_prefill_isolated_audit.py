from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks import profile_sm90_micro_prefill_isolated_audit as isolated
from benchmarks.profile_sm90_micro_prefill_audit import cases, write_checkpoint
from benchmarks.summarize_sm90_micro_prefill_audit import summarize
from tests.test_sm90_micro_prefill_audit_summary import _resolve, fixture_result


def loaded_evidence(names):
    return {
        name: dict(
            modules={name: dict(path=f"/installed/{name}.py", sha256="a" * 64, size_bytes=10)},
            binaries=[dict(path=f"/installed/{name}.so", sha256="b" * 64, size_bytes=20)],
            errors=[], resolved=True,
        ) for name in names
    }


def worker_result(baseline="torch_flash", *, complete=True, scale=1.0):
    result = fixture_result()
    result["requested_baselines"] = [baseline]
    result["loaded_binary_provenance_required"] = True
    result["complete"] = complete
    result["device"] = f"H100 worker {baseline}"
    template = result["rows"][0]
    kept = (baseline, "natural", "transposed")
    for trial in template["trials_ms"]:
        latency = trial["torch_flash"]
        for name in ("torch_flash", "flashinfer_fa3"):
            del trial[name]
        trial[baseline] = latency
        for name in trial:
            trial[name] *= scale
    template["accuracy"] = {name: deepcopy(template["accuracy"]["natural"]) for name in kept}
    template["mutation"] = {name: deepcopy(template["mutation"]["natural"]) for name in kept}
    template["unavailable_baselines"] = {}
    template["loaded_binary_provenance"] = loaded_evidence(kept)
    result["rows"] = []
    for case in cases("smoke")[:None if complete else 1]:
        row = deepcopy(template)
        row["case"] = case
        result["rows"].append(row)
    return json.loads(json.dumps(_resolve(result)))


def worker_record(result, *, returncode=0, timed_out=False):
    identity = dict(path="/evidence/log", sha256="c" * 64, size_bytes=0)
    return dict(
        baseline_id=result["requested_baselines"][0], returncode=returncode, timed_out=timed_out,
        stdout=identity, stderr=identity, result_json=dict(identity, path="/evidence/result.json"),
        result=result, result_sha256=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
        validation_error=None, artifact_state="complete" if result["complete"] else "partial",
        status="complete" if result["complete"] and returncode == 0 and not timed_out else "partial",
    )


def isolated_result(*workers):
    return dict(schema=isolated.ISOLATED_SCHEMA, provider="test", cohort="smoke",
                requested_baselines=[w["baseline_id"] for w in workers], workers=list(workers),
                finished=True, complete=all(w["status"] == "complete" for w in workers))


def args_for(tmp_path):
    return SimpleNamespace(provider="test", cohort="smoke", cutlass_root=tmp_path / "cutlass",
                           build_dir=tmp_path / "build", warmup=4, iterations=40,
                           repeats=3, worker_timeout_seconds=10)


def test_worker_cli_selects_exactly_one_backend_with_unchanged_timing(tmp_path):
    command = isolated.worker_command(args_for(tmp_path), "flashattention3", tmp_path)
    assert command[command.index("--baseline") + 1] == "flashattention3"
    assert command[command.index("--cohort") + 1] == "smoke"
    assert command[command.index("--repeats") + 1] == "3"
    assert "--baselines" not in command
    assert len(cases("smoke")) == 4
    assert Path(command[2]).resolve().parents[1] == isolated.ROOT


def test_independent_environments_only_report_within_worker_paired_ratios():
    first = worker_record(worker_result())
    second = worker_record(worker_result("flashattention3", scale=100))
    data = isolated_result(first, second)
    before = deepcopy(data)
    report = summarize(data)
    assert data == before
    assert report["complete"]
    assert not report["global_promotion"]
    assert report["comparison_scope"] == "within_worker_paired_native_only"
    assert "fastest_measured_baseline" not in report
    assert "environment_sha256" not in report
    a, b = report["workers"]
    assert a["environment_sha256"] != b["environment_sha256"]
    assert a["paired_native_ratios"] == b["paired_native_ratios"]
    assert a["paired_native_ratios"][0]["baseline_to_native_ratio"] == 1.25
    assert a["paired_native_ratios"][0]["trial_ratios"] == [1.25, 0.875, 1.625]
    assert a["v2_summary"]["groups"][0]["complete_baseline_cells"] == 0


@pytest.mark.parametrize("complete,returncode,timed_out", [(False, 9, False), (True, -6, False), (True, 0, True)])
def test_partial_or_aborted_workers_retain_valid_pairs_without_complete_claim(complete, returncode, timed_out):
    worker = worker_record(worker_result(complete=complete), returncode=returncode, timed_out=timed_out)
    report = summarize(isolated_result(worker))
    assert not report["complete"]
    assert report["workers"][0]["status"] == "partial"
    assert report["workers"][0]["paired_native_ratios"]


@pytest.mark.parametrize("damage", ["duplicate", "missing", "subset", "environment", "digest", "binary", "mutation", "empty", "coverage", "complete"])
def test_isolated_summary_rejects_forged_status_or_nested_evidence(damage):
    worker = worker_record(worker_result())
    data = isolated_result(worker)
    if damage == "duplicate":
        data["workers"].append(deepcopy(worker))
    elif damage == "missing":
        data["workers"] = []
    elif damage == "subset":
        worker["result"]["requested_baselines"] = ["flashattention3"]
    elif damage == "environment":
        worker["result"]["environment_sha256"] = "f" * 64
    elif damage == "digest":
        worker["result_sha256"] = "f" * 64
    elif damage == "binary":
        worker["result"]["rows"][0]["loaded_binary_provenance"]["torch_flash"]["binaries"] = []
    elif damage == "mutation":
        worker["result"]["rows"][0]["mutation"]["torch_flash"]["passed"] = False
    elif damage == "empty":
        worker["result"]["rows"] = []
    elif damage == "coverage":
        worker["result"]["rows"].pop()
    else:
        data["complete"] = False
    if damage != "digest":
        worker["result_sha256"] = hashlib.sha256(json.dumps(worker["result"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError):
        summarize(data)


def test_loaded_binary_changes_invalidate_measurement_revision():
    result = worker_result()
    result["rows"][0]["loaded_binary_provenance"]["torch_flash"]["binaries"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="eligibility|revision|measurements"):
        summarize(result)


def test_unresolved_loaded_identity_is_diagnostic_not_ratio_evidence():
    result = worker_result()
    for row in result["rows"]:
        row["loaded_binary_provenance"]["natural"]["binaries"] = []
        row["loaded_binary_provenance"]["natural"]["resolved"] = False
    _resolve(result)
    report = summarize(isolated_result(worker_record(result)))
    assert not report["workers"][0]["paired_native_ratios"]


@pytest.mark.parametrize("mode", ["complete", "partial", "failure", "malformed", "empty", "timeout", "invalid"])
def test_real_cpu_subprocess_outcomes_are_retained(monkeypatch, tmp_path, mode):
    args = args_for(tmp_path)
    result = worker_result(complete=mode != "partial")
    script = ["import pathlib,sys,time", "print('worker stdout', flush=True)", "print('worker stderr', file=sys.stderr, flush=True)"]
    if mode == "malformed":
        payload = "{truncated"
    else:
        if mode == "empty":
            result.update(complete=False, rows=[])
        if mode == "invalid":
            result["rows"][0]["oracle_speedup"] = 100
        payload = json.dumps(result)
    if mode != "failure":
        script.append(f"pathlib.Path(sys.argv[1]).write_text({payload!r}, encoding='utf-8')")
    if mode == "timeout":
        args.worker_timeout_seconds = 1
        script.append("time.sleep(60)")
    script.append("sys.exit(17)" if mode in ("partial", "failure") else "sys.exit(0)")
    monkeypatch.setattr(isolated, "worker_command", lambda args, baseline, directory:
                        [sys.executable, "-B", "-c", "\n".join(script), str(directory / "result.json")])
    worker = isolated.run_worker(args, "torch_flash", tmp_path / "worker")
    assert worker["pid"] > 0
    assert worker["returncode"] is not None
    assert worker["stdout"]["size_bytes"] > 0 and worker["stderr"]["size_bytes"] > 0
    assert Path(worker["stdout"]["path"]).read_text().strip() == "worker stdout"
    assert worker["timed_out"] == (mode == "timeout")
    expected = "complete" if mode == "complete" else "partial" if mode in ("partial", "timeout") else "failed"
    assert worker["status"] == expected
    if mode not in ("failure", "malformed"):
        assert worker["result"] == result
    assert summarize(isolated_result(worker))["workers"][0]["status"] == expected


def test_spawn_failure_still_preserves_logs_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(isolated.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("cannot start")))
    worker = isolated.run_worker(args_for(tmp_path), "torch_flash", tmp_path / "worker")
    assert worker["returncode"] is None
    assert worker["status"] == "failed"
    assert "cannot start" in worker["error"]
    assert summarize(isolated_result(worker))["workers"][0]["status"] == "failed"


def test_supervisor_checkpoints_after_each_worker_and_continues_failures(monkeypatch, tmp_path):
    output = tmp_path / "audit.json"
    seen = []

    def run(args, baseline, directory):
        checkpoint = json.loads(output.read_text())
        assert [w["baseline_id"] for w in checkpoint["workers"]] == seen
        assert args.cohort == "smoke"
        seen.append(baseline)
        return worker_record(worker_result(baseline), returncode=9 if len(seen) == 1 else 0)

    monkeypatch.setattr(isolated, "run_worker", run)
    status = isolated.main(["--provider", "test", "--cutlass-root", str(tmp_path),
                            "--build-dir", str(tmp_path / "build"), "--output-json", str(output),
                            "--baselines", "torch_flash", "flashattention3"])
    result = json.loads(output.read_text())
    assert status == 1 and result["finished"] and not result["complete"]
    assert seen == ["torch_flash", "flashattention3"]
    assert len(summarize(result)["workers"]) == 2


@pytest.mark.parametrize("existing", ["output", "directory"])
def test_supervisor_refuses_existing_evidence_without_launch(monkeypatch, tmp_path, existing):
    output = tmp_path / "audit.json"
    if existing == "output":
        output.write_text("preserve me")
    else:
        (tmp_path / "audit.json.workers").mkdir()
    monkeypatch.setattr(isolated, "run_worker", lambda *a: pytest.fail("must not launch"))
    with pytest.raises(FileExistsError):
        isolated.main(["--provider", "test", "--cutlass-root", str(tmp_path),
                       "--build-dir", str(tmp_path), "--output-json", str(output)])
    if existing == "output":
        assert output.read_text() == "preserve me"


def test_failed_atomic_replace_preserves_previous_checkpoint(monkeypatch, tmp_path):
    from benchmarks import profile_sm90_micro_prefill_audit as audit
    output = tmp_path / "checkpoint.json"
    output.write_text('{"previous": true}')
    monkeypatch.setattr(audit.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        write_checkpoint(output, {"new": True})
    assert json.loads(output.read_text()) == {"previous": True}
    assert list(tmp_path.iterdir()) == [output]


def test_isolated_summary_cli_from_outside_repository(tmp_path):
    artifact = tmp_path / "isolated.json"
    artifact.write_text(json.dumps(isolated_result(worker_record(worker_result()))))
    script = isolated.ROOT / "benchmarks/summarize_sm90_micro_prefill_audit.py"
    completed = subprocess.run([sys.executable, "-B", str(script), str(artifact)],
                               cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["comparison_scope"] == "within_worker_paired_native_only"
