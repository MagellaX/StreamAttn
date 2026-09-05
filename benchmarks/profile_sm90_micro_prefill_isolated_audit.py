"""Run each external baseline plus both native controls in a fresh process.

Default smoke is four cases per worker. Results remain per-process v2 artifacts;
neither absolute timings nor environment identities are merged across workers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.micro_prefill_baselines import BASELINE_IDS, file_identity, requested_baselines  # noqa: E402
from benchmarks.profile_sm90_micro_prefill_audit import write_checkpoint  # noqa: E402
from benchmarks.summarize_sm90_micro_prefill_audit import (  # noqa: E402
    ISOLATED_SCHEMA, summarize_isolated, validate_isolated_worker,
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def worker_command(args, baseline, directory):
    return [
        sys.executable, "-u", str(ROOT / "benchmarks/profile_sm90_micro_prefill_audit.py"),
        "--provider", args.provider, "--cohort", args.cohort,
        "--baseline", baseline, "--cutlass-root", str(args.cutlass_root.resolve()),
        "--build-dir", str((args.build_dir / baseline).resolve()),
        "--output-json", str(directory / "result.json"),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        "--repeats", str(args.repeats),
    ]


def stop_worker(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if process.poll() is None:
            process.kill()
    else:
        process.kill()
    process.wait()


def run_worker(args, baseline, directory):
    directory.mkdir()
    command = worker_command(args, baseline, directory)
    stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
    result_path = directory / "result.json"
    worker = dict(baseline_id=baseline, command=command, cwd=str(ROOT),
                  started_at=timestamp(), pid=None, returncode=None, signal=None,
                  timed_out=False, error=None, validation_error=None, result=None,
                  result_sha256=None, result_json=None)
    start = time.monotonic()
    process = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            options = dict(start_new_session=True) if os.name == "posix" else {}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, **options)
            worker["pid"] = process.pid
            try:
                process.wait(timeout=args.worker_timeout_seconds)
            except subprocess.TimeoutExpired:
                worker["timed_out"] = True
                worker["error"] = "worker timeout"
                stop_worker(process)
        except OSError as exc:
            worker["error"] = f"{type(exc).__name__}: {exc}"
        except BaseException:
            if process is not None and process.poll() is None:
                stop_worker(process)
            raise
        finally:
            if process is not None:
                worker["returncode"] = process.poll()
                if worker["returncode"] is not None and worker["returncode"] < 0:
                    worker["signal"] = -worker["returncode"]
    worker.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - start,
                  stdout=file_identity(stdout_path), stderr=file_identity(stderr_path))
    if result_path.exists():
        worker["result_json"] = file_identity(result_path)
        try:
            nested = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(nested, dict):
                raise ValueError("worker artifact must be a mapping")
            digest = hashlib.sha256(json.dumps(nested, sort_keys=True, allow_nan=False).encode()).hexdigest()
            worker.update(result=nested, result_sha256=digest)
        except (OSError, ValueError) as exc:
            worker["error"] = f"artifact_read:{type(exc).__name__}: {exc}"
    try:
        summary = validate_isolated_worker(worker, provider=args.provider, cohort=args.cohort)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        summary = None
        worker["validation_error"] = str(exc)
    nested = worker["result"]
    worker["artifact_state"] = (
        "invalid" if worker["validation_error"] else
        "complete" if summary is not None and nested["complete"] else
        "partial" if summary is not None else "empty" if nested is not None else "missing"
    )
    worker["status"] = (
        "complete" if worker["artifact_state"] == "complete" and worker["returncode"] == 0
                      and not worker["timed_out"] else
        "partial" if summary is not None else "failed"
    )
    return worker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--cohort", choices=("smoke", "modal", "lightning"), default="smoke")
    parser.add_argument("--baselines", nargs="+", choices=BASELINE_IDS, default=list(BASELINE_IDS))
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--worker-timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    requested = requested_baselines(args.baselines)
    if min(args.warmup, args.iterations, args.repeats, args.worker_timeout_seconds) <= 0:
        parser.error("timing counts and worker timeout must be positive")
    output = args.output_json.resolve()
    directory = output.with_name(output.name + ".workers")
    if output.exists() or directory.exists():
        raise FileExistsError("preserve existing evidence; choose a new output-json")
    output.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    with output.open("x", encoding="utf-8"):
        pass
    result = dict(schema=ISOLATED_SCHEMA, provider=args.provider, cohort=args.cohort,
                  requested_baselines=list(requested), started_at=timestamp(),
                  comparison_scope="within_worker_paired_native_only",
                  evidence_kind="isolated_calibration_not_public_promotion",
                  finished=False, complete=False, workers=[])
    write_checkpoint(output, result)
    try:
        for baseline in requested:
            worker = run_worker(args, baseline, directory / baseline)
            result["workers"].append(worker)
            write_checkpoint(output, result)
            print(json.dumps(dict(baseline=baseline, status=worker["status"],
                                  returncode=worker["returncode"], output_json=str(output))), flush=True)
    except BaseException as exc:
        result["failure"] = dict(type=type(exc).__name__, message=str(exc))
        write_checkpoint(output, result)
        raise
    result.update(finished=True, finished_at=timestamp(),
                  complete=all(w["status"] == "complete" for w in result["workers"]))
    summarize_isolated(result)
    write_checkpoint(output, result)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
