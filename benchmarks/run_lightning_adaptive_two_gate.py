"""Run a bounded single-H100 adaptive canary using existing Lightning auth."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.lightning_log_artifacts import result_from_logs
from benchmarks.profile_adaptive_two_gate import SCHEMA, SOURCE_FILES

CUTLASS_SOURCE_COMMIT = "39e616041ae6fb1243a0f6ac891e72d576b640e5"


def upload_capture(path, *, api, teamspace_id, cloud_account, remote_path):
    """Keep signed upload URLs and credentials out of retained job logs."""
    if not remote_path.startswith("uploads/"):
        raise ValueError("capture destination must be inside teamspace uploads")
    try:
        api.upload_file(teamspace_id=teamspace_id, cloud_account=cloud_account, file_path=str(path),
                        remote_path=remote_path, progress_bar=False)
    except Exception as exc:
        # Request exceptions can contain the signed URL. Never put it in job logs.
        raise RuntimeError(f"Capture upload failed: {type(exc).__name__}") from None


def job_command(args=None):
    frontier = args is not None and args.experiment == "group_frontier"
    feasibility = args is not None and args.experiment == "feasibility"
    long_context = args is not None and args.experiment == "long_context_feasibility"
    attribution = args is not None and args.experiment == "executor_attribution"
    files = SOURCE_FILES + ("tests/test_certified_attention.py", "tests/test_adaptive_two_gate_gpu.py")
    schema = SCHEMA
    if frontier:
        from benchmarks.profile_adaptive_group_frontier import SOURCE_FILES as FRONTIER_FILES, SCHEMA as FRONTIER_SCHEMA
        files += FRONTIER_FILES + ("tests/test_adaptive_real_diagnostic.py",
                                   "benchmarks/run_lightning_adaptive_two_gate.py")
        schema = FRONTIER_SCHEMA
    if feasibility or long_context or attribution:
        # Keep the controller independent of a local CUDA/PyTorch installation.
        files += ("benchmarks/profile_adaptive_feasibility.py", "tests/test_adaptive_feasibility.py",
                  "benchmarks/run_lightning_adaptive_two_gate.py")
        schema = "streamattn.adaptive_feasibility.v1"
        if long_context:
            files += ("benchmarks/profile_adaptive_long_context.py",
                      "benchmarks/profile_adaptive_real_activations.py",
                      "tests/test_adaptive_long_context.py", "docs/adaptive_long_context_protocol.md")
            schema = "streamattn.adaptive_long_context_feasibility.v1"
        if attribution:
            files += ("benchmarks/profile_adaptive_executor_attribution.py",
                      "benchmarks/profile_adaptive_long_context.py",
                      "tests/test_adaptive_executor_attribution.py",
                      "docs/adaptive_executor_attribution_protocol.md")
            schema = "streamattn.adaptive_executor_attribution.v1"
    sha = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in dict.fromkeys(files):
            archive.add(ROOT / name, arcname=name)
    payload = base64.b64encode(stream.getvalue()).decode("ascii")
    commands = [
        "set -eu",
        "python - <<'PY'",
        "import base64,io,pathlib,tarfile,urllib.request,zipfile",
        f"sha={sha!r}",
        "urllib.request.urlretrieve(f'https://github.com/MagellaX/StreamAttn/archive/{sha}.zip','/tmp/repo.zip')",
        "with zipfile.ZipFile('/tmp/repo.zip') as z: z.extractall('/root')",
        "pathlib.Path(f'/root/StreamAttn-{sha}').rename('/root/StreamAttn')",
        f"with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as z: z.extractall('/root/StreamAttn')",
        "PY",
        "python -m pip install -q pyyaml pytest",
        "cd /root/StreamAttn",
    ]
    remote_capture = None
    capture_inputs = []
    if feasibility or long_context or attribution:
        if (feasibility or attribution) and not args.capture_artifacts:
            raise ValueError("feasibility requires existing capture metadata artifacts")
        if attribution and len(args.capture_artifacts) != 1:
            raise ValueError("attribution requires exactly one existing source report")
        for path in (args.capture_artifacts or []) if feasibility or attribution else []:
            result = json.loads(path.read_text(encoding="utf-8"))
            remote = result["capture_remote_path"]
            digest = result["capture_archive_sha256"]
            if (not remote.startswith("uploads/streamattn/") or ".." in remote
                    or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("invalid capture path/hash")
            capture_inputs.append(dict(remote=remote, sha256=digest,
                                       model=result["model"], revision=result["model_revision"]))
        commands += [
            "python -m pip install -q ninja 'lightning-sdk==2026.9.18.post1' 'flashinfer-python==0.6.13' 'flashinfer-cubin==0.6.13'",
            "python - <<'PY'",
            "import pathlib,urllib.request,zipfile",
            f"sha={CUTLASS_SOURCE_COMMIT!r}",
            "urllib.request.urlretrieve(f'https://github.com/pengcuo/FlashMLA-ETAP/archive/{sha}.zip','/tmp/cutlass.zip')",
            "with zipfile.ZipFile('/tmp/cutlass.zip') as z: z.extractall('/tmp')",
            "pathlib.Path(f'/tmp/FlashMLA-ETAP-{sha}').rename('/tmp/flashmla-etap')",
            "assert pathlib.Path('/tmp/flashmla-etap/csrc/cutlass/include/cute/tensor.hpp').is_file()",
            "PY",
            "export STREAMATTN_CUTLASS_ROOT=/tmp/flashmla-etap/csrc/cutlass",
            "python -m pytest -q tests/test_adaptive_feasibility.py tests/test_adaptive_real_diagnostic.py",
        ]
        if long_context:
            remote_capture = f"uploads/streamattn/{args.output_json.stem}.captures.pt"
            commands += [
                "python -m pip install -q 'transformers==4.51.3'",
                "python -m pytest -q tests/test_adaptive_long_context.py",
                shlex.join(["python", "-u", "benchmarks/profile_adaptive_long_context.py",
                    "--output-json", "/tmp/adaptive.json", "--capture-remote", remote_capture,
                    "--teamspace-id", args.teamspace_id, "--cloud-account", args.cloud_account]),
            ]
        else:
            commands += [
                "python - <<'PY'",
                "import hashlib,pathlib",
                "from lightning_sdk.api import TeamspaceApi",
                f"inputs={capture_inputs!r}",
                "for item in inputs:",
                "    target='/tmp/'+pathlib.PurePosixPath(item['remote']).name",
                f"    TeamspaceApi().download_file(path=item['remote'],target_path=target,teamspace_id={args.teamspace_id!r},cloud_account={args.cloud_account!r},progress_bar=False)",
                "    if hashlib.sha256(pathlib.Path(target).read_bytes()).hexdigest()!=item['sha256']: raise RuntimeError('Capture hash mismatch')",
                "PY",
            ]
            paths = ["/tmp/" + Path(item["remote"]).name for item in capture_inputs]
            if attribution:
                report = args.capture_artifacts[0].resolve().relative_to(ROOT).as_posix()
                commands += [
                    "python -m pytest -q tests/test_adaptive_executor_attribution.py",
                    shlex.join(["python", "-u", "benchmarks/profile_adaptive_executor_attribution.py",
                        "--captures", paths[0], "--source-report", report,
                        "--output-json", "/tmp/adaptive.json"]),
                ]
            else:
                commands.append(shlex.join(["python", "-u", "benchmarks/profile_adaptive_feasibility.py",
                    "--captures", *paths, "--native", "--output-json", "/tmp/adaptive.json"]))
    elif frontier:
        remote_capture = f"uploads/streamattn/{args.output_json.stem}.captures.pt"
        command = ["python", "-u", "benchmarks/profile_adaptive_group_frontier.py", "--provider", "lightning",
                   "--model", args.model, "--max-seq", str(args.max_seq), "--layers",
                   *map(str, args.layers), "--output-json", "/tmp/adaptive.json"]
        if args.revision:
            command += ["--revision", args.revision]
        commands += [
            "python -m pip install -q 'transformers==4.51.3' 'lightning-sdk==2026.9.18.post1'",
            "python -m pytest -q tests/test_adaptive_real_diagnostic.py",
            shlex.join(command),
            "python - <<'PY'",
            "import hashlib,json,pathlib",
            "from lightning_sdk.api import TeamspaceApi",
            "from benchmarks.run_lightning_adaptive_two_gate import upload_capture",
            "p=pathlib.Path('/tmp/adaptive.captures.pt')",
            "r=json.loads(pathlib.Path('/tmp/adaptive.json').read_text())",
            f"remote={remote_capture!r}",
            f"upload_capture(p, api=TeamspaceApi(), teamspace_id={args.teamspace_id!r}, cloud_account={args.cloud_account!r}, remote_path=remote)",
            "r['capture_archive_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()",
            "r['capture_remote_path']=remote",
            "print(json.dumps(r),flush=True)",
            "PY",
        ]
    else:
        commands += [
            "python -m pytest -q tests/test_adaptive_two_gate_gpu.py tests/test_certified_attention.py",
            "python -u benchmarks/profile_adaptive_two_gate.py --provider lightning --output-json /tmp/adaptive.json",
        ]
    return "\n".join(commands), dict(base_sha=sha, overlay_sha256=hashlib.sha256(stream.getvalue()).hexdigest(),
                                      schema=schema, capture_remote_path=remote_capture,
                                      capture_inputs=capture_inputs,
                                      cutlass_source_commit=CUTLASS_SOURCE_COMMIT if feasibility or long_context or attribution else None)


def main():
    from importlib.metadata import version
    from packaging.version import Version
    from lightning_sdk.api.job_api import JobApiV2
    from benchmarks.run_lightning_sm90_grouped_rs_prefill_canary import (
        COMPLETED_STATES, TERMINAL_STATES, _delete_job,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--teamspace-id", default=os.getenv("LIGHTNING_TEAMSPACE_ID", "01jggw9j5v8ms266vgvgcs3q13"))
    p.add_argument("--cloud-account", default=os.getenv("LIGHTNING_CLOUD_ACCOUNT", "lightning-nebius-prod"))
    p.add_argument("--machine", default=os.getenv("LIGHTNING_MACHINE", "nb-h100-1gpu-16vcpu-200gb"))
    p.add_argument("--experiment", choices=("physical_canary", "group_frontier", "feasibility", "long_context_feasibility", "executor_attribution"), default="physical_canary")
    p.add_argument("--capture-artifacts", type=Path, nargs="+")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 16, 24, 26, 27])
    p.add_argument("--max-runtime", type=int, default=900)
    args = p.parse_args()
    if args.output_json.exists() or args.output_json.with_suffix(".captures.pt").exists():
        raise FileExistsError(args.output_json)
    if not 60 <= args.max_runtime <= 1800:
        p.error("max-runtime must be between 60 and 1800 seconds")
    if Version(version("lightning-sdk")) < Version("2026.9.18.post1"):
        raise RuntimeError("runner requires lightning-sdk>=2026.9.18.post1 for current job/artifact APIs")
    command, source = job_command(args)
    api, job = JobApiV2(), None
    environment = {"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_XET": "1"}
    if args.experiment in ("group_frontier", "feasibility", "long_context_feasibility", "executor_attribution"):
        from lightning_sdk.lightning_cloud.login import Auth
        from lightning_sdk.lightning_cloud.openapi import V1LoginRequest
        # Short-lived platform token enables capture upload to the same teamspace.
        auth = Auth()
        auth.authenticate()
        environment["LIGHTNING_AUTH_TOKEN"] = auth.auth_token or api._client.auth_service_login(
            V1LoginRequest(auth.api_key)).token
    try:
        job = api.submit_job(
            name=f"streamattn-adaptive-{int(time.time())}", command=command,
            cloud_account=args.cloud_account, teamspace_id=args.teamspace_id,
            studio_id=None, image="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
            machine=args.machine, interruptible=False, env=environment,
            image_credentials=None, cloud_account_auth=False, entrypoint="sh -c",
            path_mappings=None, max_run_attempts=1,
            max_runtime=args.max_runtime, reuse_snapshot=False, scratch_disks=None,
        )
        print(f"Lightning submitted {job.id}", flush=True)
        deadline, prior, logs, result = time.monotonic() + args.max_runtime + 600, None, "", None
        while time.monotonic() < deadline:
            current = api.get_job(job_id=job.id, teamspace_id=args.teamspace_id)
            state = str(current.state)
            if state != prior:
                print(f"state={state}", flush=True)
                prior = state
            if state in TERMINAL_STATES or current.started_at:
                try:
                    logs = api.get_logs_finished(job_id=job.id, teamspace_id=args.teamspace_id)
                    result = result_from_logs(logs, schema=source["schema"])
                except Exception as exc:
                    print(f"logs pending: {type(exc).__name__}", flush=True)
            if state in TERMINAL_STATES:
                # Platform completion can precede delivery of the final log chunks.
                if state in COMPLETED_STATES and (result is None or not result.get("complete")):
                    for _ in range(6):
                        time.sleep(5)
                        try:
                            logs = api.get_logs_finished(job_id=job.id, teamspace_id=args.teamspace_id)
                            result = result_from_logs(logs, schema=source["schema"])
                        except Exception as exc:
                            print(f"final logs pending: {type(exc).__name__}", flush=True)
                        if result is not None and result.get("complete"):
                            break
                break
            time.sleep(30)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.with_suffix(".log").write_text(logs, encoding="utf-8")
        execution = dict(provider="lightning", job_id=job.id, platform_state=str(current.state),
                         reported_cost=current.total_cost, source=source, machine=args.machine,
                         platform_message=current.message)
        if result is None:
            args.output_json.with_suffix(".failure.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError("No adaptive artifact; platform evidence and logs retained")
        result["execution"] = execution
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.experiment in ("group_frontier", "long_context_feasibility"):
            from lightning_sdk.api import TeamspaceApi
            if not result.get("capture_remote_path"):
                raise RuntimeError("Diagnostic saved but capture upload failed; inspect retained job logs")
            target = args.output_json.with_suffix(".captures.pt")
            if target.exists():
                raise FileExistsError(target)
            TeamspaceApi().download_file(path=result["capture_remote_path"], target_path=str(target),
                                        teamspace_id=args.teamspace_id, cloud_account=args.cloud_account,
                                        progress_bar=False)
            if hashlib.sha256(target.read_bytes()).hexdigest() != result["capture_archive_sha256"]:
                raise RuntimeError("Downloaded capture hash mismatch")
        print(f"artifact complete={result['complete']}; {args.output_json}", flush=True)
        if not result["complete"] or str(current.state) not in COMPLETED_STATES:
            raise RuntimeError("Adaptive canary failed; evidence retained")
    finally:
        if job is not None:
            _delete_job(api, args, job)


if __name__ == "__main__":
    main()
