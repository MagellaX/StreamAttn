"""Run a bounded single-H100 adaptive canary using existing Lightning auth."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.lightning_log_artifacts import result_from_logs
from benchmarks.profile_adaptive_two_gate import SCHEMA, SOURCE_FILES


def job_command():
    sha = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in SOURCE_FILES + ("tests/test_certified_attention.py", "tests/test_adaptive_two_gate_gpu.py"):
            archive.add(ROOT / name, arcname=name)
    payload = base64.b64encode(stream.getvalue()).decode("ascii")
    command = "\n".join([
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
        "python -m pytest -q tests/test_adaptive_two_gate_gpu.py tests/test_certified_attention.py",
        "python -u benchmarks/profile_adaptive_two_gate.py --provider lightning --output-json /tmp/adaptive.json",
    ])
    return command, dict(base_sha=sha, overlay_sha256=hashlib.sha256(stream.getvalue()).hexdigest())


def main():
    from lightning_sdk.api.job_api import JobApiV2
    from benchmarks.run_lightning_sm90_grouped_rs_prefill_canary import TERMINAL_STATES, _delete_job

    p = argparse.ArgumentParser()
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--teamspace-id", default=os.getenv("LIGHTNING_TEAMSPACE_ID", "01jggw9j5v8ms266vgvgcs3q13"))
    p.add_argument("--cloud-account", default=os.getenv("LIGHTNING_CLOUD_ACCOUNT", "lightning-nebius-prod"))
    p.add_argument("--machine", default=os.getenv("LIGHTNING_MACHINE", "nb-h100-1gpu-16vcpu-200gb"))
    args = p.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    command, source = job_command()
    api, job = JobApiV2(), None
    try:
        job = api.submit_job(
            name=f"streamattn-adaptive-{int(time.time())}", command=command,
            cloud_account=args.cloud_account, teamspace_id=args.teamspace_id,
            studio_id=None, image="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
            machine=args.machine, interruptible=False, env={"PYTHONUNBUFFERED": "1"},
            image_credentials=None, cloud_account_auth=False, entrypoint="sh -c",
            path_mappings=None, artifacts_local=None, artifacts_remote=None,
            max_runtime=900, reuse_snapshot=False, scratch_disks=None,
        )
        print(f"Lightning submitted {job.id}", flush=True)
        deadline, prior, logs, result = time.monotonic() + 1200, None, "", None
        while time.monotonic() < deadline:
            current = api.get_job(job_id=job.id, teamspace_id=args.teamspace_id)
            state = str(current.state)
            if state != prior:
                print(f"state={state}", flush=True)
                prior = state
            if state in TERMINAL_STATES or current.started_at:
                try:
                    logs = api.get_logs_finished(job_id=job.id, teamspace_id=args.teamspace_id)
                    result = result_from_logs(logs, schema=SCHEMA)
                except Exception as exc:
                    print(f"logs pending: {type(exc).__name__}", flush=True)
            # A complete artifact can arrive before the platform's state transition.
            if result is not None or state in TERMINAL_STATES:
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
        print(f"artifact complete={result['complete']}; {args.output_json}", flush=True)
        if not result["complete"]:
            raise RuntimeError("Adaptive canary failed; evidence retained")
    finally:
        if job is not None:
            _delete_job(api, args, job)


if __name__ == "__main__":
    main()
