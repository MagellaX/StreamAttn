"""H100 cross-provider micro-prefill audit, Lightning half; bounded and cleaned up."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightning_sdk.api.job_api import JobApiV2  # noqa: E402

from benchmarks.run_lightning_sm90_grouped_rs_prefill_canary import (  # noqa: E402
    TERMINAL_STATES,
    _delete_job,
    _fetch_finished_logs,
    _result_from_logs,
)

OVERLAY = (
    "benchmarks/profile_sm90_micro_prefill_audit.py",
    "benchmarks/profile_sm90_micro_prefill.py",
    "stream_attention/backends/sm90/micro_prefill.py",
    "stream_attention/backends/sm90/transposed_gqa_exact_sources.py",
)


def command(cohort):
    sha = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True
    ).strip()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in OVERLAY:
            archive.add(ROOT / path, arcname=path)
    payload = base64.b64encode(buffer.getvalue()).decode()
    return "\n".join(
        [
            "set -eu",
            "export PYTHONUNBUFFERED=1",
            "python - <<'PY'",
            "import pathlib,urllib.request,zipfile,tarfile,base64,io",
            f"sha={sha!r}",
            "urllib.request.urlretrieve(f'https://github.com/MagellaX/StreamAttn/archive/{sha}.zip','/tmp/repo.zip')",
            "with zipfile.ZipFile('/tmp/repo.zip') as z: z.extractall('/root')",
            "pathlib.Path(f'/root/StreamAttn-{sha}').rename('/root/StreamAttn')",
            f"with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as z: z.extractall('/root/StreamAttn')",
            "sha='39e616041ae6fb1243a0f6ac891e72d576b640e5'",
            "urllib.request.urlretrieve(f'https://github.com/pengcuo/FlashMLA-ETAP/archive/{sha}.zip','/tmp/cutlass.zip')",
            "with zipfile.ZipFile('/tmp/cutlass.zip') as z: z.extractall('/tmp')",
            "pathlib.Path(f'/tmp/FlashMLA-ETAP-{sha}').rename('/tmp/flashmla-etap')",
            "PY",
            "python -m pip install -q ninja flashinfer-python==0.6.17 flashinfer-cubin==0.6.17",
            "cd /root/StreamAttn",
            "python -u benchmarks/profile_sm90_micro_prefill_audit.py "
            f"--provider lightning --cohort {cohort} "
            "--cutlass-root /tmp/flashmla-etap/csrc/cutlass "
            "--build-dir /tmp/streamattn-audit-build --output-json /tmp/audit.json",
        ]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=("lightning", "smoke"), default="lightning")
    p.add_argument(
        "--teamspace-id",
        default=os.getenv("LIGHTNING_TEAMSPACE_ID", "01jggw9j5v8ms266vgvgcs3q13"),
    )
    p.add_argument(
        "--cloud-account",
        default=os.getenv("LIGHTNING_CLOUD_ACCOUNT", "lightning-nebius-prod"),
    )
    p.add_argument(
        "--machine", default=os.getenv("LIGHTNING_MACHINE", "nb-h100-1gpu-16vcpu-200gb")
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=ROOT
        / "artifacts/gate0/sm90_micro_prefill_audit_lightning_h100_20260905.json",
    )
    args = p.parse_args()
    if args.output_json.exists():
        raise FileExistsError(
            f"preserve existing evidence; choose another output: {args.output_json}"
        )
    api, job = JobApiV2(), None
    try:
        job = api.submit_job(
            name=f"streamattn-micro-audit-{int(time.time())}",
            command=command(args.cohort),
            cloud_account=args.cloud_account,
            teamspace_id=args.teamspace_id,
            studio_id=None,
            image="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
            machine=args.machine,
            interruptible=False,
            env={"PYTHONUNBUFFERED": "1"},
            image_credentials=None,
            cloud_account_auth=False,
            entrypoint="sh -c",
            path_mappings=None,
            artifacts_local=None,
            artifacts_remote=None,
            max_runtime=2700,
            reuse_snapshot=False,
            scratch_disks=None,
        )
        print(f"submitted Lightning H100 job {job.id}", flush=True)
        deadline = time.monotonic() + 3300
        old = None
        while time.monotonic() < deadline:
            current = api.get_job(job_id=job.id, teamspace_id=args.teamspace_id)
            state = str(current.state)
            if state != old:
                print(f"state={state}", flush=True)
                old = state
            if state in TERMINAL_STATES:
                print(
                    f"message={current.message}; reported_cost={current.total_cost}",
                    flush=True,
                )
                break
            time.sleep(20)
        else:
            raise TimeoutError("Lightning job exceeded local watchdog")
        if state != "completed" and not current.started_at:
            failure = dict(
                schema="streamattn.gpu_launch_failure.v1",
                provider="lightning",
                job_id=job.id,
                state=state,
                message=current.message,
                cloud_account=args.cloud_account,
                machine=args.machine,
                reported_cost=current.total_cost,
            )
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.with_suffix(".failure.json").write_text(
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(f"Lightning failed before execution: {current.message}")
        logs = _fetch_finished_logs(
            api, job_id=job.id, teamspace_id=args.teamspace_id, attempts=3, delay=10
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.with_suffix(".log").write_text(logs, encoding="utf-8")
        result = _result_from_logs(
            logs, schema="streamattn.sm90_micro_prefill_audit.v1"
        )
        if not result or not result.get("complete"):
            raise RuntimeError(f"incomplete result, state={state}; see local log")
        result["lightning_job_id"] = job.id
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output_json}", flush=True)
    finally:
        if job is not None:
            _delete_job(api, args, job)


if __name__ == "__main__":
    main()
