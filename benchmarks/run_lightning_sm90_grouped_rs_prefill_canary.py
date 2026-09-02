"""Run the complete grouped RS-PV prefill canary on Lightning H100."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Optional

from lightning_sdk.api.job_api import JobApiV2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASE_SHA = "71ab9b8bcb4d03b6f04e6339c855e1126a6d286c"
RESULT_SCHEMAS = {
    "canary": "streamattn.sm90_grouped_rs_prefill_canary.v1",
    "promotion": "streamattn.sm90_grouped_rs_prefill_promotion.v1",
}
TERMINAL_STATES = {"completed", "failed", "stop", "stopped", "cancelled", "error"}
OVERLAY_FILES = (
    "benchmarks/profile_sm90_grouped_rs_prefill_canary.py",
    "stream_attention/backends/sm90/__init__.py",
    "stream_attention/backends/sm90/grouped_gqa_prefill.py",
    "stream_attention/backends/sm90/transposed_gqa_exact_sources.py",
)


def _overlay_b64() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as archive:
        for relative in OVERLAY_FILES:
            archive.add(REPO_ROOT / relative, arcname=relative)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _result_from_logs(logs: str, *, schema: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    result = None
    for start, char in enumerate(logs):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(logs[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == schema:
            result = payload
    return result


def _fetch_finished_logs(
    api: JobApiV2,
    *,
    job_id: str,
    teamspace_id: str,
    attempts: int = 12,
    delay: float = 10.0,
) -> str:
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return api.get_logs_finished(job_id=job_id, teamspace_id=teamspace_id)
        except Exception as exc:
            last_error = exc
            print(
                f"log fetch attempt {attempt} failed: {type(exc).__name__}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"could not fetch Lightning logs: {last_error}")


def _job_command(args: argparse.Namespace) -> str:
    payload = _overlay_b64()
    batch_sizes = " ".join(str(value) for value in args.batch_sizes)
    sequence_lengths = " ".join(str(value) for value in args.sequence_lengths)
    group_sizes = " ".join(str(value) for value in args.group_sizes)
    return "\n".join(
        [
            "set -eu",
            "export PYTHONUNBUFFERED=1",
            "python - <<'PY'\n"
            "import pathlib, shutil, urllib.request, zipfile\n"
            f"sha = {BASE_SHA!r}\n"
            "archive = pathlib.Path('/tmp/streamattn.zip')\n"
            "dst = pathlib.Path('/root/StreamAttn')\n"
            "if dst.exists(): shutil.rmtree(dst)\n"
            "urllib.request.urlretrieve(f'https://github.com/MagellaX/StreamAttn/archive/{sha}.zip', archive)\n"
            "with zipfile.ZipFile(archive) as zf: zf.extractall('/root')\n"
            "(pathlib.Path('/root') / f'StreamAttn-{sha}').rename(dst)\n"
            "PY",
            "cd /root/StreamAttn",
            "python - <<'PY'\n"
            "import base64, io, tarfile\n"
            f"payload = {payload!r}\n"
            "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz') as archive:\n"
            "    archive.extractall('.')\n"
            "PY",
            "apt-get update -qq && apt-get install -y -qq ninja-build",
            "python -m pip install -q ninja",
            "python - <<'PY'\n"
            "import pathlib, shutil, urllib.request, zipfile\n"
            "sha = '39e616041ae6fb1243a0f6ac891e72d576b640e5'\n"
            "archive = pathlib.Path('/tmp/flashmla-etap.zip')\n"
            "dst = pathlib.Path('/tmp/flashmla-etap')\n"
            "urllib.request.urlretrieve(f'https://github.com/pengcuo/FlashMLA-ETAP/archive/{sha}.zip', archive)\n"
            "with zipfile.ZipFile(archive) as zf: zf.extractall('/tmp')\n"
            "if dst.exists(): shutil.rmtree(dst)\n"
            "(pathlib.Path('/tmp') / f'FlashMLA-ETAP-{sha}').rename(dst)\n"
            "PY",
            "cd /root/StreamAttn",
            "python -u benchmarks/profile_sm90_grouped_rs_prefill_canary.py "
            "--provider lightning "
            f"--matrix-kind {args.matrix_kind} "
            f"--batch-sizes {batch_sizes} "
            f"--sequence-lengths {sequence_lengths} "
            f"--group-sizes {group_sizes} "
            f"--warmup {args.warmup} --iterations {args.iterations} "
            f"--repeats {args.repeats} "
            "--cutlass-root /tmp/flashmla-etap/csrc/cutlass "
            "--build-dir /tmp/streamattn-grouped-rs-prefill-build "
            "--output-json /tmp/result.json --verbose-build",
            "cat /tmp/result.json",
        ]
    )


def _delete_job(api: JobApiV2, args: argparse.Namespace, job: Any) -> None:
    try:
        current = api.get_job(job_id=job.id, teamspace_id=args.teamspace_id)
        if str(current.state) not in TERMINAL_STATES:
            api.stop_job(job_id=job.id, teamspace_id=args.teamspace_id)
            time.sleep(3)
    except Exception as exc:
        print(f"warning: stop check failed: {type(exc).__name__}: {exc}", flush=True)
    try:
        api.delete_job(job_id=job.id, teamspace_id=args.teamspace_id, cloudspace_id="")
        print(f"deleted job id={job.id}", flush=True)
    except Exception as exc:
        print(f"warning: delete failed: {type(exc).__name__}: {exc}", flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    api = JobApiV2()
    job = None
    try:
        job = api.submit_job(
            name=args.name,
            command=_job_command(args),
            cloud_account=args.cloud_account,
            teamspace_id=args.teamspace_id,
            studio_id=None,
            image=args.image,
            machine=args.machine,
            interruptible=False,
            env={"PYTHONUNBUFFERED": "1"},
            image_credentials=None,
            cloud_account_auth=False,
            entrypoint="sh -c",
            path_mappings=None,
            artifacts_local=None,
            artifacts_remote=None,
            max_runtime=args.max_runtime,
            reuse_snapshot=False,
            scratch_disks=None,
        )
        print(f"submitted job name={job.name} id={job.id} state={job.state}", flush=True)
        started = time.time()
        while True:
            current = api.get_job(job_id=job.id, teamspace_id=args.teamspace_id)
            state = str(current.state)
            print(f"poll elapsed={int(time.time() - started)}s state={state}", flush=True)
            if state in TERMINAL_STATES:
                break
            time.sleep(args.poll_seconds)
        logs = _fetch_finished_logs(
            api,
            job_id=job.id,
            teamspace_id=args.teamspace_id,
            attempts=12,
            delay=10,
        )
        if args.log_path is not None:
            args.log_path.parent.mkdir(parents=True, exist_ok=True)
            args.log_path.write_text(logs, encoding="utf-8")
        result = _result_from_logs(logs, schema=RESULT_SCHEMAS[args.matrix_kind])
        if result is None:
            raise RuntimeError(
                "could not parse grouped RS-PV prefill canary result "
                f"from Lightning state={state}"
            )
        if state != "completed":
            print(
                f"warning: accepting complete structured result from state={state}",
                flush=True,
            )
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return result
    finally:
        if job is not None:
            _delete_job(api, args, job)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="streamattn-sm90-grouped-rs-prefill-canary")
    parser.add_argument(
        "--teamspace-id",
        default=os.environ.get("LIGHTNING_TEAMSPACE_ID", "01jggw9j5v8ms266vgvgcs3q13"),
    )
    parser.add_argument(
        "--cloud-account",
        default=os.environ.get("LIGHTNING_CLOUD_ACCOUNT", "lightning-nebius-prod"),
    )
    parser.add_argument(
        "--machine",
        default=os.environ.get("LIGHTNING_MACHINE", "nb-h100-1gpu-16vcpu-200gb"),
    )
    parser.add_argument("--image", default="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    parser.add_argument("--max-runtime", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--matrix-kind", choices=sorted(RESULT_SCHEMAS), default="canary")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "artifacts/gate0/"
            "sm90_grouped_rs_prefill_canary_lightning_h100_20260902.json"
        ),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(
            "artifacts/gate0/"
            "sm90_grouped_rs_prefill_canary_lightning_h100_20260902.log"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
