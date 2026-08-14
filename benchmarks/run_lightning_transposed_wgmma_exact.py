"""Run the promoted SM90 exact backend on Lightning and always delete the job."""

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
from typing import Any, Dict, Optional

from lightning_sdk.api.job_api import JobApiV2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_lightning_mistral_marginals import _fetch_finished_logs


BASE_SHA = "0fb66431deea9aa6025d8eb9a2910a7b907c28fb"
RESULT_SCHEMA = "streamattn.transposed_wgmma_exact_decode.v1"
TERMINAL_STATES = {"completed", "failed", "stopped", "cancelled", "error"}


def _overlay_b64() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as archive:
        for relative in (
            "benchmarks/profile_transposed_wgmma_exact_qk.py",
            "stream_attention/backends",
            "stream_attention/decode.py",
        ):
            archive.add(REPO_ROOT / relative, arcname=relative)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _result_from_logs(logs: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    result = None
    for start, char in enumerate(logs):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(logs[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == RESULT_SCHEMA:
            result = payload
    return result


def _job_command(args: argparse.Namespace) -> str:
    payload = _overlay_b64()
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
            "apt-get update -qq && apt-get install -y -qq git ninja-build",
            "python -m pip install -q einops flashinfer-python==0.6.12 flashinfer-cubin==0.6.12 ninja",
            "git clone --filter=blob:none --no-checkout https://github.com/pengcuo/FlashMLA-ETAP.git /tmp/flashmla-etap",
            "cd /tmp/flashmla-etap && git sparse-checkout init --cone && git sparse-checkout set csrc/cutlass/include && git fetch --depth=1 origin 39e616041ae6fb1243a0f6ac891e72d576b640e5 && git checkout 39e616041ae6fb1243a0f6ac891e72d576b640e5",
            "cd /root/StreamAttn",
            "python -u benchmarks/profile_transposed_wgmma_exact_qk.py "
            f"--batch 4 --kv-len 32768 --q-heads 16 --kv-heads 2 --head-dim 64 --num-splits-list 32 --warmup {args.warmup} --iters {args.iters} --repeats {args.repeats} "
            "--cutlass-root /tmp/flashmla-etap/csrc/cutlass --build-dir /tmp/streamattn-exact-build --output-json /tmp/result.json",
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
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
            api, job_id=job.id, teamspace_id=args.teamspace_id, attempts=12, delay=10
        )
        if args.log_path is not None:
            args.log_path.parent.mkdir(parents=True, exist_ok=True)
            args.log_path.write_text(logs, encoding="utf-8")
        if state != "completed":
            raise RuntimeError(f"Lightning job ended with state={state}")
        result = _result_from_logs(logs)
        if result is None:
            raise RuntimeError("could not parse promoted exact-backend result")
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
    parser.add_argument("--name", default="streamattn-sm90-exact-promotion")
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
