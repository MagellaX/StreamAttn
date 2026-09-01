"""Run the TK tensor-core exact-decode floor on Lightning and delete the job."""

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
from typing import Any, Dict, Optional

from lightning_sdk.api.job_api import JobApiV2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_SHA = "0394801d6c508ef21f674a108b98aef9173e9e30"
RESULT_SCHEMA = "streamattn.tk_tensor_core_exact_decode.v1"
MATRIX_SCHEMA = "streamattn.tk_tensor_core_exact_decode.matrix.v1"
PHASED_GATE_SCHEMA = "streamattn.sm80_d128_phased_kv_gate.matrix.v1"
TERMINAL_STATES = {"completed", "failed", "stopped", "cancelled", "error"}


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _fetch_finished_logs(
    api: JobApiV2,
    *,
    job_id: str,
    teamspace_id: str,
    attempts: int,
    delay: float,
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


def _overlay_b64() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as archive:
        for path in (
            "benchmarks/profile_tk_tensor_core_exact_decode.py",
            "benchmarks/profile_sm80_d128_phased_gate.py",
            "stream_attention/backends/sm80/__init__.py",
            "stream_attention/backends/sm80/tk_grouped_exact.py",
            "stream_attention/backends/sm80/tk_grouped_exact_sources.py",
        ):
            archive.add(REPO_ROOT / path, arcname=path)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _results_from_logs(logs: str) -> list[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    results: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for start, char in enumerate(logs):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(logs[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") in {
            RESULT_SCHEMA,
            PHASED_GATE_SCHEMA,
        }:
            key = json.dumps(payload, sort_keys=True)
            if key not in seen:
                seen.add(key)
                results.append(payload)
    return results


def _result_from_logs(logs: str) -> Optional[Dict[str, Any]]:
    results = _results_from_logs(logs)
    return results[-1] if results else None


def _matrix_specs(args: argparse.Namespace) -> list[Dict[str, Any]]:
    if not args.matrix_specs:
        return [{}]
    parsed = json.loads(args.matrix_specs)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--matrix-specs must be a non-empty JSON list")
    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError("every matrix spec must be a JSON object")
    return parsed


def _profile_command(args: argparse.Namespace, spec: Dict[str, Any], index: int) -> str:
    def value(name: str) -> Any:
        return spec.get(name, getattr(args, name))

    common = [
        f"--batch {value('batch')}",
        f"--q-heads {value('q_heads')}",
        f"--kv-heads {value('kv_heads')}",
        f"--head-dim {value('head_dim')}",
        f"--kv-len {value('kv_len')}",
        f"--dtype {_quote(str(value('dtype')))}",
        f"--seed {value('seed')}",
        f"--num-chunks {value('num_chunks')}",
        f"--flashinfer-page-size {value('flashinfer_page_size')}",
        f"--flashinfer-workspace-mb {value('flashinfer_workspace_mb')}",
        f"--cuda-arch {_quote(args.cuda_arch)}",
        f"--torch-cuda-arch-list {_quote(args.torch_cuda_arch_list)}",
        "--checkout-dir /tmp/streamattn_backend_sources",
    ]
    if args.profile == "sm80_d128_phased_gate":
        command = [
            "python -u -m benchmarks.profile_sm80_d128_phased_gate",
            *common,
            f"--warmup {value('warmup')}",
            f"--paired-trials {value('paired_trials')}",
            f"--paired-iters {value('paired_iters')}",
            f"--output-json /tmp/sm80_d128_phased_gate_{index}.json",
        ]
        if args.production_plan:
            command.append("--production-plan")
        return " ".join(command)
    return " ".join(
        [
            "python -u benchmarks/profile_tk_tensor_core_exact_decode.py",
            *common,
            f"--num-chunks-list {_quote(str(value('num_chunks_list')))}",
            f"--producer-warps-list {_quote(str(value('producer_warps_list')))}",
            f"--warmup {value('warmup')}",
            f"--iters {value('iters')}",
            f"--output-json /tmp/tk_tensor_core_exact_decode_{index}.json",
        ]
    )


def _job_command(args: argparse.Namespace) -> str:
    payload = _overlay_b64()
    specs = _matrix_specs(args)
    profiles = [_profile_command(args, spec, index) for index, spec in enumerate(specs)]
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
            "python -m pip install -q einops flashinfer-python==0.6.13 flashinfer-cubin==0.6.13 ninja",
            "python - <<'PY'\n"
            "import traceback\n"
            "try:\n"
            "    import flashinfer\n"
            "    print('flashinfer import ok', getattr(flashinfer, '__version__', 'unknown'))\n"
            "except Exception:\n"
            "    print('flashinfer import failed')\n"
            "    traceback.print_exc()\n"
            "PY",
            *profiles,
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
        state = str(job.state)
        while True:
            current = api.get_job(job_id=job.id, teamspace_id=args.teamspace_id)
            state = str(current.state)
            print(
                f"poll elapsed={int(time.time() - started)}s state={state} "
                f"message={str(getattr(current, 'message', '') or '')!r} "
                f"server_error={str(getattr(current, 'server_error', '') or '')!r}",
                flush=True,
            )
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
        if state != "completed":
            raise RuntimeError(f"Lightning job ended with state={state}")
        results = _results_from_logs(logs)
        if not results:
            raise RuntimeError("could not parse TK exact-decode benchmark result")
        if args.matrix_specs:
            specs = _matrix_specs(args)
            expected = len(specs)
            if len(results) != expected:
                raise RuntimeError(f"expected {expected} matrix results, parsed {len(results)}")
            labeled_results = []
            for index, (spec, row) in enumerate(zip(specs, results)):
                labeled = dict(row)
                labeled["matrix_name"] = str(spec.get("name", f"cell_{index}"))
                labeled_results.append(labeled)
            result_schema = (
                "streamattn.sm80_d128_phased_kv_gate.lightning_matrix.v1"
                if args.profile == "sm80_d128_phased_gate"
                else MATRIX_SCHEMA
            )
            result: Dict[str, Any] = {
                "schema": result_schema,
                "results": labeled_results,
            }
        else:
            result = results[-1]
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
    parser.add_argument("--name", required=True)
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
    parser.add_argument(
        "--profile",
        choices=("tk", "sm80_d128_phased_gate"),
        default="tk",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cuda-arch",
        default=os.environ.get("STREAMATTN_CUDA_ARCH", "sm_90a"),
    )
    parser.add_argument(
        "--torch-cuda-arch-list",
        default=os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0a"),
    )
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--num-chunks-list", default="8,16,32,64")
    parser.add_argument("--producer-warps-list", default="1,2,4,8")
    parser.add_argument("--flashinfer-page-size", type=int, default=16)
    parser.add_argument("--flashinfer-workspace-mb", type=int, default=128)
    parser.add_argument("--matrix-specs", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--paired-trials", type=int, default=15)
    parser.add_argument("--paired-iters", type=int, default=100)
    parser.add_argument("--production-plan", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
