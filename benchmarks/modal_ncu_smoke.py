"""Modal runner for a minimal Nsight Compute smoke test."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-ncu-smoke")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _run_smoke(metric: str, replay_mode: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    ncu_path = shutil.which("ncu")
    summary = {
        "ncu_path": ncu_path,
        "ncu_version": None,
    }
    if ncu_path:
        version = subprocess.run(
            ["ncu", "--version"],
            cwd="/root/StreamAttn",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        summary["ncu_version"] = version.stdout

    plain = subprocess.run(
        ["python", "/root/StreamAttn/benchmarks/ncu_smoke.py"],
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary["plain_returncode"] = plain.returncode
    summary["plain_stdout"] = plain.stdout

    report_base = Path("/tmp/ncu_smoke")
    cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--force-overwrite",
        "--export",
        str(report_base),
        "--replay-mode",
        replay_mode,
        "--metrics",
        metric,
        "python",
        "/root/StreamAttn/benchmarks/ncu_smoke.py",
    ]
    profiled = subprocess.run(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary["profile_returncode"] = profiled.returncode
    summary["profile_stdout_tail"] = profiled.stdout[-8000:]
    summary["command"] = cmd

    report = report_base.with_suffix(".ncu-rep")
    if report.exists():
        imported = subprocess.run(
            ["ncu", "--import", str(report), "--csv"],
            cwd="/root/StreamAttn",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        summary["import_returncode"] = imported.returncode
        summary["import_stdout_tail"] = imported.stdout[-8000:]
    return summary


@app.function(image=image, gpu="A100", timeout=1200)
def smoke_a100(metric: str, replay_mode: str):
    return _run_smoke(metric, replay_mode)


@app.function(image=image, gpu="H100", timeout=1200)
def smoke_h100(metric: str, replay_mode: str):
    return _run_smoke(metric, replay_mode)


@app.local_entrypoint()
def main(
    target: str = "h100",
    metric: str = "sm__pipe_tensor_active.avg.pct_of_peak_sustained_active",
    replay_mode: str = "kernel",
):
    if target == "a100":
        print(json.dumps(smoke_a100.remote(metric, replay_mode), indent=2, sort_keys=True))
    elif target == "h100":
        print(json.dumps(smoke_h100.remote(metric, replay_mode), indent=2, sort_keys=True))
    else:
        raise ValueError("target must be a100 or h100")
