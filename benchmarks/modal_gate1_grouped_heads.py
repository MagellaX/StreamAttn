"""Modal runner for actual mixed-head Gate-1 grouping simulation."""

import json
import os
import subprocess

import modal


app = modal.App("streamattn-gate1-grouped-heads")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("triton==3.1.0")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _profile():
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    output = subprocess.check_output(
        [
            "python",
            "/root/StreamAttn/benchmarks/profile_gate1_grouped_heads.py",
            "--source",
            "actual_raw_stats",
            "--per-head-active",
            "0.05,0.08,0.80,0.92",
            "--seq",
            "1024",
            "--dim",
            "64",
            "--block-size",
            "64",
            "--tile-size-q",
            "64",
            "--measure-gather-tax",
            "--warmup",
            "10",
            "--iters",
            "30",
        ],
        cwd="/root/StreamAttn",
        env=env,
        text=True,
    )
    return json.loads(output)


@app.function(image=image, gpu="A10G", timeout=900)
def profile_a10g():
    return _profile()


@app.function(image=image, gpu="A100", timeout=900)
def profile_a100():
    return _profile()


@app.function(image=image, gpu="H100", timeout=900)
def profile_h100():
    return _profile()


@app.local_entrypoint()
def main(target: str = "h100"):
    if target == "a10g":
        print(profile_a10g.remote())
    elif target == "a100":
        print(profile_a100.remote())
    elif target == "h100":
        print(profile_h100.remote())
    else:
        raise ValueError("target must be a10g, a100, or h100")
