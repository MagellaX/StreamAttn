"""Modal runner for the Gate-1 router regret benchmark."""

import os
import json
import subprocess

import modal


app = modal.App("streamattn-gate1-router-profile")

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
            "/root/StreamAttn/benchmarks/profile_gate1_router.py",
            "--seq",
            "1024",
            "--heads",
            "4",
            "--dim",
            "64",
            "--active-fraction",
            "0.0625",
            "--warmup",
            "20",
            "--iters",
            "80",
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
def main(target: str = "a10g"):
    if target == "a10g":
        print(profile_a10g.remote())
    elif target == "a100":
        print(profile_a100.remote())
    elif target == "h100":
        print(profile_h100.remote())
    else:
        raise ValueError("target must be a10g, a100, or h100")
