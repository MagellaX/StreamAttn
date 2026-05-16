"""Modal runner for Gate-1 real-shape competitiveness profiling."""

import json
import os
import subprocess

import modal


app = modal.App("streamattn-gate1-real-shapes")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("triton==3.1.0")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _profile(seqs: str, heads: str, dims: str, active_fractions: str, block_sizes: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_gate1_real_shapes.py",
        "--seq",
        *seqs.split(","),
        "--heads",
        *heads.split(","),
        "--dim",
        *dims.split(","),
        "--active-fraction",
        *active_fractions.split(","),
        "--block-size",
        *block_sizes.split(","),
        "--dtype",
        "fp16",
        "--warmup",
        "5",
        "--iters",
        "8",
        "--metadata-warmup",
        "5",
        "--metadata-iters",
        "8",
    ]
    output = subprocess.check_output(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
    )
    return json.loads(output)


@app.function(image=image, gpu="A100", timeout=2400)
def profile_a100(seqs: str, heads: str, dims: str, active_fractions: str, block_sizes: str):
    return _profile(seqs, heads, dims, active_fractions, block_sizes)


@app.function(image=image, gpu="H100", timeout=2400)
def profile_h100(seqs: str, heads: str, dims: str, active_fractions: str, block_sizes: str):
    return _profile(seqs, heads, dims, active_fractions, block_sizes)


@app.local_entrypoint()
def main(
    target: str = "h100",
    seqs: str = "4096",
    heads: str = "16",
    dims: str = "128",
    active_fractions: str = "0.0625,1.0",
    block_sizes: str = "64",
):
    if target == "a100":
        print(profile_a100.remote(seqs, heads, dims, active_fractions, block_sizes))
    elif target == "h100":
        print(profile_h100.remote(seqs, heads, dims, active_fractions, block_sizes))
    else:
        raise ValueError("target must be a100 or h100")
