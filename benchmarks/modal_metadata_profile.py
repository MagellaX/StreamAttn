"""Modal runner for value-norm metadata builder profiling."""

import json
import os
import subprocess

import modal


app = modal.App("streamattn-metadata-profile")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("triton==3.1.0")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _profile(seqs: str, heads: str, dims: str, block_sizes: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_metadata_builder.py",
        "--seq",
        *seqs.split(","),
        "--heads",
        *heads.split(","),
        "--dim",
        *dims.split(","),
        "--block-size",
        *block_sizes.split(","),
        "--dtype",
        "fp16",
        "--warmup",
        "5",
        "--iters",
        "10",
        "--attn-warmup",
        "5",
        "--attn-iters",
        "10",
    ]
    output = subprocess.check_output(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
    )
    return json.loads(output)


@app.function(image=image, gpu="A100", timeout=1800)
def profile_a100(seqs: str, heads: str, dims: str, block_sizes: str):
    return _profile(seqs, heads, dims, block_sizes)


@app.function(image=image, gpu="H100", timeout=1800)
def profile_h100(seqs: str, heads: str, dims: str, block_sizes: str):
    return _profile(seqs, heads, dims, block_sizes)


@app.local_entrypoint()
def main(
    target: str = "h100",
    seqs: str = "4096,8192",
    heads: str = "16",
    dims: str = "128",
    block_sizes: str = "64",
):
    if target == "a100":
        print(profile_a100.remote(seqs, heads, dims, block_sizes))
    elif target == "h100":
        print(profile_h100.remote(seqs, heads, dims, block_sizes))
    else:
        raise ValueError("target must be a100 or h100")
