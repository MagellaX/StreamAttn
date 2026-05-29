"""Modal runner for decode-shaped Qwen output projection microbenchmarks."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-qwen-o-proj-triton-profile")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install("triton")
    .add_local_dir(
        ".",
        remote_path="/root/StreamAttn",
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".pytest_cache/**",
            "__pycache__/**",
            "artifacts/**",
        ],
    )
)


def _json_from_output(output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, char in enumerate(output):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(output[start:])
            return payload
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"could not parse JSON from command output:\n{output[-4000:]}")


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_qwen_o_proj_triton.py",
        "--batch",
        str(kwargs["batch"]),
        "--hidden",
        str(kwargs["hidden"]),
        "--dtype",
        kwargs["dtype"],
        "--warmup",
        str(kwargs["warmup"]),
        "--iters",
        str(kwargs["iters"]),
        "--seed",
        str(kwargs["seed"]),
        "--num-warps",
        str(kwargs["num_warps"]),
        "--num-stages",
        str(kwargs["num_stages"]),
        "--variants",
        kwargs["variants"],
    ]
    if kwargs["bias"]:
        cmd.append("--bias")
    process = subprocess.run(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{process.returncode}: {' '.join(cmd)}\n{process.stdout[-6000:]}"
        )
    return _json_from_output(process.stdout)


@app.function(image=image, gpu="H100", timeout=1800)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    batch: int = 8,
    hidden: int = 2048,
    dtype: str = "fp16",
    bias: bool = False,
    warmup: int = 20,
    iters: int = 200,
    seed: int = 1234,
    num_warps: int = 4,
    num_stages: int = 3,
    variants: str = "16,32,64;16,64,64;16,128,64;16,64,128;32,64,64;32,128,64",
    output_json: str = "",
):
    result = profile_h100.remote(
        batch=batch,
        hidden=hidden,
        dtype=dtype,
        bias=bias,
        warmup=warmup,
        iters=iters,
        seed=seed,
        num_warps=num_warps,
        num_stages=num_stages,
        variants=variants,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"shape": result.get("shape"), "baseline": result.get("baseline"), "best": result.get("best")}, indent=2, sort_keys=True))
    else:
        print(text)
