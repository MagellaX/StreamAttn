"""Modal runner for Qwen routed-layer projection floor profiling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-qwen-routed-projection-floor")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install(
        "transformers>=4.45.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    )
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


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], tail: int = 7000) -> dict[str, Any]:
    print(f"[qwen-projection-floor] running: {' '.join(cmd[:5])} ...", flush=True)
    result = subprocess.run(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout
    if output.strip():
        print(output[-tail:], flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{result.returncode}: {' '.join(cmd)}\n{output[-9000:]}"
        )
    decoder = json.JSONDecoder()
    for start, char in enumerate(output):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(output[start:])
            return payload
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"could not parse JSON from command output:\n{output[-5000:]}")


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_qwen_routed_projection_floor.py",
        "--model",
        kwargs["model"],
        "--layers",
        kwargs["layers"],
        "--batch-sizes",
        kwargs["batch_sizes"],
        "--dtype",
        kwargs["dtype"],
        "--device",
        "cuda",
        "--warmup",
        str(kwargs["warmup"]),
        "--iters",
        str(kwargs["iters"]),
        "--seed",
        str(kwargs["seed"]),
    ]
    if kwargs["attn_implementation"]:
        cmd.extend(["--attn-implementation", kwargs["attn_implementation"]])
    if not kwargs["use_safetensors"]:
        cmd.append("--no-use-safetensors")
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=3600)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.function(image=image, gpu="A100", timeout=3600)
def profile_a100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    layers: str = "0,14,16,24,26,27,35",
    batch_sizes: str = "4,8,16",
    dtype: str = "fp16",
    attn_implementation: str = "",
    use_safetensors: bool = True,
    warmup: int = 5,
    iters: int = 30,
    seed: int = 1234,
    gpu: str = "h100",
    output_json: str = "",
):
    runner = profile_a100 if gpu.lower() == "a100" else profile_h100
    result = runner.remote(
        model=model,
        layers=layers,
        batch_sizes=batch_sizes,
        dtype=dtype,
        attn_implementation=attn_implementation,
        use_safetensors=use_safetensors,
        warmup=warmup,
        iters=iters,
        seed=seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        summary = {
            "schema": result.get("schema"),
            "model": result.get("model"),
            "dtype": result.get("dtype"),
            "layers": result.get("layers"),
            "batch_sizes": result.get("batch_sizes"),
            "summary": result.get("summary"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)
