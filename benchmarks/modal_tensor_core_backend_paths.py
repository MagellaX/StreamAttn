"""Modal runner for tensor-core backend path probing."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-tensor-core-backend-paths")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git")
    .pip_install(
        "flashinfer-python",
        "flashinfer-cubin",
        "ninja",
    )
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _json_from_cmd(cmd: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    print(f"[tensor-core-backend-paths] running: {' '.join(cmd)}", flush=True)
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
    print(output, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{result.returncode}: {' '.join(cmd)}\n{output[-6000:]}"
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
    raise RuntimeError(f"could not parse JSON from command output:\n{output[-4000:]}")


def _run(
    *,
    clone_cutlass: bool,
    clone_thunderkittens: bool,
    compile_smoke: bool,
    cuda_arch: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/probe_tensor_core_backend_paths.py",
        "--checkout-dir",
        "/tmp/streamattn_backend_sources",
    ]
    if clone_cutlass:
        cmd.append("--clone-cutlass")
    if clone_thunderkittens:
        cmd.append("--clone-thunderkittens")
    if compile_smoke:
        cmd.append("--compile-smoke")
    if cuda_arch:
        cmd.extend(["--cuda-arch", cuda_arch])
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=7200)
def probe_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    clone_cutlass: bool = False,
    clone_thunderkittens: bool = True,
    compile_smoke: bool = True,
    cuda_arch: str = "sm_90a",
    output_json: str = "",
):
    result = probe_h100.remote(
        clone_cutlass=clone_cutlass,
        clone_thunderkittens=clone_thunderkittens,
        compile_smoke=compile_smoke,
        cuda_arch=cuda_arch,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
