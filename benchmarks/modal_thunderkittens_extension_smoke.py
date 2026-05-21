"""Modal runner for the ThunderKittens PyTorch extension smoke."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-thunderkittens-extension-smoke")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git")
    .pip_install("ninja")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _json_from_cmd(cmd: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    print(f"[tk-extension-smoke] running: {' '.join(cmd)}", flush=True)
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
    clone_thunderkittens: bool,
    cuda_arch: str,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    seed_heads: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_thunderkittens_extension_smoke.py",
        "--checkout-dir",
        "/tmp/streamattn_backend_sources",
        "--cuda-arch",
        cuda_arch,
        "--q-heads",
        str(q_heads),
        "--kv-heads",
        str(kv_heads),
        "--head-dim",
        str(head_dim),
        "--seed-heads",
        seed_heads,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    if clone_thunderkittens:
        cmd.append("--clone-thunderkittens")
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=7200)
def smoke_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    clone_thunderkittens: bool = True,
    cuda_arch: str = "sm_90a",
    q_heads: int = 14,
    kv_heads: int = 2,
    head_dim: int = 64,
    seed_heads: str = "2,3,4,6,7",
    warmup: int = 5,
    iters: int = 50,
    output_json: str = "",
):
    result = smoke_h100.remote(
        clone_thunderkittens=clone_thunderkittens,
        cuda_arch=cuda_arch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seed_heads=seed_heads,
        warmup=warmup,
        iters=iters,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
