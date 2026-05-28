"""Modal runner for packed head-private seed-cache probe."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-packed-seed-cache")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
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
    print(f"[packed-seed-cache] running: {' '.join(cmd[:5])} ...", flush=True)
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
        "/root/StreamAttn/benchmarks/profile_seed_only_packed_seed_cache.py",
        "--dtype",
        kwargs["dtype"],
        "--batch-sizes",
        kwargs["batch_sizes"],
        "--q-heads",
        str(kwargs["q_heads"]),
        "--kv-heads",
        str(kwargs["kv_heads"]),
        "--dim",
        str(kwargs["dim"]),
        "--kv-len",
        str(kwargs["kv_len"]),
        "--block-size",
        str(kwargs["block_size"]),
        "--sink-blocks",
        str(kwargs["sink_blocks"]),
        "--recent-blocks",
        str(kwargs["recent_blocks"]),
        "--middle-seed-blocks",
        str(kwargs["middle_seed_blocks"]),
        "--block-order",
        kwargs["block_order"],
        "--direct-num-warps",
        str(kwargs["direct_num_warps"]),
        "--direct-num-stages",
        str(kwargs["direct_num_stages"]),
        "--pack-num-warps",
        str(kwargs["pack_num_warps"]),
        "--pack-num-stages",
        str(kwargs["pack_num_stages"]),
        "--refresh-num-warps",
        str(kwargs["refresh_num_warps"]),
        "--refresh-num-stages",
        str(kwargs["refresh_num_stages"]),
        "--packed-num-warps",
        str(kwargs["packed_num_warps"]),
        "--packed-num-stages",
        str(kwargs["packed_num_stages"]),
        "--ring-num-warps",
        str(kwargs["ring_num_warps"]),
        "--ring-num-stages",
        str(kwargs["ring_num_stages"]),
        "--warmup",
        str(kwargs["warmup"]),
        "--iters",
        str(kwargs["iters"]),
        "--seed",
        str(kwargs["seed"]),
    ]
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=3600)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.function(image=image, gpu="A100", timeout=3600)
def profile_a100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    dtype: str = "fp16",
    batch_sizes: str = "4,8,16",
    q_heads: int = 16,
    kv_heads: int = 2,
    dim: int = 128,
    kv_len: int = 32768,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    direct_num_warps: int = 4,
    direct_num_stages: int = 2,
    pack_num_warps: int = 4,
    pack_num_stages: int = 3,
    refresh_num_warps: int = 4,
    refresh_num_stages: int = 3,
    packed_num_warps: int = 4,
    packed_num_stages: int = 2,
    ring_num_warps: int = 4,
    ring_num_stages: int = 2,
    warmup: int = 5,
    iters: int = 30,
    seed: int = 1234,
    gpu: str = "h100",
    output_json: str = "",
):
    runner = profile_a100 if gpu.lower() == "a100" else profile_h100
    result = runner.remote(
        dtype=dtype,
        batch_sizes=batch_sizes,
        q_heads=q_heads,
        kv_heads=kv_heads,
        dim=dim,
        kv_len=kv_len,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        direct_num_warps=direct_num_warps,
        direct_num_stages=direct_num_stages,
        pack_num_warps=pack_num_warps,
        pack_num_stages=pack_num_stages,
        refresh_num_warps=refresh_num_warps,
        refresh_num_stages=refresh_num_stages,
        packed_num_warps=packed_num_warps,
        packed_num_stages=packed_num_stages,
        ring_num_warps=ring_num_warps,
        ring_num_stages=ring_num_stages,
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
            "shape": result.get("shape"),
            "best_kernel_only": result.get("best_kernel_only"),
            "best_total": result.get("best_total"),
            "best_refresh_total": result.get("best_refresh_total"),
            "best_ring_total": result.get("best_ring_total"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)
