"""Modal runner for seed-only batch scaling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-batch-scaling")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install("flashinfer-python", "flashinfer-cubin")
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


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], tail: int = 5000) -> dict[str, Any]:
    print(f"[seed-batch] running: {' '.join(cmd[:5])} ...", flush=True)
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


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_seed_only_batch_scaling.py",
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
        "--num-warps",
        str(kwargs["num_warps"]),
        "--num-stages",
        str(kwargs["num_stages"]),
        "--flashinfer-backend",
        kwargs["flashinfer_backend"],
        "--page-size",
        str(kwargs["page_size"]),
        "--workspace-mb",
        str(kwargs["workspace_mb"]),
        "--warmup",
        str(kwargs["warmup"]),
        "--iters",
        str(kwargs["iters"]),
        "--seed",
        str(kwargs["seed"]),
        "--flashinfer-tensor-cores",
    ]
    if kwargs["disable_split_kv"]:
        cmd.append("--disable-split-kv")
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
    batch_sizes: str = "1,2,4,8,16,32",
    q_heads: int = 14,
    kv_heads: int = 2,
    dim: int = 64,
    kv_len: int = 32768,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 2,
    flashinfer_backend: str = "auto",
    disable_split_kv: bool = False,
    page_size: int = 32,
    workspace_mb: int = 256,
    warmup: int = 5,
    iters: int = 20,
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
        num_warps=num_warps,
        num_stages=num_stages,
        flashinfer_backend=flashinfer_backend,
        disable_split_kv=disable_split_kv,
        page_size=page_size,
        workspace_mb=workspace_mb,
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
            "break_even_batch": result.get("break_even_batch"),
            "decision": result.get("decision"),
            "rows": [
                {
                    "batch": row.get("batch"),
                    "seed_ms": row.get("seed_direct_full_prealloc_ms"),
                    "flashinfer_ms": row.get("flashinfer_batch_tc_exact_ms"),
                    "speedup": row.get("speedup_vs_flashinfer_batch"),
                }
                for row in result.get("rows", [])
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)
