"""Modal runner for fused RoPE+append+seed-only kernel parity."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-fused-rope-parity")

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
        "/root/StreamAttn/benchmarks/profile_seed_only_fused_rope_parity.py",
        "--batch",
        str(kwargs["batch"]),
        "--q-heads",
        str(kwargs["q_heads"]),
        "--kv-heads",
        str(kwargs["kv_heads"]),
        "--head-dim",
        str(kwargs["head_dim"]),
        "--max-seq",
        str(kwargs["max_seq"]),
        "--position",
        str(kwargs["position"]),
        "--dtype",
        kwargs["dtype"],
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
        "--seed",
        str(kwargs["seed"]),
        "--atol",
        str(kwargs["atol"]),
    ]
    if kwargs["shared_cos"]:
        cmd.append("--shared-cos")
    print(f"[modal-fused-rope-parity] running: {' '.join(cmd)}", flush=True)
    output = subprocess.check_output(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stderr=subprocess.STDOUT,
    )
    return _json_from_output(output)


@app.function(image=image, gpu="H100", timeout=900)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    batch: int = 8,
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 128,
    max_seq: int = 32777,
    position: int = 32768,
    dtype: str = "fp16",
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 2,
    seed: int = 1234,
    atol: float = 2.0e-3,
    shared_cos: bool = False,
    output_json: str = "",
):
    result = profile_h100.remote(
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        max_seq=max_seq,
        position=position,
        dtype=dtype,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        num_warps=num_warps,
        num_stages=num_stages,
        seed=seed,
        atol=atol,
        shared_cos=shared_cos,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
