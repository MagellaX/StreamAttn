"""Modal runner for KV-cache update floor profiling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-kv-cache-update-floor")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install("transformers>=4.45.0,<5", "accelerate", "safetensors")
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


def _json_from_cmd(cmd: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    print(f"[modal-kv-cache-update-floor] running: {' '.join(cmd[:5])} ...", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    in_json_payload = False
    assert process.stdout is not None
    for line in process.stdout:
        chunks.append(line)
        if line.lstrip().startswith("{"):
            in_json_payload = True
        if not in_json_payload:
            print(line, end="", flush=True)
    returncode = process.wait()
    output = "".join(chunks)
    if returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{returncode}: {' '.join(cmd)}\n{output[-6000:]}"
        )
    return _json_from_output(output)


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_kv_cache_update_floor.py",
        "--device",
        "cuda",
        "--dtype",
        kwargs["dtype"],
        "--layer-count",
        str(kwargs["layer_count"]),
        "--batch-size",
        str(kwargs["batch_size"]),
        "--kv-heads",
        str(kwargs["kv_heads"]),
        "--head-dim",
        str(kwargs["head_dim"]),
        "--max-seq",
        str(kwargs["max_seq"]),
        "--update-steps",
        str(kwargs["update_steps"]),
        "--warmup-samples",
        str(kwargs["warmup_samples"]),
        "--samples",
        str(kwargs["samples"]),
        "--triton-block",
        str(kwargs["triton_block"]),
        "--methods",
        kwargs["methods"],
    ]
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=3600)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    dtype: str = "fp16",
    layer_count: int = 7,
    batch_size: int = 8,
    kv_heads: int = 2,
    head_dim: int = 128,
    max_seq: int = 32768,
    update_steps: int = 8,
    warmup_samples: int = 1,
    samples: int = 5,
    triton_block: int = 256,
    methods: str = (
        "hf_dynamic_cache,static_layer_index_copy,direct_slice_layer_loop,"
        "direct_slice_batched_layers,triton_append_batched_layers"
    ),
    output_json: str = "",
):
    result = profile_h100.remote(
        dtype=dtype,
        layer_count=layer_count,
        batch_size=batch_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        max_seq=max_seq,
        update_steps=update_steps,
        warmup_samples=warmup_samples,
        samples=samples,
        triton_block=triton_block,
        methods=methods,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        summary = {
            "schema": result.get("schema"),
            "device": result.get("device"),
            "shape": result.get("shape"),
            "results": result.get("results"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)
