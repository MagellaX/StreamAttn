"""Modal runner for FlashInfer custom-JIT head-mode smoke."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-flashinfer-custom-jit-head-mode-smoke")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install(
        "flashinfer-python",
        "flashinfer-cubin",
        "ninja",
    )
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _json_from_cmd(cmd: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    print(f"[flashinfer-custom-jit] running: {' '.join(cmd)}", flush=True)
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
    kv_len: int,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: str,
    backend: str,
    mode: str,
    use_tensor_cores: bool,
    disable_split_kv: bool,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    warmup: int,
    iters: int,
    seed_heads: str,
    uri_suffix: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_flashinfer_custom_jit_head_mode_smoke.py",
        "--kv-len",
        str(kv_len),
        "--page-size",
        str(page_size),
        "--q-heads",
        str(q_heads),
        "--kv-heads",
        str(kv_heads),
        "--head-dim",
        str(head_dim),
        "--dtype",
        dtype,
        "--backend",
        backend,
        "--mode",
        mode,
        "--block-size",
        str(block_size),
        "--sink-blocks",
        str(sink_blocks),
        "--recent-blocks",
        str(recent_blocks),
        "--middle-seed-blocks",
        str(middle_seed_blocks),
        "--block-order",
        block_order,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--seed-heads",
        seed_heads,
        "--uri-suffix",
        uri_suffix,
    ]
    if use_tensor_cores:
        cmd.append("--use-tensor-cores")
    if disable_split_kv:
        cmd.append("--disable-split-kv")
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    kv_len: int = 32768,
    page_size: int = 16,
    q_heads: int = 14,
    kv_heads: int = 2,
    head_dim: int = 64,
    dtype: str = "fp16",
    backend: str = "fa2",
    mode: str = "exact_equiv",
    use_tensor_cores: bool = True,
    disable_split_kv: bool = False,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 2,
    block_order: str = "recent_first",
    warmup: int = 8,
    iters: int = 40,
    seed_heads: str = "2,3,4,6,7",
    uri_suffix: str = "v1",
    output_json: str = "",
):
    result = profile_h100.remote(
        kv_len=kv_len,
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        backend=backend,
        mode=mode,
        use_tensor_cores=use_tensor_cores,
        disable_split_kv=disable_split_kv,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        warmup=warmup,
        iters=iters,
        seed_heads=seed_heads,
        uri_suffix=uri_suffix,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
