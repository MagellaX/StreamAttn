"""Modal runner for the head-private split-seed benchmark."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-split-seed")

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


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], tail: int = 7000) -> dict[str, Any]:
    print(f"[split-seed] running: {' '.join(cmd[:5])} ...", flush=True)
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
        "/root/StreamAttn/benchmarks/profile_seed_only_split_seed.py",
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
        "--seed-tiles",
        kwargs["seed_tiles"],
        "--block-order",
        kwargs["block_order"],
        "--direct-num-warps",
        str(kwargs["direct_num_warps"]),
        "--direct-num-stages",
        str(kwargs["direct_num_stages"]),
        "--partial-num-warps",
        str(kwargs["partial_num_warps"]),
        "--partial-num-stages",
        str(kwargs["partial_num_stages"]),
        "--merge-num-warps",
        str(kwargs["merge_num_warps"]),
        "--merge-num-stages",
        str(kwargs["merge_num_stages"]),
        "--sm-count",
        str(kwargs["sm_count"]),
        "--target-waves",
        str(kwargs["target_waves"]),
        "--duplication-byte-budget",
        str(kwargs["duplication_byte_budget"]),
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
    batch_sizes: str = "1,2,4,8",
    q_heads: int = 14,
    kv_heads: int = 2,
    dim: int = 64,
    kv_len: int = 32768,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    seed_tiles: str = "32,64,96,128,192",
    block_order: str = "recent_first",
    direct_num_warps: int = 4,
    direct_num_stages: int = 2,
    partial_num_warps: int = 4,
    partial_num_stages: int = 3,
    merge_num_warps: int = 1,
    merge_num_stages: int = 3,
    sm_count: int = 132,
    target_waves: float = 0.75,
    duplication_byte_budget: float = 0.15,
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
        seed_tiles=seed_tiles,
        block_order=block_order,
        direct_num_warps=direct_num_warps,
        direct_num_stages=direct_num_stages,
        partial_num_warps=partial_num_warps,
        partial_num_stages=partial_num_stages,
        merge_num_warps=merge_num_warps,
        merge_num_stages=merge_num_stages,
        sm_count=sm_count,
        target_waves=target_waves,
        duplication_byte_budget=duplication_byte_budget,
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
            "decision": result.get("decision"),
            "best_by_batch": {
                batch: {
                    "seed_tile_tokens": row.get("seed_tile_tokens"),
                    "csplit": row.get("csplit"),
                    "flashinfer_ms": row.get("flashinfer_exact_ms"),
                    "direct_seed_ms": row.get("direct_seed_ms"),
                    "split_seed_ms": row.get("total_split_seed_ms"),
                    "speedup_vs_flashinfer": row.get("speedup_vs_flashinfer"),
                    "max_err_vs_direct_seed": row.get("max_err_vs_direct_seed"),
                }
                for batch, row in (result.get("best_by_batch") or {}).items()
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)


if __name__ == "__main__":
    main()
