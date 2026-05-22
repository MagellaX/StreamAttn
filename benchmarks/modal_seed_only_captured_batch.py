"""Capture real prompts and profile batched seed-only on their Q/K/V rows."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-captured-batch")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install(
        "flashinfer-python",
        "flashinfer-cubin",
        "transformers>=4.45.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    )
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _prompt_for_kind(kind: str) -> str:
    if kind == "code":
        return (
            "def stream_attention_decode(q, k_cache, v_cache, policy):\n"
            "    kv_head = q_head // group_size\n"
            "    if policy.seed_only_group:\n"
            "        schedule_seed_blocks(sink, recent, middle_seed)\n"
            "    return online_softmax_merge(partial_states)\n"
        )
    if kind == "long_doc":
        return (
            "StreamAttn long context technical memorandum. "
            "The system stores cached key and value tensors, maintains online softmax state, "
            "routes true grouped-query attention heads, verifies approximation error, and "
            "falls back to exact decode when calibration is stale. "
        )
    return (
        "Needle retrieval context with cached KV metadata, online softmax, middle blocks, "
        "sink tokens, recent tokens, sparse decode routing, exact repair, and long-context retrieval. "
    )


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], tail: int = 5000) -> dict[str, Any]:
    print(f"[captured-batch] running: {' '.join(cmd[:5])} ...", flush=True)
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
    capture_dir = "/tmp/streamattn_captured_batch_qkv"
    capture_json = f"{capture_dir}/metadata.json"
    prompt_file = "/tmp/streamattn_captured_batch_prompts.jsonl"
    prompts = []
    for kind in [item.strip() for item in kwargs["prompt_kinds"].split(",") if item.strip()]:
        prompts.append((_prompt_for_kind(kind).strip() + " ") * max(1, int(kwargs["prompt_repeat"])))
    Path(prompt_file).write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    capture_cmd = [
        "python",
        "/root/StreamAttn/benchmarks/capture_real_qk_decode.py",
        "--model",
        kwargs["model"],
        "--prompt-file",
        prompt_file,
        "--max-prompts",
        str(len(prompts)),
        "--layers",
        str(kwargs["layer_id"]),
        "--device",
        "cuda",
        "--dtype",
        kwargs["dtype"],
        "--max-seq",
        str(max(kwargs["max_seq"], kwargs["kv_len"])),
        "--kv-len",
        str(kwargs["kv_len"]),
        "--query-len",
        "1",
        "--tensor-space",
        kwargs["tensor_space"],
        "--save-v",
        "--output-dir",
        capture_dir,
        "--metadata-json-out",
        capture_json,
    ]
    capture = _json_from_cmd(capture_cmd, env=env, tail=4000)
    profile = _json_from_cmd(
        [
            "python",
            "/root/StreamAttn/benchmarks/profile_seed_only_captured_batch.py",
            "--metadata-json",
            capture_json,
            "--layer-id",
            str(kwargs["layer_id"]),
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
            "--flashinfer-tensor-cores",
        ],
        env=env,
        tail=5000,
    )
    profile["capture_summary"] = {
        "prompt_kinds": kwargs["prompt_kinds"],
        "captured_rows": len([row for row in capture.get("rows", []) if not row.get("skipped")]),
    }
    return profile


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt_kinds: str = "needle,code,long_doc,needle",
    prompt_repeat: int = 3000,
    layer_id: int = 8,
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 2,
    flashinfer_backend: str = "auto",
    page_size: int = 32,
    workspace_mb: int = 256,
    warmup: int = 5,
    iters: int = 20,
    output_json: str = "",
):
    result = profile_h100.remote(
        model=model,
        prompt_kinds=prompt_kinds,
        prompt_repeat=prompt_repeat,
        layer_id=layer_id,
        max_seq=max_seq,
        kv_len=kv_len,
        dtype=dtype,
        tensor_space=tensor_space,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        num_warps=num_warps,
        num_stages=num_stages,
        flashinfer_backend=flashinfer_backend,
        page_size=page_size,
        workspace_mb=workspace_mb,
        warmup=warmup,
        iters=iters,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
