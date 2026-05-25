"""Modal runner for seed-only wrapper batch-threshold benchmarking."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-wrapper-threshold")

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
    if kind == "chat_doc":
        return (
            "User: Summarize the implementation status.\n"
            "Assistant: StreamAttn has a seed-only batched route, wrapper telemetry, "
            "distribution-aware safety checks, and dense fallback for unsupported requests.\n"
        )
    return (
        "Needle retrieval context with cached KV metadata, online softmax, middle blocks, "
        "sink tokens, recent tokens, sparse decode routing, exact repair, and long-context retrieval. "
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


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], tail: int = 5000) -> dict[str, Any]:
    print(f"[modal-wrapper-threshold] running: {' '.join(cmd[:5])} ...", flush=True)
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
    return _json_from_output(output)


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    capture_dir = "/tmp/streamattn_wrapper_threshold_qkv"
    capture_json = f"{capture_dir}/metadata.json"
    prompt_file = "/tmp/streamattn_wrapper_threshold_prompts.jsonl"
    prompt_kinds = [item.strip() for item in kwargs["prompt_kinds"].split(",") if item.strip()]
    prompts = [
        (_prompt_for_kind(kind).strip() + " ") * max(1, int(kwargs["prompt_repeat"]))
        for kind in prompt_kinds
    ]
    Path(prompt_file).write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    _json_from_cmd(
        [
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
        ],
        env=env,
        tail=2500,
    )
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_seed_only_wrapper_batch_threshold.py",
        "--metadata-json",
        capture_json,
        "--layer-id",
        str(kwargs["layer_id"]),
        "--batch-sizes",
        kwargs["batch_sizes"],
        "--product-min-batch",
        str(kwargs["product_min_batch"]),
        "--forced-min-batch",
        str(kwargs["forced_min_batch"]),
        "--dtype",
        kwargs["dtype"],
        "--policy-json",
        kwargs["policy_json"],
        "--safety-margin",
        str(kwargs["safety_margin"]),
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
    ]
    profile = _json_from_cmd(cmd, env=env, tail=5000)
    profile["capture_summary"] = {
        "prompt_kinds": prompt_kinds,
        "captured_rows": len(prompts),
    }
    return profile


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.function(image=image, gpu="A100", timeout=7200)
def profile_a100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt_kinds: str = (
        "needle,code,long_doc,chat_doc,needle,code,long_doc,chat_doc,"
        "needle,code,long_doc,chat_doc,needle,code,long_doc,chat_doc"
    ),
    prompt_repeat: int = 3000,
    layer_id: int = 8,
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    batch_sizes: str = "1,2,4,8,16",
    product_min_batch: int = 8,
    forced_min_batch: int = 1,
    policy_json: str = "/root/StreamAttn/stream_attention/policies/qwen25_05b_l8_32k_seed_only_batched.json",
    safety_margin: float = 1.10,
    flashinfer_backend: str = "auto",
    page_size: int = 32,
    workspace_mb: int = 256,
    warmup: int = 5,
    iters: int = 20,
    gpu: str = "h100",
    output_json: str = "",
):
    runner = profile_a100 if gpu.lower() == "a100" else profile_h100
    result = runner.remote(
        model=model,
        prompt_kinds=prompt_kinds,
        prompt_repeat=prompt_repeat,
        layer_id=layer_id,
        max_seq=max_seq,
        kv_len=kv_len,
        dtype=dtype,
        tensor_space=tensor_space,
        batch_sizes=batch_sizes,
        product_min_batch=product_min_batch,
        forced_min_batch=forced_min_batch,
        policy_json=policy_json,
        safety_margin=safety_margin,
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
        summary = {
            "schema": result.get("schema"),
            "thresholds": result.get("thresholds"),
            "entries": [
                {
                    "batch": entry.get("batch"),
                    "product_route": entry.get("product_route"),
                    "timing": entry.get("timing"),
                    "decision": entry.get("decision"),
                }
                for entry in result.get("entries", [])
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)


if __name__ == "__main__":
    main()
