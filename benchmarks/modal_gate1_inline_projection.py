"""Modal runner for the inline projection Gate-1 prototype."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-gate1-inline-projection")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install(
        "triton==3.1.0",
        "transformers>=4.45.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    )
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _json_from_cmd(cmd: list[str], *, env: dict[str, str]) -> dict:
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
    model: str,
    prompt: str,
    prompt_type: str,
    layers: str,
    head_index: int,
    max_seq: int,
    kv_len: int,
    dtype: str,
    tensor_space: str,
    block_size: int,
    tile_size_q: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    projection_dim: int,
    projection_metadata_dtype: str,
    filter_margin: float,
    error_budget: float,
    warmup: int,
    iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    capture_dir = "/tmp/streamattn_gate1_inline_projection_qk"
    metadata_json = f"{capture_dir}/metadata.json"
    prompt_file = "/tmp/streamattn_gate1_inline_projection_prompt.txt"
    Path(prompt_file).parent.mkdir(parents=True, exist_ok=True)
    Path(prompt_file).write_text(prompt, encoding="utf-8")
    capture_cmd = [
        "python",
        "/root/StreamAttn/benchmarks/capture_real_qk_decode.py",
        "--model",
        model,
        "--prompt-file",
        prompt_file,
        "--layers",
        layers,
        "--device",
        "cuda",
        "--dtype",
        dtype,
        "--max-seq",
        str(max_seq),
        "--kv-len",
        str(kv_len),
        "--query-len",
        "1",
        "--tensor-space",
        tensor_space,
        "--save-v",
        "--output-dir",
        capture_dir,
        "--metadata-json-out",
        metadata_json,
    ]
    capture_payload = _json_from_cmd(capture_cmd, env=env)
    rows = [row for row in capture_payload.get("rows", []) if not row.get("skipped")]
    if not rows:
        raise RuntimeError("capture produced no usable rows")
    captured = rows[0]

    profile_cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_gate1_inline_projection.py",
        "--q-path",
        captured["q_path"],
        "--k-path",
        captured["k_path"],
        "--v-path",
        captured["v_path"],
        "--head-index",
        str(head_index),
        "--dtype",
        dtype,
        "--block-size",
        str(block_size),
        "--tile-size-q",
        str(tile_size_q),
        "--sink-blocks",
        str(sink_blocks),
        "--recent-blocks",
        str(recent_blocks),
        "--middle-seed-blocks",
        str(middle_seed_blocks),
        "--block-order",
        block_order,
        "--projection-dim",
        str(projection_dim),
        "--projection-metadata-dtype",
        projection_metadata_dtype,
        "--filter-margin",
        str(filter_margin),
        "--error-budget",
        str(error_budget),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    profile_payload = _json_from_cmd(profile_cmd, env=env)
    return {
        "capture": {
            "model_id": model,
            "prompt_type": prompt_type,
            "layer_id": captured.get("layer_id"),
            "head_index": head_index,
            "shape": captured.get("shape"),
        },
        "profile": profile_payload,
        "profile_command": profile_cmd,
    }


@app.function(image=image, gpu="A100", timeout=7200)
def profile_a100(**kwargs):
    return _run(**kwargs)


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    target: str = "h100",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "Needle retrieval context with hidden key BLUE LANTERN 729 threshold middle recovery surrounded by repeated distractors about cached KV metadata online softmax block summaries post RoPE tensors middle blocks sink tokens recent tokens sparse decode routing and retrieval over long contexts. ",
    prompt_file: str = "",
    prompt_type: str = "needle",
    prompt_repeat: int = 512,
    layers: str = "8",
    head_index: int = 3,
    max_seq: int = 4096,
    kv_len: int = 4096,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    block_size: int = 16,
    tile_size_q: int = 16,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 0,
    block_order: str = "sink_recent_first",
    projection_dim: int = 8,
    projection_metadata_dtype: str = "fp16",
    filter_margin: float = 32.0,
    error_budget: float = 1e-2,
    warmup: int = 5,
    iters: int = 20,
    output_json: str = "",
):
    if prompt_file:
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    prompt = prompt * max(1, prompt_repeat)
    kwargs = {
        "model": model,
        "prompt": prompt,
        "prompt_type": prompt_type,
        "layers": layers,
        "head_index": head_index,
        "max_seq": max_seq,
        "kv_len": kv_len,
        "dtype": dtype,
        "tensor_space": tensor_space,
        "block_size": block_size,
        "tile_size_q": tile_size_q,
        "sink_blocks": sink_blocks,
        "recent_blocks": recent_blocks,
        "middle_seed_blocks": middle_seed_blocks,
        "block_order": block_order,
        "projection_dim": projection_dim,
        "projection_metadata_dtype": projection_metadata_dtype,
        "filter_margin": filter_margin,
        "error_budget": error_budget,
        "warmup": warmup,
        "iters": iters,
    }
    payload = profile_a100.remote(**kwargs) if target == "a100" else profile_h100.remote(**kwargs)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
