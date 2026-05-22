"""Modal runner for Triton seed-only backend tuning."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-triton-tune")

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


def _read_prompt_file(path: str) -> str:
    return " ".join(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], tail: int = 4000) -> dict[str, Any]:
    print(f"[triton-tune] running: {' '.join(cmd[:5])} ...", flush=True)
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


def _run(
    *,
    model: str,
    prompt: str,
    prompt_type: str,
    layers: str,
    max_seq: int,
    kv_len: int,
    dtype: str,
    tensor_space: str,
    block_sizes: str,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks_list: str,
    block_order: str,
    num_warps_list: str,
    num_stages_list: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    prompt_file = "/tmp/streamattn_seed_tune_prompt.txt"
    capture_dir = "/tmp/streamattn_seed_tune_qkv"
    capture_json = f"{capture_dir}/metadata.json"
    Path(prompt_file).write_text(prompt, encoding="utf-8")

    capture = _json_from_cmd(
        [
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
            str(max(max_seq, kv_len)),
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
            capture_json,
        ],
        env=env,
    )
    rows = [row for row in capture.get("rows", []) if not row.get("skipped")]
    if not rows:
        raise RuntimeError(f"capture produced no usable rows: {capture}")

    profiles = []
    for row in rows:
        true_kv_heads = int((row.get("meta") or {}).get("logical_num_kv_heads") or row["shape"]["heads"])
        profile = _json_from_cmd(
            [
                "python",
                "/root/StreamAttn/benchmarks/profile_seed_only_triton_tune.py",
                "--q-path",
                row["q_path"],
                "--k-path",
                row["k_path"],
                "--v-path",
                row["v_path"],
                "--true-kv-heads",
                str(true_kv_heads),
                "--dtype",
                dtype,
                "--block-sizes",
                block_sizes,
                "--sink-blocks",
                str(sink_blocks),
                "--recent-blocks",
                str(recent_blocks),
                "--middle-seed-blocks-list",
                middle_seed_blocks_list,
                "--block-order",
                block_order,
                "--num-warps-list",
                num_warps_list,
                "--num-stages-list",
                num_stages_list,
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
                "--flashinfer-tensor-cores",
            ],
            env=env,
            tail=6000,
        )
        profile["capture"] = {
            "model_id": model,
            "prompt_type": prompt_type,
            "layer_id": row.get("layer_id"),
            "shape": row.get("shape"),
            "logical_num_kv_heads": true_kv_heads,
        }
        profiles.append(profile)
    return {
        "schema": "streamattn.gate0.seed_only_triton_tune_sweep.v1",
        "capture": {
            "model_id": model,
            "prompt_type": prompt_type,
            "layers": layers,
            "kv_len": kv_len,
            "tensor_space": tensor_space,
            "row_count": len(rows),
        },
        "rows": profiles,
    }


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "needle retrieval sparse attention seed only online softmax projection metadata gate0 decode long context",
    prompt_file: str = "",
    prompt_type: str = "needle_seed_only_triton_tune_l8_32k",
    prompt_repeat: int = 2500,
    layers: str = "8",
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    block_sizes: str = "16,32,64",
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks_list: str = "0,2,4,8",
    block_order: str = "recent_first",
    num_warps_list: str = "1,2,4,8",
    num_stages_list: str = "2,3,4",
    warmup: int = 5,
    iters: int = 20,
    output_json: str = "",
):
    if prompt_file:
        prompt = _read_prompt_file(prompt_file)
    prompt = prompt * max(1, prompt_repeat)
    result = profile_h100.remote(
        model=model,
        prompt=prompt,
        prompt_type=prompt_type,
        layers=layers,
        max_seq=max_seq,
        kv_len=kv_len,
        dtype=dtype,
        tensor_space=tensor_space,
        block_sizes=block_sizes,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks_list=middle_seed_blocks_list,
        block_order=block_order,
        num_warps_list=num_warps_list,
        num_stages_list=num_stages_list,
        warmup=warmup,
        iters=iters,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
