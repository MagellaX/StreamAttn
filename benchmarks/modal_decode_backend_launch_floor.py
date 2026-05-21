"""Modal runner for decode backend launch-floor diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-decode-backend-launch-floor")

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


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], echo: bool = True) -> dict[str, Any]:
    print(f"[launch-floor] running: {' '.join(cmd[:5])} ...", flush=True)
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
    if echo:
        print(output, end="", flush=True)
    elif output.strip():
        print(output[-1200:], flush=True)
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
    seed_heads: str,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    warmup: int,
    iters: int,
    measure_kv_major_seed: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    prompt_file = "/tmp/streamattn_launch_floor_prompt.txt"
    capture_dir = "/tmp/streamattn_launch_floor_qkv"
    capture_json = f"{capture_dir}/metadata.json"
    Path(prompt_file).write_text(prompt, encoding="utf-8")

    try:
        flashinfer_config = subprocess.check_output(
            ["flashinfer", "show-config"],
            env=env,
            text=True,
        )
    except Exception as exc:
        flashinfer_config = f"flashinfer show-config failed: {type(exc).__name__}: {exc}"

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

    results = []
    for row in rows:
        true_kv_heads = int((row.get("meta") or {}).get("logical_num_kv_heads") or row["shape"]["heads"])
        profile_cmd = [
            "python",
            "/root/StreamAttn/benchmarks/profile_decode_backend_launch_floor.py",
            "--q-path",
            row["q_path"],
            "--k-path",
            row["k_path"],
            "--v-path",
            row["v_path"],
            "--true-kv-heads",
            str(true_kv_heads),
            "--seed-heads",
            seed_heads,
            "--dtype",
            dtype,
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
            "--measure-flashinfer",
            "--flashinfer-tensor-cores",
        ]
        if measure_kv_major_seed:
            profile_cmd.append("--measure-kv-major-seed")
        profile = _json_from_cmd(profile_cmd, env=env)
        profile["capture"] = {
            "model_id": model,
            "prompt_type": prompt_type,
            "layer_id": row.get("layer_id"),
            "shape": row.get("shape"),
            "logical_num_kv_heads": true_kv_heads,
        }
        results.append(profile)
    return {
        "capture": {
            "model_id": model,
            "prompt_type": prompt_type,
            "layers": layers,
            "kv_len": kv_len,
            "tensor_space": tensor_space,
            "row_count": len(rows),
        },
        "flashinfer_config": flashinfer_config,
        "rows": results,
    }


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "needle retrieval sparse attention seed only online softmax decode backend launch floor",
    prompt_file: str = "",
    prompt_type: str = "needle_launch_floor_l8_64k",
    prompt_repeat: int = 1024,
    layers: str = "8",
    max_seq: int = 65536,
    kv_len: int = 65536,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    seed_heads: str = "4,6",
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 2,
    block_order: str = "recent_first",
    warmup: int = 8,
    iters: int = 50,
    measure_kv_major_seed: bool = True,
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
        seed_heads=seed_heads,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        warmup=warmup,
        iters=iters,
        measure_kv_major_seed=measure_kv_major_seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)

