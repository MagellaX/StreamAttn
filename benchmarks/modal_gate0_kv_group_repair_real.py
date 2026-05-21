"""Modal runner for real-Qwen KV-group repair profiling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-gate0-kv-group-repair-real")

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
    print(f"[modal-kv-repair-real] running: {' '.join(cmd[:5])} ...", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    output = "".join(lines)
    if returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{returncode}: {' '.join(cmd)}\n{output[-6000:]}"
        )
    return _json_from_output(output)


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
    kv_groups: str,
    budgets: str,
    repair_counts: str,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    warmup: int,
    iters: int,
    group_warmup: int,
    group_iters: int,
    measure_triton_repair: bool,
    repair_num_chunks: int,
    repair_block_d: int,
    measure_tk_bf16: bool,
    tk_num_chunks: int,
):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    prompt_file = "/tmp/streamattn_kv_repair_prompt.txt"
    capture_dir = "/tmp/streamattn_kv_repair_qkv"
    capture_json = f"{capture_dir}/metadata.json"
    Path(prompt_file).write_text(prompt, encoding="utf-8")

    capture = _json_from_cmd(
        [
            "python",
            "-u",
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
            "-u",
            "/root/StreamAttn/benchmarks/profile_gate0_kv_group_repair_real.py",
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
            "--budgets",
            budgets,
            "--repair-counts",
            repair_counts,
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
            "--group-warmup",
            str(group_warmup),
            "--group-iters",
            str(group_iters),
            "--repair-num-chunks",
            str(repair_num_chunks),
            "--repair-block-d",
            str(repair_block_d),
        ]
        if kv_groups:
            profile_cmd.extend(["--kv-groups", kv_groups])
        if measure_triton_repair:
            profile_cmd.append("--measure-triton-repair")
        if measure_tk_bf16:
            profile_cmd.extend(["--measure-tk-bf16", "--tk-num-chunks", str(tk_num_chunks)])
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
        "rows": results,
    }


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "needle retrieval sparse attention seed only exact repair true gqa long context",
    prompt_file: str = "",
    prompt_type: str = "needle_kv_group_repair_l8_32k",
    prompt_repeat: int = 3000,
    layers: str = "8",
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    kv_groups: str = "",
    budgets: str = "strict:1e-2,moderate:1.5e-2,research:5e-2",
    repair_counts: str = "0,1,2,3,4,7",
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    warmup: int = 3,
    iters: int = 10,
    group_warmup: int = 3,
    group_iters: int = 10,
    measure_triton_repair: bool = False,
    repair_num_chunks: int = 256,
    repair_block_d: int = 32,
    measure_tk_bf16: bool = False,
    tk_num_chunks: int = 256,
    output_json: str = "",
):
    if prompt_file:
        prompt = _read_prompt_file(prompt_file)
    prompt = ((prompt.strip() + " ") * max(1, prompt_repeat)).strip()
    result = profile_h100.remote(
        model=model,
        prompt=prompt,
        prompt_type=prompt_type,
        layers=layers,
        max_seq=max_seq,
        kv_len=kv_len,
        dtype=dtype,
        tensor_space=tensor_space,
        kv_groups=kv_groups,
        budgets=budgets,
        repair_counts=repair_counts,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        warmup=warmup,
        iters=iters,
        group_warmup=group_warmup,
        group_iters=group_iters,
        measure_triton_repair=measure_triton_repair,
        repair_num_chunks=repair_num_chunks,
        repair_block_d=repair_block_d,
        measure_tk_bf16=measure_tk_bf16,
        tk_num_chunks=tk_num_chunks,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
