"""Modal runner for true-GQA hybrid scheduling profiling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-gate0-true-gqa-hybrid-schedule")

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


def _read_prompt_file(path: str) -> str:
    return " ".join(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _json_from_cmd(cmd: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    print(f"[modal-true-gqa-hybrid] running: {' '.join(cmd[:5])} ...", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        chunks.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    output = "".join(chunks)
    if returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{returncode}: {' '.join(cmd)}\n{output[-6000:]}"
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
    policy_payload: str,
    model: str,
    prompt: str,
    prompt_type: str,
    layers: str,
    max_seq: int,
    kv_len: int,
    dtype: str,
    tensor_space: str,
    trusted_heads: str,
    warmup: int,
    iters: int,
    group_warmup: int,
    group_iters: int,
    metadata_warmup: int,
    metadata_iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    prompt_file = "/tmp/streamattn_gate0_true_gqa_hybrid_prompt.txt"
    policy_file = "/tmp/streamattn_gate0_true_gqa_hybrid_policy.json"
    capture_dir = "/tmp/streamattn_gate0_true_gqa_hybrid_qkv"
    capture_json = f"{capture_dir}/metadata.json"
    Path(prompt_file).write_text(prompt, encoding="utf-8")
    Path(policy_file).write_text(policy_payload, encoding="utf-8")

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
    ]
    capture_payload = _json_from_cmd(capture_cmd, env=env)
    rows = [row for row in capture_payload.get("rows", []) if not row.get("skipped")]
    if not rows:
        raise RuntimeError(f"capture produced no usable rows: {capture_payload}")

    profile_rows = []
    for row in rows:
        true_kv_heads = int((row.get("meta") or {}).get("logical_num_kv_heads") or row["shape"]["heads"])
        print(
            "[modal-true-gqa-hybrid] profiling "
            f"layer={row.get('layer_id')} true_kv_heads={true_kv_heads} shape={row.get('shape')}",
            flush=True,
        )
        profile_cmd = [
            "python",
            "/root/StreamAttn/benchmarks/profile_gate0_true_gqa_hybrid_schedule.py",
            "--q-path",
            row["q_path"],
            "--k-path",
            row["k_path"],
            "--v-path",
            row["v_path"],
            "--policy-json",
            policy_file,
            "--policy-section",
            "stable_entries",
            "--true-kv-heads",
            str(true_kv_heads),
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--group-warmup",
            str(group_warmup),
            "--group-iters",
            str(group_iters),
            "--metadata-warmup",
            str(metadata_warmup),
            "--metadata-iters",
            str(metadata_iters),
        ]
        if trusted_heads:
            profile_cmd.extend(["--trusted-heads", trusted_heads])
        profile = _json_from_cmd(profile_cmd, env=env)
        profile.update(
            {
                "capture": {
                    "model_id": model,
                    "prompt_type": prompt_type,
                    "layer_id": row.get("layer_id"),
                    "tensor_space": row.get("tensor_space"),
                    "shape": row.get("shape"),
                    "logical_num_kv_heads": true_kv_heads,
                }
            }
        )
        profile_rows.append(profile)
    return {
        "capture": {
            "model_id": model,
            "prompt_type": prompt_type,
            "layers": layers,
            "kv_len": kv_len,
            "tensor_space": tensor_space,
            "row_count": len(rows),
        },
        "rows": profile_rows,
    }


@app.function(image=image, gpu="A100", timeout=7200)
def profile_a100(**kwargs):
    return _run(**kwargs)


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    policy_json: str,
    target: str = "h100",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "function run_decode q k v projection metadata online softmax split k exact repair sparse head dense fallback calibrated policy long context attention decode telemetry cache",
    prompt_file: str = "",
    prompt_type: str = "needle_true_gqa_hybrid_l8_32k",
    prompt_repeat: int = 512,
    layers: str = "8",
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    trusted_heads: str = "",
    warmup: int = 3,
    iters: int = 10,
    group_warmup: int = 2,
    group_iters: int = 8,
    metadata_warmup: int = 1,
    metadata_iters: int = 2,
    output_json: str = "",
):
    if prompt_file:
        prompt = _read_prompt_file(prompt_file)
    prompt = prompt * max(1, prompt_repeat)
    policy_payload = Path(policy_json).read_text(encoding="utf-8")
    kwargs = {
        "policy_payload": policy_payload,
        "model": model,
        "prompt": prompt,
        "prompt_type": prompt_type,
        "layers": layers,
        "max_seq": max_seq,
        "kv_len": kv_len,
        "dtype": dtype,
        "tensor_space": tensor_space,
        "trusted_heads": trusted_heads,
        "warmup": warmup,
        "iters": iters,
        "group_warmup": group_warmup,
        "group_iters": group_iters,
        "metadata_warmup": metadata_warmup,
        "metadata_iters": metadata_iters,
    }
    result = profile_a100.remote(**kwargs) if target == "a100" else profile_h100.remote(**kwargs)
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
