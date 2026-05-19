"""Modal runner for Gate-0 projection Nsight Compute anatomy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-gate0-projection-ncu")

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
    output = subprocess.check_output(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stderr=subprocess.STDOUT,
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


def _run_ncu(
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
    backends: str,
    threshold_modes: str,
    projection_dim: int,
    projection_metadata_dtype: str,
    scan_region: str,
    block_order: str,
    filter_margin: float,
    error_budget: float,
    blocks_per_program: int,
    words_per_program: int,
    metric_preset: str,
    replay_mode: str,
    kernel_name: str,
    warmup: int,
    iters: int,
    profile_iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    ncu_path = shutil.which("ncu")
    ncu_version = None
    if ncu_path:
        version = subprocess.run(
            ["ncu", "--version"],
            cwd="/root/StreamAttn",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        ncu_version = version.stdout

    capture_dir = "/tmp/streamattn_gate0_projection_ncu_qk"
    metadata_json = f"{capture_dir}/metadata.json"
    prompt_file = "/tmp/streamattn_gate0_projection_ncu_prompt.txt"
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

    summary_path = Path("/tmp/gate0_projection_ncu_summary.json")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_gate0_projection_ncu.py",
        "--q-path",
        captured["q_path"],
        "--k-path",
        captured["k_path"],
        "--head-index",
        str(head_index),
        "--backends",
        backends,
        "--threshold-modes",
        threshold_modes,
        "--dtype",
        dtype,
        "--projection-dim",
        str(projection_dim),
        "--projection-metadata-dtype",
        projection_metadata_dtype,
        "--block-size",
        str(block_size),
        "--scan-region",
        scan_region,
        "--block-order",
        block_order,
        "--filter-margin",
        str(filter_margin),
        "--error-budget",
        str(error_budget),
        "--blocks-per-program",
        str(blocks_per_program),
        "--words-per-program",
        str(words_per_program),
        "--metric-preset",
        metric_preset,
        "--replay-mode",
        replay_mode,
        "--kernel-name",
        kernel_name,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--profile-iters",
        str(profile_iters),
        "--continue-on-ncu-error",
        "--summary-json-out",
        str(summary_path),
    ]
    result = subprocess.run(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    payload = {
        "capture": {
            "model_id": model,
            "prompt_type": prompt_type,
            "layer_id": captured.get("layer_id"),
            "head_index": head_index,
            "shape": captured.get("shape"),
        },
        "ncu_path": ncu_path,
        "ncu_version": ncu_version,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-8000:],
        "command": cmd,
    }
    if summary_path.exists():
        payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload


@app.function(image=image, gpu="A100", timeout=7200)
def profile_a100(**kwargs):
    return _run_ncu(**kwargs)


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run_ncu(**kwargs)


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
    backends: str = "triton_mask,triton_bitmask",
    threshold_modes: str = "dynamic,static",
    projection_dim: int = 8,
    projection_metadata_dtype: str = "fp16",
    scan_region: str = "middle_only",
    block_order: str = "recent_first",
    filter_margin: float = 32.0,
    error_budget: float = 1e-2,
    blocks_per_program: int = 32,
    words_per_program: int = 4,
    metric_preset: str = "anatomy",
    replay_mode: str = "kernel",
    kernel_name: str = "regex:.*projection.*",
    warmup: int = 3,
    iters: int = 10,
    profile_iters: int = 1,
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
        "backends": backends,
        "threshold_modes": threshold_modes,
        "projection_dim": projection_dim,
        "projection_metadata_dtype": projection_metadata_dtype,
        "scan_region": scan_region,
        "block_order": block_order,
        "filter_margin": filter_margin,
        "error_budget": error_budget,
        "blocks_per_program": blocks_per_program,
        "words_per_program": words_per_program,
        "metric_preset": metric_preset,
        "replay_mode": replay_mode,
        "kernel_name": kernel_name,
        "warmup": warmup,
        "iters": iters,
        "profile_iters": profile_iters,
    }
    payload = profile_a100.remote(**kwargs) if target == "a100" else profile_h100.remote(**kwargs)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
