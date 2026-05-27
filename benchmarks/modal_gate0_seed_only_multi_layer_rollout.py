"""Modal runner for multi-layer seed-only closed-loop rollout safety."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-multi-layer-rollout")

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
    print(f"[modal-multi-layer-rollout] running: {' '.join(cmd[:5])} ...", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    in_json_payload = False
    assert process.stdout is not None
    for line in process.stdout:
        chunks.append(line)
        if line.lstrip().startswith("{"):
            in_json_payload = True
        if not in_json_payload:
            print(line, end="", flush=True)
    returncode = process.wait()
    output = "".join(chunks)
    if returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{returncode}: {' '.join(cmd)}\n{output[-6000:]}"
        )
    return _json_from_output(output)


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_gate0_seed_only_multi_layer_rollout.py",
        "--model",
        kwargs["model"],
        "--layers",
        kwargs["layers"],
        "--policy-names",
        kwargs["policy_names"],
        "--prompt-kinds",
        kwargs["prompt_kinds"],
        "--prompt-repeat",
        str(kwargs["prompt_repeat"]),
        "--batch-size",
        str(kwargs["batch_size"]),
        "--model-batch-size",
        str(kwargs["model_batch_size"]),
        "--device",
        "cuda",
        "--dtype",
        kwargs["dtype"],
        "--max-seq",
        str(kwargs["max_seq"]),
        "--steps",
        str(kwargs["steps"]),
        "--mode",
        kwargs["mode"],
        "--top-k",
        str(kwargs["top_k"]),
        "--max-kl",
        str(kwargs["max_kl"]),
        "--min-topk-overlap",
        str(kwargs["min_topk_overlap"]),
        "--max-logprob-delta",
        str(kwargs["max_logprob_delta"]),
        "--sample-temperature",
        str(kwargs["sample_temperature"]),
        "--sample-top-p",
        str(kwargs["sample_top_p"]),
        "--sample-top-k",
        str(kwargs["sample_top_k"]),
        "--sample-seed",
        str(kwargs["sample_seed"]),
    ]
    if not kwargs["use_packaged_policies"]:
        cmd.append("--no-use-packaged-policies")
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    layers: str = "0,14,16,24,26,27,29,35",
    policy_names: str = "",
    use_packaged_policies: bool = True,
    prompt_kinds: str = "needle,code,long_doc,chat_doc",
    prompt_repeat: int = 3000,
    batch_size: int = 4,
    model_batch_size: int = 1,
    max_seq: int = 32768,
    steps: int = 32,
    mode: str = "all",
    dtype: str = "fp16",
    top_k: int = 5,
    max_kl: float = 1.0e-4,
    min_topk_overlap: int = 4,
    max_logprob_delta: float = 2.0e-3,
    sample_temperature: float = 0.8,
    sample_top_p: float = 0.95,
    sample_top_k: int = 0,
    sample_seed: int = 1234,
    output_json: str = "",
):
    result = profile_h100.remote(
        model=model,
        layers=layers,
        policy_names=policy_names,
        use_packaged_policies=use_packaged_policies,
        prompt_kinds=prompt_kinds,
        prompt_repeat=prompt_repeat,
        batch_size=batch_size,
        model_batch_size=model_batch_size,
        max_seq=max_seq,
        steps=steps,
        mode=mode,
        dtype=dtype,
        top_k=top_k,
        max_kl=max_kl,
        min_topk_overlap=min_topk_overlap,
        max_logprob_delta=max_logprob_delta,
        sample_temperature=sample_temperature,
        sample_top_p=sample_top_p,
        sample_top_k=sample_top_k,
        sample_seed=sample_seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        summary = {
            "schema": result.get("schema"),
            "model": result.get("model"),
            "shape": result.get("shape"),
            "route_bundle": result.get("route_bundle"),
            "attention_runtime_estimate": result.get("attention_runtime_estimate"),
            "teacher_forced": (result.get("teacher_forced") or {}).get("summary"),
            "greedy": (result.get("greedy") or {}).get("summary"),
            "sampling": (result.get("sampling") or {}).get("summary"),
            "decision": result.get("decision"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)
