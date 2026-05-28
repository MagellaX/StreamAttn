"""Modal runner for seed-only stress attribution sweeps."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-stress-attribution")

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


def _run_cmd(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"[modal-stress-attribution] running: {' '.join(cmd[:5])} ...", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    in_json_payload = False
    assert process.stdout is not None
    for line in process.stdout:
        if line.lstrip().startswith("{"):
            in_json_payload = True
        if not in_json_payload:
            print(line, end="", flush=True)
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{returncode}: {' '.join(cmd)}"
        )


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_seed_only_stress_attribution.py",
        "--model",
        kwargs["model"],
        "--base-layers",
        kwargs["base_layers"],
        "--add-layers",
        kwargs["add_layers"],
        "--route-set",
        kwargs["route_set"],
        "--prompt-file",
        kwargs["prompt_file"],
        "--prompt-truncation-side",
        kwargs["prompt_truncation_side"],
        "--max-prompts",
        str(kwargs["max_prompts"]),
        "--batch-size",
        str(kwargs["batch_size"]),
        "--device",
        "cuda",
        "--dtype",
        kwargs["dtype"],
        "--max-seq",
        str(kwargs["max_seq"]),
        "--steps",
        str(kwargs["steps"]),
        "--warmup-steps",
        str(kwargs["warmup_steps"]),
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
        "--q-heads",
        str(kwargs["q_heads"]),
        "--kv-heads",
        str(kwargs["kv_heads"]),
        "--head-dim",
        str(kwargs["head_dim"]),
        "--output-dir",
        kwargs["remote_output_dir"],
        "--output-json",
        kwargs["remote_output_json"],
    ]
    if not kwargs["native_routed_cache"]:
        cmd.append("--no-native-routed-cache")
    if not kwargs["fused_rope_append_seed"]:
        cmd.append("--no-fused-rope-append-seed")
    if not kwargs["packed_qkv_projection"]:
        cmd.append("--no-packed-qkv-projection")
    _run_cmd(cmd, env=env)
    return json.loads(Path(kwargs["remote_output_json"]).read_text(encoding="utf-8"))


@app.function(image=image, gpu="H100", timeout=14400)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    base_layers: str = "0,14,16,24,26,27,35",
    add_layers: str = "2,29",
    route_set: str = "leaveout",
    prompt_file: str = "benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl",
    prompt_truncation_side: str = "left",
    max_prompts: int = 8,
    batch_size: int = 8,
    max_seq: int = 32768,
    steps: int = 8,
    warmup_steps: int = 0,
    dtype: str = "fp16",
    top_k: int = 5,
    max_kl: float = 1.0e-4,
    min_topk_overlap: int = 4,
    max_logprob_delta: float = 2.0e-3,
    sample_temperature: float = 0.8,
    sample_top_p: float = 0.95,
    sample_top_k: int = 0,
    sample_seed: int = 1234,
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 128,
    native_routed_cache: bool = True,
    fused_rope_append_seed: bool = True,
    packed_qkv_projection: bool = True,
    output_json: str = "",
):
    result = profile_h100.remote(
        model=model,
        base_layers=base_layers,
        add_layers=add_layers,
        route_set=route_set,
        prompt_file=prompt_file,
        prompt_truncation_side=prompt_truncation_side,
        max_prompts=max_prompts,
        batch_size=batch_size,
        max_seq=max_seq,
        steps=steps,
        warmup_steps=warmup_steps,
        dtype=dtype,
        top_k=top_k,
        max_kl=max_kl,
        min_topk_overlap=min_topk_overlap,
        max_logprob_delta=max_logprob_delta,
        sample_temperature=sample_temperature,
        sample_top_p=sample_top_p,
        sample_top_k=sample_top_k,
        sample_seed=sample_seed,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        native_routed_cache=native_routed_cache,
        fused_rope_append_seed=fused_rope_append_seed,
        packed_qkv_projection=packed_qkv_projection,
        remote_output_dir="/root/StreamAttn/artifacts/gate0/qwen25_3b_32k_b8_stress_attribution",
        remote_output_json="/root/StreamAttn/artifacts/gate0/qwen25_3b_32k_b8_stress_attribution/summary.json",
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        compact = {
            "schema": result.get("schema"),
            "base_layers": result.get("base_layers"),
            "route_set": result.get("route_set"),
            "failure_score": result.get("failure_score"),
            "leaveout_attribution": result.get("leaveout_attribution"),
            "routes": [
                {
                    "route_name": row.get("route_name"),
                    "layers": row.get("layers"),
                    "score": row.get("score"),
                    "top1_changes": row.get("top1_changes"),
                    "sample_changes": row.get("sample_changes"),
                    "kl_max": row.get("kl_max"),
                    "speedup_vs_dense_decode": row.get("speedup_vs_dense_decode"),
                    "worst_bucket": row.get("worst_bucket"),
                }
                for row in result.get("routes", [])
            ],
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
    else:
        print(text)
