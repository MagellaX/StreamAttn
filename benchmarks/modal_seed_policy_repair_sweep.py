"""Modal runner for Qwen3B targeted seed-policy repair sweeps."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-policy-repair-sweep")

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
    print(f"[modal-repair-sweep] running: {' '.join(cmd[:5])} ...", flush=True)
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
        raise RuntimeError(f"command failed with return code {returncode}: {' '.join(cmd)}")


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    remote_output_json = kwargs["remote_output_json"]
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_seed_policy_repair_sweep.py",
        "--model",
        kwargs["model"],
        "--prompt-file",
        kwargs["prompt_file"],
        "--target-buckets",
        kwargs["target_buckets"],
        "--variant-set",
        kwargs["variant_set"],
        "--prompt-truncation-side",
        kwargs["prompt_truncation_side"],
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
        "--output-dir",
        kwargs["remote_output_dir"],
        "--output-json",
        remote_output_json,
    ]
    if not kwargs["native_routed_cache"]:
        cmd.append("--no-native-routed-cache")
    if not kwargs["fused_rope_append_seed"]:
        cmd.append("--no-fused-rope-append-seed")
    if not kwargs["packed_qkv_projection"]:
        cmd.append("--no-packed-qkv-projection")
    _run_cmd(cmd, env=env)
    return json.loads(Path(remote_output_json).read_text(encoding="utf-8"))


@app.function(image=image, gpu="H100", timeout=14400)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    prompt_file: str = "benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl",
    target_buckets: str = "chat_instruction,json_tool,needle_rag,noisy_neartie",
    variant_set: str = "minimal",
    prompt_truncation_side: str = "left",
    batch_size: int = 8,
    max_seq: int = 32768,
    steps: int = 8,
    warmup_steps: int = 0,
    dtype: str = "fp16",
    native_routed_cache: bool = True,
    fused_rope_append_seed: bool = True,
    packed_qkv_projection: bool = True,
    output_json: str = "",
):
    result = profile_h100.remote(
        model=model,
        prompt_file=prompt_file,
        target_buckets=target_buckets,
        variant_set=variant_set,
        prompt_truncation_side=prompt_truncation_side,
        batch_size=batch_size,
        max_seq=max_seq,
        steps=steps,
        warmup_steps=warmup_steps,
        dtype=dtype,
        native_routed_cache=native_routed_cache,
        fused_rope_append_seed=fused_rope_append_seed,
        packed_qkv_projection=packed_qkv_projection,
        remote_output_dir="/root/StreamAttn/artifacts/gate0/qwen25_3b_32k_b8_repair_sweep",
        remote_output_json="/root/StreamAttn/artifacts/gate0/qwen25_3b_32k_b8_repair_sweep/summary.json",
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        compact = {
            "schema": result.get("schema"),
            "target_buckets": result.get("target_buckets"),
            "variant_set": result.get("variant_set"),
            "best_route": {
                "route_name": result.get("best_route", {}).get("route_name"),
                "score": result.get("best_route", {}).get("score"),
                "speedup_vs_dense_decode": result.get("best_route", {}).get("speedup_vs_dense_decode"),
                "top1_changes": result.get("best_route", {}).get("top1_changes"),
                "sample_changes": result.get("best_route", {}).get("sample_changes"),
                "kl_max": result.get("best_route", {}).get("kl_max"),
                "worst_bucket": result.get("best_route", {}).get("worst_bucket"),
            },
            "routes": [
                {
                    "route_name": row.get("route_name"),
                    "score": row.get("score"),
                    "speedup_vs_dense_decode": row.get("speedup_vs_dense_decode"),
                    "top1_changes": row.get("top1_changes"),
                    "sample_changes": row.get("sample_changes"),
                    "kl_max": row.get("kl_max"),
                    "worst_bucket": row.get("worst_bucket"),
                }
                for row in result.get("routes", [])
            ],
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
    else:
        print(text)


if __name__ == "__main__":
    main()
