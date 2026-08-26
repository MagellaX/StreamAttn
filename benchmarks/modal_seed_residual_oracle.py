"""Modal H100 runner for the synthetic residual K/V capacity oracle."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-residual-capacity-oracle")
hf_cache = modal.Volume.from_name("streamattn-hf-cache", create_if_missing=True)
hf_secret_name = os.environ.get("STREAMATTN_MODAL_HF_SECRET", "").strip()
hf_secrets = [modal.Secret.from_name(hf_secret_name)] if hf_secret_name else []

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install(
        "transformers>=4.45.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    )
    .add_local_dir(
        ".",
        remote_path="/root/StreamAttn",
        copy=True,
        ignore=[".git", ".git/**", ".pytest_cache/**", "__pycache__/**", "artifacts/**"],
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=hf_secrets,
)
def profile_h100(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", "/root/.cache/huggingface")
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_seed_residual_oracle.py",
        "--model",
        kwargs["model"],
        "--prompt-file",
        kwargs["prompt_file"],
        "--buckets",
        kwargs["buckets"],
        "--max-prompts",
        str(kwargs["max_prompts"]),
        "--layers",
        kwargs["layers"],
        "--max-seq",
        str(kwargs["max_seq"]),
        "--prompt-truncation-side",
        kwargs["prompt_truncation_side"],
        "--query-rows",
        str(kwargs["query_rows"]),
        "--train-rows",
        str(kwargs["train_rows"]),
        "--residual-sizes",
        kwargs["residual_sizes"],
        "--optimization-steps",
        str(kwargs["optimization_steps"]),
        "--learning-rate",
        str(kwargs["learning_rate"]),
        "--projection-loss-weight",
        str(kwargs["projection_loss_weight"]),
        "--linear-feature-sizes",
        kwargs["linear_feature_sizes"],
        "--linear-feature-seeds",
        kwargs["linear_feature_seeds"],
        "--dtype",
        kwargs["dtype"],
        "--device",
        "cuda",
        "--output-json",
        "/tmp/seed_residual_oracle.json",
    ]
    if kwargs["skip_static_residual_fit"]:
        cmd.append("--skip-static-residual-fit")
    print(f"[modal-seed-residual] running: {' '.join(cmd[:8])} ...", flush=True)
    subprocess.run(cmd, cwd="/root/StreamAttn", env=env, check=True)
    hf_cache.commit()
    return json.loads(Path("/tmp/seed_residual_oracle.json").read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    prompt_file: str = "benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl",
    buckets: str = "chat_instruction",
    max_prompts: int = 1,
    layers: str = "26,27",
    max_seq: int = 32768,
    prompt_truncation_side: str = "left",
    query_rows: int = 16,
    train_rows: int = 8,
    residual_sizes: str = "4,8,16,32",
    optimization_steps: int = 150,
    learning_rate: float = 0.03,
    projection_loss_weight: float = 1.0,
    skip_static_residual_fit: bool = False,
    linear_feature_sizes: str = "",
    linear_feature_seeds: str = "0",
    dtype: str = "fp16",
    output_json: str = "artifacts/gate0/qwen25_3b_32k_seed_residual_oracle_chat_l26_l27_h100.json",
):
    result = profile_h100.remote(
        model=model,
        prompt_file=prompt_file,
        buckets=buckets,
        max_prompts=max_prompts,
        layers=layers,
        max_seq=max_seq,
        prompt_truncation_side=prompt_truncation_side,
        query_rows=query_rows,
        train_rows=train_rows,
        residual_sizes=residual_sizes,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        projection_loss_weight=projection_loss_weight,
        skip_static_residual_fit=skip_static_residual_fit,
        linear_feature_sizes=linear_feature_sizes,
        linear_feature_seeds=linear_feature_seeds,
        dtype=dtype,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": result.get("schema"),
                "summary": result.get("summary"),
                "output_json": str(path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
