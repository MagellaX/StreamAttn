"""Run the bounded adaptive residual-predictability gate on an H100."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-adaptive-residual-predictability")
hf_cache = modal.Volume.from_name("streamattn-hf-cache", create_if_missing=True)
hf_secret_name = os.environ.get("STREAMATTN_MODAL_HF_SECRET", "").strip()
hf_secrets = [modal.Secret.from_name(hf_secret_name)] if hf_secret_name else []

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install("transformers>=4.45.0", "accelerate", "sentencepiece", "safetensors")
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
    timeout=14400,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=hf_secrets,
)
def profile_h100(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", "/root/.cache/huggingface")
    command = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_adaptive_residual_predictability.py",
        "--model",
        kwargs["model"],
        "--prompt-file",
        kwargs["prompt_file"],
        "--buckets",
        kwargs["buckets"],
        "--max-prompts",
        str(kwargs["max_prompts"]),
        "--max-prompts-per-bucket",
        str(kwargs["max_prompts_per_bucket"]),
        "--layers",
        kwargs["layers"],
        "--max-seq",
        str(kwargs["max_seq"]),
        "--query-rows",
        str(kwargs["query_rows"]),
        "--block-size",
        str(kwargs["block_size"]),
        "--budgets",
        kwargs["budgets"],
        "--hash-width",
        str(kwargs["hash_width"]),
        "--ridge",
        str(kwargs["ridge"]),
        "--dtype",
        kwargs["dtype"],
        "--device",
        "cuda",
        "--output-json",
        "/tmp/adaptive_residual_predictability.json",
    ]
    print(f"[residual-predictability] model={kwargs['model']} layers={kwargs['layers']}", flush=True)
    subprocess.run(command, cwd="/root/StreamAttn", env=env, check=True)
    hf_cache.commit()
    return json.loads(
        Path("/tmp/adaptive_residual_predictability.json").read_text(encoding="utf-8")
    )


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    prompt_file: str = "benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl",
    buckets: str = "chat_instruction,json_tool,needle_rag,code",
    max_prompts: int = 4,
    max_prompts_per_bucket: int = 1,
    layers: str = "14,26,27",
    max_seq: int = 32768,
    query_rows: int = 16,
    block_size: int = 64,
    budgets: str = "4,8",
    hash_width: int = 32,
    ridge: float = 1.0,
    dtype: str = "fp16",
    output_json: str = "artifacts/adaptive/qwen25_3b_32k_residual_predictability_h100.json",
):
    result = profile_h100.remote(
        model=model,
        prompt_file=prompt_file,
        buckets=buckets,
        max_prompts=max_prompts,
        max_prompts_per_bucket=max_prompts_per_bucket,
        layers=layers,
        max_seq=max_seq,
        query_rows=query_rows,
        block_size=block_size,
        budgets=budgets,
        hash_width=hash_width,
        ridge=ridge,
        dtype=dtype,
    )
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = [
        {
            "layer": row["layer"],
            "budget": row["exact_middle_blocks_per_kv_group"],
            "decision": row["decision"],
            "best_state_feature": row["best_state_feature_on_unseen_prompt"],
            "unseen_prompt": row["split_summary"].get("unseen_prompt"),
        }
        for row in result["reports"]
    ]
    print(json.dumps({"output_json": str(path), "reports": compact}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

