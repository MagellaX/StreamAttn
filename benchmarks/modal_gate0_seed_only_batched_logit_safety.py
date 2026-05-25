"""Modal runner for batched seed-only Gate-0 logit safety profiling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-batched-logit-safety")

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
    print(f"[modal-batched-logit-safety] running: {' '.join(cmd[:5])} ...", flush=True)
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
        "/root/StreamAttn/benchmarks/profile_gate0_seed_only_batched_logit_safety.py",
        "--model",
        kwargs["model"],
        "--prompt-kinds",
        kwargs["prompt_kinds"],
        "--prompt-repeat",
        str(kwargs["prompt_repeat"]),
        "--max-prompts",
        str(kwargs["max_prompts"]),
        "--layer-id",
        str(kwargs["layer_id"]),
        "--device",
        "cuda",
        "--dtype",
        kwargs["dtype"],
        "--max-seq",
        str(kwargs["max_seq"]),
        "--kv-len",
        str(kwargs["kv_len"]),
        "--position-count",
        str(kwargs["position_count"]),
        "--position-stride",
        str(kwargs["position_stride"]),
        "--block-size",
        str(kwargs["block_size"]),
        "--sink-blocks",
        str(kwargs["sink_blocks"]),
        "--recent-blocks",
        str(kwargs["recent_blocks"]),
        "--middle-seed-blocks",
        str(kwargs["middle_seed_blocks"]),
        "--block-order",
        kwargs["block_order"],
        "--num-warps",
        str(kwargs["num_warps"]),
        "--num-stages",
        str(kwargs["num_stages"]),
        "--top-k",
        str(kwargs["top_k"]),
        "--min-topk-overlap",
        str(kwargs["min_topk_overlap"]),
        "--max-kl",
        str(kwargs["max_kl"]),
        "--max-logit-delta",
        str(kwargs["max_logit_delta"]),
        "--max-top1-logprob-delta",
        str(kwargs["max_top1_logprob_delta"]),
        "--max-target-logprob-delta",
        str(kwargs["max_target_logprob_delta"]),
    ]
    if not kwargs["require_top1_match"]:
        cmd.append("--no-require-top1-match")
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt_kinds: str = "needle,code,long_doc,needle,code,long_doc,needle,code",
    prompt_repeat: int = 3000,
    max_prompts: int = 8,
    layer_id: int = 8,
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    position_count: int = 8,
    position_stride: int = 1,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 2,
    top_k: int = 5,
    require_top1_match: bool = True,
    min_topk_overlap: int = 4,
    max_kl: float = 1.0e-4,
    max_logit_delta: float = 0.0,
    max_top1_logprob_delta: float = 0.10,
    max_target_logprob_delta: float = 0.10,
    output_json: str = "",
):
    result = profile_h100.remote(
        model=model,
        prompt_kinds=prompt_kinds,
        prompt_repeat=prompt_repeat,
        max_prompts=max_prompts,
        layer_id=layer_id,
        max_seq=max_seq,
        kv_len=kv_len,
        dtype=dtype,
        position_count=position_count,
        position_stride=position_stride,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        num_warps=num_warps,
        num_stages=num_stages,
        top_k=top_k,
        require_top1_match=require_top1_match,
        min_topk_overlap=min_topk_overlap,
        max_kl=max_kl,
        max_logit_delta=max_logit_delta,
        max_top1_logprob_delta=max_top1_logprob_delta,
        max_target_logprob_delta=max_target_logprob_delta,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        summary = {
            "schema": result.get("schema"),
            "shape": result.get("shape"),
            "policy": {
                "passes_distribution_gate": (result.get("policy") or {}).get(
                    "passes_distribution_gate"
                ),
                "fallback_recommendation": (result.get("policy") or {}).get(
                    "fallback_recommendation"
                ),
                "summary_vs_model_baseline": (result.get("policy") or {}).get(
                    "summary_vs_model_baseline"
                ),
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)


if __name__ == "__main__":
    main()
