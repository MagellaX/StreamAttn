"""Modal runner for multi-layer seed-only policy expansion sweeps."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-layer-sweep")

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


def _parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"empty integer list: {raw!r}")
    return values


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
    raise RuntimeError(f"could not parse JSON from command output:\n{output[-6000:]}")


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], label: str) -> dict[str, Any]:
    print(f"[layer-sweep] {label}: {' '.join(cmd[:5])} ...", flush=True)
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
            f"{returncode}: {' '.join(cmd)}\n{output[-9000:]}"
        )
    return _json_from_output(output)


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    policy = result.get("policy") or {}
    summary = policy.get("summary_vs_model_baseline") or {}
    shape = result.get("shape") or {}
    model = result.get("model") or {}
    return {
        "layer_id": model.get("layer_id"),
        "batch": shape.get("batch"),
        "kv_len": shape.get("kv_len"),
        "passes_distribution_gate": bool(policy.get("passes_distribution_gate")),
        "fallback_recommendation": policy.get("fallback_recommendation"),
        "top1_changed_count": summary.get("top1_changed_count"),
        "topk_overlap_min": summary.get("topk_overlap_min"),
        "kl_max": summary.get("kl_max"),
        "max_logit_delta": summary.get("max_logit_delta"),
        "target_next_token_logprob_delta_max_abs": summary.get(
            "target_next_token_logprob_delta_max_abs"
        ),
        "reference_top1_logprob_delta_max_abs": summary.get(
            "reference_top1_logprob_delta_max_abs"
        ),
        "worst_case_by_kl": summary.get("worst_case_by_kl"),
    }


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for layer_id in _parse_ints(kwargs["layers"]):
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
            str(layer_id),
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
        result = _json_from_cmd(cmd, env=env, label=f"L{layer_id}")
        results[str(layer_id)] = result
        rows.append(_summary_row(result))
    passing_layers = [
        int(row["layer_id"])
        for row in rows
        if row.get("passes_distribution_gate")
    ]
    return {
        "schema": "streamattn.gate0.seed_only_layer_sweep.v1",
        "sweep": {
            "model": kwargs["model"],
            "layers": _parse_ints(kwargs["layers"]),
            "prompt_kinds": kwargs["prompt_kinds"],
            "prompt_repeat": kwargs["prompt_repeat"],
            "max_prompts": kwargs["max_prompts"],
            "dtype": kwargs["dtype"],
            "max_seq": kwargs["max_seq"],
            "kv_len": kwargs["kv_len"],
            "position_count": kwargs["position_count"],
            "position_stride": kwargs["position_stride"],
        },
        "seed_config": {
            "block_size": kwargs["block_size"],
            "sink_blocks": kwargs["sink_blocks"],
            "recent_blocks": kwargs["recent_blocks"],
            "middle_seed_blocks": kwargs["middle_seed_blocks"],
            "block_order": kwargs["block_order"],
            "num_warps": kwargs["num_warps"],
            "num_stages": kwargs["num_stages"],
        },
        "safety_gate": {
            "require_top1_match": kwargs["require_top1_match"],
            "min_topk_overlap": kwargs["min_topk_overlap"],
            "max_kl": kwargs["max_kl"],
            "max_logit_delta": kwargs["max_logit_delta"],
            "max_top1_logprob_delta": kwargs["max_top1_logprob_delta"],
            "max_target_logprob_delta": kwargs["max_target_logprob_delta"],
        },
        "passing_layers": passing_layers,
        "rows": rows,
        "results_by_layer": results,
    }


@app.function(image=image, gpu="H100", timeout=14400)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    layers: str = "6,7,8,9,10",
    prompt_kinds: str = "needle,code,long_doc,chat_doc",
    prompt_repeat: int = 3000,
    max_prompts: int = 4,
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    position_count: int = 32,
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
        layers=layers,
        prompt_kinds=prompt_kinds,
        prompt_repeat=prompt_repeat,
        max_prompts=max_prompts,
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
            "sweep": result.get("sweep"),
            "passing_layers": result.get("passing_layers"),
            "rows": result.get("rows"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)


if __name__ == "__main__":
    main()
