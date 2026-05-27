"""Modal runner for actual-model seed-only decode marginal profiling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-model-decode-marginals")

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
    print(f"[modal-model-decode-marginals] running: {' '.join(cmd[:5])} ...", flush=True)
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
        "/root/StreamAttn/benchmarks/profile_seed_only_model_decode_marginals.py",
        "--model",
        kwargs["model"],
        "--base-layers",
        kwargs["base_layers"],
        "--candidate-layers",
        kwargs["candidate_layers"],
        "--case-modes",
        kwargs["case_modes"],
        "--max-cases",
        str(kwargs["max_cases"]),
        "--prompt-kinds",
        kwargs["prompt_kinds"],
        "--prompt-repeat",
        str(kwargs["prompt_repeat"]),
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
        "--assumed-region-speedup",
        str(kwargs["assumed_region_speedup"]),
        "--target-speedups",
        kwargs["target_speedups"],
    ]
    if kwargs["attn_implementation"]:
        cmd.extend(["--attn-implementation", kwargs["attn_implementation"]])
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=14400)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    base_layers: str = "0,14,16,24,26,27,35",
    candidate_layers: str = "0,2,14,16,18,24,26,27,29,35",
    case_modes: str = "base,add,leave_one_out",
    max_cases: int = 0,
    prompt_kinds: str = "needle,code,long_doc,chat_doc,needle,code,long_doc,chat_doc",
    prompt_repeat: int = 3000,
    batch_size: int = 8,
    max_seq: int = 32768,
    steps: int = 8,
    warmup_steps: int = 2,
    dtype: str = "fp16",
    attn_implementation: str = "",
    top_k: int = 5,
    max_kl: float = 1.0e-4,
    min_topk_overlap: int = 4,
    max_logprob_delta: float = 2.0e-3,
    sample_temperature: float = 0.8,
    sample_top_p: float = 0.95,
    sample_top_k: int = 0,
    sample_seed: int = 1234,
    assumed_region_speedup: float = 3.0,
    target_speedups: str = "1.05,1.10,1.20",
    output_json: str = "",
):
    result = profile_h100.remote(
        model=model,
        base_layers=base_layers,
        candidate_layers=candidate_layers,
        case_modes=case_modes,
        max_cases=max_cases,
        prompt_kinds=prompt_kinds,
        prompt_repeat=prompt_repeat,
        batch_size=batch_size,
        max_seq=max_seq,
        steps=steps,
        warmup_steps=warmup_steps,
        dtype=dtype,
        attn_implementation=attn_implementation,
        top_k=top_k,
        max_kl=max_kl,
        min_topk_overlap=min_topk_overlap,
        max_logprob_delta=max_logprob_delta,
        sample_temperature=sample_temperature,
        sample_top_p=sample_top_p,
        sample_top_k=sample_top_k,
        sample_seed=sample_seed,
        assumed_region_speedup=assumed_region_speedup,
        target_speedups=target_speedups,
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
            "dense": result.get("dense"),
            "base_layers": result.get("base_layers"),
            "candidate_layers": result.get("candidate_layers"),
            "coverage": result.get("coverage"),
            "cases": [
                {
                    "name": row["name"],
                    "kind": row["kind"],
                    "layers": row["layers"],
                    "speedup_vs_dense": row["speedup_vs_dense"],
                    "saved_ms_per_token": row["saved_ms_per_token"],
                    "kl_max": row["safety"]["kl_max"],
                    "passed": row["decision"]["passed"],
                    "recommendation": row["recommendation"],
                    "marginal_vs_base_ms_total": row["marginal_vs_base_ms_total"],
                    "marginal_vs_base_kl": row["marginal_vs_base_kl"],
                }
                for row in result.get("cases", [])
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)


if __name__ == "__main__":
    main()
