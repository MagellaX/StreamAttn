"""Modal runner for real-K Gate-0 candidate-filter profiling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-gate0-candidate-filters")

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
    )
    return json.loads(output)


def _run(
    *,
    model: str,
    prompt: str,
    prompt_type: str,
    layers: str,
    max_seq: int,
    kv_len: int,
    query_len: int,
    dtype: str,
    tensor_space: str,
    allow_rope_fallback: bool,
    use_safetensors: bool,
    trust_remote_code: bool,
    block_size: str,
    filter_mode: str,
    projection_dim: str,
    filter_margin: str,
    scan_region: str,
    block_order: str,
    projection_scan_backend: str,
    blocks_per_program: int,
    head_indices: str,
    seed: int,
    error_budget: float,
    max_false_skip_rate: float,
    min_recovery: float,
    max_scan_over_qk: float,
    warmup: int,
    iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    capture_dir = "/tmp/streamattn_gate0_candidate_qk"
    metadata_json = f"{capture_dir}/metadata.json"
    prompt_file = "/tmp/streamattn_gate0_candidate_prompt.txt"
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
        str(query_len),
        "--tensor-space",
        tensor_space,
        "--save-v",
        "--output-dir",
        capture_dir,
        "--metadata-json-out",
        metadata_json,
    ]
    if allow_rope_fallback:
        capture_cmd.append("--allow-rope-fallback")
    if not use_safetensors:
        capture_cmd.append("--no-use-safetensors")
    if trust_remote_code:
        capture_cmd.append("--trust-remote-code")

    capture_payload = _json_from_cmd(capture_cmd, env=env)
    rows = []
    profile_errors = []
    for captured in capture_payload.get("rows", []):
        if captured.get("skipped"):
            profile_errors.append(captured)
            continue
        profile_cmd = [
            "python",
            "/root/StreamAttn/benchmarks/profile_gate0_candidate_filters.py",
            "--q-path",
            captured["q_path"],
            "--k-path",
            captured["k_path"],
            "--tensor-format",
            "pt",
            "--tensor-space",
            captured["tensor_space"],
            "--model-id",
            model,
            "--layer-id",
            str(captured["layer_id"]),
            "--per-head",
            "--block-size",
            block_size,
            "--filter-mode",
            filter_mode,
            "--projection-dim",
            projection_dim,
            "--filter-margin",
            filter_margin,
            "--scan-region",
            scan_region,
            "--block-order",
            block_order,
            "--projection-scan-backend",
            projection_scan_backend,
            "--blocks-per-program",
            str(blocks_per_program),
            "--seed",
            str(seed),
            "--dtype",
            dtype,
            "--error-budget",
            str(error_budget),
            "--max-false-skip-rate",
            str(max_false_skip_rate),
            "--min-recovery",
            str(min_recovery),
            "--max-scan-over-qk",
            str(max_scan_over_qk),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
        ]
        if head_indices:
            profile_cmd.extend(["--head-indices", head_indices])
        try:
            profile_payload = _json_from_cmd(profile_cmd, env=env)
            for row in profile_payload.get("rows", []):
                row["prompt_type"] = prompt_type
                rows.append(row)
        except Exception as exc:
            profile_errors.append(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "model_id": model,
                    "layer_id": captured.get("layer_id"),
                    "prompt_type": prompt_type,
                    "q_path": captured.get("q_path"),
                    "k_path": captured.get("k_path"),
                }
            )

    return {
        "model_id": model,
        "prompt_type": prompt_type,
        "tensor_space": tensor_space,
        "capture": capture_payload,
        "profile_errors": profile_errors,
        "rows": rows,
    }


@app.function(image=image, gpu="A100", timeout=7200)
def profile_a100(**kwargs):
    return _run(**kwargs)


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    target: str = "h100",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "Needle retrieval context with cached KV metadata, online softmax, middle blocks, sink tokens, recent tokens, sparse decode routing, and long-context retrieval. ",
    prompt_file: str = "",
    prompt_type: str = "default",
    prompt_repeat: int = 256,
    layers: str = "8",
    max_seq: int = 4096,
    kv_len: int = 4096,
    query_len: int = 1,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    allow_rope_fallback: bool = False,
    use_safetensors: bool = True,
    trust_remote_code: bool = False,
    block_size: str = "16",
    filter_mode: str = "projection_random,projection_hadamard",
    projection_dim: str = "8,16,32",
    filter_margin: str = "0,2,4,8",
    scan_region: str = "all,middle_only",
    block_order: str = "recent_first",
    projection_scan_backend: str = "triton_mask",
    blocks_per_program: int = 32,
    head_indices: str = "",
    seed: int = 0,
    error_budget: float = 1e-2,
    max_false_skip_rate: float = 0.01,
    min_recovery: float = 0.50,
    max_scan_over_qk: float = 0.25,
    warmup: int = 1,
    iters: int = 3,
    output_json: str = "",
    print_full_json: bool = False,
):
    if prompt_file:
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    prompt = prompt * max(1, prompt_repeat)
    kwargs = {
        "model": model,
        "prompt": prompt,
        "prompt_type": prompt_type,
        "layers": layers,
        "max_seq": max_seq,
        "kv_len": kv_len,
        "query_len": query_len,
        "dtype": dtype,
        "tensor_space": tensor_space,
        "allow_rope_fallback": allow_rope_fallback,
        "use_safetensors": use_safetensors,
        "trust_remote_code": trust_remote_code,
        "block_size": block_size,
        "filter_mode": filter_mode,
        "projection_dim": projection_dim,
        "filter_margin": filter_margin,
        "scan_region": scan_region,
        "block_order": block_order,
        "projection_scan_backend": projection_scan_backend,
        "blocks_per_program": blocks_per_program,
        "head_indices": head_indices,
        "seed": seed,
        "error_budget": error_budget,
        "max_false_skip_rate": max_false_skip_rate,
        "min_recovery": min_recovery,
        "max_scan_over_qk": max_scan_over_qk,
        "warmup": warmup,
        "iters": iters,
    }
    if target == "a100":
        result = profile_a100.remote(**kwargs)
    elif target == "h100":
        result = profile_h100.remote(**kwargs)
    else:
        raise ValueError("target must be a100 or h100")

    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    summary = {
        "model_id": result.get("model_id"),
        "prompt_type": result.get("prompt_type"),
        "tensor_space": result.get("tensor_space"),
        "captured_layers": len(result.get("capture", {}).get("rows", [])),
        "rows": len(result.get("rows", [])),
        "profile_errors": len(result.get("profile_errors", [])),
        "output_json": output_json or None,
    }
    print(json.dumps(result if print_full_json else summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
