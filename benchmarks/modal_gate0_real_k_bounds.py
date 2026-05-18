"""Modal runner for real-K Gate-0 bound profiling.

The capture artifacts are produced and consumed inside one remote function so
large Q/K/V tensors do not need to be copied back to the local machine. The
returned JSON has a top-level ``rows`` field compatible with
``summarize_gate0_summary_bounds.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-gate0-real-k-bounds")

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


def _parse_csv(raw: str) -> list[str]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(item)
    return values


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
    summary_outliers: str,
    block_order: str,
    head_indices: str,
    scan_backend: str,
    blocks_per_program: str,
    error_budget: str,
    warmup: int,
    iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    capture_dir = "/tmp/streamattn_gate0_real_qk"
    metadata_json = f"{capture_dir}/metadata.json"
    prompt_file = "/tmp/streamattn_gate0_prompt.txt"
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
        for budget in _parse_csv(error_budget):
            profile_cmd = [
                "python",
                "/root/StreamAttn/benchmarks/profile_gate0_summary_bounds.py",
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
                "--summary-outliers",
                summary_outliers,
                "--block-order",
                block_order,
                "--scan-backend",
                scan_backend,
                "--blocks-per-program",
                blocks_per_program,
                "--dtype",
                dtype,
                "--error-budget",
                budget,
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
            ]
            if head_indices:
                profile_cmd.extend(["--head-indices", head_indices])
            if captured.get("v_path"):
                profile_cmd.extend(["--v-path", captured["v_path"]])
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
                        "error_budget": budget,
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
    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    prompt: str = "A long technical note about attention kernels, cached KV metadata, online softmax, block summaries, sparse decode routing, and retrieval over long documents. ",
    prompt_file: str = "",
    prompt_type: str = "default",
    prompt_repeat: int = 256,
    layers: str = "0,4,8,12",
    max_seq: int = 4096,
    kv_len: int = 4096,
    query_len: int = 1,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    allow_rope_fallback: bool = False,
    use_safetensors: bool = True,
    trust_remote_code: bool = False,
    block_size: str = "64,128",
    summary_outliers: str = "0,1,2,4",
    block_order: str = "sequential,recent_first,sink_recent_first,summary_desc",
    head_indices: str = "",
    scan_backend: str = "triton",
    blocks_per_program: str = "16,32,64",
    error_budget: str = "1e-3",
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
        "summary_outliers": summary_outliers,
        "block_order": block_order,
        "head_indices": head_indices,
        "scan_backend": scan_backend,
        "blocks_per_program": blocks_per_program,
        "error_budget": error_budget,
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
