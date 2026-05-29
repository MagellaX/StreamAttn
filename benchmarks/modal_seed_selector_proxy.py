"""Modal runner for seed selector proxy diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-selector-proxy")

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
    print(f"[modal-seed-selector-proxy] running: {' '.join(cmd[:5])} ...", flush=True)
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

    remote_failure_artifact = ""
    failure_artifact_json = kwargs.get("failure_artifact_json") or ""
    if failure_artifact_json:
        remote_failure_artifact = "/root/StreamAttn/artifacts/gate0/seed_selector_failure_source.json"
        path = Path(remote_failure_artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(failure_artifact_json, encoding="utf-8")

    remote_output_json = kwargs["remote_output_json"]
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_seed_selector_proxy.py",
        "--model",
        kwargs["model"],
        "--prompt-file",
        kwargs["prompt_file"],
        "--prompt-truncation-side",
        kwargs["prompt_truncation_side"],
        "--max-prompts",
        str(kwargs["max_prompts"]),
        "--batch-size",
        str(kwargs["batch_size"]),
        "--max-seq",
        str(kwargs["max_seq"]),
        "--device",
        "cuda",
        "--dtype",
        kwargs["dtype"],
        "--target-layers",
        kwargs["target_layers"],
        "--routed-layers",
        kwargs["routed_layers"],
        "--target-buckets",
        kwargs["target_buckets"],
        "--selector-profiles",
        kwargs["selector_profiles"],
        "--failure-artifact",
        remote_failure_artifact,
        "--max-rows",
        str(kwargs["max_rows"]),
        "--output-json",
        remote_output_json,
    ]
    if not kwargs["include_step0"]:
        cmd.append("--no-include-step0")
    _run_cmd(cmd, env=env)
    return json.loads(Path(remote_output_json).read_text(encoding="utf-8"))


@app.function(image=image, gpu="H100", timeout=14400)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    prompt_file: str = "benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl",
    prompt_truncation_side: str = "left",
    max_prompts: int = 8,
    batch_size: int = 8,
    max_seq: int = 32768,
    dtype: str = "fp16",
    target_layers: str = "26,27",
    routed_layers: str = "0,14,16,24,26,27,35",
    target_buckets: str = "chat_instruction,noisy_neartie,json_tool,needle_rag",
    selector_profiles: str = (
        "fixed_policy,block_mean_proxy,block_l2_bound_proxy,"
        "support_top2_norm,support_top4_norm,support_top2_norm_refine16,"
        "support_top2_norm_refine32,support_top4_norm_refine16,"
        "support_top4_norm_refine32,support_extreme2_mean,"
        "support_extreme4_mean,support_extreme2_mean_refine32,"
        "support_extreme4_mean_refine32,support_rand4,support_rand8,"
        "support_rand4_refine32,support_rand8_refine32,qk_block_max,"
        "exact_mass_oracle,value_residual_oracle"
    ),
    failure_artifact: str = "artifacts/gate0/qwen25_3b_32k_b8_model_decode/strict7_stress_pack_left_b8_h100.json",
    include_step0: bool = True,
    max_rows: int = 0,
    output_json: str = "",
):
    failure_artifact_json = ""
    if failure_artifact:
        path = Path(failure_artifact)
        if path.exists():
            failure_artifact_json = path.read_text(encoding="utf-8")
        else:
            print(f"[modal-seed-selector-proxy] warning: failure artifact not found: {failure_artifact}", flush=True)

    result = profile_h100.remote(
        model=model,
        prompt_file=prompt_file,
        prompt_truncation_side=prompt_truncation_side,
        max_prompts=max_prompts,
        batch_size=batch_size,
        max_seq=max_seq,
        dtype=dtype,
        target_layers=target_layers,
        routed_layers=routed_layers,
        target_buckets=target_buckets,
        selector_profiles=selector_profiles,
        failure_artifact_json=failure_artifact_json,
        include_step0=include_step0,
        max_rows=max_rows,
        remote_output_json="/root/StreamAttn/artifacts/gate0/seed_selector_proxy/l26_l27.json",
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        compact = {
            "schema": result.get("schema"),
            "target_layers": result.get("target_layers"),
            "routed_layers": result.get("routed_layers"),
            "target_buckets": result.get("target_buckets"),
            "selector_profiles": result.get("selector_profiles"),
            "capture_steps": result.get("capture_steps"),
            "summary": result.get("summary"),
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
    else:
        print(text)


if __name__ == "__main__":
    main()
