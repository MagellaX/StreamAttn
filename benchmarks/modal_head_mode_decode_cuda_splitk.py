"""Modal runner for split-K CUDA head-mode decode prototype."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-head-mode-decode-cuda-splitk")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install(
        "flashinfer-python",
        "flashinfer-cubin",
        "ninja",
        "transformers>=4.45.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    )
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _read_prompt_file(path: str) -> str:
    return " ".join(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], prefix: str) -> dict[str, Any]:
    print(f"[{prefix}] running: {' '.join(cmd[:5])} ...", flush=True)
    result = subprocess.run(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout
    print(output, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{result.returncode}: {' '.join(cmd)}\n{output[-6000:]}"
        )
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


def _profile_base_cmd(
    *,
    dtype: str,
    seed_heads: str,
    num_chunks: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    threads: int,
    warmup: int,
    iters: int,
    flashinfer_tensor_cores: bool,
) -> list[str]:
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_head_mode_decode_cuda_splitk.py",
        "--dtype",
        dtype,
        "--seed-heads",
        seed_heads,
        "--num-chunks",
        str(num_chunks),
        "--block-size",
        str(block_size),
        "--sink-blocks",
        str(sink_blocks),
        "--recent-blocks",
        str(recent_blocks),
        "--middle-seed-blocks",
        str(middle_seed_blocks),
        "--block-order",
        block_order,
        "--threads",
        str(threads),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    if flashinfer_tensor_cores:
        cmd.append("--flashinfer-tensor-cores")
    return cmd


def _run_synthetic(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    cmd = _profile_base_cmd(
        dtype=kwargs["dtype"],
        seed_heads=kwargs["seed_heads"],
        num_chunks=kwargs["num_chunks"],
        block_size=kwargs["block_size"],
        sink_blocks=kwargs["sink_blocks"],
        recent_blocks=kwargs["recent_blocks"],
        middle_seed_blocks=kwargs["middle_seed_blocks"],
        block_order=kwargs["block_order"],
        threads=kwargs["threads"],
        warmup=kwargs["warmup"],
        iters=kwargs["iters"],
        flashinfer_tensor_cores=kwargs["flashinfer_tensor_cores"],
    )
    cmd.extend(
        [
            "--kv-len",
            str(kwargs["kv_len"]),
            "--q-heads",
            str(kwargs["q_heads"]),
            "--kv-heads",
            str(kwargs["kv_heads"]),
            "--head-dim",
            str(kwargs["head_dim"]),
        ]
    )
    return _json_from_cmd(cmd, env=env, prefix="head-mode-splitk-synth")


def _run_real(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

    prompt_file = "/tmp/streamattn_head_mode_splitk_prompt.txt"
    capture_dir = "/tmp/streamattn_head_mode_splitk_qkv"
    capture_json = f"{capture_dir}/metadata.json"
    Path(prompt_file).write_text(kwargs["prompt"], encoding="utf-8")

    capture = _json_from_cmd(
        [
            "python",
            "/root/StreamAttn/benchmarks/capture_real_qk_decode.py",
            "--model",
            kwargs["model"],
            "--prompt-file",
            prompt_file,
            "--layers",
            kwargs["layers"],
            "--device",
            "cuda",
            "--dtype",
            kwargs["dtype"],
            "--max-seq",
            str(max(kwargs["max_seq"], kwargs["kv_len"])),
            "--kv-len",
            str(kwargs["kv_len"]),
            "--query-len",
            "1",
            "--tensor-space",
            kwargs["tensor_space"],
            "--save-v",
            "--output-dir",
            capture_dir,
            "--metadata-json-out",
            capture_json,
        ],
        env=env,
        prefix="head-mode-splitk-capture",
    )
    rows = [row for row in capture.get("rows", []) if not row.get("skipped")]
    if not rows:
        raise RuntimeError(f"capture produced no usable rows: {capture}")

    results = []
    for row in rows:
        true_kv_heads = int((row.get("meta") or {}).get("logical_num_kv_heads") or row["shape"]["heads"])
        cmd = _profile_base_cmd(
            dtype=kwargs["dtype"],
            seed_heads=kwargs["seed_heads"],
            num_chunks=kwargs["num_chunks"],
            block_size=kwargs["block_size"],
            sink_blocks=kwargs["sink_blocks"],
            recent_blocks=kwargs["recent_blocks"],
            middle_seed_blocks=kwargs["middle_seed_blocks"],
            block_order=kwargs["block_order"],
            threads=kwargs["threads"],
            warmup=kwargs["warmup"],
            iters=kwargs["iters"],
            flashinfer_tensor_cores=kwargs["flashinfer_tensor_cores"],
        )
        cmd.extend(
            [
                "--q-path",
                row["q_path"],
                "--k-path",
                row["k_path"],
                "--v-path",
                row["v_path"],
                "--true-kv-heads",
                str(true_kv_heads),
            ]
        )
        profile = _json_from_cmd(cmd, env=env, prefix="head-mode-splitk-profile")
        profile["capture"] = {
            "model_id": kwargs["model"],
            "prompt_type": kwargs["prompt_type"],
            "layer_id": row.get("layer_id"),
            "shape": row.get("shape"),
            "logical_num_kv_heads": true_kv_heads,
        }
        results.append(profile)
    return {
        "capture": {
            "model_id": kwargs["model"],
            "prompt_type": kwargs["prompt_type"],
            "layers": kwargs["layers"],
            "kv_len": kwargs["kv_len"],
            "tensor_space": kwargs["tensor_space"],
            "row_count": len(rows),
        },
        "rows": results,
    }


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    synthetic = kwargs.pop("synthetic")
    if synthetic:
        synthetic_keys = {
            "kv_len",
            "q_heads",
            "kv_heads",
            "head_dim",
            "dtype",
            "seed_heads",
            "num_chunks",
            "block_size",
            "sink_blocks",
            "recent_blocks",
            "middle_seed_blocks",
            "block_order",
            "threads",
            "warmup",
            "iters",
            "flashinfer_tensor_cores",
        }
        return _run_synthetic(**{key: value for key, value in kwargs.items() if key in synthetic_keys})
    real_keys = {
        "model",
        "prompt",
        "prompt_type",
        "layers",
        "max_seq",
        "kv_len",
        "dtype",
        "tensor_space",
        "seed_heads",
        "num_chunks",
        "block_size",
        "sink_blocks",
        "recent_blocks",
        "middle_seed_blocks",
        "block_order",
        "threads",
        "warmup",
        "iters",
        "flashinfer_tensor_cores",
    }
    return _run_real(**{key: value for key, value in kwargs.items() if key in real_keys})


@app.local_entrypoint()
def main(
    synthetic: bool = True,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "needle retrieval sparse attention seed only online softmax splitk backend long context",
    prompt_file: str = "",
    prompt_type: str = "needle_head_mode_splitk_l8_32k",
    prompt_repeat: int = 4096,
    layers: str = "8",
    max_seq: int = 32768,
    kv_len: int = 32768,
    q_heads: int = 14,
    kv_heads: int = 2,
    head_dim: int = 64,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    seed_heads: str = "2,3,4,6,7",
    num_chunks: int = 64,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 2,
    block_order: str = "recent_first",
    threads: int = 128,
    warmup: int = 4,
    iters: int = 20,
    flashinfer_tensor_cores: bool = True,
    output_json: str = "",
):
    if prompt_file:
        prompt = _read_prompt_file(prompt_file)
    prompt = prompt * max(1, prompt_repeat)
    result = profile_h100.remote(
        synthetic=synthetic,
        model=model,
        prompt=prompt,
        prompt_type=prompt_type,
        layers=layers,
        max_seq=max_seq,
        kv_len=kv_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        tensor_space=tensor_space,
        seed_heads=seed_heads,
        num_chunks=num_chunks,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        threads=threads,
        warmup=warmup,
        iters=iters,
        flashinfer_tensor_cores=flashinfer_tensor_cores,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
