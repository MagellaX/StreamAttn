"""Modal runner for seed-only backend floor diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-backend-floor")

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
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _read_prompt_file(path: str) -> str:
    return " ".join(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _parse_seed_sets(raw: str) -> list[dict[str, str]]:
    out = []
    for idx, chunk in enumerate(raw.split(";")):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            name, heads = chunk.split("=", 1)
        else:
            name, heads = f"set{idx}", chunk
        out.append({"name": name.strip(), "heads": heads.strip()})
    if not out:
        raise ValueError("at least one seed-head set is required")
    return out


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], tail: int = 2000) -> dict[str, Any]:
    print(f"[seed-floor] running: {' '.join(cmd[:5])} ...", flush=True)
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
    if output.strip():
        print(output[-tail:], flush=True)
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


def _compact_row(*, profile: dict[str, Any], capture_row: dict[str, Any], seed_set: dict[str, str]) -> dict[str, Any]:
    timing = profile.get("timing") or {}
    quality = profile.get("quality") or {}
    return {
        "layer_id": capture_row.get("layer_id"),
        "seed_set": seed_set["name"],
        "seed_heads": seed_set["heads"],
        "flashinfer_tc_exact_ms": timing.get("flashinfer_tc_exact_ms"),
        "empty_launch_ms": timing.get("empty_launch_ms"),
        "q_only_ms": timing.get("q_only_ms"),
        "qkv_seed_load_no_softmax_ms": timing.get("qkv_seed_load_no_softmax_ms"),
        "seed_direct_full_ms": timing.get("seed_direct_full_ms"),
        "seed_selected_compact_ms": timing.get("seed_selected_compact_ms"),
        "seed_selected_kv_major_ms": timing.get("seed_selected_kv_major_ms"),
        "scatter_selected_only_ms": timing.get("scatter_selected_only_ms"),
        "copy_full_only_ms": timing.get("copy_full_only_ms"),
        "seed_direct_full_speedup_vs_flashinfer": timing.get("seed_direct_full_speedup_vs_flashinfer"),
        "seed_selected_compact_speedup_vs_flashinfer": timing.get(
            "seed_selected_compact_speedup_vs_flashinfer"
        ),
        "seed_selected_kv_major_speedup_vs_flashinfer": timing.get(
            "seed_selected_kv_major_speedup_vs_flashinfer"
        ),
        "empty_launch_fraction_of_flashinfer": timing.get("empty_launch_fraction_of_flashinfer"),
        "seed_direct_full_max_abs_error": (
            quality.get("seed_direct_full_vs_flashinfer") or {}
        ).get("max_abs_error"),
        "decision": profile.get("decision"),
    }


def _run(
    *,
    model: str,
    prompt: str,
    prompt_type: str,
    layers: str,
    max_seq: int,
    kv_len: int,
    dtype: str,
    tensor_space: str,
    seed_head_sets: str,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    prompt_file = "/tmp/streamattn_seed_floor_prompt.txt"
    capture_dir = "/tmp/streamattn_seed_floor_qkv"
    capture_json = f"{capture_dir}/metadata.json"
    Path(prompt_file).write_text(prompt, encoding="utf-8")

    capture = _json_from_cmd(
        [
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
            str(max(max_seq, kv_len)),
            "--kv-len",
            str(kv_len),
            "--query-len",
            "1",
            "--tensor-space",
            tensor_space,
            "--save-v",
            "--output-dir",
            capture_dir,
            "--metadata-json-out",
            capture_json,
        ],
        env=env,
        tail=4000,
    )
    captured_rows = [row for row in capture.get("rows", []) if not row.get("skipped")]
    if not captured_rows:
        raise RuntimeError(f"capture produced no usable rows: {capture}")

    seed_sets = _parse_seed_sets(seed_head_sets)
    compact_rows = []
    raw_rows = []
    total = len(captured_rows) * len(seed_sets)
    idx = 0
    for capture_row in captured_rows:
        true_kv_heads = int(
            (capture_row.get("meta") or {}).get("logical_num_kv_heads")
            or capture_row["shape"]["heads"]
        )
        for seed_set in seed_sets:
            idx += 1
            print(
                "[seed-floor] "
                f"{idx}/{total} layer={capture_row.get('layer_id')} "
                f"set={seed_set['name']} heads={seed_set['heads']}",
                flush=True,
            )
            profile_cmd = [
                "python",
                "/root/StreamAttn/benchmarks/profile_seed_only_backend_floor.py",
                "--q-path",
                capture_row["q_path"],
                "--k-path",
                capture_row["k_path"],
                "--v-path",
                capture_row["v_path"],
                "--true-kv-heads",
                str(true_kv_heads),
                "--seed-heads",
                seed_set["heads"],
                "--dtype",
                dtype,
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
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
                "--flashinfer-tensor-cores",
            ]
            profile = _json_from_cmd(profile_cmd, env=env, tail=3200)
            row = _compact_row(profile=profile, capture_row=capture_row, seed_set=seed_set)
            raw_rows.append(profile)
            compact_rows.append(row)
            print(
                "[seed-floor] result "
                f"flash={row['flashinfer_tc_exact_ms']:.4f} "
                f"direct={row['seed_direct_full_ms']:.4f} "
                f"selected={row['seed_selected_compact_ms']:.4f} "
                f"empty={row['empty_launch_ms']:.4f} "
                f"decision={row['decision']}",
                flush=True,
            )

    return {
        "schema": "streamattn.gate0.seed_only_backend_floor_sweep.v1",
        "capture": {
            "model_id": model,
            "prompt_type": prompt_type,
            "layers": layers,
            "kv_len": kv_len,
            "tensor_space": tensor_space,
            "row_count": len(captured_rows),
        },
        "seed_sets": seed_sets,
        "rows": compact_rows,
        "raw_rows": raw_rows,
    }


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "needle retrieval sparse attention seed only online softmax projection metadata gate0 decode long context",
    prompt_file: str = "",
    prompt_type: str = "needle_seed_only_backend_floor_l8_32k",
    prompt_repeat: int = 2500,
    layers: str = "8",
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    seed_head_sets: str = "trusted=2,3,4;kv0=0,1,2,3,4,5,6;all=0,1,2,3,4,5,6,7,8,9,10,11,12,13",
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    warmup: int = 5,
    iters: int = 20,
    output_json: str = "",
):
    if prompt_file:
        prompt = _read_prompt_file(prompt_file)
    prompt = prompt * max(1, prompt_repeat)
    result = profile_h100.remote(
        model=model,
        prompt=prompt,
        prompt_type=prompt_type,
        layers=layers,
        max_seq=max_seq,
        kv_len=kv_len,
        dtype=dtype,
        tensor_space=tensor_space,
        seed_head_sets=seed_head_sets,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        warmup=warmup,
        iters=iters,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
