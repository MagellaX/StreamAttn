"""Modal sweep runner for inline projection Gate-1 ablations."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-gate1-inline-projection-sweep")

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


def _parse_values(raw: str, cast):
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def _parse_str_values(raw: str):
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _head_values(raw: str, captured: dict) -> list[int]:
    if str(raw).strip().lower() == "all":
        heads = int((captured.get("shape") or {}).get("heads") or 0)
        if heads <= 0:
            raise ValueError("head_indices=all requires captured shape metadata with a positive head count")
        return list(range(heads))
    return _parse_values(raw, int)


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], check: bool = True) -> tuple[dict | None, str, int]:
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
    if result.returncode != 0 and check:
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
            return payload, output, result.returncode
        except json.JSONDecodeError:
            continue
    if check:
        raise RuntimeError(f"could not parse JSON from command output:\n{output[-4000:]}")
    return None, output, result.returncode


def _run(
    *,
    model: str,
    prompt: str,
    prompt_type: str,
    layers: str,
    head_indices: str,
    max_seq: int,
    kv_len: int,
    kv_lens: str,
    dtype: str,
    tensor_space: str,
    block_size: str,
    tile_size_q: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: str,
    block_order: str,
    projection_dim: int,
    projection_metadata_dtype: str,
    qproj_mode: str,
    filter_margin: str,
    error_budget: float,
    warmup: int,
    iters: int,
    max_cases: int,
    continue_on_error: bool,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    prompt_file = "/tmp/streamattn_gate1_inline_projection_sweep_prompt.txt"
    Path(prompt_file).parent.mkdir(parents=True, exist_ok=True)
    Path(prompt_file).write_text(prompt, encoding="utf-8")
    results = []
    capture_summaries = []
    all_cases = []
    for current_kv_len in _parse_values(kv_lens or str(kv_len), int):
        capture_dir = f"/tmp/streamattn_gate1_inline_projection_sweep_qk/kv{current_kv_len}"
        metadata_json = f"{capture_dir}/metadata.json"
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
            str(max(max_seq, current_kv_len)),
            "--kv-len",
            str(current_kv_len),
            "--query-len",
            "1",
            "--tensor-space",
            tensor_space,
            "--save-v",
            "--output-dir",
            capture_dir,
            "--metadata-json-out",
            metadata_json,
        ]
        capture_payload, _capture_stdout, _capture_code = _json_from_cmd(capture_cmd, env=env)
        rows = [row for row in (capture_payload or {}).get("rows", []) if not row.get("skipped")]
        if not rows:
            skipped_rows = (capture_payload or {}).get("rows", [])
            errors = (capture_payload or {}).get("errors", [])
            raise RuntimeError(
                f"capture produced no usable rows for kv_len={current_kv_len}. "
                f"skipped_rows={skipped_rows[:4]} errors={errors[:4]}"
            )
        capture_summaries.append({"kv_len": current_kv_len, "row_count": len(rows)})
        all_cases.extend(
            (
                (row, current_kv_len, head_index, bs, seed_blocks, margin, order)
                for row in rows
                for head_index in _head_values(head_indices, row)
                for bs, seed_blocks, margin, order in itertools.product(
                    _parse_values(block_size, int),
                    _parse_values(middle_seed_blocks, int),
                    _parse_values(filter_margin, float),
                    _parse_str_values(block_order),
                )
            )
        )

    cases = list(all_cases)
    if max_cases > 0:
        cases = cases[:max_cases]

    for captured, current_kv_len, head_index, bs, seed_blocks, margin, order in cases:
        profile_cmd = [
            "python",
            "/root/StreamAttn/benchmarks/profile_gate1_inline_projection.py",
            "--q-path",
            captured["q_path"],
            "--k-path",
            captured["k_path"],
            "--v-path",
            captured["v_path"],
            "--head-index",
            str(head_index),
            "--dtype",
            dtype,
            "--block-size",
            str(bs),
            "--tile-size-q",
            str(tile_size_q),
            "--sink-blocks",
            str(sink_blocks),
            "--recent-blocks",
            str(recent_blocks),
            "--middle-seed-blocks",
            str(seed_blocks),
            "--block-order",
            order,
            "--projection-dim",
            str(projection_dim),
            "--projection-metadata-dtype",
            projection_metadata_dtype,
            "--qproj-mode",
            qproj_mode,
            "--filter-margin",
            str(margin),
            "--error-budget",
            str(error_budget),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
        ]
        profile_payload, profile_stdout, returncode = _json_from_cmd(
            profile_cmd,
            env=env,
            check=not continue_on_error,
        )
        result = {
            "model_id": model,
            "prompt_type": prompt_type,
            "layer_id": captured.get("layer_id"),
            "head_index": head_index,
            "kv_len": current_kv_len,
            "capture_shape": captured.get("shape"),
            "block_size": bs,
            "middle_seed_blocks": seed_blocks,
            "filter_margin": margin,
            "block_order": order,
            "returncode": returncode,
            "profile_command": profile_cmd,
        }
        if profile_payload is not None:
            profile = profile_payload
            stats = profile.get("stats") or {}
            result.update(
                {
                    "dense_ms": profile.get("dense_ms"),
                    "gate1_mass_ms": profile.get("gate1_mass_ms"),
                    "inline_kernel_ms": profile.get("inline_kernel_ms"),
                    "inline_projection_ms": profile.get("inline_projection_ms"),
                    "q_projection_ms": profile.get("q_projection_ms"),
                    "q_projection_reference_ms": profile.get("q_projection_reference_ms"),
                    "qproj_mode": profile.get("qproj_mode"),
                    "inline_total_ms": profile.get("inline_total_ms"),
                    "inline_vs_gate1_speedup": profile.get("inline_vs_gate1_speedup"),
                    "inline_total_vs_gate1_speedup": profile.get("inline_total_vs_gate1_speedup"),
                    "inline_vs_dense_speedup": profile.get("inline_vs_dense_speedup"),
                    "inline_total_vs_dense_speedup": profile.get("inline_total_vs_dense_speedup"),
                    "projection_skip_fraction": stats.get("projection_skip_fraction"),
                    "pv_executed_fraction": stats.get("pv_executed_fraction"),
                    "projection_skipped_blocks": stats.get("projection_skipped_blocks"),
                    "pv_executed_blocks": stats.get("pv_executed_blocks"),
                    "max_abs_error": profile.get("max_abs_error"),
                    "mean_abs_error": profile.get("mean_abs_error"),
                    "per_head_stats": profile.get("per_head_stats"),
                }
            )
        else:
            result["stdout_tail"] = profile_stdout[-6000:]
        results.append(result)

    best_inline = min(
        (row for row in results if row.get("inline_projection_ms") is not None),
        key=lambda row: row["inline_projection_ms"],
        default=None,
    )
    best_total = min(
        (row for row in results if row.get("inline_total_ms") is not None),
        key=lambda row: row["inline_total_ms"],
        default=None,
    )
    return {
        "capture": {
            "model_id": model,
            "prompt_type": prompt_type,
            "layers": layers,
            "kv_lens": kv_lens or str(kv_len),
            "tensor_space": tensor_space,
            "captures": capture_summaries,
        },
        "sweep": {
            "block_size": block_size,
            "middle_seed_blocks": middle_seed_blocks,
            "filter_margin": filter_margin,
            "block_order": block_order,
            "projection_dim": projection_dim,
            "projection_metadata_dtype": projection_metadata_dtype,
            "qproj_mode": qproj_mode,
            "head_indices": head_indices,
            "kv_lens": kv_lens or str(kv_len),
        },
        "results": results,
        "best_inline_kernel": best_inline,
        "best_inline_total": best_total,
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
    prompt: str = "Needle retrieval context with hidden key BLUE LANTERN 729 threshold middle recovery surrounded by repeated distractors about cached KV metadata online softmax block summaries post RoPE tensors middle blocks sink tokens recent tokens sparse decode routing and retrieval over long contexts. ",
    prompt_file: str = "",
    prompt_type: str = "needle",
    prompt_repeat: int = 512,
    layers: str = "8",
    head_index: int = 3,
    head_indices: str = "",
    max_seq: int = 4096,
    kv_len: int = 4096,
    kv_lens: str = "",
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    block_size: str = "16,32,64",
    tile_size_q: int = 16,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: str = "0,4,8,16",
    block_order: str = "recent_first,sink_recent_first",
    projection_dim: int = 8,
    projection_metadata_dtype: str = "fp16",
    qproj_mode: str = "precomputed",
    filter_margin: str = "8,16,24,32,48",
    error_budget: float = 1e-2,
    warmup: int = 3,
    iters: int = 10,
    max_cases: int = 0,
    continue_on_error: bool = False,
    output_json: str = "",
):
    if prompt_file:
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    prompt = prompt * max(1, prompt_repeat)
    kwargs = {
        "model": model,
        "prompt": prompt,
        "prompt_type": prompt_type,
        "layers": layers,
        "head_indices": head_indices or str(head_index),
        "max_seq": max_seq,
        "kv_len": kv_len,
        "kv_lens": kv_lens,
        "dtype": dtype,
        "tensor_space": tensor_space,
        "block_size": block_size,
        "tile_size_q": tile_size_q,
        "sink_blocks": sink_blocks,
        "recent_blocks": recent_blocks,
        "middle_seed_blocks": middle_seed_blocks,
        "block_order": block_order,
        "projection_dim": projection_dim,
        "projection_metadata_dtype": projection_metadata_dtype,
        "qproj_mode": qproj_mode,
        "filter_margin": filter_margin,
        "error_budget": error_budget,
        "warmup": warmup,
        "iters": iters,
        "max_cases": max_cases,
        "continue_on_error": continue_on_error,
    }
    payload = profile_a100.remote(**kwargs) if target == "a100" else profile_h100.remote(**kwargs)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
