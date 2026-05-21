"""Modal sweep for seed-only Gate-0 against FlashInfer tensor-core exact.

The point of this runner is to stop optimizing against the weaker PyTorch SDPA
baseline.  It captures real post-RoPE Q/K/V once, sweeps seed-only policies, and
reports whether any configuration has a positive oracle margin versus
FlashInfer TC exact.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-gate0-seed-flashinfer-sweep")

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


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_seed_sets(raw: str) -> list[dict[str, str]]:
    """Parse ``name=heads;name=heads`` or plain ``heads;heads``."""

    sets = []
    for idx, chunk in enumerate(raw.split(";")):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            name, heads = chunk.split("=", 1)
            name = name.strip()
            heads = heads.strip()
        else:
            heads = chunk
            name = f"set{idx}"
        sets.append({"name": name, "heads": heads})
    if not sets:
        raise ValueError("at least one seed-head set is required")
    return sets


def _json_from_cmd(cmd: list[str], *, env: dict[str, str], echo_json: bool = False) -> dict[str, Any]:
    print(f"[seed-sweep] running: {' '.join(cmd[:5])} ...", flush=True)
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
    if echo_json:
        print(output, end="", flush=True)
    elif output.strip():
        print(output[-1200:], end="\n", flush=True)
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


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = (row.get("timing") or {}).get(key)
    return float(value) if value is not None else None


def _quality(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("quality") or {}).get("hybrid_seed_selected_error_vs_true_dense") or {}


def _per_head_quality(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("quality") or {}).get("hybrid_seed_selected_error_vs_true_dense_per_head") or {}


def _compact_row(
    *,
    profile: dict[str, Any],
    capture_row: dict[str, Any],
    seed_set: dict[str, str],
    block_size: int,
    middle_seed_blocks: int,
) -> dict[str, Any]:
    timing = profile.get("timing") or {}
    quality = _quality(profile)
    per_head = _per_head_quality(profile)
    return {
        "layer_id": capture_row.get("layer_id"),
        "seed_set": seed_set["name"],
        "seed_heads": seed_set["heads"],
        "block_size": block_size,
        "middle_seed_blocks": middle_seed_blocks,
        "reference_exact_backend": timing.get("reference_exact_backend"),
        "reference_exact_ms": timing.get("reference_exact_ms"),
        "flashinfer_all_true_gqa_ms": timing.get("flashinfer_all_true_gqa_ms"),
        "seed_only_selected_ms": timing.get("seed_only_selected_ms"),
        "seed_only_group_parallel_oracle_ms": timing.get("seed_only_group_parallel_oracle_ms"),
        "exact_remaining_flashinfer_group_parallel_oracle_ms": timing.get(
            "exact_remaining_flashinfer_group_parallel_oracle_ms"
        ),
        "fused_hybrid_parallel_flashinfer_oracle_ms": timing.get(
            "fused_hybrid_parallel_flashinfer_oracle_ms"
        ),
        "fused_hybrid_group_parallel_flashinfer_oracle_ms": timing.get(
            "fused_hybrid_group_parallel_flashinfer_oracle_ms"
        ),
        "seed_only_selected_margin_vs_reference_exact_ms": timing.get(
            "seed_only_selected_margin_vs_reference_exact_ms"
        ),
        "fused_hybrid_parallel_flashinfer_oracle_margin_vs_reference_exact_ms": timing.get(
            "fused_hybrid_parallel_flashinfer_oracle_margin_vs_reference_exact_ms"
        ),
        "fused_hybrid_group_parallel_flashinfer_oracle_margin_vs_reference_exact_ms": timing.get(
            "fused_hybrid_group_parallel_flashinfer_oracle_margin_vs_reference_exact_ms"
        ),
        "seed_only_selected_speedup_vs_reference_exact": timing.get(
            "seed_only_selected_speedup_vs_reference_exact"
        ),
        "fused_hybrid_group_parallel_flashinfer_oracle_speedup_vs_reference_exact": timing.get(
            "fused_hybrid_group_parallel_flashinfer_oracle_speedup_vs_reference_exact"
        ),
        "max_abs_error": quality.get("max_abs_error"),
        "mean_abs_error": quality.get("mean_abs_error"),
        "worst_head": per_head.get("worst_head"),
        "flashinfer_errors": profile.get("flashinfer_errors") or [],
    }


def _best_by_budget(rows: list[dict[str, Any]], budgets: list[float]) -> dict[str, Any]:
    out = {}
    for budget in budgets:
        valid = [
            row
            for row in rows
            if row.get("max_abs_error") is not None
            and float(row["max_abs_error"]) <= budget
        ]
        valid.sort(
            key=lambda row: (
                row.get("fused_hybrid_group_parallel_flashinfer_oracle_margin_vs_reference_exact_ms")
                if row.get("fused_hybrid_group_parallel_flashinfer_oracle_margin_vs_reference_exact_ms")
                is not None
                else -1e9
            ),
            reverse=True,
        )
        out[str(budget)] = valid[0] if valid else None
    return out


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
    block_sizes: str,
    middle_seed_blocks: str,
    warmup: int,
    iters: int,
    group_warmup: int,
    group_iters: int,
    budgets: str,
    echo_profile_json: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    prompt_file = "/tmp/streamattn_seed_sweep_prompt.txt"
    capture_dir = "/tmp/streamattn_seed_sweep_qkv"
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
        echo_json=echo_profile_json,
    )
    captured_rows = [row for row in capture.get("rows", []) if not row.get("skipped")]
    if not captured_rows:
        raise RuntimeError(f"capture produced no usable rows: {capture}")

    seed_sets = _parse_seed_sets(seed_head_sets)
    block_size_values = _parse_csv_ints(block_sizes)
    middle_seed_values = _parse_csv_ints(middle_seed_blocks)
    budget_values = [float(item) for item in budgets.split(",") if item.strip()]
    compact_rows = []
    raw_rows = []
    total = len(captured_rows) * len(seed_sets) * len(block_size_values) * len(middle_seed_values)
    idx = 0

    for capture_row in captured_rows:
        true_kv_heads = int(
            (capture_row.get("meta") or {}).get("logical_num_kv_heads")
            or capture_row["shape"]["heads"]
        )
        for seed_set in seed_sets:
            for block_size in block_size_values:
                for middle_seed in middle_seed_values:
                    idx += 1
                    print(
                        "[seed-sweep] "
                        f"{idx}/{total} layer={capture_row.get('layer_id')} "
                        f"set={seed_set['name']} heads={seed_set['heads']} "
                        f"block={block_size} middle_seed={middle_seed}",
                        flush=True,
                    )
                    profile_cmd = [
                        "python",
                        "/root/StreamAttn/benchmarks/profile_gate0_seed_only_true_gqa.py",
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
                        "--middle-seed-blocks",
                        str(middle_seed),
                        "--warmup",
                        str(warmup),
                        "--iters",
                        str(iters),
                        "--group-warmup",
                        str(group_warmup),
                        "--group-iters",
                        str(group_iters),
                        "--measure-flashinfer",
                        "--flashinfer-tensor-cores",
                    ]
                    profile = _json_from_cmd(
                        profile_cmd,
                        env=env,
                        echo_json=echo_profile_json,
                    )
                    row = _compact_row(
                        profile=profile,
                        capture_row=capture_row,
                        seed_set=seed_set,
                        block_size=block_size,
                        middle_seed_blocks=middle_seed,
                    )
                    raw_rows.append(profile)
                    compact_rows.append(row)
                    print(
                        "[seed-sweep] result "
                        f"ref={row['reference_exact_ms']:.4f} "
                        f"seed={row['seed_only_selected_ms']:.4f} "
                        f"group_oracle={row['fused_hybrid_group_parallel_flashinfer_oracle_ms']:.4f} "
                        f"margin={row['fused_hybrid_group_parallel_flashinfer_oracle_margin_vs_reference_exact_ms']:.4f} "
                        f"err={row['max_abs_error']}",
                        flush=True,
                    )

    sorted_rows = sorted(
        compact_rows,
        key=lambda row: (
            row.get("fused_hybrid_group_parallel_flashinfer_oracle_margin_vs_reference_exact_ms")
            if row.get("fused_hybrid_group_parallel_flashinfer_oracle_margin_vs_reference_exact_ms")
            is not None
            else -1e9
        ),
        reverse=True,
    )
    return {
        "capture": {
            "model_id": model,
            "prompt_type": prompt_type,
            "layers": layers,
            "kv_len": kv_len,
            "tensor_space": tensor_space,
            "row_count": len(captured_rows),
        },
        "sweep": {
            "seed_head_sets": seed_sets,
            "block_sizes": block_size_values,
            "middle_seed_blocks": middle_seed_values,
            "budgets": budget_values,
            "row_count": len(compact_rows),
        },
        "best_by_budget": _best_by_budget(sorted_rows, budget_values),
        "top_rows": sorted_rows[:20],
        "rows": compact_rows,
        "raw_rows": raw_rows if echo_profile_json else [],
    }


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "needle retrieval sparse attention seed only online softmax projection metadata gate0 decode long context",
    prompt_file: str = "",
    prompt_type: str = "needle_seed_flashinfer_sweep_l8_32k",
    prompt_repeat: int = 512,
    layers: str = "8",
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    seed_head_sets: str = "h7=7;moderate=2,3,4,6,7;kv0=2,3,4,6;aggressive=2,3,4,6,7,8,9,11",
    block_sizes: str = "32,64",
    middle_seed_blocks: str = "0,2,4,8",
    warmup: int = 2,
    iters: int = 5,
    group_warmup: int = 2,
    group_iters: int = 5,
    budgets: str = "0.005,0.01,0.015,0.05",
    echo_profile_json: bool = False,
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
        block_sizes=block_sizes,
        middle_seed_blocks=middle_seed_blocks,
        warmup=warmup,
        iters=iters,
        group_warmup=group_warmup,
        group_iters=group_iters,
        budgets=budgets,
        echo_profile_json=echo_profile_json,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
