"""Modal runner for split-K inline projection profiling."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-gate1-inline-projection-splitk")

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
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def _head_values(raw: str, fallback: int, captured: dict[str, Any]) -> list[int]:
    if raw:
        values = _parse_values(raw, int)
    else:
        values = [int(fallback)]
    heads = ((captured.get("shape") or {}).get("heads")) or 0
    expanded: list[int] = []
    for value in values:
        if value < 0:
            expanded.extend(range(int(heads)))
        else:
            expanded.append(value)
    return sorted(set(expanded))


def _head_groups(raw: str, fallback: int, captured: dict[str, Any]) -> list[str]:
    if raw:
        return [group.strip() for group in str(raw).split(";") if group.strip()]
    return [str(value) for value in _head_values("", fallback, captured)]


def _json_from_cmd(cmd: list[str], *, env: dict[str, str]) -> dict:
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


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in results:
        stats = row.get("splitk_stats") or {}
        error = row.get("splitk_error_vs_dense") or {}
        projection_skip = float(stats.get("projection_skip_fraction") or 0.0)
        max_error = float(error.get("max_abs_error") or 0.0)
        rows.append(
            {
                "kv_len": row.get("kv_len"),
                "head_index": row.get("head_index"),
                "head_indices": row.get("head_indices"),
                "selected_head_count": row.get("selected_head_count"),
                "projection_dim": row.get("projection_dim"),
                "projection_seed": row.get("projection_seed"),
                "filter_margin": row.get("filter_margin"),
                "chunk_anchor_blocks": row.get("chunk_anchor_blocks"),
                "num_chunks": row.get("num_chunks"),
                "splitk_ms": row.get("splitk_ms"),
                "splitk_breakdown": row.get("splitk_breakdown"),
                "splitk_workspace": row.get("splitk_workspace"),
                "splitk_vs_dense_speedup": row.get("splitk_vs_dense_speedup"),
                "splitk_vs_serial_speedup": row.get("splitk_vs_serial_speedup"),
                "projection_skip_fraction": projection_skip,
                "pv_executed_fraction": float(stats.get("pv_executed_fraction") or 0.0),
                "gate1_post_qk_skipped_blocks": stats.get("gate1_post_qk_skipped_blocks"),
                "max_abs_error": max_error,
                "mean_abs_error": float(error.get("mean_abs_error") or 0.0),
            }
        )
    zero_error = [row for row in rows if row["max_abs_error"] == 0.0]
    strict = [row for row in rows if row["max_abs_error"] <= 1.0e-3]
    moderate = [row for row in rows if row["max_abs_error"] <= 1.0e-2]
    by_projection_skip = sorted(
        rows,
        key=lambda row: (
            row["projection_skip_fraction"],
            -(row["max_abs_error"]),
            row["splitk_vs_dense_speedup"] or 0.0,
        ),
        reverse=True,
    )
    strict_by_projection_skip = sorted(
        strict,
        key=lambda row: (row["projection_skip_fraction"], row["splitk_vs_dense_speedup"] or 0.0),
        reverse=True,
    )
    return {
        "row_count": len(rows),
        "zero_error_count": len(zero_error),
        "strict_count": len(strict),
        "moderate_count": len(moderate),
        "max_projection_skip_fraction": by_projection_skip[0]["projection_skip_fraction"] if by_projection_skip else 0.0,
        "max_strict_projection_skip_fraction": strict_by_projection_skip[0]["projection_skip_fraction"] if strict_by_projection_skip else 0.0,
        "top_projection_skip_rows": by_projection_skip[:8],
        "top_strict_projection_skip_rows": strict_by_projection_skip[:8],
    }


def _run(
    *,
    model: str,
    prompt: str,
    prompt_type: str,
    layers: str,
    head_index: int,
    head_indices: str,
    head_groups: str,
    max_seq: int,
    kv_lens: str,
    dtype: str,
    tensor_space: str,
    block_size: int,
    tile_size_q: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    chunk_anchor_blocks: int,
    chunk_anchor_blocks_values: str,
    block_order: str,
    num_chunks: str,
    seed_strategy: str,
    projection_dim: int,
    projection_dims: str,
    projection_seeds: str,
    projection_metadata_dtype: str,
    qproj_mode: str,
    splitk_breakdown: bool,
    splitk_workspace: str,
    filter_margin: float,
    filter_margins: str,
    error_budget: float,
    warmup: int,
    iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    prompt_file = "/tmp/streamattn_gate1_inline_projection_splitk_prompt.txt"
    Path(prompt_file).parent.mkdir(parents=True, exist_ok=True)
    Path(prompt_file).write_text(prompt, encoding="utf-8")
    results = []
    captures = []

    for kv_len in _parse_values(kv_lens, int):
        capture_dir = f"/tmp/streamattn_gate1_inline_projection_splitk_qk/kv{kv_len}"
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
            metadata_json,
        ]
        capture_payload = _json_from_cmd(capture_cmd, env=env)
        rows = [row for row in capture_payload.get("rows", []) if not row.get("skipped")]
        if not rows:
            raise RuntimeError(
                f"capture produced no usable rows for kv_len={kv_len}: "
                f"rows={capture_payload.get('rows', [])[:4]} errors={capture_payload.get('errors', [])[:4]}"
            )
        captured = rows[0]
        captures.append({"kv_len": kv_len, "row_count": len(rows), "shape": captured.get("shape")})
        chunk_values = _parse_values(num_chunks, int)
        anchor_values = (
            _parse_values(chunk_anchor_blocks_values, int)
            if chunk_anchor_blocks_values
            else [int(chunk_anchor_blocks)]
        )
        dim_values = _parse_values(projection_dims, int) if projection_dims else [int(projection_dim)]
        seed_values = _parse_values(projection_seeds, int)
        margin_values = _parse_values(filter_margins, float) if filter_margins else [float(filter_margin)]
        group_values = (
            _head_groups(head_groups, head_index, captured)
            if head_groups
            else [str(value) for value in _head_values(head_indices, head_index, captured)]
        )
        for current_group, chunks, anchors, proj_dim, proj_seed, margin in itertools.product(
            group_values,
            chunk_values,
            anchor_values,
            dim_values,
            seed_values,
            margin_values,
        ):
            group_heads = _head_values(current_group, -1, captured)
            is_group = len(group_heads) != 1
            profile_cmd = [
                "python",
                "/root/StreamAttn/benchmarks/profile_gate1_inline_projection_splitk.py",
                "--q-path",
                captured["q_path"],
                "--k-path",
                captured["k_path"],
                "--v-path",
                captured["v_path"],
                "--dtype",
                dtype,
                "--block-size",
                str(block_size),
                "--tile-size-q",
                str(tile_size_q),
                "--sink-blocks",
                str(sink_blocks),
                "--recent-blocks",
                str(recent_blocks),
                "--middle-seed-blocks",
                str(middle_seed_blocks),
                "--chunk-anchor-blocks",
                str(anchors),
                "--block-order",
                block_order,
                "--num-chunks",
                str(chunks),
                "--seed-strategy",
                seed_strategy,
                "--projection-dim",
                str(proj_dim),
                "--projection-metadata-dtype",
                projection_metadata_dtype,
                "--qproj-mode",
                qproj_mode,
                "--splitk-workspace",
                splitk_workspace,
                "--filter-margin",
                str(margin),
                "--error-budget",
                str(error_budget),
                "--seed",
                str(proj_seed),
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
            ]
            if splitk_breakdown:
                profile_cmd.append("--splitk-breakdown")
            if is_group:
                profile_cmd.extend(["--head-indices", current_group])
            else:
                profile_cmd.extend(["--head-index", str(group_heads[0])])
            profile = _json_from_cmd(profile_cmd, env=env)
            profile.update(
                {
                    "model_id": model,
                    "prompt_type": prompt_type,
                    "layer_id": captured.get("layer_id"),
                    "head_index": None if is_group else group_heads[0],
                    "head_indices": group_heads,
                    "head_group": current_group,
                    "kv_len": kv_len,
                    "capture_shape": captured.get("shape"),
                    "profile_command": profile_cmd,
                }
            )
            results.append(profile)

    return {
        "capture": {
            "model_id": model,
            "prompt_type": prompt_type,
            "layers": layers,
            "tensor_space": tensor_space,
            "captures": captures,
        },
        "sweep": {
            "kv_lens": kv_lens,
            "head_index": head_index,
            "head_indices": head_indices,
            "head_groups": head_groups,
            "block_size": block_size,
            "middle_seed_blocks": middle_seed_blocks,
            "chunk_anchor_blocks": chunk_anchor_blocks,
            "chunk_anchor_blocks_values": chunk_anchor_blocks_values,
            "filter_margin": filter_margin,
            "filter_margins": filter_margins,
            "block_order": block_order,
            "num_chunks": num_chunks,
            "seed_strategy": seed_strategy,
            "projection_dim": projection_dim,
            "projection_dims": projection_dims,
            "projection_seeds": projection_seeds,
            "projection_metadata_dtype": projection_metadata_dtype,
            "qproj_mode": qproj_mode,
            "splitk_breakdown": splitk_breakdown,
            "splitk_workspace": splitk_workspace,
        },
        "summary": _summarize_results(results),
        "results": results,
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
    prompt: str = "function run_decode q k v for block in ordered blocks if gate0 skip q block continue scan qk q block if gate1 skip block continue accumulate pv block metadata cache projection summaries online softmax dense fallback router telemetry kernel dispatch sparse decode long context attention",
    prompt_file: str = "",
    prompt_type: str = "code",
    prompt_repeat: int = 512,
    layers: str = "8",
    head_index: int = -1,
    head_indices: str = "",
    head_groups: str = "",
    max_seq: int = 4096,
    kv_lens: str = "4096",
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    block_size: int = 32,
    tile_size_q: int = 16,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    chunk_anchor_blocks: int = 0,
    chunk_anchor_blocks_values: str = "",
    block_order: str = "recent_first",
    num_chunks: str = "2,4,8,16",
    seed_strategy: str = "recompute_seed",
    projection_dim: int = 8,
    projection_dims: str = "",
    projection_seeds: str = "0",
    projection_metadata_dtype: str = "fp16",
    qproj_mode: str = "fused",
    splitk_breakdown: bool = False,
    splitk_workspace: str = "none",
    filter_margin: float = 32.0,
    filter_margins: str = "",
    error_budget: float = 1e-2,
    warmup: int = 2,
    iters: int = 10,
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
        "head_index": head_index,
        "head_indices": head_indices,
        "head_groups": head_groups,
        "max_seq": max_seq,
        "kv_lens": kv_lens,
        "dtype": dtype,
        "tensor_space": tensor_space,
        "block_size": block_size,
        "tile_size_q": tile_size_q,
        "sink_blocks": sink_blocks,
        "recent_blocks": recent_blocks,
        "middle_seed_blocks": middle_seed_blocks,
        "chunk_anchor_blocks": chunk_anchor_blocks,
        "chunk_anchor_blocks_values": chunk_anchor_blocks_values,
        "block_order": block_order,
        "num_chunks": num_chunks,
        "seed_strategy": seed_strategy,
        "projection_dim": projection_dim,
        "projection_dims": projection_dims,
        "projection_seeds": projection_seeds,
        "projection_metadata_dtype": projection_metadata_dtype,
        "qproj_mode": qproj_mode,
        "splitk_breakdown": splitk_breakdown,
        "splitk_workspace": splitk_workspace,
        "filter_margin": filter_margin,
        "filter_margins": filter_margins,
        "error_budget": error_budget,
        "warmup": warmup,
        "iters": iters,
    }
    payload = profile_a100.remote(**kwargs) if target == "a100" else profile_h100.remote(**kwargs)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
