"""Modal runner for Gate-0 hybrid correction profiling."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-gate0-hybrid-correction")

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


def _groups(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(";") if item.strip()]


def _read_prompt_file(path: str) -> str:
    return " ".join(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _json_from_cmd(cmd: list[str], *, env: dict[str, str]) -> dict:
    print(f"[modal-hybrid] running: {' '.join(cmd[:4])} ...", flush=True)
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


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in results:
        timing = row.get("timing") or {}
        quality = row.get("quality") or {}
        corrected = quality.get("corrected_error_vs_dense_all") or {}
        rows.append(
            {
                "trusted_heads": (row.get("policy") or {}).get("trusted_sparse_heads"),
                "exact_heads": (row.get("policy") or {}).get("exact_heads"),
                "dense_all_ms": timing.get("dense_all_ms"),
                "sparse_union_ms": timing.get("sparse_union_ms"),
                "dense_exact_ms": timing.get("dense_exact_ms"),
                "serial_corrected_ms": timing.get("serial_corrected_ms"),
                "oracle_max_ms": timing.get("oracle_max_ms"),
                "fused_hybrid_ms": timing.get("fused_hybrid_ms"),
                "parallel_stream_ms": timing.get("parallel_stream_ms"),
                "oracle_max_speedup_vs_dense_all": timing.get("oracle_max_speedup_vs_dense_all"),
                "fused_hybrid_speedup_vs_dense_all": timing.get("fused_hybrid_speedup_vs_dense_all"),
                "parallel_stream_speedup_vs_dense_all": timing.get("parallel_stream_speedup_vs_dense_all"),
                "corrected_max_abs_error": corrected.get("max_abs_error"),
                "corrected_mean_abs_error": corrected.get("mean_abs_error"),
            }
        )
    by_oracle = sorted(
        rows,
        key=lambda item: item.get("oracle_max_speedup_vs_dense_all") or 0.0,
        reverse=True,
    )
    by_parallel = sorted(
        rows,
        key=lambda item: item.get("parallel_stream_speedup_vs_dense_all") or 0.0,
        reverse=True,
    )
    by_fused = sorted(
        rows,
        key=lambda item: item.get("fused_hybrid_speedup_vs_dense_all") or 0.0,
        reverse=True,
    )
    return {
        "row_count": len(rows),
        "best_oracle": by_oracle[0] if by_oracle else None,
        "best_fused": by_fused[0] if by_fused else None,
        "best_parallel": by_parallel[0] if by_parallel else None,
        "rows": rows,
    }


def _run(
    *,
    model: str,
    prompt: str,
    prompt_type: str,
    layers: str,
    max_seq: int,
    kv_lens: str,
    dtype: str,
    tensor_space: str,
    aggressive_heads: str,
    trusted_head_groups: str,
    block_size: int,
    tile_size_q: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    chunk_anchor_blocks: int,
    block_order: str,
    num_chunks: int,
    seed_strategy: str,
    projection_dim: int,
    projection_seed: int,
    projection_metadata_dtype: str,
    splitk_workspace: str,
    filter_margin: float,
    error_budget: float,
    measure_parallel_streams: bool,
    warmup: int,
    iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    prompt_file = "/tmp/streamattn_gate0_hybrid_prompt.txt"
    Path(prompt_file).parent.mkdir(parents=True, exist_ok=True)
    Path(prompt_file).write_text(prompt, encoding="utf-8")
    results = []
    captures = []

    for kv_len in _parse_values(kv_lens, int):
        capture_dir = f"/tmp/streamattn_gate0_hybrid_qk/kv{kv_len}"
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
        print(
            f"[modal-hybrid] captured kv_len={kv_len} rows={len(rows)} shape={captured.get('shape')}",
            flush=True,
        )
        captures.append({"kv_len": kv_len, "row_count": len(rows), "shape": captured.get("shape")})
        group_values = _groups(trusted_head_groups)
        for case_index, trusted_heads in enumerate(group_values, start=1):
            print(
                f"[modal-hybrid] profile {case_index}/{len(group_values)} "
                f"kv_len={kv_len} aggressive={aggressive_heads} trusted={trusted_heads}",
                flush=True,
            )
            profile_cmd = [
                "python",
                "/root/StreamAttn/benchmarks/profile_gate0_hybrid_correction.py",
                "--q-path",
                captured["q_path"],
                "--k-path",
                captured["k_path"],
                "--v-path",
                captured["v_path"],
                "--dtype",
                dtype,
                "--aggressive-heads",
                aggressive_heads,
                "--trusted-heads",
                trusted_heads,
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
                str(chunk_anchor_blocks),
                "--block-order",
                block_order,
                "--num-chunks",
                str(num_chunks),
                "--seed-strategy",
                seed_strategy,
                "--projection-dim",
                str(projection_dim),
                "--projection-metadata-dtype",
                projection_metadata_dtype,
                "--splitk-workspace",
                splitk_workspace,
                "--filter-margin",
                str(filter_margin),
                "--error-budget",
                str(error_budget),
                "--seed",
                str(projection_seed),
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
            ]
            if measure_parallel_streams:
                profile_cmd.append("--measure-parallel-streams")
            profile = _json_from_cmd(profile_cmd, env=env)
            profile.update(
                {
                    "model_id": model,
                    "prompt_type": prompt_type,
                    "layer_id": captured.get("layer_id"),
                    "kv_len": kv_len,
                    "capture_shape": captured.get("shape"),
                    "profile_command": profile_cmd,
                }
            )
            results.append(profile)
            timing = profile.get("timing") or {}
            quality = profile.get("quality") or {}
            corrected = quality.get("corrected_error_vs_dense_all") or {}
            print(
                f"[modal-hybrid] done {case_index}/{len(group_values)} "
                f"dense_all={timing.get('dense_all_ms')} sparse={timing.get('sparse_union_ms')} "
                f"exact={timing.get('dense_exact_ms')} oracle_speed={timing.get('oracle_max_speedup_vs_dense_all')} "
                f"fused_speed={timing.get('fused_hybrid_speedup_vs_dense_all')} "
                f"parallel_speed={timing.get('parallel_stream_speedup_vs_dense_all')} "
                f"corrected_err={corrected.get('max_abs_error')}",
                flush=True,
            )

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
            "aggressive_heads": aggressive_heads,
            "trusted_head_groups": trusted_head_groups,
            "block_size": block_size,
            "middle_seed_blocks": middle_seed_blocks,
            "chunk_anchor_blocks": chunk_anchor_blocks,
            "filter_margin": filter_margin,
            "block_order": block_order,
            "num_chunks": num_chunks,
            "seed_strategy": seed_strategy,
            "projection_dim": projection_dim,
            "projection_seed": projection_seed,
            "projection_metadata_dtype": projection_metadata_dtype,
            "splitk_workspace": splitk_workspace,
            "measure_parallel_streams": measure_parallel_streams,
        },
        "summary": _summarize(results),
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
    prompt: str = "Needle retrieval scenario with StreamAttn Gate-0 projection routing and long-context decode.",
    prompt_file: str = "",
    prompt_type: str = "needle_hybrid_correction",
    prompt_repeat: int = 512,
    layers: str = "8",
    max_seq: int = 32768,
    kv_lens: str = "32768",
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    aggressive_heads: str = "2,3,4,6,7,8,9,11",
    trusted_head_groups: str = "6,7;6;7",
    block_size: int = 32,
    tile_size_q: int = 16,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    chunk_anchor_blocks: int = 0,
    block_order: str = "recent_first",
    num_chunks: int = 32,
    seed_strategy: str = "recompute_seed",
    projection_dim: int = 8,
    projection_seed: int = 1,
    projection_metadata_dtype: str = "fp16",
    splitk_workspace: str = "reuse",
    filter_margin: float = 64.0,
    error_budget: float = 1e-2,
    measure_parallel_streams: bool = True,
    warmup: int = 1,
    iters: int = 6,
    output_json: str = "",
):
    if prompt_file:
        prompt = _read_prompt_file(prompt_file)
    prompt = prompt * max(1, prompt_repeat)
    kwargs = {
        "model": model,
        "prompt": prompt,
        "prompt_type": prompt_type,
        "layers": layers,
        "max_seq": max_seq,
        "kv_lens": kv_lens,
        "dtype": dtype,
        "tensor_space": tensor_space,
        "aggressive_heads": aggressive_heads,
        "trusted_head_groups": trusted_head_groups,
        "block_size": block_size,
        "tile_size_q": tile_size_q,
        "sink_blocks": sink_blocks,
        "recent_blocks": recent_blocks,
        "middle_seed_blocks": middle_seed_blocks,
        "chunk_anchor_blocks": chunk_anchor_blocks,
        "block_order": block_order,
        "num_chunks": num_chunks,
        "seed_strategy": seed_strategy,
        "projection_dim": projection_dim,
        "projection_seed": projection_seed,
        "projection_metadata_dtype": projection_metadata_dtype,
        "splitk_workspace": splitk_workspace,
        "filter_margin": filter_margin,
        "error_budget": error_budget,
        "measure_parallel_streams": measure_parallel_streams,
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
