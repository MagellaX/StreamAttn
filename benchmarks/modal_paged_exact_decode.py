"""Run native paged exact decode correctness and latency gates on NVIDIA GPUs."""

from __future__ import annotations

import json
from pathlib import Path

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ninja-build")
    .pip_install(
        "flashinfer-python==0.6.17",
        "ninja",
    )
    .run_commands(
        "python -m pip install flashinfer-cubin==0.6.17 "
        "--index-url https://flashinfer.ai/whl/"
    )
    .run_commands(
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/pengcuo/FlashMLA-ETAP.git /opt/flashmla-etap && "
        "cd /opt/flashmla-etap && "
        "git sparse-checkout init --cone && "
        "git sparse-checkout set csrc/cutlass/include && "
        "git fetch --depth=1 origin 39e616041ae6fb1243a0f6ac891e72d576b640e5 && "
        "git checkout 39e616041ae6fb1243a0f6ac891e72d576b640e5"
    )
    .run_commands(
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/NVIDIA/cutlass.git /opt/cutlass && "
        "cd /opt/cutlass && "
        "git sparse-checkout init --cone && "
        "git sparse-checkout set include && "
        "git fetch --depth=1 origin 7107b05535f8977f5ecb9d01ee203205b1fd9bc4 && "
        "git checkout 7107b05535f8977f5ecb9d01ee203205b1fd9bc4"
    )
    .add_local_dir("benchmarks", remote_path="/root/StreamAttn/benchmarks", copy=True)
    .add_local_dir(
        "stream_attention", remote_path="/root/StreamAttn/stream_attention", copy=True
    )
)

app = modal.App("streamattn-paged-exact-decode")


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_batch_splits(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        batch, splits = item.split(":", 1)
        result[int(batch.strip())] = int(splits.strip())
    return result


def _profile_gpu(
    *,
    batches: str,
    kv_lens: str,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    layout: str,
    dtype: str,
    split_counts: str,
    batch_splits: str,
    batch_merge_segments: str,
    token_tiles: str,
    partial_num_warps: int,
    sm80_cp_async_experimental: bool,
    sm80_merge_segments: str,
    sm80_grouped_experimental: bool,
    sm100_grouped_experimental: bool,
    sm100_tgv_experimental: bool,
    sm90_fragmented_experimental: bool,
    sm90_fragmented_ragged_experimental: bool,
    length_profiles: str,
    flashinfer_backends: str,
    workspace_mb: int,
    warmup: int,
    repeats: int,
    paired_trials: int,
    paired_repeats: int,
    atol: float,
    seed: int,
) -> dict[str, object]:
    import argparse
    import os
    import sys

    os.chdir("/root/StreamAttn")
    sys.path.insert(0, "/root/StreamAttn")

    from benchmarks.profile_paged_exact_decode import profile

    cells: list[dict[str, object]] = []
    split_values: list[int | None] = (
        [None] if split_counts.strip().lower() == "auto" else _parse_ints(split_counts)
    )
    batch_split_map = _parse_batch_splits(batch_splits)
    batch_merge_map = _parse_batch_splits(batch_merge_segments)
    for batch in _parse_ints(batches):
        batch_split_values = (
            [batch_split_map[batch]] if batch in batch_split_map else split_values
        )
        merge_segment_values = (
            [batch_merge_map[batch]]
            if batch in batch_merge_map
            else (
                [None]
                if sm80_merge_segments.strip().lower() == "auto"
                else _parse_ints(sm80_merge_segments)
            )
        )
        for kv_len in _parse_ints(kv_lens):
            for length_profile in (
                item.strip() for item in length_profiles.split(",") if item.strip()
            ):
                for merge_segments in merge_segment_values:
                    for splits in batch_split_values:
                        for tokens_per_tile in _parse_ints(token_tiles):
                            args = argparse.Namespace(
                                batch=batch,
                                kv_len=kv_len,
                                q_heads=q_heads,
                                kv_heads=kv_heads,
                                head_dim=head_dim,
                                page_size=page_size,
                                layout=layout,
                                dtype=dtype,
                                splits=splits,
                                tokens_per_tile=tokens_per_tile,
                                partial_num_warps=partial_num_warps,
                                sm80_cp_async_experimental=(
                                    sm80_cp_async_experimental
                                ),
                                sm80_merge_segments=merge_segments,
                                sm80_grouped_experimental=(
                                    sm80_grouped_experimental
                                ),
                                sm100_grouped_experimental=(
                                    sm100_grouped_experimental
                                ),
                                sm100_tgv_experimental=sm100_tgv_experimental,
                                sm90_fragmented_experimental=(
                                    sm90_fragmented_experimental
                                ),
                                sm90_fragmented_ragged_experimental=(
                                    sm90_fragmented_ragged_experimental
                                ),
                                length_profile=length_profile,
                                flashinfer_backends=flashinfer_backends,
                                workspace_mb=workspace_mb,
                                warmup=warmup,
                                repeats=repeats,
                                paired_trials=paired_trials,
                                paired_repeats=paired_repeats,
                                atol=atol,
                                seed=seed,
                            )
                            print(
                                f"[paged-exact] starting B={batch} N={kv_len} "
                                f"profile={length_profile} C={splits or 'auto'} "
                                f"merge={merge_segments or 'auto'} "
                                f"T={tokens_per_tile} W={partial_num_warps} "
                                f"Hq={q_heads} Hkv={kv_heads} D={head_dim} {dtype}",
                                flush=True,
                            )
                            result = profile(args)
                            cells.append(result)
                            print(
                                f"[paged-exact] B={batch} N={kv_len} "
                                f"profile={length_profile} "
                                f"splits={result['splits']} "
                                f"merge={result['sm80_merge_segments']} "
                                f"tile={result['tokens_per_tile']} "
                                f"stream={result['streamattn_ms']:.5f} ms "
                                f"flashinfer={result['flashinfer_ms']:.5f} ms "
                                f"speedup={result['speedup_vs_flashinfer']:.3f}x "
                                f"max_err={result['max_abs_error']:.3e}",
                                flush=True,
                            )

    correct = [cell for cell in cells if float(cell["max_abs_error"]) <= atol]
    winners = [cell for cell in correct if float(cell["speedup_vs_flashinfer"]) > 1.0]
    device_name = str(cells[0]["device"]) if cells else "unknown"
    capability = cells[0].get("compute_capability") if cells else None
    return {
        "schema": "streamattn.paged_exact_decode_matrix.v2",
        "backend": "gpu_remote",
        "device": device_name,
        "compute_capability": capability,
        "shape_family": {
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "group_size": q_heads // kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "dtype": dtype,
            "length_profiles": [
                item.strip() for item in length_profiles.split(",") if item.strip()
            ],
        },
        "summary": {
            "cells": len(cells),
            "correct_cells": len(correct),
            "streamattn_wins": len(winners),
            "flashinfer_wins": len(correct) - len(winners),
        },
        "cells": cells,
    }


@app.function(image=image, gpu="H100!", timeout=60 * 60)
def profile_h100(kwargs: dict[str, object]) -> dict[str, object]:
    return _profile_gpu(**kwargs)


@app.function(image=image, gpu="A100-80GB", timeout=60 * 60)
def profile_a100(kwargs: dict[str, object]) -> dict[str, object]:
    return _profile_gpu(**kwargs)


@app.function(image=image, gpu="B200", timeout=60 * 60)
def profile_b200(kwargs: dict[str, object]) -> dict[str, object]:
    return _profile_gpu(**kwargs)


@app.local_entrypoint()
def main(
    gpu_type: str = "H100",
    batches: str = "4",
    kv_lens: str = "32768",
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 64,
    page_size: int = 16,
    layout: str = "NHD",
    dtype: str = "bf16",
    split_counts: str = "4",
    batch_splits: str = "",
    batch_merge_segments: str = "",
    token_tiles: str = "128",
    partial_num_warps: int = 4,
    sm80_cp_async_experimental: bool = False,
    sm80_merge_segments: str = "auto",
    sm80_grouped_experimental: bool = False,
    sm100_grouped_experimental: bool = False,
    sm100_tgv_experimental: bool = False,
    sm90_fragmented_experimental: bool = False,
    sm90_fragmented_ragged_experimental: bool = False,
    length_profiles: str = "full",
    flashinfer_backends: str = "auto",
    workspace_mb: int = 128,
    warmup: int = 10,
    repeats: int = 30,
    paired_trials: int = 9,
    paired_repeats: int = 10,
    atol: float = 1e-2,
    seed: int = 17,
    output_json: str = "",
) -> None:
    profiles = {
        "H100": profile_h100,
        "A100": profile_a100,
        "A100-80GB": profile_a100,
        "B200": profile_b200,
    }
    normalized_gpu = gpu_type.strip().upper()
    if normalized_gpu not in profiles:
        raise ValueError("gpu_type must be H100, A100-80GB, or B200")
    result = profiles[normalized_gpu].remote(
        {
            "batches": batches,
            "kv_lens": kv_lens,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "layout": layout,
            "dtype": dtype,
            "split_counts": split_counts,
            "batch_splits": batch_splits,
            "batch_merge_segments": batch_merge_segments,
            "token_tiles": token_tiles,
            "partial_num_warps": partial_num_warps,
            "sm80_cp_async_experimental": sm80_cp_async_experimental,
            "sm80_merge_segments": sm80_merge_segments,
            "sm80_grouped_experimental": sm80_grouped_experimental,
            "sm100_grouped_experimental": sm100_grouped_experimental,
            "sm100_tgv_experimental": sm100_tgv_experimental,
            "sm90_fragmented_experimental": sm90_fragmented_experimental,
            "sm90_fragmented_ragged_experimental": (
                sm90_fragmented_ragged_experimental
            ),
            "length_profiles": length_profiles,
            "flashinfer_backends": flashinfer_backends,
            "workspace_mb": workspace_mb,
            "warmup": warmup,
            "repeats": repeats,
            "paired_trials": paired_trials,
            "paired_repeats": paired_repeats,
            "atol": atol,
            "seed": seed,
        }
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        output = Path(output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"artifact": str(output), "summary": result["summary"]},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(payload)
