"""Run native paged exact decode correctness and latency gates on Modal H100."""

from __future__ import annotations

import json
from pathlib import Path

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ninja-build")
    .pip_install(
        "flashinfer-python==0.6.12",
        "flashinfer-cubin==0.6.12",
        "ninja",
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
    .add_local_dir("benchmarks", remote_path="/root/StreamAttn/benchmarks", copy=True)
    .add_local_dir(
        "stream_attention", remote_path="/root/StreamAttn/stream_attention", copy=True
    )
)

app = modal.App("streamattn-paged-exact-decode")


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@app.function(image=image, gpu="H100", timeout=60 * 60)
def profile_h100(
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
    token_tiles: str,
    partial_num_warps: int,
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
    for batch in _parse_ints(batches):
        for kv_len in _parse_ints(kv_lens):
            for splits in split_values:
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
                        f"C={splits or 'auto'} T={tokens_per_tile} W={partial_num_warps} "
                        f"Hq={q_heads} Hkv={kv_heads} D={head_dim} {dtype}",
                        flush=True,
                    )
                    result = profile(args)
                    cells.append(result)
                    print(
                        f"[paged-exact] B={batch} N={kv_len} splits={result['splits']} "
                        f"tile={result['tokens_per_tile']} "
                        f"stream={result['streamattn_ms']:.5f} ms "
                        f"flashinfer={result['flashinfer_ms']:.5f} ms "
                        f"speedup={result['speedup_vs_flashinfer']:.3f}x "
                        f"max_err={result['max_abs_error']:.3e}",
                        flush=True,
                    )

    correct = [cell for cell in cells if float(cell["max_abs_error"]) <= atol]
    winners = [cell for cell in correct if float(cell["speedup_vs_flashinfer"]) > 1.0]
    return {
        "schema": "streamattn.paged_exact_decode_matrix.v1",
        "backend": "modal_h100",
        "shape_family": {
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "group_size": q_heads // kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "dtype": dtype,
        },
        "summary": {
            "cells": len(cells),
            "correct_cells": len(correct),
            "flashinfer_wins": len(winners),
        },
        "cells": cells,
    }


@app.local_entrypoint()
def main(
    batches: str = "4",
    kv_lens: str = "32768",
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 64,
    page_size: int = 16,
    layout: str = "NHD",
    dtype: str = "bf16",
    split_counts: str = "4",
    token_tiles: str = "128",
    partial_num_warps: int = 4,
    workspace_mb: int = 128,
    warmup: int = 10,
    repeats: int = 30,
    paired_trials: int = 9,
    paired_repeats: int = 10,
    atol: float = 1e-2,
    seed: int = 17,
    output_json: str = "",
) -> None:
    result = profile_h100.remote(
        batches=batches,
        kv_lens=kv_lens,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        layout=layout,
        dtype=dtype,
        split_counts=split_counts,
        token_tiles=token_tiles,
        partial_num_warps=partial_num_warps,
        workspace_mb=workspace_mb,
        warmup=warmup,
        repeats=repeats,
        paired_trials=paired_trials,
        paired_repeats=paired_repeats,
        atol=atol,
        seed=seed,
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
