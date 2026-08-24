"""Run selected paged WGMMA correctness and latency gates on H100."""

from __future__ import annotations

import json
from pathlib import Path

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ninja-build")
    .pip_install("flashinfer-python==0.6.17", "ninja")
    .run_commands(
        "python -m pip install flashinfer-cubin==0.6.17 "
        "--index-url https://flashinfer.ai/whl/"
    )
    .run_commands(
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/pengcuo/FlashMLA-ETAP.git /opt/flashmla-etap && "
        "cd /opt/flashmla-etap && git sparse-checkout init --cone && "
        "git sparse-checkout set csrc/cutlass/include && "
        "git fetch --depth=1 origin 39e616041ae6fb1243a0f6ac891e72d576b640e5 && "
        "git checkout 39e616041ae6fb1243a0f6ac891e72d576b640e5"
    )
    .add_local_dir("benchmarks", remote_path="/root/StreamAttn/benchmarks", copy=True)
    .add_local_dir(
        "stream_attention", remote_path="/root/StreamAttn/stream_attention", copy=True
    )
)

app = modal.App("streamattn-paged-selected-decode")


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@app.function(image=image, gpu="H100!", timeout=60 * 60)
def profile_h100(kwargs: dict[str, object]) -> dict[str, object]:
    import argparse
    import os
    import sys

    os.chdir("/root/StreamAttn")
    sys.path.insert(0, "/root/StreamAttn")
    from benchmarks.profile_paged_selected_decode import profile

    cells: list[dict[str, object]] = []
    batch_values = _parse_ints(str(kwargs["batches"]))
    selected_values = _parse_ints(str(kwargs["selected_tokens"]))
    base_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"batches", "selected_tokens"}
    }
    for batch in batch_values:
        for selected_tokens in selected_values:
            args = argparse.Namespace(
                batch=batch,
                selected_tokens=selected_tokens,
                **base_kwargs,
            )
            print(
                f"[selected-paged] B={batch} N={args.kv_len} S={selected_tokens} "
                f"Hq={args.q_heads} Hkv={args.kv_heads} D={args.head_dim}",
                flush=True,
            )
            row = profile(args)
            cells.append(row)
            print(
                f"[selected-paged] stream={row['streamattn_ms']:.5f} ms "
                f"flashinfer={row['flashinfer_ms']:.5f} ms "
                f"speedup={row['speedup_vs_flashinfer']:.3f}x "
                f"paired_min={row['paired']['speedup_min']:.3f}x",
                flush=True,
            )
    return {
        "schema": "streamattn.paged_selected_decode_matrix.v1",
        "device": cells[0]["device"] if cells else "unknown",
        "summary": {
            "cells": len(cells),
            "correct_cells": sum(
                float(row["max_abs_error_vs_selected_reference"])
                <= float(base_kwargs["atol"])
                for row in cells
            ),
            "paired_winning_cells": sum(
                float(row["paired"]["speedup_min"]) > 1.0 for row in cells
            ),
        },
        "cells": cells,
    }


@app.local_entrypoint()
def main(
    batches: str = "1,4,8",
    selected_tokens: str = "384,2048,8192,16384,32768",
    kv_len: int = 32768,
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 128,
    page_size: int = 16,
    layout: str = "NHD",
    route_mode: str = "all_heads",
    flashinfer_backends: str = "auto,fa2,fa3",
    workspace_mb: int = 128,
    warmup: int = 10,
    repeats: int = 30,
    paired_trials: int = 9,
    paired_repeats: int = 10,
    atol: float = 2e-2,
    seed: int = 17,
    output_json: str = "",
) -> None:
    result = profile_h100.remote(
        {
            "batches": batches,
            "selected_tokens": selected_tokens,
            "kv_len": kv_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "layout": layout,
            "route_mode": route_mode,
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
        print(json.dumps({"artifact": str(output), "summary": result["summary"]}))
    else:
        print(payload)
