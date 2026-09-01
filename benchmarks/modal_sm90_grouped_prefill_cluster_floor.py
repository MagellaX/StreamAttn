"""Bounded Modal H100 runner for the two-CTA TMA multicast floor."""

from __future__ import annotations

from pathlib import Path

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ninja-build")
    .pip_install("ninja")
    .run_commands(
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/pengcuo/FlashMLA-ETAP.git /opt/flashmla-etap && "
        "cd /opt/flashmla-etap && "
        "git sparse-checkout init --cone && "
        "git sparse-checkout set csrc/cutlass/include && "
        "git fetch --depth=1 origin 39e616041ae6fb1243a0f6ac891e72d576b640e5 && "
        "git checkout 39e616041ae6fb1243a0f6ac891e72d576b640e5"
    )
    .add_local_file(
        "benchmarks/profile_sm90_grouped_prefill_cluster_floor.py",
        remote_path=(
            "/root/StreamAttn/benchmarks/"
            "profile_sm90_grouped_prefill_cluster_floor.py"
        ),
        copy=True,
    )
    .add_local_dir(
        "stream_attention", remote_path="/root/StreamAttn/stream_attention", copy=True
    )
)

app = modal.App("streamattn-sm90-grouped-prefill-cluster-floor")


@app.function(image=image, gpu="H100", timeout=30 * 60)
def run(
    *,
    total_tiles: int,
    tiles_per_group: str,
    warmup: int,
    iters: int,
    repeats: int,
) -> str:
    import os
    import subprocess

    os.chdir("/root/StreamAttn")
    command = [
        "python",
        "-u",
        "benchmarks/profile_sm90_grouped_prefill_cluster_floor.py",
        "--total-tiles",
        str(total_tiles),
        "--tiles-per-group",
        tiles_per_group,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--repeats",
        str(repeats),
        "--cutlass-root",
        "/opt/flashmla-etap/csrc/cutlass",
        "--build-dir",
        "/tmp/streamattn-prefill-cluster-floor-build",
        "--output-json",
        "/tmp/result.json",
        "--verbose-build",
    ]
    subprocess.run(command, check=True)
    return Path("/tmp/result.json").read_text(encoding="utf-8")


@app.local_entrypoint()
def main(
    total_tiles: int = 4096,
    tiles_per_group: str = "1,2,4,8",
    warmup: int = 10,
    iters: int = 100,
    repeats: int = 9,
    output_json: str = (
        "artifacts/gate0/"
        "sm90_grouped_prefill_cluster_floor_modal_h100_20260902.json"
    ),
) -> None:
    payload = run.remote(
        total_tiles=total_tiles,
        tiles_per_group=tiles_per_group,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
    )
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"wrote {output}")
