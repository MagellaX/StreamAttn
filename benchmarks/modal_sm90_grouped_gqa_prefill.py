"""Bounded Modal H100 gate for natural-orientation grouped exact prefill."""

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
        "benchmarks/profile_sm90_grouped_gqa_prefill.py",
        remote_path="/root/StreamAttn/benchmarks/profile_sm90_grouped_gqa_prefill.py",
        copy=True,
    )
    .add_local_dir(
        "stream_attention",
        remote_path="/root/StreamAttn/stream_attention",
        copy=True,
    )
)

app = modal.App("streamattn-sm90-grouped-prefill")


@app.function(image=image, gpu="H100", timeout=45 * 60)
def run(
    *,
    batches: str,
    sequence_lengths: str,
    group_sizes: str,
    head_dim: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> str:
    import os
    import subprocess

    os.chdir("/root/StreamAttn")
    command = [
        "python",
        "-u",
        "benchmarks/profile_sm90_grouped_gqa_prefill.py",
        "--batches",
        *batches.split(","),
        "--sequence-lengths",
        *sequence_lengths.split(","),
        "--group-sizes",
        *group_sizes.split(","),
        "--head-dim",
        str(head_dim),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--repeats",
        str(repeats),
        "--cutlass-root",
        "/opt/flashmla-etap/csrc/cutlass",
        "--build-dir",
        "/tmp/streamattn-sm90-grouped-prefill-build",
        "--output-json",
        "/tmp/result.json",
        "--verbose-build",
    ]
    subprocess.run(command, check=True)
    return Path("/tmp/result.json").read_text(encoding="utf-8")


@app.local_entrypoint()
def main(
    batches: str = "1",
    sequence_lengths: str = "128,256,512,1024,2048",
    group_sizes: str = "4,8",
    head_dim: int = 128,
    warmup: int = 3,
    iterations: int = 20,
    repeats: int = 5,
    output_json: str = (
        "artifacts/gate0/sm90_grouped_gqa_prefill_gate_modal_h100_20260902.json"
    ),
) -> None:
    payload = run.remote(
        batches=batches,
        sequence_lengths=sequence_lengths,
        group_sizes=group_sizes,
        head_dim=head_dim,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"wrote {output}")
