"""Run the complete grouped RS-PV prefill canary on a Modal H100."""

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
        "benchmarks/profile_sm90_grouped_rs_prefill_canary.py",
        remote_path=(
            "/root/StreamAttn/benchmarks/"
            "profile_sm90_grouped_rs_prefill_canary.py"
        ),
        copy=True,
    )
    .add_local_dir(
        "stream_attention", remote_path="/root/StreamAttn/stream_attention", copy=True
    )
)

app = modal.App("streamattn-sm90-grouped-rs-prefill-canary")


@app.function(image=image, gpu="H100", timeout=30 * 60)
def run(
    *,
    batch_sizes: str,
    sequence_lengths: str,
    group_sizes: str,
    matrix_kind: str,
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
        "benchmarks/profile_sm90_grouped_rs_prefill_canary.py",
        "--provider",
        "modal",
        "--matrix-kind",
        matrix_kind,
        "--batch-sizes",
        *batch_sizes.split(","),
        "--sequence-lengths",
        *sequence_lengths.split(","),
        "--group-sizes",
        *group_sizes.split(","),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--repeats",
        str(repeats),
        "--cutlass-root",
        "/opt/flashmla-etap/csrc/cutlass",
        "--build-dir",
        "/tmp/streamattn-grouped-rs-prefill-build",
        "--output-json",
        "/tmp/result.json",
        "--verbose-build",
    ]
    subprocess.run(command, check=True)
    return Path("/tmp/result.json").read_text(encoding="utf-8")


@app.local_entrypoint()
def main(
    batch_sizes: str = "1",
    sequence_lengths: str = "2048,4096",
    group_sizes: str = "4,8",
    matrix_kind: str = "canary",
    warmup: int = 5,
    iterations: int = 30,
    repeats: int = 7,
    output_json: str = (
        "artifacts/gate0/"
        "sm90_grouped_rs_prefill_canary_modal_h100_20260902.json"
    ),
) -> None:
    payload = run.remote(
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        group_sizes=group_sizes,
        matrix_kind=matrix_kind,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"wrote {output}")
