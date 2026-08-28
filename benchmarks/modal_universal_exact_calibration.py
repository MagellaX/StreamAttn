"""Run first universal-exact calibration campaigns on H100 and B200."""

from __future__ import annotations

import json
from pathlib import Path

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ninja-build")
    .pip_install("flashinfer-python==0.6.17", "ninja", "pyyaml>=6.0")
    .run_commands(
        "python -m pip install flashinfer-cubin==0.6.17 "
        "--index-url https://flashinfer.ai/whl/"
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
    .env(
        {
            "STREAMATTN_CUTLASS_ROOT": "/opt/cutlass",
            "STREAMATTN_SM100_CUTLASS_ROOT": "/opt/cutlass",
            "STREAMATTN_SM100_BUILD_DIR": "/tmp/streamattn-sm100-build",
        }
    )
    .add_local_dir("benchmarks", remote_path="/root/StreamAttn/benchmarks", copy=True)
    .add_local_dir(
        "stream_attention", remote_path="/root/StreamAttn/stream_attention", copy=True
    )
)

app = modal.App("streamattn-universal-exact-calibration")


@app.function(image=image, gpu="H100!", timeout=60 * 60)
def calibrate_sm90(
    warmup: int,
    repeats: int,
    paired_trials: int,
    paired_repeats: int,
) -> list[dict[str, object]]:
    import os
    import sys

    os.chdir("/root/StreamAttn")
    sys.path.insert(0, "/root/StreamAttn")
    from benchmarks.profile_universal_exact_calibration import profile_sm90

    return [
        row.as_dict()
        for row in profile_sm90(
            warmup=warmup,
            repeats=repeats,
            paired_trials=paired_trials,
            paired_repeats=paired_repeats,
        )
    ]


@app.function(image=image, gpu="B200", timeout=60 * 60)
def calibrate_sm100(
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[dict[str, object]]:
    import os
    import sys

    os.chdir("/root/StreamAttn")
    sys.path.insert(0, "/root/StreamAttn")
    from benchmarks.profile_universal_exact_calibration import profile_sm100

    return [
        row.as_dict()
        for row in profile_sm100(
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
            build_dir=Path("/tmp/streamattn-sm100-build"),
        )
    ]


@app.local_entrypoint()
def main(
    architecture: str = "both",
    output_json: str = "artifacts/universal_exact/calibration_evidence.json",
    warmup: int = 5,
    iterations: int = 20,
    repeats: int = 9,
    paired_trials: int = 9,
    paired_repeats: int = 10,
) -> None:
    normalized = architecture.strip().lower()
    if normalized not in {"sm90", "sm100", "both"}:
        raise ValueError("architecture must be sm90, sm100, or both")
    rows: list[dict[str, object]] = []
    if normalized in {"sm90", "both"}:
        rows.extend(
            calibrate_sm90.remote(
                warmup,
                repeats,
                paired_trials,
                paired_repeats,
            )
        )
    if normalized in {"sm100", "both"}:
        rows.extend(calibrate_sm100.remote(warmup, iterations, repeats))
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "evidence": rows}
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "architectures": normalized,
                "evidence_rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
