"""Run universal-exact calibration campaigns on A100, H100, and B200."""

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
            "STREAMATTN_SM80_BUILD_DIR": "/tmp/streamattn-sm80-build",
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


@app.function(image=image, gpu="A100-80GB", timeout=60 * 60)
def calibrate_sm80_decode(
    warmup: int,
    repeats: int,
    paired_trials: int,
    paired_repeats: int,
) -> list[dict[str, object]]:
    import os
    import sys

    os.chdir("/root/StreamAttn")
    sys.path.insert(0, "/root/StreamAttn")
    from benchmarks.profile_universal_exact_calibration import (
        profile_sm80_decode_surface,
    )

    return [
        row.as_dict()
        for row in profile_sm80_decode_surface(
            warmup=warmup,
            repeats=repeats,
            paired_trials=paired_trials,
            paired_repeats=paired_repeats,
        )
    ]


@app.function(image=image, gpu="A100-80GB", timeout=60 * 60)
def calibrate_sm80_prefill(
    warmup: int,
    repeats: int,
    paired_repeats: int,
) -> list[dict[str, object]]:
    import os
    import sys

    os.chdir("/root/StreamAttn")
    sys.path.insert(0, "/root/StreamAttn")
    from benchmarks.profile_universal_exact_calibration import (
        profile_sm80_prefill_surface,
    )

    return [
        row.as_dict()
        for row in profile_sm80_prefill_surface(
            warmup=warmup,
            repeats=repeats,
            paired_repeats=paired_repeats,
        )
    ]


@app.function(image=image, gpu="A100-80GB", timeout=60 * 60)
def calibrate_sm80_training(
    warmup: int,
    repeats: int,
    iterations: int,
    skip_native: bool,
) -> list[dict[str, object]]:
    import os
    import sys

    os.chdir("/root/StreamAttn")
    sys.path.insert(0, "/root/StreamAttn")
    from benchmarks.profile_universal_exact_calibration import (
        _profile_sm80_training,
    )

    return [
        row.as_dict()
        for row in _profile_sm80_training(
            warmup=warmup,
            repeats=repeats,
            iterations=iterations,
            skip_native=skip_native,
        )
    ]


@app.function(image=image, gpu="A100-80GB", timeout=60 * 60)
def calibrate_sm80_training_dropout(
    warmup: int,
    repeats: int,
    iterations: int,
) -> list[dict[str, object]]:
    import os
    import sys

    os.chdir("/root/StreamAttn")
    sys.path.insert(0, "/root/StreamAttn")
    from benchmarks.profile_universal_exact_calibration import (
        _profile_sm80_training_dropout,
    )

    return [
        row.as_dict()
        for row in _profile_sm80_training_dropout(
            warmup=warmup,
            repeats=repeats,
            iterations=iterations,
        )
    ]


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
    choices = {
        "sm80",
        "sm80-decode",
        "sm80-prefill",
        "sm80-training",
        "sm80-training-external",
        "sm80-dropout",
        "sm90",
        "sm100",
        "both",
        "all",
    }
    if normalized not in choices:
        raise ValueError(f"architecture must be one of {sorted(choices)}")
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def persist() -> None:
        output.write_text(
            json.dumps(
                {"schema_version": 1, "evidence": rows},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if normalized in {"sm80", "sm80-decode", "all"}:
        rows.extend(
            calibrate_sm80_decode.remote(
                warmup,
                repeats,
                paired_trials,
                paired_repeats,
            )
        )
        persist()
    if normalized in {"sm80", "sm80-prefill", "all"}:
        rows.extend(calibrate_sm80_prefill.remote(warmup, repeats, paired_repeats))
        persist()
    if normalized in {
        "sm80",
        "sm80-training",
        "sm80-training-external",
        "sm80-dropout",
        "all",
    }:
        training_warmup = max(3, min(warmup, 5))
        training_iterations = max(3, min(paired_repeats, 5))
        training_repeats = max(5, min(repeats, 9))
    if normalized in {"sm80", "sm80-dropout", "all"}:
        rows.extend(
            calibrate_sm80_training_dropout.remote(
                training_warmup,
                training_repeats,
                training_iterations,
            )
        )
        persist()
    if normalized in {"sm80", "sm80-training", "sm80-training-external", "all"}:
        rows.extend(
            calibrate_sm80_training.remote(
                training_warmup,
                training_repeats,
                training_iterations,
                normalized == "sm80-training-external",
            )
        )
        persist()
    if normalized in {"sm90", "both", "all"}:
        rows.extend(
            calibrate_sm90.remote(
                warmup,
                repeats,
                paired_trials,
                paired_repeats,
            )
        )
        persist()
    if normalized in {"sm100", "both", "all"}:
        rows.extend(calibrate_sm100.remote(warmup, iterations, repeats))
        persist()
    persist()
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
