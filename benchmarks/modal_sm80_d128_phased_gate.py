"""Modal A100 runner for the strict SM80 D128 phased-K/V gate."""

from __future__ import annotations

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ninja-build")
    .pip_install(
        "einops",
        "flashinfer-python==0.6.13",
        "flashinfer-cubin==0.6.13",
        "ninja",
    )
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)

app = modal.App("streamattn-sm80-d128-phased-kv-gate")
volume = modal.Volume.from_name("streamattn-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60,
    volumes={"/artifacts": volume},
)
def run(
    *,
    matrix_specs: str,
    paired_trials: int = 15,
    paired_iters: int = 100,
    production_plan: bool = False,
    output_json: str = "/artifacts/gate0/sm80_d128_phased_kv_gate_modal.json",
) -> str:
    import os
    import subprocess

    os.chdir("/root/StreamAttn")
    os.environ["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "python",
        "-m",
        "benchmarks.profile_sm80_d128_phased_gate",
        "--matrix-specs",
        matrix_specs,
        "--paired-trials",
        str(paired_trials),
        "--paired-iters",
        str(paired_iters),
        "--checkout-dir",
        "/artifacts/backend_sources",
        "--cuda-arch",
        "sm_80",
        "--torch-cuda-arch-list",
        "8.0",
        "--output-json",
        output_json,
    ]
    if production_plan:
        cmd.append("--production-plan")
    subprocess.run(cmd, check=True)
    volume.commit()
    return "ok"


@app.local_entrypoint()
def main(
    matrix_specs: str,
    paired_trials: int = 15,
    paired_iters: int = 100,
    production_plan: bool = False,
    output_json: str = "/artifacts/gate0/sm80_d128_phased_kv_gate_modal.json",
) -> None:
    print(
        run.remote(
            matrix_specs=matrix_specs,
            paired_trials=paired_trials,
            paired_iters=paired_iters,
            production_plan=production_plan,
            output_json=output_json,
        )
    )
