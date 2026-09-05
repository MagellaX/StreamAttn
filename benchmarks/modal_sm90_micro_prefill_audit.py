"""H100 cross-provider micro-prefill audit, Modal half."""

from pathlib import Path

import modal

if modal.is_local():
    from benchmarks.modal_sm90_micro_prefill import image as base_image

    image = (
        base_image.pip_install(
            "flashinfer-python==0.6.13", "flashinfer-cubin==0.6.13", "pyyaml"
        )
        .run_commands(
            "python -m pip install --no-deps xformers==0.0.31 --index-url https://download.pytorch.org/whl/cu128"
        )
        .add_local_file(
            "benchmarks/profile_sm90_micro_prefill_audit.py",
            "/root/StreamAttn/benchmarks/profile_sm90_micro_prefill_audit.py",
            copy=True,
        )
    )
    for source in ("micro_prefill_baselines.py", "micro_prefill_optional_baselines.py"):
        image = image.add_local_file(
            f"benchmarks/{source}", f"/root/StreamAttn/benchmarks/{source}", copy=True
        )
else:
    image = None
app = modal.App("streamattn-sm90-micro-prefill-audit")


@app.function(image=image, gpu="H100", timeout=2700)
def run(cohort: str) -> str:
    import subprocess

    subprocess.run(
        [
            "python",
            "-u",
            "benchmarks/profile_sm90_micro_prefill_audit.py",
            "--provider",
            "modal",
            "--cohort",
            cohort,
            "--cutlass-root",
            "/opt/flashmla-etap/csrc/cutlass",
            "--build-dir",
            "/tmp/streamattn-audit-build",
            "--output-json",
            "/tmp/audit.json",
        ],
        cwd="/root/StreamAttn",
        check=True,
    )
    return Path("/tmp/audit.json").read_text()


@app.local_entrypoint()
def main(
    cohort: str = "modal",
    output_json: str = "artifacts/gate0/sm90_micro_prefill_audit_modal_h100_20260905.json",
):
    path = Path(output_json)
    if path.exists():
        raise FileExistsError(
            f"preserve existing evidence; choose another output: {path}"
        )
    data = run.remote(cohort)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    print(f"wrote {path}")
