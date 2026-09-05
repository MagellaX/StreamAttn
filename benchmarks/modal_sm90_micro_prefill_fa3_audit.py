"""Direct exact audit with a pinned, forward-only standalone FA3 build.

The source build runs before GPU provisioning and is bounded to 30 minutes.
It retains split-K, packed GQA, and cluster scheduling for BF16 D64/D128.
"""

from pathlib import Path
import os

import modal


FA3_BUILD = r"""
set -eu
git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git /tmp/fa3-v283
cd /tmp/fa3-v283
test "$(git rev-parse HEAD)" = 060c9188beec3a8b62b33a3bfa6d5d2d44975fab
git submodule update --init --depth 1 csrc/cutlass
cd hopper
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE
for feature in BACKWARD FP16 FP8 SM80 HDIM96 HDIM192 HDIM256 PAGEDKV APPENDKV LOCAL SOFTCAP VARLEN; do
  export "FLASH_ATTENTION_DISABLE_${feature}=TRUE"
done
for feature in HDIM64 HDIM128 SPLIT PACKGQA CLUSTER; do
  export "FLASH_ATTENTION_DISABLE_${feature}=FALSE"
done
export FLASH_ATTENTION_ENABLE_VCOLMAJOR=FALSE MAX_JOBS=2 NVCC_THREADS=1
timeout --signal=TERM --kill-after=30s 1800s python -m pip install -v --no-build-isolation --no-deps .
"""


def build_fa3(script: str):
    import subprocess

    subprocess.run(["bash", "-c", script], check=True)


if modal.is_local():
    cached_image = os.getenv("STREAMATTN_FA3_IMAGE_ID")
    if cached_image:
        image = modal.Image.from_id(cached_image)
    else:
        from benchmarks.modal_sm90_micro_prefill_audit import image as audit_image

        image = audit_image.run_function(
            build_fa3,
            args=(FA3_BUILD,),
            cpu=4,
            memory=32768,
            timeout=2000,
        )
    # Refresh benchmark code without rebuilding the compiled FA3 toolchain.
    image = image.add_local_dir(
        "stream_attention", "/root/StreamAttn/stream_attention", copy=True
    )
    for name in (
        "profile_sm90_micro_prefill.py",
        "profile_sm90_micro_prefill_audit.py",
        "micro_prefill_baselines.py",
        "micro_prefill_optional_baselines.py",
        "profile_sm90_micro_prefill_isolated_audit.py",
        "summarize_sm90_micro_prefill_audit.py",
    ):
        image = image.add_local_file(
            f"benchmarks/{name}", f"/root/StreamAttn/benchmarks/{name}", copy=True
        )
else:
    image = None

app = modal.App("streamattn-sm90-micro-prefill-fa3-audit")


@app.function(image=image, gpu="H100", timeout=2700)
def run(cohort: str, baselines: str) -> str:
    import subprocess

    command = [
        "python",
        "-u",
        "benchmarks/profile_sm90_micro_prefill_isolated_audit.py",
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
    ]
    if baselines:
        command += ["--baselines", *baselines.split(",")]
    completed = subprocess.run(
        command,
        cwd="/root/StreamAttn",
        check=False,
    )
    import json

    result = json.loads(Path("/tmp/audit.json").read_text())
    result["subprocess_exit_code"] = completed.returncode
    for worker in result.get("workers", []):
        for stream in ("stdout", "stderr"):
            identity = worker.get(stream)
            if identity:
                worker[stream + "_text"] = Path(identity["path"]).read_text(
                    errors="replace"
                )
    return json.dumps(result, indent=2)


@app.local_entrypoint()
def main(
    cohort: str = "smoke",
    baselines: str = "",
    output_json: str = "artifacts/gate0/sm90_micro_prefill_isolated_audit_modal_h100_20260905.json",
):
    path = Path(output_json)
    if path.exists():
        raise FileExistsError(f"preserve existing evidence: {path}")
    data = run.remote(cohort, baselines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    print(f"wrote {path}")
