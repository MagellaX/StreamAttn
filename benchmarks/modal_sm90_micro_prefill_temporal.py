"""Bounded H100 R64 scheduling experiment, with retained compiler diagnostics."""

import base64
import json
from pathlib import Path

import modal

if modal.is_local():
    from benchmarks.modal_sm90_micro_prefill import image as base_image

    image = base_image
    for name in (
        "profile_sm90_micro_prefill_128.py",
        "profile_sm90_micro_prefill_temporal.py",
        "sm90_binary_diagnostics.py",
    ):
        path = Path("benchmarks") / name
        if path.exists():
            image = image.add_local_file(
                str(path), f"/root/StreamAttn/benchmarks/{name}", copy=True
            )
else:
    image = None

app = modal.App("streamattn-sm90-r64-temporal")


@app.function(image=image, gpu="H100", timeout=2700)
def run(mode: str, suite: str) -> dict:
    import io
    import subprocess
    import zipfile

    build = Path("/tmp/streamattn-temporal")
    build.mkdir(exist_ok=True)
    command = [
        "python",
        "-u",
        "benchmarks/profile_sm90_micro_prefill_temporal.py",
        "--mode",
        mode,
        "--suite",
        suite,
        "--cutlass-root",
        "/opt/flashmla-etap/csrc/cutlass",
        "--build-dir",
        str(build),
        "--output",
        "/tmp/temporal.json",
    ]
    with (build / "compiler_and_run.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd="/root/StreamAttn",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        code = process.wait()
    path = Path("/tmp/temporal.json")
    result = (
        json.loads(path.read_text()) if path.exists() else dict(complete=False, rows=[])
    )
    result.update(provider="modal", subprocess_exit_code=code)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in build.rglob("*"):
            if file.is_file() and file.suffix in (
                ".log",
                ".json",
                ".sass",
                ".ptx",
                ".so",
                ".cubin",
                ".ninja",
                ".cu",
                ".cpp",
                ".txt",
                ".gz",
            ):
                archive.write(file, file.relative_to(build))
    return dict(
        result=result, diagnostics_zip_base64=base64.b64encode(data.getvalue()).decode()
    )


@app.local_entrypoint()
def main(
    mode: str = "benchmark",
    suite: str = "smoke",
    output_json: str = "artifacts/gate0/sm90_r64_temporal_smoke_modal_h100_20260905.json",
):
    output = Path(output_json)
    diagnostics = output.with_suffix(".diagnostics.zip")
    if output.exists() or diagnostics.exists():
        raise FileExistsError("preserve existing evidence")
    payload = run.remote(mode, suite)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload["result"], indent=2) + "\n", encoding="utf-8")
    diagnostics.write_bytes(base64.b64decode(payload["diagnostics_zip_base64"]))
    print(f"wrote {output} and {diagnostics}")
    if payload["result"].get("subprocess_exit_code"):
        raise RuntimeError(
            "GPU experiment failed; partial evidence and diagnostics were retained"
        )
