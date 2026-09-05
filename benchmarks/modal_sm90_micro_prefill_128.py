"""Bounded H100 M128 resources/correctness/performance canary."""

import json
import base64
from pathlib import Path

import modal

if modal.is_local():
    from benchmarks.modal_sm90_micro_prefill import image as base_image

    image = base_image.add_local_file(
        "benchmarks/profile_sm90_micro_prefill_128.py",
        "/root/StreamAttn/benchmarks/profile_sm90_micro_prefill_128.py",
        copy=True,
    )
    image = image.add_local_file(
        "benchmarks/sm90_binary_diagnostics.py",
        "/root/StreamAttn/benchmarks/sm90_binary_diagnostics.py",
        copy=True,
    )
else:
    image = None

app = modal.App("streamattn-sm90-micro-prefill-128")


@app.function(image=image, gpu="H100", timeout=2700)
def run(mode: str, suite: str, protocol: str, binary_diagnostics: bool) -> dict:
    import io
    import subprocess
    import zipfile

    build = Path("/tmp/streamattn-micro128-build")
    build.mkdir(exist_ok=True)
    command = [
        "python",
        "-u",
        "benchmarks/profile_sm90_micro_prefill_128.py",
        "--mode",
        mode,
        "--suite",
        suite,
        "--protocol",
        protocol,
        "--matches-splits",
        "--cutlass-root",
        "/opt/flashmla-etap/csrc/cutlass",
        "--build-dir",
        "/tmp/streamattn-micro128-build",
        "--output",
        "/tmp/result.json",
    ]
    if binary_diagnostics:
        command.append("--binary-diagnostics")
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
    path = Path("/tmp/result.json")
    result = (
        json.loads(path.read_text()) if path.exists() else dict(passed=False, rows=[])
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
                ".txt",
                ".cu",
                ".cpp",
                ".gz",
            ):
                archive.write(file, file.relative_to(build))
    return dict(
        result=result, diagnostics_zip_base64=base64.b64encode(data.getvalue()).decode()
    )


@app.local_entrypoint()
def main(
    mode: str = "resources",
    suite: str = "smoke",
    protocol: str = "both",
    binary_diagnostics: bool = False,
    output_json: str = "artifacts/gate0/sm90_micro_prefill_128_resources_modal_h100_20260905.json",
):
    path = Path(output_json)
    diagnostics = path.with_suffix(".diagnostics.zip")
    if path.exists() or diagnostics.exists():
        raise FileExistsError(f"preserve existing evidence: {path}")
    data = run.remote(mode, suite, protocol, binary_diagnostics)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data["result"], indent=2) + "\n", encoding="utf-8")
    diagnostics.write_bytes(base64.b64decode(data["diagnostics_zip_base64"]))
    print(f"wrote {path}")
    if data["result"].get("subprocess_exit_code"):
        raise RuntimeError("GPU experiment failed; diagnostics retained")
