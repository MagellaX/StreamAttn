"""Bounded backup canary when the preferred GPU provider cannot provision."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("streamattn-adaptive-two-gate")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install("pyyaml")
    .add_local_dir(str(ROOT / "stream_attention"), remote_path="/root/StreamAttn/stream_attention", copy=True)
    .add_local_file(str(ROOT / "benchmarks/profile_adaptive_two_gate.py"),
                    remote_path="/root/StreamAttn/benchmarks/profile_adaptive_two_gate.py", copy=True)
)


@app.function(image=image, gpu="H100", timeout=600, max_containers=1)
def canary(precision_only: bool = False):
    import subprocess
    result_path = Path("/tmp/adaptive.json")
    command = ["python", "-u", "/root/StreamAttn/benchmarks/profile_adaptive_two_gate.py",
               "--provider", "modal", "--output-json", str(result_path)]
    if precision_only:
        command.append("--diagnose-causal")
    process = subprocess.run(command, timeout=540, check=False)
    if not result_path.exists():
        raise RuntimeError(f"Canary exited {process.returncode} without an artifact")
    result = json.loads(result_path.read_text())
    result["process_returncode"] = process.returncode
    return result


@app.local_entrypoint()
def main(output_json: str, precision_only: bool = False):
    path = Path(output_json)
    if path.exists():
        raise FileExistsError(path)
    result = canary.remote(precision_only)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}: complete={result['complete']}")
    if not result["complete"]:
        raise RuntimeError("Canary failed; artifact retained")
