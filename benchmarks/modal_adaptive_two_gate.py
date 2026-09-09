"""Bounded backup canary when the preferred GPU provider cannot provision."""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("streamattn-adaptive-two-gate")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install("pyyaml", "pytest")
    .add_local_dir(str(ROOT / "stream_attention"), remote_path="/root/StreamAttn/stream_attention", copy=True)
    .add_local_file(str(ROOT / "benchmarks/profile_adaptive_two_gate.py"),
                    remote_path="/root/StreamAttn/benchmarks/profile_adaptive_two_gate.py", copy=True)
    .add_local_file(str(ROOT / "benchmarks/adaptive_known_support.py"),
                    remote_path="/root/StreamAttn/benchmarks/adaptive_known_support.py", copy=True)
    .add_local_file(str(ROOT / "tests/test_adaptive_two_gate_gpu.py"),
                    remote_path="/root/StreamAttn/tests/test_adaptive_two_gate_gpu.py", copy=True)
)

hf_cache = modal.Volume.from_name("streamattn-hf-cache", create_if_missing=True)
secret_name = os.environ.get("STREAMATTN_MODAL_HF_SECRET", "").strip()
real_image = (
    image.pip_install("transformers==4.51.3", "accelerate", "sentencepiece", "safetensors")
    .add_local_file(str(ROOT / "benchmarks/profile_adaptive_real_activations.py"),
                    remote_path="/root/StreamAttn/benchmarks/profile_adaptive_real_activations.py", copy=True)
    .add_local_file(str(ROOT / "benchmarks/profile_real_llm_gate1_heads.py"),
                    remote_path="/root/StreamAttn/benchmarks/profile_real_llm_gate1_heads.py", copy=True)
)


@app.function(image=real_image, gpu="H100", timeout=900, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name(secret_name)] if secret_name else [])
def real_activations():
    import subprocess
    path = Path("/tmp/adaptive_real.json")
    subprocess.run(["python", "-u", "/root/StreamAttn/benchmarks/profile_adaptive_real_activations.py",
                    "--output-json", str(path)], timeout=840, check=True)
    hf_cache.commit()
    return json.loads(path.read_text()), path.with_suffix(".captures.pt").read_bytes()


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
    tests = subprocess.run(["python", "-m", "pytest", "-q", "tests/test_adaptive_two_gate_gpu.py"],
                           cwd="/root/StreamAttn", timeout=120, check=False)
    result["gpu_pytest_returncode"] = tests.returncode
    result["complete"] = result["complete"] and tests.returncode == 0
    return result


@app.local_entrypoint()
def main(output_json: str, precision_only: bool = False, real: bool = False):
    path = Path(output_json)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if real:
        result, captures = real_activations.remote()
        path.with_suffix(".captures.pt").write_bytes(captures)
    else:
        result = canary.remote(precision_only)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}: complete={result['complete']}")
    if not result["complete"]:
        raise RuntimeError("Canary failed; artifact retained")
