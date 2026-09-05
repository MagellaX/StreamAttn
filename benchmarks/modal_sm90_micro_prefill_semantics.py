"""Independent H100 replay of the explicit-position semantics matrix."""

import json
from pathlib import Path

import modal

if modal.is_local():
    from benchmarks.modal_sm90_micro_prefill import image as base_image

    image = base_image.add_local_file(
        "benchmarks/profile_sm90_micro_prefill_semantics.py",
        "/root/StreamAttn/benchmarks/profile_sm90_micro_prefill_semantics.py", copy=True,
    )
    image = image.add_local_file(
        "benchmarks/profile_sm90_micro_prefill_deferred_sum.py",
        "/root/StreamAttn/benchmarks/profile_sm90_micro_prefill_deferred_sum.py", copy=True,
    )
    image = image.add_local_file(
        "benchmarks/profile_sm90_micro_prefill_paged.py",
        "/root/StreamAttn/benchmarks/profile_sm90_micro_prefill_paged.py", copy=True,
    )
    image = image.add_local_file(
        "benchmarks/profile_sm90_micro_prefill_counters.py",
        "/root/StreamAttn/benchmarks/profile_sm90_micro_prefill_counters.py", copy=True,
    )
else:
    image = None

app = modal.App("streamattn-sm90-micro-semantics")


@app.function(image=image, gpu="H100", timeout=2700)
def run(suite: str, experiment: str) -> dict:
    import subprocess

    script = f"profile_sm90_micro_prefill_{experiment}.py"
    options = ["--suite", suite, "--provider", "modal", "--seed", "9613"] if experiment in ("semantics", "paged") else []
    proc = subprocess.run([
        "python", "-u", "benchmarks/" + script, *options,
        "--cutlass-root", "/opt/flashmla-etap/csrc/cutlass",
        "--build-dir", "/tmp/micro-semantics", "--output-json", "/tmp/semantics.json",
    ], cwd="/root/StreamAttn", check=False)
    path = Path("/tmp/semantics.json")
    result = json.loads(path.read_text()) if path.exists() else dict(complete=False)
    result["subprocess_exit_code"] = proc.returncode
    return result


@app.local_entrypoint()
def main(suite: str = "smoke", experiment: str = "semantics",
         output_json: str = "artifacts/gate0/sm90_micro_semantics_modal_h100_20260905.json"):
    if experiment not in ("semantics", "deferred_sum", "paged", "counters"):
        raise ValueError("experiment must be semantics, deferred_sum, paged or counters")
    path = Path(output_json)
    if path.exists():
        raise FileExistsError("preserve existing evidence")
    result = run.remote(suite, experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    if not result.get("complete") or result["subprocess_exit_code"]:
        raise RuntimeError("GPU matrix failed; partial evidence was retained")
