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
    image = image.pip_install("flashinfer-python==0.6.13", "flashinfer-cubin==0.6.13", "pyyaml")
    for filename in ("profile_sm90_micro_prefill_mixed.py", "micro_prefill_baselines.py",
                     "sm90_mixed_attribution.py", "profile_sm90_mixed_attribution_counters.py"):
        image = image.add_local_file("benchmarks/" + filename,
                                     "/root/StreamAttn/benchmarks/" + filename, copy=True)
else:
    image = None

app = modal.App("streamattn-sm90-micro-semantics")


@app.function(image=image, gpu="H100", cpu=8, timeout=2700)
def run(suite: str, experiment: str, seed: int = 9613) -> dict:
    import subprocess

    script = f"profile_sm90_micro_prefill_{'counters' if experiment == 'source_counters' else 'mixed' if experiment in ('attribution', 'producer_copy', 'page_pair') else experiment}.py"
    options = ["--suite", suite, "--provider", "modal", "--seed", str(seed)] if experiment in ("semantics", "paged", "mixed") else []
    if experiment == "source_counters":
        options = ["--source-correlated"]
    if experiment in ("attribution", "producer_copy", "page_pair"):
        options = ["--suite", suite, "--provider", "modal", "--seed", str(seed), "--attribution"]
        if experiment == "producer_copy":
            options += ["--producer-copy"]
        if experiment == "page_pair":
            options += ["--page-pair"]
    if experiment in ("attribution_counters", "paged_source_counters", "page_pair_counters"):
        script = "profile_sm90_mixed_attribution_counters.py"
        options = ["--source-correlated"] if experiment != "attribution_counters" else []
        if experiment == "page_pair_counters":
            options += ["--page-pair"]
    proc = subprocess.run([
        "python", "-u", "benchmarks/" + script, *options,
        "--cutlass-root", "/opt/flashmla-etap/csrc/cutlass",
        "--build-dir", "/tmp/micro-semantics", "--output-json", "/tmp/semantics.json",
    ], cwd="/root/StreamAttn", check=False)
    path = Path("/tmp/semantics.json")
    result = json.loads(path.read_text()) if path.exists() else dict(complete=False)
    result["subprocess_exit_code"] = proc.returncode
    if experiment == "producer_copy" and result.get("complete") and not proc.returncode:
        counters = subprocess.run([
            "python", "-u", "benchmarks/profile_sm90_mixed_attribution_counters.py",
            "--source-correlated", "--producer-copy",
            "--cutlass-root", "/opt/flashmla-etap/csrc/cutlass",
            "--build-dir", "/tmp/micro-semantics", "--output-json", "/tmp/copy-counters.json",
        ], cwd="/root/StreamAttn", check=False)
        counter_path = Path("/tmp/copy-counters.json")
        result["source_counters"] = json.loads(counter_path.read_text()) if counter_path.exists() else dict(complete=False)
        result["counter_subprocess_exit_code"] = counters.returncode
    return result


@app.local_entrypoint()
def main(suite: str = "smoke", experiment: str = "semantics",
         output_json: str = "artifacts/gate0/sm90_micro_semantics_modal_h100_20260905.json",
         seed: int = 9613):
    if experiment not in ("semantics", "deferred_sum", "paged", "counters", "source_counters", "mixed",
                           "attribution", "producer_copy", "page_pair", "page_pair_counters", "attribution_counters", "paged_source_counters"):
        raise ValueError("unknown semantics experiment")
    path = Path(output_json)
    if path.exists():
        raise FileExistsError("preserve existing evidence")
    result = run.remote(suite, experiment, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    if not result.get("complete") or result["subprocess_exit_code"] or result.get("counter_subprocess_exit_code"):
        raise RuntimeError("GPU matrix failed; partial evidence was retained")
