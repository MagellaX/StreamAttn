"""Modal runner for Gate-0 summary-bound profiling."""

import json
import os
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-gate0-summary-bounds")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("triton==3.1.0")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _profile(
    *,
    query_len: int,
    kv_lens: str,
    heads: str,
    dim: int,
    dtype: str,
    pattern: str,
    active_fraction: str,
    block_size: str,
    summary_outliers: str,
    peak: float,
    sink_blocks: int,
    recent_blocks: int,
    error_budget: float,
    bound_tolerance: float,
    min_predicted_skip_fraction: float,
    max_false_negative_rate: float,
    warmup: int,
    iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_gate0_summary_bounds.py",
        "--query-len",
        str(query_len),
        "--kv-lens",
        kv_lens,
        "--heads",
        heads,
        "--dim",
        str(dim),
        "--dtype",
        dtype,
        "--pattern",
        pattern,
        "--active-fraction",
        active_fraction,
        "--block-size",
        block_size,
        "--summary-outliers",
        summary_outliers,
        "--peak",
        str(peak),
        "--sink-blocks",
        str(sink_blocks),
        "--recent-blocks",
        str(recent_blocks),
        "--error-budget",
        str(error_budget),
        "--bound-tolerance",
        str(bound_tolerance),
        "--min-predicted-skip-fraction",
        str(min_predicted_skip_fraction),
        "--max-false-negative-rate",
        str(max_false_negative_rate),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    output = subprocess.check_output(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
    )
    return json.loads(output)


@app.function(image=image, gpu="A100", timeout=2400)
def profile_a100(**kwargs):
    return _profile(**kwargs)


@app.function(image=image, gpu="H100", timeout=2400)
def profile_h100(**kwargs):
    return _profile(**kwargs)


@app.local_entrypoint()
def main(
    target: str = "h100",
    query_len: int = 1,
    kv_lens: str = "8192,16384,32768",
    heads: str = "16,32",
    dim: int = 128,
    dtype: str = "fp16",
    pattern: str = "peaked",
    active_fraction: str = "0.0625,0.125,0.25,1.0",
    block_size: str = "64,128",
    summary_outliers: str = "0,1,2,4",
    peak: float = 8.0,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    error_budget: float = 1e-3,
    bound_tolerance: float = 1e-4,
    min_predicted_skip_fraction: float = 0.25,
    max_false_negative_rate: float = 0.0,
    warmup: int = 1,
    iters: int = 3,
    output_json: str = "",
):
    kwargs = {
        "query_len": query_len,
        "kv_lens": kv_lens,
        "heads": heads,
        "dim": dim,
        "dtype": dtype,
        "pattern": pattern,
        "active_fraction": active_fraction,
        "block_size": block_size,
        "summary_outliers": summary_outliers,
        "peak": peak,
        "sink_blocks": sink_blocks,
        "recent_blocks": recent_blocks,
        "error_budget": error_budget,
        "bound_tolerance": bound_tolerance,
        "min_predicted_skip_fraction": min_predicted_skip_fraction,
        "max_false_negative_rate": max_false_negative_rate,
        "warmup": warmup,
        "iters": iters,
    }
    if target == "a100":
        result = profile_a100.remote(**kwargs)
    elif target == "h100":
        result = profile_h100.remote(**kwargs)
    else:
        raise ValueError("target must be a100 or h100")

    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
