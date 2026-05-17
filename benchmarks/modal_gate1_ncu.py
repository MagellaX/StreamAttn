"""Modal runner for Gate-1 Nsight Compute mode profiling."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-gate1-ncu")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("triton==3.1.0")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _run_ncu(
    *,
    modes: str,
    seq: int,
    heads: int,
    dim: int,
    active_fraction: float,
    block_size: int,
    tile_size_q: int,
    num_warps: int,
    num_stages: int,
    metric_preset: str,
    replay_mode: str,
    kernel: str,
    skip_predicate: str,
    warmup: int,
    iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    ncu_path = shutil.which("ncu")
    ncu_version = None
    if ncu_path is not None:
        version = subprocess.run(
            ["ncu", "--version"],
            cwd="/root/StreamAttn",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        ncu_version = version.stdout

    summary_path = Path("/tmp/gate1_ncu_summary.json")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_gate1_ncu.py",
        "--modes",
        modes,
        "--seq-q",
        str(seq),
        "--seq-k",
        str(seq),
        "--heads",
        str(heads),
        "--dim",
        str(dim),
        "--active-fraction",
        str(active_fraction),
        "--block-size",
        str(block_size),
        "--tile-size-q",
        str(tile_size_q),
        "--num-warps",
        str(num_warps),
        "--num-stages",
        str(num_stages),
        "--kernel",
        kernel,
        "--skip-predicate",
        skip_predicate,
        "--metric-preset",
        metric_preset,
        "--replay-mode",
        replay_mode,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--collect-stats",
        "--continue-on-ncu-error",
        "--summary-json-out",
        str(summary_path),
    ]
    result = subprocess.run(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    payload = {
        "ncu_path": ncu_path,
        "ncu_version": ncu_version,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-8000:],
        "command": cmd,
    }
    if summary_path.exists():
        payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload


@app.function(image=image, gpu="A100", timeout=3600)
def profile_a100(
    modes: str,
    seq: int,
    heads: int,
    dim: int,
    active_fraction: float,
    block_size: int,
    tile_size_q: int,
    num_warps: int,
    num_stages: int,
    metric_preset: str,
    replay_mode: str,
    kernel: str,
    skip_predicate: str,
    warmup: int,
    iters: int,
):
    return _run_ncu(
        modes=modes,
        seq=seq,
        heads=heads,
        dim=dim,
        active_fraction=active_fraction,
        block_size=block_size,
        tile_size_q=tile_size_q,
        num_warps=num_warps,
        num_stages=num_stages,
        metric_preset=metric_preset,
        replay_mode=replay_mode,
        kernel=kernel,
        skip_predicate=skip_predicate,
        warmup=warmup,
        iters=iters,
    )


@app.function(image=image, gpu="H100", timeout=3600)
def profile_h100(
    modes: str,
    seq: int,
    heads: int,
    dim: int,
    active_fraction: float,
    block_size: int,
    tile_size_q: int,
    num_warps: int,
    num_stages: int,
    metric_preset: str,
    replay_mode: str,
    kernel: str,
    skip_predicate: str,
    warmup: int,
    iters: int,
):
    return _run_ncu(
        modes=modes,
        seq=seq,
        heads=heads,
        dim=dim,
        active_fraction=active_fraction,
        block_size=block_size,
        tile_size_q=tile_size_q,
        num_warps=num_warps,
        num_stages=num_stages,
        metric_preset=metric_preset,
        replay_mode=replay_mode,
        kernel=kernel,
        skip_predicate=skip_predicate,
        warmup=warmup,
        iters=iters,
    )


@app.local_entrypoint()
def main(
    target: str = "h100",
    modes: str = "5,7,8,9,0",
    seq: int = 4096,
    heads: int = 16,
    dim: int = 128,
    active_fraction: float = 0.0625,
    block_size: int = 64,
    tile_size_q: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
    metric_preset: str = "basic",
    replay_mode: str = "kernel",
    kernel: str = "generic",
    skip_predicate: str = "mass",
    warmup: int = 5,
    iters: int = 20,
):
    kwargs = {
        "modes": modes,
        "seq": seq,
        "heads": heads,
        "dim": dim,
        "active_fraction": active_fraction,
        "block_size": block_size,
        "tile_size_q": tile_size_q,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "metric_preset": metric_preset,
        "replay_mode": replay_mode,
        "kernel": kernel,
        "skip_predicate": skip_predicate,
        "warmup": warmup,
        "iters": iters,
    }
    if target == "a100":
        print(profile_a100.remote(**kwargs))
    elif target == "h100":
        print(profile_h100.remote(**kwargs))
    else:
        raise ValueError("target must be a100 or h100")
