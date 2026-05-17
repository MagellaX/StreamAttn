"""Modal runner for Gate-1 long-KV decode profiling."""

import json
import os
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-gate1-decode")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("triton==3.1.0")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _profile(
    *,
    query_lens: str,
    kv_lens: str,
    heads: str,
    kv_heads: str,
    dim: int,
    dtype: str,
    attention_type: str,
    pattern: str,
    active_fraction: str,
    skip_predicate: str,
    auto_skip_predicate: str,
    mode: str,
    causal_mode: str,
    block_size: int,
    tile_size_q: int,
    num_warps: int,
    num_stages: int,
    peak: float,
    sink_blocks: int,
    recent_blocks: int,
    error_budget: float,
    warmup: int,
    iters: int,
    metadata_warmup: int,
    metadata_iters: int,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_gate1_decode.py",
        "--query-lens",
        query_lens,
        "--kv-lens",
        kv_lens,
        "--heads",
        heads,
        "--kv-heads",
        kv_heads,
        "--dim",
        str(dim),
        "--dtype",
        dtype,
        "--attention-type",
        attention_type,
        "--pattern",
        pattern,
        "--active-fraction",
        active_fraction,
        "--skip-predicate",
        skip_predicate,
        "--auto-skip-predicate",
        auto_skip_predicate,
        "--mode",
        mode,
        "--causal-mode",
        causal_mode,
        "--block-size",
        str(block_size),
        "--tile-size-q",
        str(tile_size_q),
        "--num-warps",
        str(num_warps),
        "--num-stages",
        str(num_stages),
        "--peak",
        str(peak),
        "--sink-blocks",
        str(sink_blocks),
        "--recent-blocks",
        str(recent_blocks),
        "--error-budget",
        str(error_budget),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--metadata-warmup",
        str(metadata_warmup),
        "--metadata-iters",
        str(metadata_iters),
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
    query_lens: str = "1,4,8,16",
    kv_lens: str = "4096,8192,16384",
    heads: str = "16",
    kv_heads: str = "16",
    dim: int = 128,
    dtype: str = "fp16",
    attention_type: str = "mha",
    pattern: str = "peaked",
    active_fraction: str = "0.0625,0.25,1.0",
    skip_predicate: str = "both",
    auto_skip_predicate: str = "mass",
    mode: str = "all",
    causal_mode: str = "none",
    block_size: int = 64,
    tile_size_q: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
    peak: float = 8.0,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    error_budget: float = 1e-3,
    warmup: int = 5,
    iters: int = 10,
    metadata_warmup: int = 3,
    metadata_iters: int = 8,
    output_json: str = "",
):
    kwargs = {
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "heads": heads,
        "kv_heads": kv_heads,
        "dim": dim,
        "dtype": dtype,
        "attention_type": attention_type,
        "pattern": pattern,
        "active_fraction": active_fraction,
        "skip_predicate": skip_predicate,
        "auto_skip_predicate": auto_skip_predicate,
        "mode": mode,
        "causal_mode": causal_mode,
        "block_size": block_size,
        "tile_size_q": tile_size_q,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "peak": peak,
        "sink_blocks": sink_blocks,
        "recent_blocks": recent_blocks,
        "error_budget": error_budget,
        "warmup": warmup,
        "iters": iters,
        "metadata_warmup": metadata_warmup,
        "metadata_iters": metadata_iters,
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
