"""Modal runner for real-LLM Gate-1 per-head telemetry."""

import json
import os
import subprocess

import modal


app = modal.App("streamattn-real-llm-gate1-heads")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install(
        "triton==3.1.0",
        "transformers>=4.45.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    )
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _profile(
    *,
    model: str,
    prompt: str,
    layers: str,
    max_seq: int,
    min_seq: int,
    block_size: int,
    tile_size_q: int,
    dtype: str,
    no_rope: bool,
    use_safetensors: bool,
    trust_remote_code: bool,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_real_llm_gate1_heads.py",
        "--model",
        model,
        "--prompt",
        prompt,
        "--layers",
        layers,
        "--max-seq",
        str(max_seq),
        "--min-seq",
        str(min_seq),
        "--block-size",
        str(block_size),
        "--tile-size-q",
        str(tile_size_q),
        "--dtype",
        dtype,
    ]
    if no_rope:
        cmd.append("--no-rope")
    if not use_safetensors:
        cmd.append("--no-use-safetensors")
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    output = subprocess.check_output(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
    )
    return json.loads(output)


@app.function(image=image, gpu="A100", timeout=3600)
def profile_a100(
    model: str,
    prompt: str,
    layers: str,
    max_seq: int,
    min_seq: int,
    block_size: int,
    tile_size_q: int,
    dtype: str,
    no_rope: bool,
    use_safetensors: bool,
    trust_remote_code: bool,
):
    return _profile(
        model=model,
        prompt=prompt,
        layers=layers,
        max_seq=max_seq,
        min_seq=min_seq,
        block_size=block_size,
        tile_size_q=tile_size_q,
        dtype=dtype,
        no_rope=no_rope,
        use_safetensors=use_safetensors,
        trust_remote_code=trust_remote_code,
    )


@app.function(image=image, gpu="H100", timeout=3600)
def profile_h100(
    model: str,
    prompt: str,
    layers: str,
    max_seq: int,
    min_seq: int,
    block_size: int,
    tile_size_q: int,
    dtype: str,
    no_rope: bool,
    use_safetensors: bool,
    trust_remote_code: bool,
):
    return _profile(
        model=model,
        prompt=prompt,
        layers=layers,
        max_seq=max_seq,
        min_seq=min_seq,
        block_size=block_size,
        tile_size_q=tile_size_q,
        dtype=dtype,
        no_rope=no_rope,
        use_safetensors=use_safetensors,
        trust_remote_code=trust_remote_code,
    )


@app.local_entrypoint()
def main(
    target: str = "h100",
    model: str = "HuggingFaceTB/SmolLM2-135M",
    prompt: str = "attention kernels need hardware aligned sparse routing " * 32,
    layers: str = "0",
    max_seq: int = 256,
    min_seq: int = 1,
    block_size: int = 16,
    tile_size_q: int = 16,
    dtype: str = "fp16",
    no_rope: bool = False,
    use_safetensors: bool = True,
    trust_remote_code: bool = False,
):
    kwargs = {
        "model": model,
        "prompt": prompt,
        "layers": layers,
        "max_seq": max_seq,
        "min_seq": min_seq,
        "block_size": block_size,
        "tile_size_q": tile_size_q,
        "dtype": dtype,
        "no_rope": no_rope,
        "use_safetensors": use_safetensors,
        "trust_remote_code": trust_remote_code,
    }
    if target == "a100":
        print(profile_a100.remote(**kwargs))
    elif target == "h100":
        print(profile_h100.remote(**kwargs))
    else:
        raise ValueError("target must be a100 or h100")
