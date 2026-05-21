"""Modal runner for the TK tensor-core exact true-GQA decode baseline."""

from __future__ import annotations

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ninja-build")
    .pip_install(
        "einops",
        "flashinfer-python",
        "flashinfer-cubin",
        "ninja",
    )
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)

app = modal.App("streamattn-tk-tensor-core-exact-decode")
volume = modal.Volume.from_name("streamattn-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/artifacts": volume},
)
def run(
    *,
    kv_len: int = 32768,
    q_heads: int = 14,
    kv_heads: int = 2,
    head_dim: int = 128,
    dtype: str = "bf16",
    seed: int = 0,
    warmup: int = 5,
    iters: int = 20,
    output_json: str = "/artifacts/gate0/tk_tensor_core_exact_decode_h100.json",
) -> str:
    import os
    import subprocess

    os.chdir("/root/StreamAttn")
    tk_root = "/artifacts/backend_sources/ThunderKittens"
    checkout_dir = "/artifacts/backend_sources"
    cmd = [
        "python",
        "benchmarks/profile_tk_tensor_core_exact_decode.py",
        "--kv-len",
        str(kv_len),
        "--q-heads",
        str(q_heads),
        "--kv-heads",
        str(kv_heads),
        "--head-dim",
        str(head_dim),
        "--dtype",
        dtype,
        "--seed",
        str(seed),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--tk-root",
        tk_root,
        "--checkout-dir",
        checkout_dir,
        "--output-json",
        output_json,
    ]
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"TK tensor-core exact decode benchmark failed with code {proc.returncode}")
    volume.commit()
    return proc.stdout


@app.local_entrypoint()
def main(
    kv_len: int = 32768,
    q_heads: int = 14,
    kv_heads: int = 2,
    head_dim: int = 128,
    dtype: str = "bf16",
    seed: int = 0,
    warmup: int = 5,
    iters: int = 20,
    output_json: str = "/artifacts/gate0/tk_tensor_core_exact_decode_h100.json",
) -> None:
    print(
        run.remote(
            kv_len=kv_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            seed=seed,
            warmup=warmup,
            iters=iters,
            output_json=output_json,
        )
    )
