"""Run the transposed m64n8 exact-QK milestone on a Modal H100."""

from __future__ import annotations

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ninja-build")
    .pip_install(
        "einops",
        "flashinfer-python==0.6.12",
        "flashinfer-cubin==0.6.12",
        "ninja",
    )
    .run_commands(
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/pengcuo/FlashMLA-ETAP.git /opt/flashmla-etap && "
        "cd /opt/flashmla-etap && "
        "git sparse-checkout init --cone && "
        "git sparse-checkout set csrc/cutlass/include && "
        "git fetch --depth=1 origin 39e616041ae6fb1243a0f6ac891e72d576b640e5 && "
        "git checkout 39e616041ae6fb1243a0f6ac891e72d576b640e5"
    )
    .add_local_dir("benchmarks", remote_path="/root/StreamAttn/benchmarks", copy=True)
    .add_local_dir("stream_attention", remote_path="/root/StreamAttn/stream_attention", copy=True)
)

app = modal.App("streamattn-transposed-wgmma-exact-qk")
volume = modal.Volume.from_name("streamattn-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/artifacts": volume},
)
def run(
    *,
    batch: int = 4,
    kv_len: int = 32768,
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 64,
    num_splits_list: str = "8,16,17,32,33,64,128,256,512",
    warmup: int = 10,
    iters: int = 100,
    repeats: int = 3,
    output_json: str = "/artifacts/gate0/transposed_wgmma_exact_qk_h100.json",
) -> str:
    import os
    import subprocess

    os.chdir("/root/StreamAttn")
    os.environ["PYTHONUNBUFFERED"] = "1"
    cutlass_root = "/opt/flashmla-etap/csrc/cutlass"
    cmd = [
        "python",
        "benchmarks/profile_transposed_wgmma_exact_qk.py",
        "--batch", str(batch),
        "--kv-len", str(kv_len),
        "--q-heads", str(q_heads),
        "--kv-heads", str(kv_heads),
        "--head-dim", str(head_dim),
        "--num-splits-list", num_splits_list,
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--repeats", str(repeats),
        "--cutlass-root", cutlass_root,
        "--build-dir", "/artifacts/torch_extensions/transposed_wgmma_exact_qk",
        "--output-json", output_json,
    ]
    print("[modal-transposed-qk] launching benchmark", flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()
    return "ok"


@app.local_entrypoint()
def main(
    batch: int = 4,
    kv_len: int = 32768,
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 64,
    num_splits_list: str = "8,16,17,32,33,64,128,256,512",
    warmup: int = 10,
    iters: int = 100,
    repeats: int = 3,
    output_json: str = "/artifacts/gate0/transposed_wgmma_exact_qk_h100.json",
) -> None:
    print(run.remote(
        batch=batch,
        kv_len=kv_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        num_splits_list=num_splits_list,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        output_json=output_json,
    ))
