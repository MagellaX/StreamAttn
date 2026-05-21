"""Modal runner for the TK KV-group repair break-even benchmark."""

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

app = modal.App("streamattn-tk-kv-group-repair")
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
    q_heads: int = 7,
    kv_heads: int = 1,
    head_dim: int = 64,
    dtype: str = "bf16",
    seed: int = 0,
    warmup: int = 3,
    iters: int = 10,
    num_chunks: int = 256,
    repair_threads: int = 128,
    repair_num_chunks: int = 256,
    repair_block_d: int = 32,
    repair_counts: str = "0,1,2,3,4,7",
    repair_order: str = "",
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 2,
    block_order: str = "recent_first",
    output_json: str = "/artifacts/gate0/tk_kv_group_repair_d64_bf16_h100.json",
) -> str:
    import os
    import subprocess

    os.chdir("/root/StreamAttn")
    os.environ["PYTHONUNBUFFERED"] = "1"
    tk_root = "/artifacts/backend_sources/ThunderKittens"
    checkout_dir = "/artifacts/backend_sources"
    cmd = [
        "python",
        "benchmarks/profile_tk_kv_group_repair.py",
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
        "--num-chunks",
        str(num_chunks),
        "--repair-threads",
        str(repair_threads),
        "--repair-num-chunks",
        str(repair_num_chunks),
        "--repair-block-d",
        str(repair_block_d),
        "--repair-counts",
        repair_counts,
        "--block-size",
        str(block_size),
        "--sink-blocks",
        str(sink_blocks),
        "--recent-blocks",
        str(recent_blocks),
        "--middle-seed-blocks",
        str(middle_seed_blocks),
        "--block-order",
        block_order,
        "--tk-root",
        tk_root,
        "--checkout-dir",
        checkout_dir,
        "--output-json",
        output_json,
    ]
    if repair_order:
        cmd.extend(["--repair-order", repair_order])
    print("[modal-tk-repair] launching benchmark subprocess", flush=True)
    proc = subprocess.run(cmd, check=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"TK KV-group repair benchmark failed with code {proc.returncode}")
    volume.commit()
    return "ok"


@app.local_entrypoint()
def main(
    kv_len: int = 32768,
    q_heads: int = 7,
    kv_heads: int = 1,
    head_dim: int = 64,
    dtype: str = "bf16",
    seed: int = 0,
    warmup: int = 3,
    iters: int = 10,
    num_chunks: int = 256,
    repair_threads: int = 128,
    repair_num_chunks: int = 256,
    repair_block_d: int = 32,
    repair_counts: str = "0,1,2,3,4,7",
    repair_order: str = "",
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 2,
    block_order: str = "recent_first",
    output_json: str = "/artifacts/gate0/tk_kv_group_repair_d64_bf16_h100.json",
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
            num_chunks=num_chunks,
            repair_threads=repair_threads,
            repair_num_chunks=repair_num_chunks,
            repair_block_d=repair_block_d,
            repair_counts=repair_counts,
            repair_order=repair_order,
            block_size=block_size,
            sink_blocks=sink_blocks,
            recent_blocks=recent_blocks,
            middle_seed_blocks=middle_seed_blocks,
            block_order=block_order,
            output_json=output_json,
        )
    )
