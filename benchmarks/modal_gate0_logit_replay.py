"""Modal runner for Gate-0 logit replay profiling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-gate0-logit-replay")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install(
        "flashinfer-python",
        "flashinfer-cubin",
        "transformers>=4.45.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    )
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _read_prompt_file(path: str) -> str:
    return " ".join(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _prompt_for_kind(kind: str, fallback: str) -> str:
    if kind == "code":
        return (
            "def stream_attention_decode(q, k_cache, v_cache, policy):\n"
            "    kv_head = q_head // group_size\n"
            "    if policy.seed_only_group:\n"
            "        schedule_seed_blocks(sink, recent, middle_seed)\n"
            "    else:\n"
            "        schedule_exact_blocks(k_cache, v_cache)\n"
            "    return online_softmax_merge(partial_states)\n"
        )
    if kind == "long_doc":
        return (
            "StreamAttn long context technical memorandum. "
            "The system stores cached key and value tensors, maintains online softmax state, "
            "routes true grouped-query attention heads, verifies approximation error, and "
            "falls back to exact decode when calibration is stale. "
        )
    if kind == "needle":
        return (
            "Needle retrieval context with cached KV metadata, online softmax, middle blocks, "
            "sink tokens, recent tokens, sparse decode routing, exact repair, and long-context retrieval. "
        )
    return fallback


def _json_from_output(output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, char in enumerate(output):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(output[start:])
            return payload
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"could not parse JSON from command output:\n{output[-4000:]}")


def _json_from_cmd(cmd: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    print(f"[modal-logit-replay] running: {' '.join(cmd[:5])} ...", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        chunks.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    output = "".join(chunks)
    if returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{returncode}: {' '.join(cmd)}\n{output[-6000:]}"
        )
    return _json_from_output(output)


def _run(
    *,
    model: str,
    prompt: str,
    prompt_type: str,
    layer_id: int,
    max_seq: int,
    kv_len: int,
    dtype: str,
    seed_kv_groups: str,
    trusted_seed_heads: str,
    repair_counts: str,
    max_policies: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")

    prompt_file = "/tmp/streamattn_logit_replay_prompt.txt"
    Path(prompt_file).write_text(" ".join(prompt.split()), encoding="utf-8")
    profile = _json_from_cmd(
        [
            "python",
            "-u",
            "/root/StreamAttn/benchmarks/profile_gate0_logit_replay.py",
            "--model",
            model,
            "--prompt-file",
            prompt_file,
            "--layer-id",
            str(layer_id),
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--max-seq",
            str(max(max_seq, kv_len)),
            "--kv-len",
            str(kv_len),
            "--seed-kv-groups",
            seed_kv_groups,
            "--trusted-seed-heads",
            trusted_seed_heads,
            "--repair-counts",
            repair_counts,
            "--max-policies",
            str(max_policies),
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
        ],
        env=env,
    )
    profile["capture"] = {
        "model_id": model,
        "prompt_type": prompt_type,
        "layer_id": layer_id,
        "kv_len": kv_len,
    }
    return profile


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "needle retrieval sparse attention seed only exact repair true gqa long context",
    prompt_kind: str = "",
    prompt_file: str = "",
    prompt_type: str = "needle_logit_replay_l8_32k",
    prompt_repeat: int = 3000,
    layer_id: int = 8,
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    seed_kv_groups: str = "0",
    trusted_seed_heads: str = "2,3,4",
    repair_counts: str = "0,2,4,7,11",
    max_policies: int = 18,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    output_json: str = "",
):
    if prompt_file:
        prompt = _read_prompt_file(prompt_file)
    elif prompt_kind:
        prompt = _prompt_for_kind(prompt_kind, prompt)
    prompt = ((prompt.strip() + " ") * max(1, prompt_repeat)).strip()
    result = profile_h100.remote(
        model=model,
        prompt=prompt,
        prompt_type=prompt_type,
        layer_id=layer_id,
        max_seq=max_seq,
        kv_len=kv_len,
        dtype=dtype,
        seed_kv_groups=seed_kv_groups,
        trusted_seed_heads=trusted_seed_heads,
        repair_counts=repair_counts,
        max_policies=max_policies,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
