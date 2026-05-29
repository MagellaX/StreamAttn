"""Modal runner for end-to-end seed-only route-bundle decode timing."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-only-route-bundle-decode")

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
    .add_local_dir(
        ".",
        remote_path="/root/StreamAttn",
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".pytest_cache/**",
            "__pycache__/**",
            "artifacts/**",
        ],
    )
)


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
    print(f"[modal-route-bundle-decode] running: {' '.join(cmd[:5])} ...", flush=True)
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
    in_json_payload = False
    assert process.stdout is not None
    for line in process.stdout:
        chunks.append(line)
        if line.lstrip().startswith("{"):
            in_json_payload = True
        if not in_json_payload:
            print(line, end="", flush=True)
    returncode = process.wait()
    output = "".join(chunks)
    if returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{returncode}: {' '.join(cmd)}\n{output[-6000:]}"
        )
    return _json_from_output(output)


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_seed_only_route_bundle_decode.py",
        "--model",
        kwargs["model"],
        "--layers",
        kwargs["layers"],
        "--policy-names",
        kwargs["policy_names"],
        "--bucket-route-policy",
        kwargs["bucket_route_policy"],
        "--prompt-kinds",
        kwargs["prompt_kinds"],
        "--prompt-repeat",
        str(kwargs["prompt_repeat"]),
        "--batch-size",
        str(kwargs["batch_size"]),
        "--device",
        "cuda",
        "--dtype",
        kwargs["dtype"],
        "--max-seq",
        str(kwargs["max_seq"]),
        "--steps",
        str(kwargs["steps"]),
        "--warmup-steps",
        str(kwargs["warmup_steps"]),
        "--top-k",
        str(kwargs["top_k"]),
        "--max-kl",
        str(kwargs["max_kl"]),
        "--min-topk-overlap",
        str(kwargs["min_topk_overlap"]),
        "--max-logprob-delta",
        str(kwargs["max_logprob_delta"]),
        "--sample-temperature",
        str(kwargs["sample_temperature"]),
        "--sample-top-p",
        str(kwargs["sample_top_p"]),
        "--sample-top-k",
        str(kwargs["sample_top_k"]),
        "--sample-seed",
        str(kwargs["sample_seed"]),
        "--q-heads",
        str(kwargs["q_heads"]),
        "--kv-heads",
        str(kwargs["kv_heads"]),
        "--head-dim",
        str(kwargs["head_dim"]),
        "--block-size",
        str(kwargs["block_size"]),
        "--sink-blocks",
        str(kwargs["sink_blocks"]),
        "--recent-blocks",
        str(kwargs["recent_blocks"]),
        "--middle-seed-blocks",
        str(kwargs["middle_seed_blocks"]),
        "--block-order",
        kwargs["block_order"],
        "--num-warps",
        str(kwargs["num_warps"]),
        "--num-stages",
        str(kwargs["num_stages"]),
    ]
    if kwargs["prompt_file"]:
        cmd.extend(["--prompt-file", kwargs["prompt_file"]])
    if kwargs["prompt_file_kinds"]:
        cmd.extend(["--prompt-file-kinds", kwargs["prompt_file_kinds"]])
    if kwargs["prompt_file_rows_per_kind"] > 0:
        cmd.extend(["--prompt-file-rows-per-kind", str(kwargs["prompt_file_rows_per_kind"])])
    if kwargs["max_prompts"] > 0:
        cmd.extend(["--max-prompts", str(kwargs["max_prompts"])])
    if kwargs["prompt_truncation_side"]:
        cmd.extend(["--prompt-truncation-side", kwargs["prompt_truncation_side"]])
    if kwargs["layer_seed_overrides"]:
        cmd.extend(["--layer-seed-overrides", kwargs["layer_seed_overrides"]])
    if kwargs["profile_patch_timing"]:
        cmd.append("--profile-patch-timing")
    if kwargs["explicit_cache_position"]:
        cmd.append("--explicit-cache-position")
    if kwargs["native_routed_cache"]:
        cmd.append("--native-routed-cache")
    if kwargs["native_cache_hf_sync_layers"]:
        cmd.extend(["--native-cache-hf-sync-layers", kwargs["native_cache_hf_sync_layers"]])
    if kwargs["native_cache_attach_hf_views"]:
        cmd.append("--native-cache-attach-hf-views")
    if kwargs["native_attention_module"]:
        cmd.append("--native-attention-module")
    if kwargs["fused_rope_append_seed"]:
        cmd.append("--fused-rope-append-seed")
    if kwargs["packed_qkv_projection"]:
        cmd.append("--packed-qkv-projection")
    if kwargs["packed_qkv_fused_input"]:
        cmd.append("--packed-qkv-fused-input")
    if kwargs["direct_o_proj"]:
        cmd.append("--direct-o-proj")
    if kwargs["triton_o_proj"]:
        cmd.append("--triton-o-proj")
    if kwargs["dynamic_selector_layers"]:
        cmd.extend(["--dynamic-selector-layers", kwargs["dynamic_selector_layers"]])
    if kwargs["dynamic_selector_profile"]:
        cmd.extend(["--dynamic-selector-profile", kwargs["dynamic_selector_profile"]])
    if kwargs["allow_mixed_seed_configs"]:
        cmd.append("--allow-mixed-seed-configs")
    if not kwargs["product_strict"]:
        cmd.append("--no-product-strict")
    if kwargs["attn_implementation"]:
        cmd.extend(["--attn-implementation", kwargs["attn_implementation"]])
    if not kwargs["use_packaged_policies"]:
        cmd.append("--no-use-packaged-policies")
    return _json_from_cmd(cmd, env=env)


@app.function(image=image, gpu="H100", timeout=7200)
def profile_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    layers: str = "0,14,16,24,26,27,35",
    policy_names: str = "",
    use_packaged_policies: bool = True,
    bucket_route_policy: str = "none",
    product_strict: bool = True,
    prompt_kinds: str = "needle,code,long_doc,chat_doc",
    prompt_file: str = "",
    prompt_file_kinds: str = "",
    prompt_file_rows_per_kind: int = 0,
    max_prompts: int = 0,
    prompt_truncation_side: str = "right",
    prompt_repeat: int = 3000,
    batch_size: int = 4,
    max_seq: int = 32768,
    steps: int = 32,
    warmup_steps: int = 2,
    dtype: str = "fp16",
    attn_implementation: str = "",
    top_k: int = 5,
    max_kl: float = 1.0e-4,
    min_topk_overlap: int = 4,
    max_logprob_delta: float = 2.0e-3,
    sample_temperature: float = 0.8,
    sample_top_p: float = 0.95,
    sample_top_k: int = 0,
    sample_seed: int = 1234,
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 128,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    layer_seed_overrides: str = "",
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 2,
    profile_patch_timing: bool = False,
    explicit_cache_position: bool = False,
    native_routed_cache: bool = False,
    native_cache_hf_sync_layers: str = "",
    native_cache_attach_hf_views: bool = False,
    native_attention_module: bool = False,
    fused_rope_append_seed: bool = False,
    packed_qkv_projection: bool = False,
    packed_qkv_fused_input: bool = False,
    direct_o_proj: bool = False,
    triton_o_proj: bool = False,
    dynamic_selector_layers: str = "",
    dynamic_selector_profile: str = "",
    allow_mixed_seed_configs: bool = False,
    output_json: str = "",
):
    result = profile_h100.remote(
        model=model,
        layers=layers,
        policy_names=policy_names,
        use_packaged_policies=use_packaged_policies,
        bucket_route_policy=bucket_route_policy,
        product_strict=product_strict,
        prompt_kinds=prompt_kinds,
        prompt_file=prompt_file,
        prompt_file_kinds=prompt_file_kinds,
        prompt_file_rows_per_kind=prompt_file_rows_per_kind,
        max_prompts=max_prompts,
        prompt_truncation_side=prompt_truncation_side,
        prompt_repeat=prompt_repeat,
        batch_size=batch_size,
        max_seq=max_seq,
        steps=steps,
        warmup_steps=warmup_steps,
        dtype=dtype,
        attn_implementation=attn_implementation,
        top_k=top_k,
        max_kl=max_kl,
        min_topk_overlap=min_topk_overlap,
        max_logprob_delta=max_logprob_delta,
        sample_temperature=sample_temperature,
        sample_top_p=sample_top_p,
        sample_top_k=sample_top_k,
        sample_seed=sample_seed,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        layer_seed_overrides=layer_seed_overrides,
        block_order=block_order,
        num_warps=num_warps,
        num_stages=num_stages,
        profile_patch_timing=profile_patch_timing,
        explicit_cache_position=explicit_cache_position,
        native_routed_cache=native_routed_cache,
        native_cache_hf_sync_layers=native_cache_hf_sync_layers,
        native_cache_attach_hf_views=native_cache_attach_hf_views,
        native_attention_module=native_attention_module,
        fused_rope_append_seed=fused_rope_append_seed,
        packed_qkv_projection=packed_qkv_projection,
        packed_qkv_fused_input=packed_qkv_fused_input,
        direct_o_proj=direct_o_proj,
        triton_o_proj=triton_o_proj,
        dynamic_selector_layers=dynamic_selector_layers,
        dynamic_selector_profile=dynamic_selector_profile,
        allow_mixed_seed_configs=allow_mixed_seed_configs,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        summary = {
            "schema": result.get("schema"),
            "model": result.get("model"),
            "shape": result.get("shape"),
            "route_bundle": result.get("route_bundle"),
            "timing": result.get("timing"),
            "safety": result.get("safety"),
            "decision": result.get("decision"),
            "patch_counts": result.get("patch_counts"),
            "patch_timing": result.get("patch_timing"),
            "prompts": result.get("prompts"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)
