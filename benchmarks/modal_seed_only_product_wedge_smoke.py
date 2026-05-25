"""Modal H100 product-gate smoke for the first StreamAttn seed-only wedge.

This gate validates the deployable route, not a research microbenchmark:

* Qwen2.5-0.5B L8, 32K, true GQA, fp16, batch=8
* packaged all-seed-only policy is selected by the wrapper
* wrapper beats FlashInfer by a configurable minimum speedup
* mixed-prompt logit replay passes the distribution-aware safety gate
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import modal


app = modal.App("streamattn-seed-only-product-wedge-smoke")

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


def _prompt_for_kind(kind: str) -> str:
    if kind == "code":
        return (
            "def stream_attention_decode(q, k_cache, v_cache, policy):\n"
            "    kv_head = q_head // group_size\n"
            "    if policy.seed_only_group:\n"
            "        schedule_seed_blocks(sink, recent, middle_seed)\n"
            "    return online_softmax_merge(partial_states)\n"
        )
    if kind == "long_doc":
        return (
            "StreamAttn long context technical memorandum. "
            "The system stores cached key and value tensors, maintains online softmax state, "
            "routes true grouped-query attention heads, verifies approximation error, and "
            "falls back to exact decode when calibration is stale. "
        )
    if kind == "chat_doc":
        return (
            "User: Summarize the implementation status.\n"
            "Assistant: StreamAttn has a seed-only batched route, wrapper telemetry, "
            "distribution-aware safety checks, and dense fallback for unsupported requests.\n"
        )
    return (
        "Needle retrieval context with cached KV metadata, online softmax, middle blocks, "
        "sink tokens, recent tokens, sparse decode routing, exact repair, and long-context retrieval. "
    )


def _json_from_output(output: str) -> Dict[str, Any]:
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


def _json_from_cmd(cmd: List[str], *, env: Dict[str, str], tail: int = 5000) -> Dict[str, Any]:
    print(f"[modal-product-wedge-smoke] running: {' '.join(cmd[:5])} ...", flush=True)
    process = subprocess.run(
        cmd,
        cwd="/root/StreamAttn",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = process.stdout
    if output.strip():
        print(output[-tail:], flush=True)
    if process.returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{process.returncode}: {' '.join(cmd)}\n{output[-6000:]}"
        )
    return _json_from_output(output)


def _prompt_file(prompt_kinds: List[str], *, repeat: int, path: str) -> None:
    prompts = [
        (_prompt_for_kind(kind).strip() + " ") * max(1, int(repeat))
        for kind in prompt_kinds
    ]
    Path(path).write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )


def _wrapper_gate(kwargs: Dict[str, Any], *, env: Dict[str, str], prompt_kinds: List[str]) -> Dict[str, Any]:
    capture_dir = "/tmp/streamattn_product_wedge_qkv"
    capture_json = f"{capture_dir}/metadata.json"
    prompt_file = "/tmp/streamattn_product_wedge_prompts.jsonl"
    _prompt_file(prompt_kinds, repeat=kwargs["prompt_repeat"], path=prompt_file)
    _json_from_cmd(
        [
            "python",
            "/root/StreamAttn/benchmarks/capture_real_qk_decode.py",
            "--model",
            kwargs["model"],
            "--prompt-file",
            prompt_file,
            "--max-prompts",
            str(len(prompt_kinds)),
            "--layers",
            str(kwargs["layer_id"]),
            "--device",
            "cuda",
            "--dtype",
            kwargs["dtype"],
            "--max-seq",
            str(max(int(kwargs["max_seq"]), int(kwargs["kv_len"]))),
            "--kv-len",
            str(kwargs["kv_len"]),
            "--query-len",
            "1",
            "--tensor-space",
            "post_rope",
            "--save-v",
            "--output-dir",
            capture_dir,
            "--metadata-json-out",
            capture_json,
        ],
        env=env,
        tail=2500,
    )
    return _json_from_cmd(
        [
            "python",
            "/root/StreamAttn/benchmarks/profile_seed_only_wrapper_batch_threshold.py",
            "--metadata-json",
            capture_json,
            "--policy-json",
            "/root/StreamAttn/stream_attention/policies/qwen25_05b_l8_32k_seed_only_batched.json",
            "--layer-id",
            str(kwargs["layer_id"]),
            "--batch-sizes",
            f"{kwargs['fallback_batch_size']},{kwargs['batch_size']}",
            "--product-min-batch",
            str(kwargs["batch_size"]),
            "--forced-min-batch",
            "1",
            "--dtype",
            kwargs["dtype"],
            "--safety-margin",
            str(kwargs["safety_margin"]),
            "--flashinfer-backend",
            kwargs["flashinfer_backend"],
            "--page-size",
            str(kwargs["page_size"]),
            "--workspace-mb",
            str(kwargs["workspace_mb"]),
            "--warmup",
            str(kwargs["warmup"]),
            "--iters",
            str(kwargs["iters"]),
            "--flashinfer-tensor-cores",
        ],
        env=env,
        tail=5000,
    )


def _safety_gate(kwargs: Dict[str, Any], *, env: Dict[str, str]) -> Dict[str, Any]:
    return _json_from_cmd(
        [
            "python",
            "-u",
            "/root/StreamAttn/benchmarks/profile_gate0_seed_only_batched_logit_safety.py",
            "--model",
            kwargs["model"],
            "--prompt-kinds",
            kwargs["prompt_kinds"],
            "--prompt-repeat",
            str(kwargs["prompt_repeat"]),
            "--max-prompts",
            str(kwargs["batch_size"]),
            "--layer-id",
            str(kwargs["layer_id"]),
            "--device",
            "cuda",
            "--dtype",
            kwargs["dtype"],
            "--max-seq",
            str(kwargs["max_seq"]),
            "--kv-len",
            str(kwargs["kv_len"]),
            "--position-count",
            str(kwargs["position_count"]),
            "--position-stride",
            str(kwargs["position_stride"]),
            "--block-size",
            "32",
            "--sink-blocks",
            "2",
            "--recent-blocks",
            "2",
            "--middle-seed-blocks",
            "8",
            "--block-order",
            "recent_first",
            "--num-warps",
            "4",
            "--num-stages",
            "2",
            "--top-k",
            "5",
            "--min-topk-overlap",
            str(kwargs["min_topk_overlap"]),
            "--max-kl",
            str(kwargs["max_kl"]),
            "--max-logit-delta",
            str(kwargs["max_logit_delta"]),
            "--max-top1-logprob-delta",
            str(kwargs["max_top1_logprob_delta"]),
            "--max-target-logprob-delta",
            str(kwargs["max_target_logprob_delta"]),
        ],
        env=env,
        tail=5000,
    )


def _summarize(wrapper: Dict[str, Any], safety: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    entries = wrapper.get("entries") or []
    if not entries:
        raise RuntimeError("wrapper benchmark returned no entries")
    entry = next(
        (row for row in entries if int(row.get("batch", -1)) == int(kwargs["batch_size"])),
        None,
    )
    if entry is None:
        raise RuntimeError(f"wrapper benchmark returned no batch={kwargs['batch_size']} entry")
    fallback_entry = next(
        (
            row
            for row in entries
            if int(row.get("batch", -1)) == int(kwargs["fallback_batch_size"])
        ),
        None,
    )
    if fallback_entry is None:
        raise RuntimeError(
            f"wrapper benchmark returned no fallback batch={kwargs['fallback_batch_size']} entry"
        )
    timing = entry.get("timing") or {}
    decision = entry.get("decision") or {}
    product_route = entry.get("product_route") or {}
    product_runtime = entry.get("product_wrapper_route") or {}
    fallback_route = fallback_entry.get("product_route") or {}
    fallback_runtime = fallback_entry.get("product_wrapper_route") or {}
    safety_policy = safety.get("policy") or {}
    safety_summary = safety_policy.get("summary_vs_model_baseline") or {}

    route_ok = (
        product_route.get("backend") == "gate0_seed_only_batched"
        and product_runtime.get("backend_used") == "gate0_seed_only_batched"
    )
    fallback_ok = (
        fallback_route.get("backend") == "dense"
        and fallback_runtime.get("backend_used") == "flashinfer_dense"
        and fallback_runtime.get("fallback_reason") == "batch_below_min"
    )
    speedup = float(timing.get("product_wrapper_speedup_vs_flashinfer") or 0.0)
    speed_ok = speedup >= float(kwargs["min_speedup"])
    safety_ok = (
        bool(safety_policy.get("passes_distribution_gate"))
        and int(safety_summary.get("top1_changed_count", 1)) == 0
        and int(safety_summary.get("topk_overlap_min", 0)) >= int(kwargs["min_topk_overlap"])
        and float(safety_summary.get("kl_max", 1.0)) <= float(kwargs["max_kl"])
    )
    gate = {
        "passed": bool(route_ok and fallback_ok and speed_ok and safety_ok),
        "route_ok": bool(route_ok),
        "fallback_ok": bool(fallback_ok),
        "speed_ok": bool(speed_ok),
        "safety_ok": bool(safety_ok),
        "min_speedup": float(kwargs["min_speedup"]),
    }
    return {
        "schema": "streamattn.gate0.seed_only_product_wedge_smoke.v1",
        "commit": kwargs.get("commit") or None,
        "model": kwargs["model"],
        "gate": gate,
        "shape": {
            "batch": entry.get("batch"),
            "kv_len": (entry.get("shape") or {}).get("kv_len"),
            "layer_id": kwargs["layer_id"],
            "dtype": kwargs["dtype"],
        },
        "fallback_smoke": {
            "batch": fallback_entry.get("batch"),
            "product_route": fallback_route,
            "product_wrapper_route": fallback_runtime,
            "passed": bool(fallback_ok),
        },
        "route": {
            "product_route": product_route,
            "product_wrapper_route": product_runtime,
            "decision": decision,
        },
        "timing": timing,
        "safety": {
            "passes_distribution_gate": safety_policy.get("passes_distribution_gate"),
            "fallback_recommendation": safety_policy.get("fallback_recommendation"),
            "summary_vs_model_baseline": safety_summary,
        },
        "raw": {
            "wrapper": wrapper,
            "safety": safety,
        },
    }


def _run(**kwargs) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    prompt_kinds = [item.strip() for item in kwargs["prompt_kinds"].split(",") if item.strip()]
    if len(prompt_kinds) < int(kwargs["batch_size"]):
        raise ValueError("prompt_kinds must provide at least batch_size rows")
    prompt_kinds = prompt_kinds[: int(kwargs["batch_size"])]
    wrapper = _wrapper_gate(kwargs, env=env, prompt_kinds=prompt_kinds)
    safety = _safety_gate({**kwargs, "prompt_kinds": ",".join(prompt_kinds)}, env=env)
    result = _summarize(wrapper, safety, kwargs)
    if kwargs["require_pass"] and not result["gate"]["passed"]:
        raise RuntimeError(json.dumps(result["gate"], indent=2, sort_keys=True))
    return result


@app.function(image=image, gpu="H100", timeout=7200)
def smoke_h100(**kwargs):
    return _run(**kwargs)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt_kinds: str = "needle,code,long_doc,chat_doc,needle,code,long_doc,chat_doc",
    prompt_repeat: int = 3000,
    layer_id: int = 8,
    max_seq: int = 32768,
    kv_len: int = 32768,
    batch_size: int = 8,
    fallback_batch_size: int = 4,
    dtype: str = "fp16",
    commit: str = "",
    position_count: int = 8,
    position_stride: int = 1,
    min_speedup: float = 1.10,
    min_topk_overlap: int = 4,
    max_kl: float = 1.0e-4,
    max_logit_delta: float = 0.2,
    max_top1_logprob_delta: float = 0.001,
    max_target_logprob_delta: float = 0.001,
    safety_margin: float = 1.10,
    flashinfer_backend: str = "auto",
    page_size: int = 32,
    workspace_mb: int = 256,
    warmup: int = 5,
    iters: int = 20,
    require_pass: bool = True,
    output_json: str = "",
):
    result = smoke_h100.remote(
        model=model,
        prompt_kinds=prompt_kinds,
        prompt_repeat=prompt_repeat,
        layer_id=layer_id,
        max_seq=max_seq,
        kv_len=kv_len,
        batch_size=batch_size,
        fallback_batch_size=fallback_batch_size,
        dtype=dtype,
        commit=commit,
        position_count=position_count,
        position_stride=position_stride,
        min_speedup=min_speedup,
        min_topk_overlap=min_topk_overlap,
        max_kl=max_kl,
        max_logit_delta=max_logit_delta,
        max_top1_logprob_delta=max_top1_logprob_delta,
        max_target_logprob_delta=max_target_logprob_delta,
        safety_margin=safety_margin,
        flashinfer_backend=flashinfer_backend,
        page_size=page_size,
        workspace_mb=workspace_mb,
        warmup=warmup,
        iters=iters,
        require_pass=require_pass,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        compact = {
            key: result.get(key)
            for key in (
                "schema",
                "commit",
                "model",
                "gate",
                "shape",
                "fallback_smoke",
                "route",
                "timing",
                "safety",
            )
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
    else:
        print(text)


if __name__ == "__main__":
    main()
