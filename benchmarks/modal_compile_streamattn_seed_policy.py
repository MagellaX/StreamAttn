"""Modal pipeline for seed-only policy discovery and compilation.

This orchestrates the full product path for a model bucket:

1. run the layer sweep,
2. run closed-loop rollout only for sweep-passing candidates,
3. compile green policy metadata from the produced artifacts.

The local entrypoint writes all returned artifacts into the requested output
directory.  Use this for model expansion before manually adding any policy cell.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("streamattn-seed-policy-compiler")

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


REMOTE_ROOT = Path("/root/StreamAttn")
REMOTE_ARTIFACT_DIR = REMOTE_ROOT / "artifacts" / "gate0"


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
    raise RuntimeError(f"could not parse JSON from command output:\n{output[-8000:]}")


def _json_from_cmd(
    cmd: list[str],
    *,
    env: dict[str, str],
    label: str,
) -> dict[str, Any]:
    print(f"[seed-policy-pipeline] {label}: {' '.join(cmd[:7])} ...", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(REMOTE_ROOT),
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
            f"{returncode}: {' '.join(cmd)}\n{output[-10000:]}"
        )
    return _json_from_output(output)


def _parse_layers(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = [int(part.strip()) for part in item.split("-", 1)]
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(item))
    if not values:
        raise ValueError(f"empty layer list: {raw!r}")
    return sorted(dict.fromkeys(values))


def _model_slug(model: str) -> str:
    tail = model.split("/")[-1].lower()
    tail = tail.replace("qwen2.5", "qwen25")
    tail = tail.replace("-instruct", "")
    tail = tail.replace(".", "_").replace("-", "_")
    tail = re.sub(r"(?<=_)0_5b$", "05b", tail)
    tail = re.sub(r"(?<=_)1_5b$", "15b", tail)
    return re.sub(r"[^a-z0-9_]+", "_", tail).strip("_")


def _kv_label(kv_len: int) -> str:
    return f"{kv_len // 1024}k" if kv_len % 1024 == 0 else str(kv_len)


def _artifact_paths(*, model_slug: str, kv_len: int, batch: int) -> dict[str, Path]:
    prefix = f"seed_only_{model_slug}_{_kv_label(kv_len)}_b{batch}"
    return {
        "sweep": REMOTE_ARTIFACT_DIR / f"{prefix}_layer_sweep_h100.json",
        "compiled": REMOTE_ARTIFACT_DIR / f"{prefix}_compiled_policy_h100.json",
    }


def _closed_loop_artifact_path(*, model_slug: str, layer_id: int, batch: int) -> Path:
    return REMOTE_ARTIFACT_DIR / f"seed_only_{model_slug}_l{layer_id}_b{batch}_closed_loop_h100.json"


def _write_remote_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_remote_path(path: Path) -> str:
    return path.relative_to(REMOTE_ROOT).as_posix()


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    policy = result.get("policy") or {}
    summary = policy.get("summary_vs_model_baseline") or {}
    shape = result.get("shape") or {}
    model = result.get("model") or {}
    return {
        "layer_id": model.get("layer_id"),
        "batch": shape.get("batch"),
        "kv_len": shape.get("kv_len"),
        "passes_distribution_gate": bool(policy.get("passes_distribution_gate")),
        "fallback_recommendation": policy.get("fallback_recommendation"),
        "top1_changed_count": summary.get("top1_changed_count"),
        "topk_overlap_min": summary.get("topk_overlap_min"),
        "kl_max": summary.get("kl_max"),
        "max_logit_delta": summary.get("max_logit_delta"),
        "target_next_token_logprob_delta_max_abs": summary.get(
            "target_next_token_logprob_delta_max_abs"
        ),
        "reference_top1_logprob_delta_max_abs": summary.get(
            "reference_top1_logprob_delta_max_abs"
        ),
        "worst_case_by_kl": summary.get("worst_case_by_kl"),
    }


def _run_layer_sweep(kwargs: dict[str, Any], env: dict[str, str], sweep_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for layer_id in _parse_layers(kwargs["layers"]):
        cmd = [
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
            str(kwargs["max_prompts"]),
            "--max-seq",
            str(kwargs["max_seq"]),
            "--kv-len",
            str(kwargs["kv_len"]),
            "--layer-id",
            str(layer_id),
            "--device",
            "cuda",
            "--dtype",
            kwargs["dtype"],
            "--position-count",
            str(kwargs["position_count"]),
            "--position-stride",
            str(kwargs["position_stride"]),
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
            "--top-k",
            str(kwargs["top_k"]),
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
        ]
        if not kwargs["require_top1_match"]:
            cmd.append("--no-require-top1-match")
        result = _json_from_cmd(cmd, env=env, label=f"L{layer_id} sweep")
        results[str(layer_id)] = result
        rows.append(_summary_row(result))
    passing_layers = [
        int(row["layer_id"])
        for row in rows
        if row.get("passes_distribution_gate")
    ]
    payload = {
        "schema": "streamattn.gate0.seed_only_layer_sweep.v1",
        "sweep": {
            "model": kwargs["model"],
            "layers": _parse_layers(kwargs["layers"]),
            "prompt_kinds": kwargs["prompt_kinds"],
            "prompt_repeat": kwargs["prompt_repeat"],
            "max_prompts": kwargs["max_prompts"],
            "dtype": kwargs["dtype"],
            "max_seq": kwargs["max_seq"],
            "kv_len": kwargs["kv_len"],
            "position_count": kwargs["position_count"],
            "position_stride": kwargs["position_stride"],
        },
        "seed_config": {
            "block_size": kwargs["block_size"],
            "sink_blocks": kwargs["sink_blocks"],
            "recent_blocks": kwargs["recent_blocks"],
            "middle_seed_blocks": kwargs["middle_seed_blocks"],
            "block_order": kwargs["block_order"],
            "num_warps": kwargs["num_warps"],
            "num_stages": kwargs["num_stages"],
        },
        "safety_gate": {
            "require_top1_match": kwargs["require_top1_match"],
            "min_topk_overlap": kwargs["min_topk_overlap"],
            "max_kl": kwargs["max_kl"],
            "max_logit_delta": kwargs["max_logit_delta"],
            "max_top1_logprob_delta": kwargs["max_top1_logprob_delta"],
            "max_target_logprob_delta": kwargs["max_target_logprob_delta"],
        },
        "passing_layers": passing_layers,
        "rows": rows,
        "results_by_layer": results,
    }
    _write_remote_json(sweep_path, payload)
    return payload


def _read_remote_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_closed_loop(
    kwargs: dict[str, Any],
    env: dict[str, str],
    *,
    layer_id: int,
    output_path: Path,
) -> dict[str, Any]:
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/profile_gate0_seed_only_closed_loop_rollout.py",
        "--model",
        kwargs["model"],
        "--prompt-kinds",
        kwargs["closed_loop_prompt_kinds"] or kwargs["prompt_kinds"],
        "--prompt-repeat",
        str(kwargs["prompt_repeat"]),
        "--batch-size",
        str(kwargs["batch_size"]),
        "--model-batch-size",
        str(kwargs["model_batch_size"]),
        "--steps",
        str(kwargs["steps"]),
        "--mode",
        kwargs["closed_loop_mode"],
        "--layer-id",
        str(layer_id),
        "--max-seq",
        str(kwargs["max_seq"]),
        "--dtype",
        kwargs["dtype"],
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
        "--top-k",
        str(kwargs["top_k"]),
        "--max-kl",
        str(kwargs["max_kl"]),
        "--min-topk-overlap",
        str(kwargs["min_topk_overlap"]),
        "--sample-temperature",
        str(kwargs["sample_temperature"]),
        "--sample-top-p",
        str(kwargs["sample_top_p"]),
        "--sample-top-k",
        str(kwargs["sample_top_k"]),
        "--sample-seed",
        str(kwargs["sample_seed"]),
        "--output-json",
        str(output_path),
    ]
    payload = _json_from_cmd(cmd, env=env, label=f"L{layer_id} closed loop")
    if not output_path.exists():
        _write_remote_json(output_path, payload)
    return _read_remote_json(output_path)


def _run_compiler(
    kwargs: dict[str, Any],
    env: dict[str, str],
    *,
    sweep_path: Path,
    compiled_path: Path,
    model_slug: str,
) -> dict[str, Any]:
    cmd = [
        "python",
        "-u",
        "/root/StreamAttn/benchmarks/compile_streamattn_seed_policy.py",
        "--sweep-json",
        _relative_remote_path(sweep_path),
        "--closed-loop-dir",
        _relative_remote_path(REMOTE_ARTIFACT_DIR),
        "--policy-dir",
        "stream_attention/policies",
        "--registry-json",
        "stream_attention/policies/registry.json",
        "--model-slug",
        model_slug,
        "--model-id",
        kwargs["model"],
        "--dtype",
        kwargs["dtype"],
        "--min-batch",
        str(kwargs["batch_size"]),
        "--max-kl",
        str(kwargs["max_kl"]),
        "--min-top5-overlap",
        str(kwargs["min_topk_overlap"]),
        "--summary-json",
        _relative_remote_path(compiled_path),
    ]
    if not kwargs["require_top1_match"]:
        cmd.append("--allow-top1-changes")
    payload = _json_from_cmd(cmd, env=env, label="policy compiler")
    if not compiled_path.exists():
        _write_remote_json(compiled_path, payload)
    return _read_remote_json(compiled_path)


def _summarize_closed_loop(layer_id: int, rollout: dict[str, Any]) -> dict[str, Any]:
    greedy = (rollout.get("greedy") or {}).get("summary") or {}
    sampling = (rollout.get("sampling") or {}).get("summary") or {}
    teacher = (rollout.get("teacher_forced") or {}).get("summary") or {}
    return {
        "layer_id": layer_id,
        "teacher_forced_kl_max": teacher.get("kl_max"),
        "greedy_kl_max": greedy.get("kl_max"),
        "greedy_diverged_rows": greedy.get("diverged_row_count"),
        "greedy_sequence_exact_match_rate": greedy.get("sequence_exact_match_rate"),
        "sampling_kl_max": sampling.get("kl_max"),
        "sample_token_changed_count": sampling.get("sample_token_changed_count"),
        "sample_sequence_exact_match_rate": sampling.get("sample_sequence_exact_match_rate"),
        "top5_overlap_min": min(
            int(teacher.get("topk_overlap_min", 999)),
            int(greedy.get("topk_overlap_min", 999)),
            int(sampling.get("topk_overlap_min", 999)),
        ),
    }


def _run(**kwargs) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REMOTE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    slug = kwargs["model_slug"] or _model_slug(kwargs["model"])
    paths = _artifact_paths(
        model_slug=slug,
        kv_len=int(kwargs["kv_len"]),
        batch=int(kwargs["batch_size"]),
    )
    REMOTE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    sweep = _run_layer_sweep(kwargs, env, paths["sweep"])
    candidate_layers = [int(layer) for layer in sweep.get("passing_layers") or []]
    if kwargs["max_candidate_layers"] > 0:
        candidate_layers = candidate_layers[: int(kwargs["max_candidate_layers"])]

    closed_loop_payloads: dict[str, dict[str, Any]] = {}
    closed_loop_artifacts: dict[str, str] = {}
    closed_loop_rows: list[dict[str, Any]] = []
    for layer_id in candidate_layers:
        rollout_path = _closed_loop_artifact_path(
            model_slug=slug,
            layer_id=layer_id,
            batch=int(kwargs["batch_size"]),
        )
        rollout = _run_closed_loop(kwargs, env, layer_id=layer_id, output_path=rollout_path)
        closed_loop_payloads[str(layer_id)] = rollout
        closed_loop_artifacts[str(layer_id)] = _relative_remote_path(rollout_path)
        closed_loop_rows.append(_summarize_closed_loop(layer_id, rollout))

    compiled = _run_compiler(
        kwargs,
        env,
        sweep_path=paths["sweep"],
        compiled_path=paths["compiled"],
        model_slug=slug,
    )
    return {
        "schema": "streamattn.seed_policy_modal_pipeline.v1",
        "model": kwargs["model"],
        "model_slug": slug,
        "layers": _parse_layers(kwargs["layers"]),
        "candidate_layers": candidate_layers,
        "green_layers": compiled.get("green_layers", []),
        "artifacts": {
            "sweep": _relative_remote_path(paths["sweep"]),
            "compiled": _relative_remote_path(paths["compiled"]),
            "closed_loop": closed_loop_artifacts,
        },
        "sweep": sweep,
        "closed_loop_rows": closed_loop_rows,
        "closed_loop_by_layer": closed_loop_payloads,
        "compiled": compiled,
    }


@app.function(image=image, gpu="H100", timeout=21600)
def profile_h100(**kwargs):
    return _run(**kwargs)


def _write_local_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = result.get("artifacts") or {}
    sweep_name = Path(str(artifacts.get("sweep", "sweep.json"))).name
    compiled_name = Path(str(artifacts.get("compiled", "compiled.json"))).name
    (output_dir / sweep_name).write_text(
        json.dumps(result["sweep"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / compiled_name).write_text(
        json.dumps(result["compiled"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for layer_id, payload in sorted(
        (result.get("closed_loop_by_layer") or {}).items(),
        key=lambda item: int(item[0]),
    ):
        remote_name = Path(str(artifacts.get("closed_loop", {}).get(layer_id, ""))).name
        name = remote_name or f"seed_only_l{layer_id}_closed_loop_h100.json"
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "schema": result.get("schema"),
        "model": result.get("model"),
        "model_slug": result.get("model_slug"),
        "layers": result.get("layers"),
        "candidate_layers": result.get("candidate_layers"),
        "green_layers": result.get("green_layers"),
        "artifacts": result.get("artifacts"),
        "closed_loop_rows": result.get("closed_loop_rows"),
        "compiled_policies": (result.get("compiled") or {}).get("policies"),
        "rejected_layers": (result.get("compiled") or {}).get("rejected_layers"),
    }
    (output_dir / "seed_policy_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    model_slug: str = "",
    layers: str = "0-23",
    prompt_kinds: str = "needle,code,long_doc,chat_doc",
    closed_loop_prompt_kinds: str = "",
    prompt_repeat: int = 3000,
    max_prompts: int = 4,
    max_seq: int = 32768,
    kv_len: int = 32768,
    dtype: str = "fp16",
    batch_size: int = 4,
    model_batch_size: int = 1,
    steps: int = 32,
    closed_loop_mode: str = "all",
    position_count: int = 32,
    position_stride: int = 1,
    block_size: int = 32,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    block_order: str = "recent_first",
    num_warps: int = 4,
    num_stages: int = 2,
    top_k: int = 5,
    require_top1_match: bool = True,
    min_topk_overlap: int = 4,
    max_kl: float = 1.0e-4,
    max_logit_delta: float = 0.0,
    max_top1_logprob_delta: float = 0.10,
    max_target_logprob_delta: float = 0.10,
    sample_temperature: float = 0.8,
    sample_top_p: float = 0.95,
    sample_top_k: int = 0,
    sample_seed: int = 1234,
    max_candidate_layers: int = 0,
    output_dir: str = "artifacts/gate0",
):
    result = profile_h100.remote(
        model=model,
        model_slug=model_slug,
        layers=layers,
        prompt_kinds=prompt_kinds,
        closed_loop_prompt_kinds=closed_loop_prompt_kinds,
        prompt_repeat=prompt_repeat,
        max_prompts=max_prompts,
        max_seq=max_seq,
        kv_len=kv_len,
        dtype=dtype,
        batch_size=batch_size,
        model_batch_size=model_batch_size,
        steps=steps,
        closed_loop_mode=closed_loop_mode,
        position_count=position_count,
        position_stride=position_stride,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        num_warps=num_warps,
        num_stages=num_stages,
        top_k=top_k,
        require_top1_match=require_top1_match,
        min_topk_overlap=min_topk_overlap,
        max_kl=max_kl,
        max_logit_delta=max_logit_delta,
        max_top1_logprob_delta=max_top1_logprob_delta,
        max_target_logprob_delta=max_target_logprob_delta,
        sample_temperature=sample_temperature,
        sample_top_p=sample_top_p,
        sample_top_k=sample_top_k,
        sample_seed=sample_seed,
        max_candidate_layers=max_candidate_layers,
    )
    _write_local_artifacts(result, Path(output_dir))
    summary = {
        "schema": result.get("schema"),
        "model": result.get("model"),
        "model_slug": result.get("model_slug"),
        "candidate_layers": result.get("candidate_layers"),
        "green_layers": result.get("green_layers"),
        "artifacts": result.get("artifacts"),
        "closed_loop_rows": result.get("closed_loop_rows"),
        "compiled_policies": (result.get("compiled") or {}).get("policies"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
