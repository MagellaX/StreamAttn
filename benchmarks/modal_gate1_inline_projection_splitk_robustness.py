"""Modal robustness runner for workspace split-K inline projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from benchmarks.modal_gate1_inline_projection_splitk import _run as _run_splitk
from benchmarks.modal_gate1_inline_projection_splitk import image
from benchmarks.summarize_gate1_inline_projection_splitk_robustness import (
    DEFAULT_ROBUSTNESS_BUDGETS,
    collect_rows_from_payload,
    parse_budgets,
    summarize_rows,
)


app = modal.App("streamattn-gate1-inline-projection-splitk-robustness")


DEFAULT_PROMPTS = {
    "needle": (
        "Needle retrieval context with hidden key BLUE LANTERN 729 surrounded by repeated "
        "distractors about cached KV metadata online softmax block summaries post RoPE "
        "tensors middle blocks sink tokens recent tokens sparse decode routing and retrieval "
        "over long contexts. "
    ),
    "code": (
        "function run_decode(q, k, v, projection, state) { for block in ordered_blocks "
        "projection_filter(block); if (skip) continue; qk_scan(block); softmax_update(state); "
        "pv_accumulate(v); } metadata cache splitk merge workspace calibrated sparse heads. "
    ),
    "long_doc": (
        "This document describes a long-context serving system. The engine stores cached key "
        "and value tensors, maintains per-head telemetry, routes decode requests through dense "
        "fallbacks when confidence is low, and uses reusable workspaces to avoid allocation "
        "overhead during split-K attention. "
    ),
}


def _parse_prompt_types(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or ["needle"]


def _prompt_for(prompt_type: str, prompt_overrides: dict[str, str]) -> str:
    if prompt_type in prompt_overrides:
        return prompt_overrides[prompt_type]
    if prompt_type in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[prompt_type]
    return prompt_type + " "


def _run_robustness(
    *,
    model: str,
    prompt_types: str,
    prompt_overrides: dict[str, str],
    prompt_repeat: int,
    layers: str,
    head_groups: str,
    max_seq: int,
    kv_lens: str,
    dtype: str,
    tensor_space: str,
    block_size: int,
    tile_size_q: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    chunk_anchor_blocks: int,
    chunk_anchor_blocks_values: str,
    block_order: str,
    num_chunks: str,
    seed_strategy: str,
    projection_dim: int,
    projection_dims: str,
    projection_seeds: str,
    projection_metadata_dtype: str,
    qproj_mode: str,
    splitk_breakdown: bool,
    splitk_workspace: str,
    filter_margin: float,
    filter_margins: str,
    error_budget: float,
    warmup: int,
    iters: int,
    budgets: str,
    min_skip_fraction: float,
) -> dict[str, Any]:
    runs = []
    rows = []
    for prompt_type in _parse_prompt_types(prompt_types):
        prompt = _prompt_for(prompt_type, prompt_overrides) * max(1, prompt_repeat)
        payload = _run_splitk(
            model=model,
            prompt=prompt,
            prompt_type=prompt_type,
            layers=layers,
            head_index=-1,
            head_indices="",
            head_groups=head_groups,
            max_seq=max_seq,
            kv_lens=kv_lens,
            dtype=dtype,
            tensor_space=tensor_space,
            block_size=block_size,
            tile_size_q=tile_size_q,
            sink_blocks=sink_blocks,
            recent_blocks=recent_blocks,
            middle_seed_blocks=middle_seed_blocks,
            chunk_anchor_blocks=chunk_anchor_blocks,
            chunk_anchor_blocks_values=chunk_anchor_blocks_values,
            block_order=block_order,
            num_chunks=num_chunks,
            seed_strategy=seed_strategy,
            projection_dim=projection_dim,
            projection_dims=projection_dims,
            projection_seeds=projection_seeds,
            projection_metadata_dtype=projection_metadata_dtype,
            qproj_mode=qproj_mode,
            splitk_breakdown=splitk_breakdown,
            splitk_workspace=splitk_workspace,
            filter_margin=filter_margin,
            filter_margins=filter_margins,
            error_budget=error_budget,
            warmup=warmup,
            iters=iters,
        )
        runs.append(payload)
        rows.extend(collect_rows_from_payload(payload))

    summary = summarize_rows(
        rows,
        parse_budgets(budgets),
        min_skip_fraction=min_skip_fraction,
    )
    return {
        "sweep": {
            "model": model,
            "prompt_types": prompt_types,
            "layers": layers,
            "head_groups": head_groups,
            "kv_lens": kv_lens,
            "block_size": block_size,
            "middle_seed_blocks": middle_seed_blocks,
            "block_order": block_order,
            "num_chunks": num_chunks,
            "filter_margins": filter_margins or str(filter_margin),
            "splitk_workspace": splitk_workspace,
            "budgets": budgets,
            "min_skip_fraction": min_skip_fraction,
        },
        "summary": summary,
        "runs": runs,
    }


@app.function(image=image, gpu="A100", timeout=14400)
def profile_a100(**kwargs):
    return _run_robustness(**kwargs)


@app.function(image=image, gpu="H100", timeout=14400)
def profile_h100(**kwargs):
    return _run_robustness(**kwargs)


@app.local_entrypoint()
def main(
    target: str = "h100",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt_types: str = "needle,code,long_doc",
    prompt_file_json: str = "",
    prompt_repeat: int = 2048,
    layers: str = "4,8,12",
    head_groups: str = "2,3,4,5,6,7,8,9,10,11,12,13;-1",
    max_seq: int = 32768,
    kv_lens: str = "8192,16384,32768",
    dtype: str = "fp16",
    tensor_space: str = "post_rope",
    block_size: int = 32,
    tile_size_q: int = 16,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    middle_seed_blocks: int = 8,
    chunk_anchor_blocks: int = 0,
    chunk_anchor_blocks_values: str = "",
    block_order: str = "recent_first",
    num_chunks: str = "8,16",
    seed_strategy: str = "recompute_seed",
    projection_dim: int = 8,
    projection_dims: str = "",
    projection_seeds: str = "1",
    projection_metadata_dtype: str = "fp16",
    qproj_mode: str = "fused",
    splitk_breakdown: bool = False,
    splitk_workspace: str = "reuse",
    filter_margin: float = 64.0,
    filter_margins: str = "48,64,80",
    error_budget: float = 1e-2,
    warmup: int = 1,
    iters: int = 8,
    budgets: str = DEFAULT_ROBUSTNESS_BUDGETS,
    min_skip_fraction: float = 0.25,
    output_json: str = "",
):
    prompt_overrides = {}
    if prompt_file_json:
        prompt_overrides = json.loads(Path(prompt_file_json).read_text(encoding="utf-8"))
    kwargs = {
        "model": model,
        "prompt_types": prompt_types,
        "prompt_overrides": prompt_overrides,
        "prompt_repeat": prompt_repeat,
        "layers": layers,
        "head_groups": head_groups,
        "max_seq": max_seq,
        "kv_lens": kv_lens,
        "dtype": dtype,
        "tensor_space": tensor_space,
        "block_size": block_size,
        "tile_size_q": tile_size_q,
        "sink_blocks": sink_blocks,
        "recent_blocks": recent_blocks,
        "middle_seed_blocks": middle_seed_blocks,
        "chunk_anchor_blocks": chunk_anchor_blocks,
        "chunk_anchor_blocks_values": chunk_anchor_blocks_values,
        "block_order": block_order,
        "num_chunks": num_chunks,
        "seed_strategy": seed_strategy,
        "projection_dim": projection_dim,
        "projection_dims": projection_dims,
        "projection_seeds": projection_seeds,
        "projection_metadata_dtype": projection_metadata_dtype,
        "qproj_mode": qproj_mode,
        "splitk_breakdown": splitk_breakdown,
        "splitk_workspace": splitk_workspace,
        "filter_margin": filter_margin,
        "filter_margins": filter_margins,
        "error_budget": error_budget,
        "warmup": warmup,
        "iters": iters,
        "budgets": budgets,
        "min_skip_fraction": min_skip_fraction,
    }
    payload = profile_a100.remote(**kwargs) if target == "a100" else profile_h100.remote(**kwargs)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
