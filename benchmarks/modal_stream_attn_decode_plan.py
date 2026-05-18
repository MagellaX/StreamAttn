"""Modal runner for planned StreamAttn decode profiling."""

import json
import os
import subprocess
from pathlib import Path

import modal


app = modal.App("streamattn-decode-plan")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("triton==3.1.0")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _profile(
    *,
    decode_cost_payload: str,
    query_lens: str,
    kv_lens: str,
    heads: str,
    kv_heads: str,
    dim: int,
    dtype: str,
    attention_type: str,
    pattern: str,
    active_fraction: str,
    plan_modes: str,
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
    plan_warmup: int,
    plan_iters: int,
    metadata_warmup: int,
    metadata_iters: int,
    metadata_update_backend: str,
    decode_steps: int,
    step_warmup: int,
    step_iters: int,
    safety_margin: float,
    max_router_regret_pct: float,
    prefer_value_bound_if_within: float,
    collect_telemetry_every: int,
    min_kv_len_for_gate1: int,
    max_active_fraction_mass: float,
    max_active_fraction_value_bound: float,
    min_confidence: float,
    disable_mass: bool,
    disable_value_bound: bool,
    allow_value_bound_without_metadata: bool,
):
    cost_path = "/tmp/streamattn_decode_cost.json"
    Path(cost_path).write_text(decode_cost_payload, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/StreamAttn" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "python",
        "/root/StreamAttn/benchmarks/profile_stream_attn_decode_plan.py",
        "--decode-cost-json",
        cost_path,
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
        "--plan-modes",
        plan_modes,
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
        "--plan-warmup",
        str(plan_warmup),
        "--plan-iters",
        str(plan_iters),
        "--metadata-warmup",
        str(metadata_warmup),
        "--metadata-iters",
        str(metadata_iters),
        "--metadata-update-backend",
        metadata_update_backend,
        "--decode-steps",
        str(decode_steps),
        "--step-warmup",
        str(step_warmup),
        "--step-iters",
        str(step_iters),
        "--safety-margin",
        str(safety_margin),
        "--max-router-regret-pct",
        str(max_router_regret_pct),
        "--prefer-value-bound-if-within",
        str(prefer_value_bound_if_within),
        "--collect-telemetry-every",
        str(collect_telemetry_every),
        "--min-kv-len-for-gate1",
        str(min_kv_len_for_gate1),
        "--max-active-fraction-mass",
        str(max_active_fraction_mass),
        "--max-active-fraction-value-bound",
        str(max_active_fraction_value_bound),
        "--min-confidence",
        str(min_confidence),
    ]
    if disable_mass:
        cmd.append("--disable-mass")
    if disable_value_bound:
        cmd.append("--disable-value-bound")
    if allow_value_bound_without_metadata:
        cmd.append("--allow-value-bound-without-metadata")
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
    decode_cost_json: str,
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
    plan_modes: str = "cold_plan,hint_plan,prev_token_plan",
    block_size: int = 128,
    tile_size_q: int = 16,
    num_warps: int = 4,
    num_stages: int = 3,
    peak: float = 8.0,
    sink_blocks: int = 2,
    recent_blocks: int = 2,
    error_budget: float = 1e-3,
    warmup: int = 5,
    iters: int = 10,
    plan_warmup: int = 10,
    plan_iters: int = 100,
    metadata_warmup: int = 3,
    metadata_iters: int = 8,
    metadata_update_backend: str = "auto",
    decode_steps: int = 0,
    step_warmup: int = 1,
    step_iters: int = 3,
    safety_margin: float = 1.10,
    max_router_regret_pct: float = 0.05,
    prefer_value_bound_if_within: float = 1.10,
    collect_telemetry_every: int = 0,
    min_kv_len_for_gate1: int = 4096,
    max_active_fraction_mass: float = 0.35,
    max_active_fraction_value_bound: float = 0.30,
    min_confidence: float = 0.70,
    disable_mass: bool = False,
    disable_value_bound: bool = False,
    allow_value_bound_without_metadata: bool = False,
    output_json: str = "",
):
    decode_cost_payload = Path(decode_cost_json).read_text(encoding="utf-8")
    kwargs = {
        "decode_cost_payload": decode_cost_payload,
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "heads": heads,
        "kv_heads": kv_heads,
        "dim": dim,
        "dtype": dtype,
        "attention_type": attention_type,
        "pattern": pattern,
        "active_fraction": active_fraction,
        "plan_modes": plan_modes,
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
        "plan_warmup": plan_warmup,
        "plan_iters": plan_iters,
        "metadata_warmup": metadata_warmup,
        "metadata_iters": metadata_iters,
        "metadata_update_backend": metadata_update_backend,
        "decode_steps": decode_steps,
        "step_warmup": step_warmup,
        "step_iters": step_iters,
        "safety_margin": safety_margin,
        "max_router_regret_pct": max_router_regret_pct,
        "prefer_value_bound_if_within": prefer_value_bound_if_within,
        "collect_telemetry_every": collect_telemetry_every,
        "min_kv_len_for_gate1": min_kv_len_for_gate1,
        "max_active_fraction_mass": max_active_fraction_mass,
        "max_active_fraction_value_bound": max_active_fraction_value_bound,
        "min_confidence": min_confidence,
        "disable_mass": disable_mass,
        "disable_value_bound": disable_value_bound,
        "allow_value_bound_without_metadata": allow_value_bound_without_metadata,
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
