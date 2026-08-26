"""Test whether omitted context is compressible into shared synthetic K/V.

This is a capacity oracle, not a production compactor.  For each captured
layer/context it fits one residual bank per true KV head on early query rows,
then evaluates the same bank on held-out later rows.  The temporal holdout is
important: a separate one-token residual can reconstruct any single query
exactly, so fitting and evaluating the same query would provide no evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_real_llm_gate1_heads import (  # noqa: E402
    _capture_attention_inputs,
    _import_transformers,
    _shape_qkv,
)
from stream_attention.decode import load_packaged_gate0_seed_only_batched_policy  # noqa: E402
from stream_attention.residuals import (  # noqa: E402
    construct_query_exact_residual,
    merge_normalized_attention_states,
    merge_seed_with_residual,
)
from stream_attention.seed_selectors import seed_indices  # noqa: E402


QWEN3B_POLICY_BY_LAYER = {
    24: "qwen25_3b_l24_32k_seed_only_batched",
    26: "qwen25_3b_l26_32k_seed_only_batched",
    27: "qwen25_3b_l27_32k_seed_only_batched",
}


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.replace(";", ",").split(",") if item.strip()]


def _parse_residual_sizes(raw: str) -> list[int]:
    values = sorted(set(_parse_ints(raw)))
    if not values or values[0] <= 0:
        raise ValueError("residual sizes must contain positive integers")
    return values


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _load_prompt_rows(
    *,
    prompt_file: str,
    prompt: list[str] | None,
    buckets: set[str],
    max_prompts: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, text in enumerate(prompt or []):
        rows.append({"id": f"cli_{index}", "bucket": "cli", "text": text})
    if prompt_file:
        path = Path(prompt_file)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if path.suffix.lower() == ".jsonl":
                payload = json.loads(line)
                text = str(payload.get("text") or payload.get("prompt") or "")
                bucket = str(payload.get("bucket") or payload.get("kind") or "unknown")
                prompt_id = str(payload.get("id") or f"row_{line_no}")
            else:
                text = line
                bucket = "unknown"
                prompt_id = f"row_{line_no}"
            if not text or (buckets and bucket not in buckets):
                continue
            rows.append({"id": prompt_id, "bucket": bucket, "text": text})
            if len(rows) >= max_prompts:
                break
    if not rows:
        raise ValueError("no prompts matched the requested prompt source/buckets")
    return rows[:max_prompts]


def _temporal_split(query_rows: int, train_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    if query_rows < 2:
        raise ValueError("query_rows must be at least 2 for held-out evaluation")
    if train_rows <= 0 or train_rows >= query_rows:
        raise ValueError("train_rows must be in [1, query_rows - 1]")
    return torch.arange(train_rows), torch.arange(train_rows, query_rows)


def _error_metrics(
    candidate: torch.Tensor,
    target: torch.Tensor,
    *,
    indices: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> dict[str, float]:
    candidate = candidate.index_select(0, indices.to(candidate.device)).float()
    target = target.index_select(0, indices.to(target.device)).float()
    diff = candidate - target
    rel_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(target).clamp_min(1.0e-12)
    row_rel = torch.linalg.vector_norm(diff, dim=(-2, -1)) / torch.linalg.vector_norm(
        target, dim=(-2, -1)
    ).clamp_min(1.0e-12)
    candidate_projected = F.linear(candidate.flatten(1), o_proj_weight.float())
    target_projected = F.linear(target.flatten(1), o_proj_weight.float())
    projected_diff = candidate_projected - target_projected
    projected_rel_l2 = torch.linalg.vector_norm(projected_diff) / torch.linalg.vector_norm(
        target_projected
    ).clamp_min(1.0e-12)
    projected_row_rel = torch.linalg.vector_norm(projected_diff, dim=-1) / torch.linalg.vector_norm(
        target_projected, dim=-1
    ).clamp_min(1.0e-12)
    return {
        "attention_relative_l2": float(rel_l2.item()),
        "attention_row_relative_l2_p95": float(torch.quantile(row_rel, 0.95).item()),
        "attention_max_abs": float(diff.abs().max().item()),
        "post_o_proj_relative_l2": float(projected_rel_l2.item()),
        "post_o_proj_row_relative_l2_p95": float(torch.quantile(projected_row_rel, 0.95).item()),
        "post_o_proj_max_abs": float(projected_diff.abs().max().item()),
    }


def _attention_states(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    policy: Any,
) -> dict[str, torch.Tensor]:
    steps, q_heads, dim = q.shape
    kv_heads, seq_len, _ = k.shape
    if q_heads % kv_heads != 0:
        raise ValueError("Q heads must be divisible by KV heads")
    k_q = k.repeat_interleave(q_heads // kv_heads, dim=0)
    v_q = v.repeat_interleave(q_heads // kv_heads, dim=0)
    scores = torch.einsum("thd,hnd->thn", q.float(), k_q.float()) / math.sqrt(dim)
    query_positions = torch.arange(seq_len - steps, seq_len, device=q.device)
    token_positions = torch.arange(seq_len, device=q.device)
    causal_mask = token_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    seed_mask = torch.zeros(steps, seq_len, device=q.device, dtype=torch.bool)
    for step, position in enumerate(query_positions.tolist()):
        selected = seed_indices(
            seq_len=position + 1,
            block_size=int(policy.block_size),
            sink_blocks=int(policy.sink_blocks),
            recent_blocks=int(policy.recent_blocks),
            middle_seed_blocks=int(policy.middle_seed_blocks),
            block_order=str(policy.block_order),
        ).to(q.device)
        seed_mask[step, selected] = True
    full_scores = scores.masked_fill(~causal_mask[:, None, :], -torch.inf)
    seed_scores = scores.masked_fill(~seed_mask[:, None, :], -torch.inf)
    omitted_mask = causal_mask & ~seed_mask
    omitted_scores = scores.masked_fill(~omitted_mask[:, None, :], -torch.inf)

    full_probs = torch.softmax(full_scores, dim=-1)
    seed_probs = torch.softmax(seed_scores, dim=-1)
    omitted_probs = torch.softmax(omitted_scores, dim=-1)
    return {
        "full_output": torch.einsum("thn,hnd->thd", full_probs, v_q.float()),
        "seed_output": torch.einsum("thn,hnd->thd", seed_probs, v_q.float()),
        "seed_log_partition": torch.logsumexp(seed_scores, dim=-1),
        "omitted_output": torch.einsum("thn,hnd->thd", omitted_probs, v_q.float()),
        "omitted_log_partition": torch.logsumexp(omitted_scores, dim=-1),
        "seed_mask": seed_mask,
    }


def _initialize_residual_bank(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    residual_tokens: int,
    final_seed_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    omitted = torch.nonzero(~final_seed_mask, as_tuple=False).flatten()
    if omitted.numel() < residual_tokens:
        raise ValueError("residual token count exceeds omitted token count")
    slots = torch.linspace(0, omitted.numel() - 1, residual_tokens, device=k.device).round().long()
    selected = omitted.index_select(0, slots)
    return k.index_select(1, selected).float().clone(), v.index_select(1, selected).float().clone()


def _gaussian_orthogonal_projection(
    *,
    rows: int,
    dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if rows <= 0:
        raise ValueError("feature rows must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    blocks = []
    remaining = rows
    while remaining > 0:
        matrix = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
        q, _r = torch.linalg.qr(matrix, mode="reduced")
        take = min(remaining, dim)
        blocks.append(q[:take])
        remaining -= take
    projection = torch.cat(blocks, dim=0)
    multiplier = torch.randn(rows, dim, generator=generator).norm(dim=1)
    return (projection * multiplier.unsqueeze(-1)).to(device=device)


def _positive_softmax_features(
    data: torch.Tensor,
    *,
    projection: torch.Tensor,
    is_query: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FAVOR+-style positive features with explicit recoverable log scale."""

    dim = data.shape[-1]
    if projection.shape[1] != dim:
        raise ValueError("projection dim must match Q/K dim")
    data_normalizer = dim ** -0.25
    dash = torch.einsum("...nd,rd->...nr", data.float(), projection.float() * data_normalizer)
    diag = data.float().square().sum(dim=-1, keepdim=True) * (dim ** -0.5) / 2.0
    dash = torch.cat([dash, -dash], dim=-1)
    feature_count = dash.shape[-1]
    if is_query:
        max_dash = dash.amax(dim=-1, keepdim=True)
        features = torch.exp(dash - max_dash)
        log_scale = -diag + max_dash - 0.5 * math.log(feature_count)
    else:
        raw = dash - diag - 0.5 * math.log(feature_count)
        max_raw = raw.amax(dim=(-2, -1), keepdim=True)
        features = torch.exp(raw - max_raw)
        log_scale = max_raw
    return features, log_scale


def _linear_omitted_residual(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    states: dict[str, torch.Tensor],
    feature_count: int,
    projection_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if feature_count % 2:
        raise ValueError("cosh softmax feature count must be even")
    projection = _gaussian_orthogonal_projection(
        rows=feature_count // 2,
        dim=q.shape[-1],
        seed=projection_seed,
        device=q.device,
    )
    q_features, q_log_scale = _positive_softmax_features(
        q, projection=projection, is_query=True
    )
    k_features, k_log_scale = _positive_softmax_features(
        k, projection=projection, is_query=False
    )
    full_z = k_features.sum(dim=1)
    full_kv = torch.einsum("hnm,hnd->hmd", k_features, v.float())
    steps, q_heads, _dim = q.shape
    kv_heads = k.shape[0]
    group_size = q_heads // kv_heads
    outputs = []
    log_partitions = []
    seq_len = k.shape[1]
    query_positions = torch.arange(seq_len - steps, seq_len, device=q.device)
    for step, position in enumerate(query_positions.tolist()):
        omitted_mask = torch.zeros(seq_len, device=q.device, dtype=torch.bool)
        omitted_mask[: position + 1] = True
        omitted_mask[states["seed_mask"][step]] = False
        excluded = torch.nonzero(~omitted_mask, as_tuple=False).flatten()
        excluded_features = k_features.index_select(1, excluded)
        z_omitted = full_z - excluded_features.sum(dim=1)
        kv_omitted = full_kv - torch.einsum(
            "hnm,hnd->hmd", excluded_features, v.index_select(1, excluded).float()
        )
        z_for_q = z_omitted.repeat_interleave(group_size, dim=0)
        kv_for_q = kv_omitted.repeat_interleave(group_size, dim=0)
        qf = q_features[step]
        unscaled_z = torch.einsum("hm,hm->h", qf, z_for_q).clamp_min(1.0e-30)
        unscaled_num = torch.einsum("hm,hmd->hd", qf, kv_for_q)
        outputs.append(unscaled_num / unscaled_z.unsqueeze(-1))
        key_scale = k_log_scale[:, 0, 0].repeat_interleave(group_size, dim=0)
        log_partitions.append(
            unscaled_z.log() + q_log_scale[step, :, 0] + key_scale
        )
    linear_output = torch.stack(outputs, dim=0)
    linear_log_partition = torch.stack(log_partitions, dim=0)
    merged = merge_normalized_attention_states(
        states["seed_log_partition"],
        states["seed_output"],
        linear_log_partition,
        linear_output,
    )
    return merged, {
        "feature_count": float(feature_count),
        "projection_seed": float(projection_seed),
        "decode_summary_scalars_per_kv_head": float(feature_count * (q.shape[-1] + 1)),
    }


def _fit_shared_residual(
    *,
    q: torch.Tensor,
    states: dict[str, torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    o_proj_weight: torch.Tensor,
    train_indices: torch.Tensor,
    residual_tokens: int,
    optimization_steps: int,
    learning_rate: float,
    projection_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    init_k, init_v = _initialize_residual_bank(
        k,
        v,
        residual_tokens=residual_tokens,
        final_seed_mask=states["seed_mask"][-1],
    )
    residual_k = torch.nn.Parameter(init_k)
    residual_v = torch.nn.Parameter(init_v)
    optimizer = torch.optim.Adam([residual_k, residual_v], lr=learning_rate)
    target = states["full_output"].detach().float()
    train = train_indices.to(q.device)
    target_train = target.index_select(0, train)
    target_projected = F.linear(target_train.flatten(1), o_proj_weight.float()).detach()
    attention_scale = target_train.square().mean().clamp_min(1.0e-8)
    projection_scale = target_projected.square().mean().clamp_min(1.0e-8)
    losses: list[float] = []
    for iteration in range(optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        candidate = merge_seed_with_residual(
            q,
            states["seed_log_partition"],
            states["seed_output"],
            residual_k,
            residual_v,
        ).index_select(0, train)
        attention_loss = (candidate - target_train).square().mean() / attention_scale
        candidate_projected = F.linear(candidate.flatten(1), o_proj_weight.float())
        projection_loss = (candidate_projected - target_projected).square().mean() / projection_scale
        loss = attention_loss + projection_loss_weight * projection_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([residual_k, residual_v], max_norm=10.0)
        optimizer.step()
        if iteration in {0, optimization_steps - 1} or (iteration + 1) % 50 == 0:
            losses.append(float(loss.detach().item()))
    return residual_k.detach(), residual_v.detach(), losses


def _profile_capture(capture, *, args, prompt_row: dict[str, str]) -> dict[str, Any]:
    with torch.no_grad():
        q_all, k_all, v_all, meta = _shape_qkv(capture, apply_rope=True)
    if not meta.get("rope_applied"):
        raise RuntimeError(f"RoPE capture failed: {meta.get('rope_error')}")
    seq_len = int(q_all.shape[1])
    if seq_len < args.query_rows:
        raise ValueError("captured sequence is shorter than query_rows")
    group_size = int(meta["q_per_kv"])
    q = q_all[0, -args.query_rows :, :, :].float()
    k = k_all[0, :, ::group_size, :].permute(1, 0, 2).contiguous().float()
    v = v_all[0, :, ::group_size, :].permute(1, 0, 2).contiguous().float()
    policy_name = QWEN3B_POLICY_BY_LAYER.get(int(capture.layer_id))
    if policy_name is None:
        raise ValueError(f"no seed policy configured for layer {capture.layer_id}")
    policy = load_packaged_gate0_seed_only_batched_policy(policy_name)
    with torch.no_grad():
        states = _attention_states(q, k, v, policy=policy)
    train_indices, test_indices = _temporal_split(args.query_rows, args.train_rows)
    o_proj_weight = capture.module.o_proj.weight.detach().float()

    seed_train = _error_metrics(
        states["seed_output"], states["full_output"], indices=train_indices, o_proj_weight=o_proj_weight
    )
    seed_test = _error_metrics(
        states["seed_output"], states["full_output"], indices=test_indices, o_proj_weight=o_proj_weight
    )
    oracle_k, oracle_v = construct_query_exact_residual(
        q, states["omitted_log_partition"], states["omitted_output"]
    )
    query_oracle = merge_seed_with_residual(
        q, states["seed_log_partition"], states["seed_output"], oracle_k, oracle_v
    )
    oracle_test = _error_metrics(
        query_oracle, states["full_output"], indices=test_indices, o_proj_weight=o_proj_weight
    )

    residual_rows = []
    for residual_tokens in ([] if args.skip_static_residual_fit else args.residual_sizes):
        residual_k, residual_v, losses = _fit_shared_residual(
            q=q,
            states=states,
            k=k,
            v=v,
            o_proj_weight=o_proj_weight,
            train_indices=train_indices,
            residual_tokens=residual_tokens,
            optimization_steps=args.optimization_steps,
            learning_rate=args.learning_rate,
            projection_loss_weight=args.projection_loss_weight,
        )
        candidate = merge_seed_with_residual(
            q,
            states["seed_log_partition"],
            states["seed_output"],
            residual_k,
            residual_v,
        )
        train_metrics = _error_metrics(
            candidate, states["full_output"], indices=train_indices, o_proj_weight=o_proj_weight
        )
        test_metrics = _error_metrics(
            candidate, states["full_output"], indices=test_indices, o_proj_weight=o_proj_weight
        )
        baseline = seed_test["post_o_proj_relative_l2"]
        repaired = test_metrics["post_o_proj_relative_l2"]
        residual_rows.append(
            {
                "residual_tokens": residual_tokens,
                "total_seed_plus_residual_tokens": int(states["seed_mask"][-1].sum().item())
                + residual_tokens,
                "token_fraction_of_context": (
                    int(states["seed_mask"][-1].sum().item()) + residual_tokens
                )
                / seq_len,
                "optimizer_loss_trace": losses,
                "train": train_metrics,
                "test": test_metrics,
                "heldout_post_o_proj_error_reduction": 1.0 - repaired / max(baseline, 1.0e-12),
            }
        )
    linear_rows = []
    for feature_count in args.linear_feature_sizes:
        for projection_seed in args.linear_feature_seeds:
            with torch.no_grad():
                candidate, work = _linear_omitted_residual(
                    q=q,
                    k=k,
                    v=v,
                    states=states,
                    feature_count=feature_count,
                    projection_seed=projection_seed,
                )
            test_metrics = _error_metrics(
                candidate, states["full_output"], indices=test_indices, o_proj_weight=o_proj_weight
            )
            baseline = seed_test["post_o_proj_relative_l2"]
            repaired = test_metrics["post_o_proj_relative_l2"]
            linear_rows.append(
                {
                    **work,
                    "test": test_metrics,
                    "heldout_post_o_proj_error_reduction": 1.0
                    - repaired / max(baseline, 1.0e-12),
                    "equivalent_seed_plus_feature_fraction": (
                        int(states["seed_mask"][-1].sum().item()) + feature_count
                    )
                    / seq_len,
                }
            )
    candidates = [
        ("static_kv", int(row["residual_tokens"]), row)
        for row in residual_rows
    ] + [
        ("linear_summary", int(row["feature_count"]), row)
        for row in linear_rows
    ]
    if candidates:
        best_kind, best_size, best = min(
            candidates, key=lambda item: item[2]["test"]["post_o_proj_relative_l2"]
        )
    else:
        best_kind, best_size, best = "seed_only", 0, {
            "test": seed_test,
            "heldout_post_o_proj_error_reduction": 0.0,
        }
    return {
        "prompt_id": prompt_row["id"],
        "bucket": prompt_row["bucket"],
        "layer": int(capture.layer_id),
        "shape": {
            "seq_len": seq_len,
            "query_rows": args.query_rows,
            "train_rows": args.train_rows,
            "test_rows": args.query_rows - args.train_rows,
            "q_heads": int(q.shape[1]),
            "kv_heads": int(k.shape[0]),
            "head_dim": int(q.shape[-1]),
            "seed_tokens_final_row": int(states["seed_mask"][-1].sum().item()),
        },
        "seed_only": {"train": seed_train, "test": seed_test},
        "query_specific_one_token_oracle_test": oracle_test,
        "residual_banks": residual_rows,
        "linear_residual_summaries": linear_rows,
        "best_method": best_kind,
        "best_method_size": best_size,
        "best_heldout_post_o_proj_relative_l2": float(best["test"]["post_o_proj_relative_l2"]),
        "best_heldout_post_o_proj_error_reduction": float(
            best["heldout_post_o_proj_error_reduction"]
        ),
    }


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {}
    reductions = [float(row["best_heldout_post_o_proj_error_reduction"]) for row in rows]
    heldout = [float(row["best_heldout_post_o_proj_relative_l2"]) for row in rows]
    return {
        "captures": len(rows),
        "best_error_reduction_min": min(reductions),
        "best_error_reduction_mean": sum(reductions) / len(reductions),
        "best_heldout_post_o_proj_relative_l2_max": max(heldout),
        "feasibility_signal": min(reductions) >= 0.8 and max(heldout) <= 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--buckets", default="chat_instruction,needle_rag,json_tool,noisy_neartie")
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--layers", default="26,27")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--prompt-truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--query-rows", type=int, default=16)
    parser.add_argument("--train-rows", type=int, default=8)
    parser.add_argument("--residual-sizes", default="4,8,16,32")
    parser.add_argument("--optimization-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--projection-loss-weight", type=float, default=1.0)
    parser.add_argument("--skip-static-residual-fit", action="store_true")
    parser.add_argument("--linear-feature-sizes", default="")
    parser.add_argument("--linear-feature-seeds", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    args.residual_sizes = _parse_residual_sizes(args.residual_sizes)
    args.linear_feature_sizes = (
        _parse_residual_sizes(args.linear_feature_sizes) if args.linear_feature_sizes else []
    )
    args.linear_feature_seeds = _parse_ints(args.linear_feature_seeds)
    layers = set(_parse_ints(args.layers))
    buckets = set(item.strip() for item in args.buckets.split(",") if item.strip())
    train_indices, test_indices = _temporal_split(args.query_rows, args.train_rows)
    del train_indices, test_indices
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    prompt_rows = _load_prompt_rows(
        prompt_file=args.prompt_file,
        prompt=args.prompt,
        buckets=buckets,
        max_prompts=args.max_prompts,
    )
    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer.truncation_side = args.prompt_truncation_side
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    rows: list[dict[str, Any]] = []
    for prompt_row in prompt_rows:
        captured, handles = _capture_attention_inputs(model, layers)
        try:
            tokens = tokenizer(
                prompt_row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq,
            ).to(args.device)
            with torch.inference_mode():
                model(**tokens, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if not captured:
            raise RuntimeError("no target attention modules were captured")
        for capture in captured:
            rows.append(_profile_capture(capture, args=args, prompt_row=prompt_row))
        del captured, tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "schema": "streamattn.seed_residual_capacity_oracle.v1",
        "model": args.model,
        "layers": sorted(layers),
        "residual_sizes": args.residual_sizes,
        "query_rows": args.query_rows,
        "train_rows": args.train_rows,
        "rows": rows,
        "summary": _aggregate(rows),
        "interpretation": (
            "capacity oracle only: residuals are optimized per captured context on early query rows; "
            "a deployable mode requires an amortized compactor and model-level replay"
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
