"""Measure StreamAttn's hardware-constrained output-sufficiency frontier.

This is a semantic reference benchmark, not a production attention backend. It
compares hard-drop routes against residual-complete attention at matched exact
block budgets, using real model Q/K/V captures and the module output projection.
CUDA/Triton work should follow only after one estimator wins this frontier.
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
from stream_attention.adaptive_frontier import (  # noqa: E402
    BlockAttentionStates,
    control_variate_attention,
    diagonal_gaussian_block_states,
    exact_block_attention_states,
    gqa_topk_mask,
    merge_block_attention_states,
    poisson_tail_sample,
    post_wo_gqa_greedy_mask,
)


def _parse_ints(raw: str, *, allow_zero: bool = False) -> list[int]:
    values = sorted(set(int(item.strip()) for item in raw.replace(";", ",").split(",") if item.strip()))
    minimum = 0 if allow_zero else 1
    if not values or values[0] < minimum:
        raise ValueError(f"values must be unique integers >= {minimum}")
    return values


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _load_prompt_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    buckets = {item.strip() for item in args.buckets.split(",") if item.strip()}
    for index, prompt in enumerate(args.prompt or []):
        rows.append({"id": f"cli_{index}", "bucket": "cli", "text": prompt})
    if args.prompt_file:
        path = Path(args.prompt_file)
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
                text, bucket, prompt_id = line, "unknown", f"row_{line_no}"
            if text and (not buckets or bucket in buckets):
                rows.append({"id": prompt_id, "bucket": bucket, "text": text})
            if len(rows) >= args.max_prompts:
                break
    if not rows:
        rows = [
            {
                "id": "default_0",
                "bucket": "long_doc",
                "text": (
                    "Explain how exact online-softmax attention differs from an adaptive "
                    "attention estimator that restores the omitted numerator and denominator. "
                )
                * 256,
            }
        ]
    return rows[: args.max_prompts]


def _base_mask(
    *,
    valid_lengths: torch.Tensor,
    q_heads: int,
    blocks: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
) -> torch.Tensor:
    mask = torch.zeros(
        (int(valid_lengths.numel()), q_heads, blocks),
        device=valid_lengths.device,
        dtype=torch.bool,
    )
    for row, length in enumerate(valid_lengths.tolist()):
        valid_blocks = math.ceil(int(length) / block_size)
        sink_end = min(sink_blocks, valid_blocks)
        if sink_end:
            mask[row, :, :sink_end] = True
        recent_start = max(sink_end, valid_blocks - recent_blocks)
        if recent_start < valid_blocks:
            mask[row, :, recent_start:valid_blocks] = True
    return mask


def _qk_block_max_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    valid_lengths: torch.Tensor,
) -> torch.Tensor:
    rows, q_heads, dim = map(int, q.shape)
    kv_heads, tokens, _ = map(int, k.shape)
    blocks = math.ceil(tokens / block_size)
    padded = blocks * block_size
    if padded != tokens:
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (kv_heads, padded - tokens, dim), device=k.device, dtype=k.dtype
                ),
            ],
            dim=1,
        )
    group_size = q_heads // kv_heads
    kq = k.reshape(kv_heads, blocks, block_size, dim).repeat_interleave(group_size, dim=0)
    scores = torch.einsum("rhd,hbsd->rhbs", q.float(), kq.float()) / math.sqrt(dim)
    positions = torch.arange(padded, device=q.device).reshape(blocks, block_size)
    valid = positions[None, None, :, :] < valid_lengths[:, None, None, None]
    return scores.masked_fill(~valid, -torch.inf).amax(dim=-1)


def _select_with_base(
    scores: torch.Tensor,
    *,
    base: torch.Tensor,
    kv_heads: int,
    extra_blocks: int,
) -> torch.Tensor:
    candidate_scores = scores.masked_fill(base, -torch.inf)
    dynamic = gqa_topk_mask(
        candidate_scores, kv_heads=kv_heads, blocks_per_group=extra_blocks
    )
    return base | dynamic


def _metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    o_proj_weight: torch.Tensor,
) -> dict[str, float]:
    diff = candidate.float() - reference.float()
    row_rel = torch.linalg.vector_norm(diff, dim=(-2, -1)) / torch.linalg.vector_norm(
        reference.float(), dim=(-2, -1)
    ).clamp_min(1.0e-12)
    projected_candidate = F.linear(candidate.flatten(1).float(), o_proj_weight.float())
    projected_reference = F.linear(reference.flatten(1).float(), o_proj_weight.float())
    projected_diff = projected_candidate - projected_reference
    projected_row_rel = torch.linalg.vector_norm(
        projected_diff, dim=-1
    ) / torch.linalg.vector_norm(projected_reference, dim=-1).clamp_min(1.0e-12)
    return {
        "attention_relative_l2": float(
            (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference).clamp_min(1.0e-12)).item()
        ),
        "attention_row_relative_l2_p95": float(torch.quantile(row_rel, 0.95).item()),
        "post_wo_relative_l2": float(
            (
                torch.linalg.vector_norm(projected_diff)
                / torch.linalg.vector_norm(projected_reference).clamp_min(1.0e-12)
            ).item()
        ),
        "post_wo_row_relative_l2_p95": float(
            torch.quantile(projected_row_rel, 0.95).item()
        ),
        "post_wo_max_abs": float(projected_diff.abs().max().item()),
    }


def _physical_stats(mask: torch.Tensor, *, kv_heads: int) -> dict[str, float]:
    rows, q_heads, _blocks = map(int, mask.shape)
    group_size = q_heads // kv_heads
    grouped = mask.reshape(rows, kv_heads, group_size, -1).any(dim=2)
    per_row = grouped.sum(dim=(-2, -1)).float()
    head_blocks = mask.sum(dim=(-2, -1)).float()
    return {
        "physical_union_blocks_mean": float(per_row.mean().item()),
        "physical_union_blocks_max": float(per_row.max().item()),
        "q_head_blocks_mean": float(head_blocks.mean().item()),
        "head_union_efficiency": float(
            (head_blocks.sum() / (per_row.sum() * group_size).clamp_min(1.0)).item()
        ),
    }


def _group_mask(mask: torch.Tensor, *, kv_heads: int) -> torch.Tensor:
    rows, q_heads, blocks = map(int, mask.shape)
    group_size = q_heads // kv_heads
    return mask.reshape(rows, kv_heads, group_size, blocks).any(dim=2)


def _expand_group_tensor(value: torch.Tensor, *, q_heads: int) -> torch.Tensor:
    return value.repeat_interleave(q_heads // int(value.shape[1]), dim=1)


def _oracle_residual_priority(
    exact: BlockAttentionStates,
    approximate: BlockAttentionStates,
    *,
    full_output: torch.Tensor,
    full_log_partition: torch.Tensor,
    o_proj_weight: torch.Tensor,
    kv_heads: int,
) -> torch.Tensor:
    """Score blocks by their exact post-WO control-variate correction.

    This deliberately uses exact states and is therefore not deployable.  It is
    a variance lower-bound for sampling policies: if this priority cannot make
    stochastic correction useful, improving a cheap priority proxy will not.
    """

    rows, q_heads, blocks, dim = map(int, exact.output.shape)
    group_size = q_heads // kv_heads
    exact_mass = torch.exp(exact.log_partition - full_log_partition.unsqueeze(-1))
    approximate_mass = torch.exp(
        approximate.log_partition - full_log_partition.unsqueeze(-1)
    ).nan_to_num(0.0)
    exact_centered = exact.output.float() - full_output[:, :, None, :].float()
    approximate_centered = approximate.output.float() - full_output[:, :, None, :].float()
    correction = (
        exact_mass[..., None] * exact_centered
        - approximate_mass[..., None] * approximate_centered
    )
    priorities = []
    for group in range(kv_heads):
        start_head = group * group_size
        end_head = (group + 1) * group_size
        start_col = start_head * dim
        end_col = end_head * dim
        group_correction = correction[:, start_head:end_head].permute(0, 2, 1, 3)
        projected = F.linear(
            group_correction.flatten(2), o_proj_weight[:, start_col:end_col].float()
        )
        priorities.append(torch.linalg.vector_norm(projected, dim=-1))
    return torch.stack(priorities, dim=1).clamp_min(1.0e-20)


def _profile_tensors(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_lengths: torch.Tensor,
    o_proj_weight: torch.Tensor,
    block_size: int,
    budgets: list[int],
    tail_samples: list[int],
    sink_blocks: int,
    recent_blocks: int,
    sample_repeats: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    rows, q_heads, _dim = map(int, q.shape)
    kv_heads = int(k.shape[0])
    exact = exact_block_attention_states(
        q, k, v, block_size=block_size, valid_lengths=valid_lengths
    )
    moment = diagonal_gaussian_block_states(
        q, k, v, block_size=block_size, valid_lengths=valid_lengths
    )
    full = merge_block_attention_states(exact)
    qk_scores = _qk_block_max_scores(
        q, k, block_size=block_size, valid_lengths=valid_lengths
    )
    base = _base_mask(
        valid_lengths=valid_lengths,
        q_heads=q_heads,
        blocks=int(exact.log_partition.shape[-1]),
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
    )
    result: list[dict[str, Any]] = []

    for budget in budgets:
        masks = {
            "qk_hard_drop": _select_with_base(
                qk_scores, base=base, kv_heads=kv_heads, extra_blocks=budget
            ),
            "mass_hard_drop": _select_with_base(
                exact.log_partition, base=base, kv_heads=kv_heads, extra_blocks=budget
            ),
            "post_wo_greedy_hard_drop": post_wo_gqa_greedy_mask(
                exact,
                full_output=full.output,
                o_proj_weight=o_proj_weight,
                kv_heads=kv_heads,
                blocks_per_group=budget,
                base_mask=base,
            ),
        }
        for method, mask in masks.items():
            merged = merge_block_attention_states(exact, mask)
            result.append(
                {
                    "method": method,
                    "exact_middle_blocks_per_kv_group": budget,
                    "tail_samples_expected_per_kv_group": 0,
                    "valid_fraction": float(merged.valid.float().mean().item()),
                    **_physical_stats(mask, kv_heads=kv_heads),
                    **_metrics(merged.output, full.output, o_proj_weight=o_proj_weight),
                }
            )

        selected = masks["qk_hard_drop"]
        completed = control_variate_attention(exact, moment, selected=selected)
        result.append(
            {
                "method": "qk_exact_peaks_plus_moment_tail",
                "exact_middle_blocks_per_kv_group": budget,
                "tail_samples_expected_per_kv_group": 0,
                "valid_fraction": float(completed.valid.float().mean().item()),
                "denominator_scaled_min": float(completed.scaled_denominator.min().item()),
                **_physical_stats(selected, kv_heads=kv_heads),
                **_metrics(completed.output, full.output, o_proj_weight=o_proj_weight),
            }
        )

        selected_group = _group_mask(selected, kv_heads=kv_heads)
        grouped_log_z = moment.log_partition.reshape(
            rows, kv_heads, q_heads // kv_heads, -1
        ).amax(dim=2)
        mass_priority = torch.exp(
            grouped_log_z - grouped_log_z.amax(dim=-1, keepdim=True)
        ).nan_to_num(0.0)
        oracle_priority = _oracle_residual_priority(
            exact,
            moment,
            full_output=full.output,
            full_log_partition=full.log_partition,
            o_proj_weight=o_proj_weight,
            kv_heads=kv_heads,
        )
        for samples in tail_samples:
            if samples == 0:
                continue
            for priority_name, priority in (
                ("mass_priority", mass_priority),
                ("oracle_residual_priority", oracle_priority),
            ):
                for repeat in range(sample_repeats):
                    device_type = q.device.type if q.device.type in {"cpu", "cuda"} else "cpu"
                    generator = torch.Generator(device=device_type).manual_seed(
                        sample_seed
                        + budget * 1009
                        + samples * 97
                        + repeat
                        + (0 if priority_name == "mass_priority" else 1_000_003)
                    )
                    sampled_group, probability_group = poisson_tail_sample(
                        priority,
                        selected=selected_group,
                        expected_samples=samples,
                        generator=generator,
                    )
                    sampled = _expand_group_tensor(sampled_group, q_heads=q_heads)
                    probability = _expand_group_tensor(probability_group, q_heads=q_heads)
                    corrected = control_variate_attention(
                        exact,
                        moment,
                        selected=selected,
                        sampled=sampled,
                        inclusion_probability=probability,
                    )
                    executed = selected | sampled
                    result.append(
                        {
                            "method": f"qk_exact_peaks_plus_moment_tail_ht_{priority_name}",
                            "priority_is_oracle": priority_name == "oracle_residual_priority",
                            "repeat": repeat,
                            "exact_middle_blocks_per_kv_group": budget,
                            "tail_samples_expected_per_kv_group": samples,
                            "tail_samples_realized_per_kv_group": float(
                                sampled_group.sum(dim=-1).float().mean().item()
                            ),
                            "valid_fraction": float(corrected.valid.float().mean().item()),
                            "denominator_scaled_min": float(
                                corrected.scaled_denominator.min().item()
                            ),
                            **_physical_stats(executed, kv_heads=kv_heads),
                            **_metrics(corrected.output, full.output, o_proj_weight=o_proj_weight),
                        }
                    )
    return result


def _aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["method"],
            row["exact_middle_blocks_per_kv_group"],
            row["tail_samples_expected_per_kv_group"],
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        summary.append(
            {
                "method": key[0],
                "exact_middle_blocks_per_kv_group": key[1],
                "tail_samples_expected_per_kv_group": key[2],
                "measurements": len(members),
                "post_wo_relative_l2_mean": sum(
                    float(row["post_wo_relative_l2"]) for row in members
                )
                / len(members),
                "post_wo_relative_l2_max": max(
                    float(row["post_wo_relative_l2"]) for row in members
                ),
                "post_wo_row_relative_l2_p95_max": max(
                    float(row["post_wo_row_relative_l2_p95"]) for row in members
                ),
                "physical_union_blocks_mean": sum(
                    float(row["physical_union_blocks_mean"]) for row in members
                )
                / len(members),
                "valid_fraction_min": min(float(row["valid_fraction"]) for row in members),
            }
        )
    return summary


def _synthetic_capture(device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device.type).manual_seed(31)
    q = torch.randn(3, 4, 8, generator=generator, device=device)
    k = torch.randn(2, 32, 8, generator=generator, device=device)
    v = 0.6 * k + 0.4 * torch.randn(2, 32, 8, generator=generator, device=device)
    lengths = torch.tensor([24, 29, 32], device=device)
    o_proj_weight = torch.randn(32, 32, generator=generator, device=device) / math.sqrt(32)
    return q, k, v, lengths, o_proj_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--buckets", default="")
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--layers", default="24,26,27")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--query-rows", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--budgets", default="4,8,16")
    parser.add_argument("--tail-samples", default="0,4,8,16")
    parser.add_argument("--sample-repeats", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--sink-blocks", type=int, default=1)
    parser.add_argument("--recent-blocks", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    budgets = _parse_ints(args.budgets)
    tail_samples = _parse_ints(args.tail_samples, allow_zero=True)
    layers = set(_parse_ints(args.layers, allow_zero=True))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows: list[dict[str, Any]] = []
    if args.synthetic:
        q, k, v, lengths, o_proj_weight = _synthetic_capture(device)
        rows.extend(
            {
                "prompt_id": "synthetic",
                "bucket": "synthetic",
                "layer": -1,
                **row,
            }
            for row in _profile_tensors(
                q=q,
                k=k,
                v=v,
                valid_lengths=lengths,
                o_proj_weight=o_proj_weight,
                block_size=min(args.block_size, 8),
                budgets=[min(value, 2) for value in budgets],
                tail_samples=[min(value, 2) for value in tail_samples],
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                sample_repeats=args.sample_repeats,
                sample_seed=args.sample_seed,
            )
        )
    else:
        prompt_rows = _load_prompt_rows(args)
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        tokenizer.truncation_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=_dtype(args.dtype),
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        model.eval()
        for prompt_row in prompt_rows:
            captured, handles = _capture_attention_inputs(model, layers)
            try:
                tokens = tokenizer(
                    prompt_row["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_seq,
                ).to(device)
                with torch.inference_mode():
                    model(**tokens, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            for capture in captured:
                with torch.no_grad():
                    q_all, k_all, v_all, meta = _shape_qkv(capture, apply_rope=True)
                if not meta.get("rope_applied"):
                    raise RuntimeError(f"RoPE capture failed: {meta.get('rope_error')}")
                seq_len = int(q_all.shape[1])
                query_rows = min(args.query_rows, seq_len)
                group_size = int(meta["q_per_kv"])
                q = q_all[0, -query_rows:].float()
                k = k_all[0, :, ::group_size, :].permute(1, 0, 2).contiguous().float()
                v = v_all[0, :, ::group_size, :].permute(1, 0, 2).contiguous().float()
                lengths = torch.arange(
                    seq_len - query_rows + 1, seq_len + 1, device=device
                )
                result = _profile_tensors(
                    q=q,
                    k=k,
                    v=v,
                    valid_lengths=lengths,
                    o_proj_weight=capture.module.o_proj.weight.detach().float(),
                    block_size=args.block_size,
                    budgets=budgets,
                    tail_samples=tail_samples,
                    sink_blocks=args.sink_blocks,
                    recent_blocks=args.recent_blocks,
                    sample_repeats=args.sample_repeats,
                    sample_seed=args.sample_seed,
                )
                rows.extend(
                    {
                        "prompt_id": prompt_row["id"],
                        "bucket": prompt_row["bucket"],
                        "layer": int(capture.layer_id),
                        "seq_len": seq_len,
                        "q_heads": int(q.shape[1]),
                        "kv_heads": int(k.shape[0]),
                        "head_dim": int(q.shape[-1]),
                        **row,
                    }
                    for row in result
                )
            del captured, tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "schema": "streamattn.adaptive_output_sufficiency_frontier.v1",
        "model": "synthetic" if args.synthetic else args.model,
        "attention_semantics": (
            "exact block states; hard-drop is exact only over selected blocks; residual methods "
            "estimate omitted numerator and denominator under the same final normalization"
        ),
        "block_size": min(args.block_size, 8) if args.synthetic else args.block_size,
        "rows": rows,
        "summary": _aggregate(rows),
        "promotion_status": "reference_only_not_runtime_promoted",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
