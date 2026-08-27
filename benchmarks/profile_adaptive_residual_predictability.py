"""Measure conditional predictive rank for StreamAttn adaptive residuals.

This is a bounded semantic gate.  It tests whether exact omitted attention
state can be predicted from features available without reading omitted KV
tokens during decode.  It does not implement or promote a runtime backend.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_adaptive_output_sufficiency_frontier import (  # noqa: E402
    _base_mask,
    _qk_block_max_scores,
    _select_with_base,
)
from benchmarks.profile_real_llm_gate1_heads import (  # noqa: E402
    _capture_attention_inputs,
    _import_transformers,
    _shape_qkv,
)
from stream_attention.adaptive_frontier import (  # noqa: E402
    diagonal_gaussian_block_states,
    exact_block_attention_states,
    merge_block_attention_states,
)
from stream_attention.residual_predictability import (  # noqa: E402
    binary_roc_auc,
    calibrate_canary_threshold,
    canary_gate_report,
    canonical_correlations,
    correction_report,
    fit_ridge,
    merge_selected_with_omitted_innovation,
    predict_ridge,
    rank_report,
    relative_l2,
    signed_hash_projection,
)


def _parse_ints(raw: str, *, allow_zero: bool = False) -> list[int]:
    values = sorted(set(int(item.strip()) for item in raw.replace(";", ",").split(",") if item.strip()))
    minimum = 0 if allow_zero else 1
    if not values or values[0] < minimum:
        raise ValueError(f"values must be unique integers >= {minimum}")
    return values


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _load_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    buckets = [item.strip() for item in args.buckets.split(",") if item.strip()]
    allowed = set(buckets)
    per_bucket: dict[str, int] = {}
    for index, text in enumerate(args.prompt or []):
        rows.append({"id": f"cli_{index}", "bucket": "cli", "text": text})
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
            if not text or (allowed and bucket not in allowed):
                continue
            if per_bucket.get(bucket, 0) >= args.max_prompts_per_bucket:
                continue
            rows.append({"id": prompt_id, "bucket": bucket, "text": text})
            per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
            if len(rows) >= args.max_prompts:
                break
    if not rows:
        base = (
            "A long-context attention engine must preserve the omitted softmax mass and value state. "
            "The selected exact blocks are merged with a positive omitted sufficient state. "
        )
        rows = [
            {"id": f"default_{index}", "bucket": "default", "text": base * (128 + index * 8)}
            for index in range(max(2, min(args.max_prompts, 4)))
        ]
    return rows[: args.max_prompts]


def _temporal_features(value: torch.Tensor) -> torch.Tensor:
    previous = torch.cat([value[:1], value[:-1]], dim=0)
    return torch.cat([value, value - previous, previous], dim=-1)


def _build_feature_sets(
    *,
    q: torch.Tensor,
    selected_output: torch.Tensor,
    selected_log_z: torch.Tensor,
    approximate_omitted_output: torch.Tensor,
    approximate_omitted_log_z: torch.Tensor,
    qk_scores: torch.Tensor,
    selected_mask: torch.Tensor,
    valid_lengths: torch.Tensor,
    kv_heads: int,
    hash_width: int,
    feature_seed: int,
) -> dict[str, torch.Tensor]:
    rows, q_heads, _dim = map(int, q.shape)
    group_size = q_heads // kv_heads
    q_hash = signed_hash_projection(q, width=hash_width, seed=feature_seed)
    q_norm = torch.linalg.vector_norm(q.float(), dim=-1)
    position = valid_lengths.float().reshape(rows, 1) / valid_lengths.max().clamp_min(1)
    f0 = torch.cat([q_hash, q_norm, position], dim=-1)

    selected_hash = signed_hash_projection(
        selected_output, width=hash_width, seed=feature_seed + 1
    )
    selected_norm = torch.linalg.vector_norm(selected_output.float(), dim=-1)
    grouped_scores = qk_scores.reshape(rows, kv_heads, group_size, -1).amax(dim=2)
    grouped_mask = selected_mask.reshape(rows, kv_heads, group_size, -1).any(dim=2)
    selected_scores = grouped_scores.masked_fill(~grouped_mask, -torch.inf)
    top_count = min(4, int(selected_scores.shape[-1]))
    top_values = torch.topk(selected_scores, k=top_count, dim=-1).values
    top_values = torch.nan_to_num(top_values, neginf=-32.0, posinf=32.0).flatten(1)
    top_group = torch.topk(grouped_scores, k=min(2, int(grouped_scores.shape[-1])), dim=-1).values
    if top_group.shape[-1] == 1:
        margin = torch.zeros_like(top_group[..., 0])
    else:
        margin = top_group[..., 0] - top_group[..., 1]
    f1 = torch.cat(
        [f0, selected_hash, selected_log_z.float(), selected_norm, top_values, margin], dim=-1
    )

    omitted_hash = signed_hash_projection(
        approximate_omitted_output, width=hash_width, seed=feature_seed + 2
    )
    approximate_rho = approximate_omitted_log_z.float() - selected_log_z.float()
    approximate_norm = torch.linalg.vector_norm(approximate_omitted_output.float(), dim=-1)
    f2 = torch.cat([f1, omitted_hash, approximate_rho, approximate_norm], dim=-1)
    return {
        "f0_query": f0,
        "f1_selected_state": f1,
        "f2_block_moments": f2,
        "f3_temporal": _temporal_features(f2),
    }


def _capture_group(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_lengths: torch.Tensor,
    o_proj_weight: torch.Tensor,
    block_size: int,
    budget: int,
    sink_blocks: int,
    recent_blocks: int,
    hash_width: int,
    feature_seed: int,
) -> dict[str, Any]:
    rows, q_heads, dim = map(int, q.shape)
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
    selected_mask = _select_with_base(
        qk_scores, base=base, kv_heads=kv_heads, extra_blocks=budget
    )
    selected = merge_block_attention_states(exact, selected_mask)
    omitted_mask = ~selected_mask & torch.isfinite(exact.log_partition)
    omitted = merge_block_attention_states(exact, omitted_mask)
    approximate_omitted = merge_block_attention_states(moment, omitted_mask)
    if not bool(omitted.valid.all()):
        raise RuntimeError("capture contains a row/head with no omitted state")

    weight = o_proj_weight.float()
    full_projected = F.linear(full.output.flatten(1), weight)
    selected_projected = F.linear(selected.output.flatten(1), weight)
    innovation = omitted.output.float() - selected.output.float()
    innovation_projected = F.linear(innovation.flatten(1), weight)
    rho = omitted.log_partition.float() - selected.log_partition.float()
    alpha = torch.sigmoid(rho)
    residual = full_projected - selected_projected
    reconstructed = selected.output.float() + alpha[..., None] * innovation
    torch.testing.assert_close(reconstructed, full.output.float(), rtol=2.0e-5, atol=2.0e-5)
    features = _build_feature_sets(
        q=q,
        selected_output=selected.output,
        selected_log_z=selected.log_partition,
        approximate_omitted_output=approximate_omitted.output,
        approximate_omitted_log_z=approximate_omitted.log_partition,
        qk_scores=qk_scores,
        selected_mask=selected_mask,
        valid_lengths=valid_lengths,
        kv_heads=kv_heads,
        hash_width=hash_width,
        feature_seed=feature_seed,
    )
    return {
        "features": {name: value.detach().cpu() for name, value in features.items()},
        "residual": residual.detach().cpu(),
        "rho": rho.detach().cpu(),
        "alpha": alpha.detach().cpu(),
        "omitted_output": omitted.output.flatten(1).detach().cpu(),
        "innovation": innovation.flatten(1).detach().cpu(),
        "innovation_projected": innovation_projected.detach().cpu(),
        "selected_output": selected.output.detach().cpu(),
        "full_projected": full_projected.detach().cpu(),
        "selected_projected": selected_projected.detach().cpu(),
        "o_proj_weight": weight.detach().cpu(),
        "q_heads": q_heads,
        "head_dim": dim,
        "kv_heads": kv_heads,
        "baseline_post_wo_relative_l2": relative_l2(selected_projected, full_projected),
    }


def _indices(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def _folds(metadata: list[dict[str, Any]]) -> dict[str, list[tuple[str, torch.Tensor, torch.Tensor]]]:
    prompt_rows: dict[str, list[int]] = {}
    bucket_rows: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        prompt_rows.setdefault(str(row["prompt_id"]), []).append(index)
        bucket_rows.setdefault(str(row["bucket"]), []).append(index)

    early: list[int] = []
    late: list[int] = []
    for rows in prompt_rows.values():
        split = max(1, len(rows) // 2)
        early.extend(rows[:split])
        late.extend(rows[split:])
    result: dict[str, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
    if early and late:
        result["same_prompt_future"] = [("temporal", _indices(early), _indices(late))]
    if len(prompt_rows) >= 2:
        all_rows = set(range(len(metadata)))
        result["unseen_prompt"] = []
        for prompt_id, test_rows in sorted(prompt_rows.items()):
            train_rows = sorted(all_rows - set(test_rows))
            result["unseen_prompt"].append(
                (prompt_id, _indices(train_rows), _indices(test_rows))
            )
    if len(bucket_rows) >= 2:
        all_rows = set(range(len(metadata)))
        result["unseen_bucket"] = []
        for bucket, test_rows in sorted(bucket_rows.items()):
            train_rows = sorted(all_rows - set(test_rows))
            result["unseen_bucket"].append((bucket, _indices(train_rows), _indices(test_rows)))
    return result


def _state_prediction_report(
    *,
    features_train: torch.Tensor,
    features_test: torch.Tensor,
    rho_train: torch.Tensor,
    rho_test: torch.Tensor,
    innovation_train: torch.Tensor,
    innovation_test: torch.Tensor,
    selected_output_test: torch.Tensor,
    selected_projected_test: torch.Tensor,
    full_projected_test: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_heads: int,
    head_dim: int,
    ridge: float,
) -> dict[str, Any]:
    rho_model = fit_ridge(features_train, rho_train, ridge=ridge)
    innovation_model = fit_ridge(features_train, innovation_train, ridge=ridge)
    predicted_rho = predict_ridge(rho_model, features_test)
    predicted_innovation = predict_ridge(innovation_model, features_test).reshape(
        -1, q_heads, head_dim
    )
    candidate_output = merge_selected_with_omitted_innovation(
        selected_output_test,
        predicted_rho,
        predicted_innovation,
    )
    candidate_projected = F.linear(candidate_output.flatten(1), o_proj_weight.float())
    baseline_error = full_projected_test.float() - selected_projected_test.float()
    candidate_error = full_projected_test.float() - candidate_projected
    row_baseline = torch.linalg.vector_norm(baseline_error, dim=-1).clamp_min(1.0e-12)
    row_candidate = torch.linalg.vector_norm(candidate_error, dim=-1)
    total_ratio = torch.linalg.vector_norm(candidate_error) / torch.linalg.vector_norm(
        baseline_error
    ).clamp_min(1.0e-12)
    return {
        "rho": correction_report(predicted_rho, rho_test),
        "value_innovation": correction_report(
            predicted_innovation.flatten(1), innovation_test
        ),
        "post_wo_error_ratio_vs_hard_drop": float(total_ratio.item()),
        "post_wo_error_reduction_vs_hard_drop": float((1.0 - total_ratio).item()),
        "row_error_ratio_p50": float(torch.quantile(row_candidate / row_baseline, 0.50).item()),
        "row_error_ratio_p95": float(torch.quantile(row_candidate / row_baseline, 0.95).item()),
        "row_error_ratio_p99": float(torch.quantile(row_candidate / row_baseline, 0.99).item()),
    }


def _state_candidate(
    group: dict[str, Any],
    *,
    feature_name: str,
    train: torch.Tensor,
    test: torch.Tensor,
    ridge: float,
) -> dict[str, torch.Tensor]:
    feature = group["features"][feature_name]
    x_train = feature.index_select(0, train)
    x_test = feature.index_select(0, test)
    rho_model = fit_ridge(x_train, group["rho"].index_select(0, train), ridge=ridge)
    innovation_model = fit_ridge(
        x_train, group["innovation"].index_select(0, train), ridge=ridge
    )
    rho = predict_ridge(rho_model, x_test)
    innovation = predict_ridge(innovation_model, x_test).reshape(
        -1, group["q_heads"], group["head_dim"]
    )
    selected = group["selected_output"].index_select(0, test)
    output = merge_selected_with_omitted_innovation(selected, rho, innovation)
    projected = F.linear(output.flatten(1), group["o_proj_weight"].float())
    standardized = (x_test - rho_model.feature_mean) / rho_model.feature_scale
    return {
        "rho": rho,
        "innovation": innovation,
        "output": output,
        "projected": projected,
        "feature_z_rms": standardized.square().mean(dim=-1).sqrt(),
        "feature_z_max": standardized.abs().amax(dim=-1),
    }


_CANARY_FEATURES = (
    "primary_feature_z_rms",
    "primary_feature_z_max",
    "secondary_feature_z_rms",
    "secondary_feature_z_max",
    "primary_alpha_mean",
    "primary_alpha_std",
    "primary_alpha_min",
    "primary_alpha_max",
    "rho_disagreement_rms",
    "innovation_norm_ratio",
    "innovation_disagreement_ratio",
    "correction_norm_ratio",
    "secondary_correction_norm_ratio",
    "projected_disagreement_ratio",
    "correction_cosine",
)


def _canary_runtime_features(
    group: dict[str, Any],
    *,
    test: torch.Tensor,
    primary: dict[str, torch.Tensor],
    secondary: dict[str, torch.Tensor],
) -> torch.Tensor:
    selected_output = group["selected_output"].index_select(0, test).float()
    selected_projected = group["selected_projected"].index_select(0, test).float()
    output_norm = torch.linalg.vector_norm(selected_output.flatten(1), dim=-1).clamp_min(1.0e-6)
    projected_norm = torch.linalg.vector_norm(selected_projected, dim=-1).clamp_min(1.0e-6)
    primary_correction = primary["projected"] - selected_projected
    secondary_correction = secondary["projected"] - selected_projected
    primary_correction_norm = torch.linalg.vector_norm(primary_correction, dim=-1)
    secondary_correction_norm = torch.linalg.vector_norm(secondary_correction, dim=-1)
    correction_cosine = (
        (primary_correction * secondary_correction).sum(dim=-1)
        / (primary_correction_norm * secondary_correction_norm).clamp_min(1.0e-6)
    )
    alpha = torch.sigmoid(primary["rho"])
    columns = (
        torch.log1p(primary["feature_z_rms"]),
        torch.log1p(primary["feature_z_max"]),
        torch.log1p(secondary["feature_z_rms"]),
        torch.log1p(secondary["feature_z_max"]),
        alpha.mean(dim=-1),
        alpha.std(dim=-1, unbiased=False),
        alpha.amin(dim=-1),
        alpha.amax(dim=-1),
        (primary["rho"] - secondary["rho"]).square().mean(dim=-1).sqrt(),
        torch.log1p(
            torch.linalg.vector_norm(primary["innovation"].flatten(1), dim=-1) / output_norm
        ),
        torch.log1p(
            torch.linalg.vector_norm(
                (primary["innovation"] - secondary["innovation"]).flatten(1), dim=-1
            )
            / output_norm
        ),
        torch.log1p(primary_correction_norm / projected_norm),
        torch.log1p(secondary_correction_norm / projected_norm),
        torch.log1p(
            torch.linalg.vector_norm(primary["projected"] - secondary["projected"], dim=-1)
            / projected_norm
        ),
        correction_cosine,
    )
    return torch.stack(columns, dim=-1)


def _candidate_error_ratio(
    group: dict[str, Any], *, test: torch.Tensor, projected: torch.Tensor
) -> torch.Tensor:
    full = group["full_projected"].index_select(0, test).float()
    selected = group["selected_projected"].index_select(0, test).float()
    baseline = torch.linalg.vector_norm(full - selected, dim=-1).clamp_min(1.0e-12)
    return torch.linalg.vector_norm(full - projected.float(), dim=-1) / baseline


def _canary_observations(
    group: dict[str, Any],
    *,
    train: torch.Tensor,
    test: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    primary = _state_candidate(
        group, feature_name="f3_temporal", train=train, test=test, ridge=ridge
    )
    secondary = _state_candidate(
        group, feature_name="f1_selected_state", train=train, test=test, ridge=ridge
    )
    diagnostics = _canary_runtime_features(
        group, test=test, primary=primary, secondary=secondary
    )
    ratios = _candidate_error_ratio(group, test=test, projected=primary["projected"])
    return diagnostics, ratios


def _nested_canary_report(group: dict[str, Any], *, ridge: float) -> dict[str, Any]:
    metadata = group["metadata"]
    outer_folds = _folds(metadata).get("unseen_prompt", [])
    if len(outer_folds) < 3:
        return {"decision": "insufficient_prompt_folds", "folds": []}

    fold_reports = []
    all_scores = []
    all_ratios = []
    strict_masks = []
    margin_masks = []
    for outer_name, outer_train, outer_test in outer_folds:
        inner_prompts: dict[str, list[int]] = {}
        for row in outer_train.tolist():
            inner_prompts.setdefault(str(metadata[row]["prompt_id"]), []).append(row)
        if len(inner_prompts) < 2:
            continue

        calibration_diagnostics = []
        calibration_ratios = []
        calibration_prompt_ids = []
        outer_train_set = set(outer_train.tolist())
        for prompt_id, held_rows in sorted(inner_prompts.items()):
            inner_test = _indices(held_rows)
            inner_train = _indices(sorted(outer_train_set - set(held_rows)))
            diagnostics, ratios = _canary_observations(
                group, train=inner_train, test=inner_test, ridge=ridge
            )
            calibration_diagnostics.append(diagnostics)
            calibration_ratios.append(ratios)
            calibration_prompt_ids.extend([prompt_id] * len(held_rows))
        calibration_x = torch.cat(calibration_diagnostics, dim=0)
        calibration_ratio = torch.cat(calibration_ratios, dim=0)

        calibration_risk_oof = torch.empty_like(calibration_ratio)
        for prompt_id in sorted(inner_prompts):
            held = torch.tensor(
                [item == prompt_id for item in calibration_prompt_ids], dtype=torch.bool
            )
            risk_model = fit_ridge(
                calibration_x[~held],
                calibration_ratio[~held].log().unsqueeze(-1),
                ridge=ridge,
            )
            calibration_risk_oof[held] = predict_ridge(
                risk_model, calibration_x[held]
            ).squeeze(-1)

        final_risk_model = fit_ridge(
            calibration_x,
            calibration_ratio.log().unsqueeze(-1),
            ridge=ridge,
        )
        test_x, test_ratio = _canary_observations(
            group, train=outer_train, test=outer_test, ridge=ridge
        )
        test_score = predict_ridge(final_risk_model, test_x).squeeze(-1)
        strict_calibration = calibrate_canary_threshold(
            calibration_risk_oof, calibration_ratio, unsafe_limit=1.0
        )
        margin_calibration = calibrate_canary_threshold(
            calibration_risk_oof, calibration_ratio, unsafe_limit=0.95
        )
        strict = canary_gate_report(
            test_score,
            test_ratio,
            threshold=float(strict_calibration["threshold"]),
            unsafe_limit=1.0,
        )
        margin = canary_gate_report(
            test_score,
            test_ratio,
            threshold=float(margin_calibration["threshold"]),
            unsafe_limit=1.0,
        )
        fold_reports.append(
            {
                "fold": outer_name,
                "test_rows": int(outer_test.numel()),
                "candidate_mean_error_ratio": float(test_ratio.mean().item()),
                "candidate_worst_error_ratio": float(test_ratio.max().item()),
                "candidate_regressing_rows": int((test_ratio > 1.0).sum().item()),
                "calibration_risk_auc": binary_roc_auc(
                    calibration_risk_oof, calibration_ratio > 1.0
                ),
                "test_risk_auc": binary_roc_auc(test_score, test_ratio > 1.0),
                "strict_calibration": strict_calibration,
                "margin_calibration": margin_calibration,
                "strict_gate": strict,
                "margin_gate": margin,
            }
        )
        all_scores.append(test_score)
        all_ratios.append(test_ratio)
        strict_masks.append(test_score < float(strict_calibration["threshold"]))
        margin_masks.append(test_score < float(margin_calibration["threshold"]))

    ratios = torch.cat(all_ratios)
    scores = torch.cat(all_scores)

    def aggregate(masks: list[torch.Tensor]) -> dict[str, Any]:
        accepted = torch.cat(masks)
        accepted_ratios = ratios[accepted]
        return {
            "rows": int(ratios.numel()),
            "accepted_rows": int(accepted.sum().item()),
            "accepted_unsafe_rows": int((accepted_ratios > 1.0).sum().item()),
            "coverage": float(accepted.float().mean().item()),
            "mean_accepted_error_ratio": (
                float(accepted_ratios.mean().item()) if accepted_ratios.numel() else None
            ),
            "worst_accepted_error_ratio": (
                float(accepted_ratios.max().item()) if accepted_ratios.numel() else None
            ),
        }

    strict = aggregate(strict_masks)
    margin = aggregate(margin_masks)
    calibration_failures_present = all(
        int(fold["strict_calibration"]["calibration_unsafe_rows"]) > 0
        for fold in fold_reports
    )
    promotable = (
        calibration_failures_present
        and int(strict["accepted_unsafe_rows"]) == 0
        and float(strict["coverage"]) >= 0.25
    )
    return {
        "decision": (
            "continue_runtime_cost_gate" if promotable else "stop_exact_canary_predictor"
        ),
        "feature_contract": (
            "risk score uses only F1/F3 feature distance, predicted omitted mass, "
            "predicted innovation, correction magnitude, and predictor disagreement"
        ),
        "risk_feature_names": list(_CANARY_FEATURES),
        "calibration_contract": (
            "outer leave-one-prompt-out with inner prompt-held-out state predictions and "
            "prompt-held-out risk calibration"
        ),
        "candidate": {
            "rows": int(ratios.numel()),
            "mean_error_ratio": float(ratios.mean().item()),
            "worst_error_ratio": float(ratios.max().item()),
            "regressing_rows": int((ratios > 1.0).sum().item()),
            "oracle_safe_coverage": float((ratios <= 1.0).float().mean().item()),
            "risk_auc": binary_roc_auc(scores, ratios > 1.0),
        },
        "strict_gate": strict,
        "margin_gate": margin,
        "folds": fold_reports,
    }


def _analyze_fold(
    group: dict[str, Any],
    *,
    fold_name: str,
    train: torch.Tensor,
    test: torch.Tensor,
    ridge: float,
    ranks: list[int],
) -> dict[str, Any]:
    targets = {
        "post_wo_residual": group["residual"],
        "omitted_log_mass_ratio": group["rho"],
        "omitted_mass": group["alpha"],
        "omitted_output": group["omitted_output"],
        "value_innovation": group["innovation"],
        "projected_value_innovation": group["innovation_projected"],
    }
    rank = {
        name: rank_report(value.index_select(0, train), value.index_select(0, test), ranks=ranks)
        for name, value in targets.items()
    }
    predictors: dict[str, Any] = {}
    for feature_name, feature in group["features"].items():
        x_train = feature.index_select(0, train)
        x_test = feature.index_select(0, test)
        residual_train = group["residual"].index_select(0, train)
        residual_test = group["residual"].index_select(0, test)
        direct_model = fit_ridge(x_train, residual_train, ridge=ridge)
        direct = predict_ridge(direct_model, x_test)
        predictors[feature_name] = {
            "feature_width": int(feature.shape[1]),
            "train_canonical_correlations": canonical_correlations(
                x_train, residual_train, count=min(8, int(x_train.shape[0]))
            ),
            "direct_post_wo_residual": correction_report(direct, residual_test),
            "omitted_state": _state_prediction_report(
                features_train=x_train,
                features_test=x_test,
                rho_train=group["rho"].index_select(0, train),
                rho_test=group["rho"].index_select(0, test),
                innovation_train=group["innovation"].index_select(0, train),
                innovation_test=group["innovation"].index_select(0, test),
                selected_output_test=group["selected_output"].index_select(0, test),
                selected_projected_test=group["selected_projected"].index_select(0, test),
                full_projected_test=group["full_projected"].index_select(0, test),
                o_proj_weight=group["o_proj_weight"],
                q_heads=group["q_heads"],
                head_dim=group["head_dim"],
                ridge=ridge,
            ),
        }
    return {
        "fold": fold_name,
        "samples_train": int(train.numel()),
        "samples_test": int(test.numel()),
        "rank": rank,
        "predictors": predictors,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _summarize_split(folds: list[dict[str, Any]], *, rank: int) -> dict[str, Any]:
    feature_names = sorted(folds[0]["predictors"])
    predictor_summary: dict[str, Any] = {}
    for feature in feature_names:
        direct_reduction = [
            float(fold["predictors"][feature]["direct_post_wo_residual"]["relative_l2_reduction"])
            for fold in folds
        ]
        direct_energy = [
            float(fold["predictors"][feature]["direct_post_wo_residual"]["predictable_energy_fraction"])
            for fold in folds
        ]
        state_reduction = [
            float(fold["predictors"][feature]["omitted_state"]["post_wo_error_reduction_vs_hard_drop"])
            for fold in folds
        ]
        state_p95 = [
            float(fold["predictors"][feature]["omitted_state"]["row_error_ratio_p95"])
            for fold in folds
        ]
        predictor_summary[feature] = {
            "direct_error_reduction_mean": _mean(direct_reduction),
            "direct_error_reduction_min": min(direct_reduction),
            "direct_predictable_energy_mean": _mean(direct_energy),
            "state_error_reduction_mean": _mean(state_reduction),
            "state_error_reduction_min": min(state_reduction),
            "state_row_error_ratio_p95_worst": max(state_p95),
        }
    rank_energy = []
    for fold in folds:
        projections = fold["rank"]["post_wo_residual"]["heldout_projection"]
        row = next((item for item in projections if int(item["rank"]) == rank), projections[-1])
        rank_energy.append(float(row["heldout_representable_energy"]))
    return {
        "folds": len(folds),
        "rank_heldout_representable_energy_mean": _mean(rank_energy),
        "rank_heldout_representable_energy_min": min(rank_energy),
        "predictors": predictor_summary,
    }


def _analyze_group(
    captures: list[dict[str, Any]],
    *,
    ridge: float,
    ranks: list[int],
    decision_rank: int,
    canary: bool,
) -> dict[str, Any]:
    metadata: list[dict[str, Any]] = []
    features: dict[str, list[torch.Tensor]] = {}
    tensor_names = (
        "residual",
        "rho",
        "alpha",
        "omitted_output",
        "innovation",
        "innovation_projected",
        "selected_output",
        "full_projected",
        "selected_projected",
    )
    tensors: dict[str, list[torch.Tensor]] = {name: [] for name in tensor_names}
    baseline = []
    reference_weight = captures[0]["o_proj_weight"]
    reference_shape = (captures[0]["q_heads"], captures[0]["head_dim"])
    for capture in captures:
        if (capture["q_heads"], capture["head_dim"]) != reference_shape:
            raise ValueError("all captures in a group must share the same attention shape")
        torch.testing.assert_close(capture["o_proj_weight"], reference_weight)
        row_count = int(capture["residual"].shape[0])
        metadata.extend(
            {
                "prompt_id": capture["prompt_id"],
                "bucket": capture["bucket"],
                "row": row,
            }
            for row in range(row_count)
        )
        for name, value in capture["features"].items():
            features.setdefault(name, []).append(value)
        for name in tensor_names:
            tensors[name].append(capture[name])
        baseline.append(float(capture["baseline_post_wo_relative_l2"]))
    group = {
        "metadata": metadata,
        "features": {name: torch.cat(values, dim=0) for name, values in features.items()},
        **{name: torch.cat(values, dim=0) for name, values in tensors.items()},
        "o_proj_weight": reference_weight,
        "q_heads": captures[0]["q_heads"],
        "head_dim": captures[0]["head_dim"],
    }
    split_rows: dict[str, Any] = {}
    split_summary: dict[str, Any] = {}
    for split_name, definitions in _folds(metadata).items():
        folds = [
            _analyze_fold(
                group,
                fold_name=fold_name,
                train=train,
                test=test,
                ridge=ridge,
                ranks=ranks,
            )
            for fold_name, train, test in definitions
        ]
        split_rows[split_name] = folds
        split_summary[split_name] = _summarize_split(folds, rank=decision_rank)

    unseen = split_summary.get("unseen_prompt")
    decision = "insufficient_cross_prompt_evidence"
    best_feature = None
    if unseen:
        best_feature, best = max(
            unseen["predictors"].items(),
            key=lambda item: float(item[1]["state_error_reduction_mean"]),
        )
        rank_ok = float(unseen["rank_heldout_representable_energy_mean"]) >= 0.90
        state_ok = (
            float(best["state_error_reduction_mean"]) >= 0.50
            and float(best["state_error_reduction_min"]) >= 0.25
            and float(best["state_row_error_ratio_p95_worst"]) <= 1.0
        )
        direct_ok = (
            float(best["direct_predictable_energy_mean"]) >= 0.70
            and float(best["direct_error_reduction_mean"]) >= 0.50
        )
        if rank_ok and state_ok:
            decision = "continue_omitted_state_runtime_research"
        elif rank_ok and direct_ok:
            decision = "direct_residual_only_diagnostic_not_attention_state_ready"
        else:
            temporal = split_summary.get("same_prompt_future", {}).get("predictors", {}).get(best_feature, {})
            if float(temporal.get("state_error_reduction_mean", -1.0)) >= 0.50:
                decision = "temporal_canary_only_candidate"
            else:
                decision = "stop_training_free_predictor_for_this_cell"
    result = {
        "samples": len(metadata),
        "prompts": len({row["prompt_id"] for row in metadata}),
        "buckets": sorted({row["bucket"] for row in metadata}),
        "baseline_post_wo_relative_l2_mean": _mean(baseline),
        "split_summary": split_summary,
        "folds": split_rows,
        "decision": decision,
        "best_state_feature_on_unseen_prompt": best_feature,
    }
    if canary:
        result["exact_canary"] = _nested_canary_report(group, ridge=ridge)
    return result


def _synthetic_prompts(
    *,
    device: torch.device,
    prompt_count: int,
    query_rows: int,
) -> list[tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    result = []
    weight_generator = torch.Generator(device=device.type).manual_seed(97)
    weight = torch.randn(32, 32, generator=weight_generator, device=device) / math.sqrt(32)
    for prompt in range(prompt_count):
        generator = torch.Generator(device=device.type).manual_seed(101 + prompt)
        q = torch.randn(query_rows, 4, 8, generator=generator, device=device)
        k = torch.randn(2, 64, 8, generator=generator, device=device)
        v = 0.7 * k + 0.3 * torch.randn(2, 64, 8, generator=generator, device=device)
        lengths = torch.arange(64 - query_rows + 1, 65, device=device)
        result.append((f"synthetic_{prompt}", "synthetic", q, k, v, lengths, weight))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--buckets", default="chat_instruction,json_tool,needle_rag,code")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-prompts-per-bucket", type=int, default=1)
    parser.add_argument("--layers", default="14,26,27")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--query-rows", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--budgets", default="4,8")
    parser.add_argument("--sink-blocks", type=int, default=1)
    parser.add_argument("--recent-blocks", type=int, default=1)
    parser.add_argument("--hash-width", type=int, default=32)
    parser.add_argument("--feature-seed", type=int, default=37)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--ranks", default="1,2,4,8,16,32")
    parser.add_argument("--decision-rank", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    layers = set(_parse_ints(args.layers, allow_zero=True))
    budgets = _parse_ints(args.budgets)
    ranks = _parse_ints(args.ranks)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    if args.synthetic:
        sources = _synthetic_prompts(
            device=device, prompt_count=max(2, args.max_prompts), query_rows=args.query_rows
        )
        for prompt_id, bucket, q, k, v, lengths, weight in sources:
            for budget in budgets:
                capture = _capture_group(
                    q=q,
                    k=k,
                    v=v,
                    valid_lengths=lengths,
                    o_proj_weight=weight,
                    block_size=min(args.block_size, 8),
                    budget=min(budget, 2),
                    sink_blocks=args.sink_blocks,
                    recent_blocks=args.recent_blocks,
                    hash_width=args.hash_width,
                    feature_seed=args.feature_seed,
                )
                capture.update(prompt_id=prompt_id, bucket=bucket, layer=-1)
                groups.setdefault((-1, min(budget, 2)), []).append(capture)
    else:
        prompts = _load_prompts(args)
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
        for prompt in prompts:
            captured, handles = _capture_attention_inputs(model, layers)
            try:
                tokens = tokenizer(
                    prompt["text"], return_tensors="pt", truncation=True, max_length=args.max_seq
                ).to(device)
                with torch.inference_mode():
                    model(**tokens, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            for raw in captured:
                with torch.no_grad():
                    q_all, k_all, v_all, meta = _shape_qkv(raw, apply_rope=True)
                if not meta.get("rope_applied"):
                    raise RuntimeError(f"RoPE capture failed: {meta.get('rope_error')}")
                seq_len = int(q_all.shape[1])
                query_rows = min(args.query_rows, seq_len)
                group_size = int(meta["q_per_kv"])
                q = q_all[0, -query_rows:].float()
                k = k_all[0, :, ::group_size, :].permute(1, 0, 2).contiguous().float()
                v = v_all[0, :, ::group_size, :].permute(1, 0, 2).contiguous().float()
                lengths = torch.arange(seq_len - query_rows + 1, seq_len + 1, device=device)
                for budget in budgets:
                    capture = _capture_group(
                        q=q,
                        k=k,
                        v=v,
                        valid_lengths=lengths,
                        o_proj_weight=raw.module.o_proj.weight.detach().float(),
                        block_size=args.block_size,
                        budget=budget,
                        sink_blocks=args.sink_blocks,
                        recent_blocks=args.recent_blocks,
                        hash_width=args.hash_width,
                        feature_seed=args.feature_seed + int(raw.layer_id) * 101,
                    )
                    capture.update(
                        prompt_id=prompt["id"], bucket=prompt["bucket"], layer=int(raw.layer_id)
                    )
                    groups.setdefault((int(raw.layer_id), budget), []).append(capture)
            del captured, tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    reports = []
    for (layer, budget), captures in sorted(groups.items()):
        report = _analyze_group(
            captures,
            ridge=args.ridge,
            ranks=ranks,
            decision_rank=args.decision_rank,
            canary=args.canary,
        )
        reports.append(
            {
                "layer": layer,
                "exact_middle_blocks_per_kv_group": budget,
                **report,
            }
        )
    payload = {
        "schema": "streamattn.adaptive_residual_predictability.v1",
        "model": "synthetic" if args.synthetic else args.model,
        "target_contract": (
            "primary target predicts omitted log-mass ratio and omitted value innovation "
            "(o_U - o_A), then merges with the selected state using positive online-softmax weights; "
            "direct post-o_proj residual is diagnostic only"
        ),
        "route_contract": (
            "exact QK block-max selection is an offline favorable upper bound in this gate; "
            "a positive result must later survive the deployed support-summary selector"
        ),
        "feature_contract": (
            "predictor targets never enter its features; block-moment features assume persistent "
            "summaries and must not read omitted KV tokens during deployed decode"
        ),
        "reports": reports,
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
