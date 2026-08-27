"""Reference analysis for conditional attention-residual predictability.

The adaptive engine can only exploit a compact omitted-context representation
when that representation is predictable from information available without
scanning the omitted KV tail.  This module keeps the statistical gate separate
from any runtime kernel implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class RidgeModel:
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    target_mean: torch.Tensor
    weights: torch.Tensor


def _matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.dim() != 2:
        raise ValueError(f"{name} must be a matrix")
    if value.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    return value.float()


def signed_hash_projection(
    value: torch.Tensor,
    *,
    width: int,
    seed: int,
) -> torch.Tensor:
    """Project the final dimension with a deterministic signed feature hash."""

    if width <= 0:
        raise ValueError("width must be positive")
    flat = value.float().flatten(start_dim=1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    columns = int(flat.shape[1])
    buckets = torch.randint(width, (columns,), generator=generator)
    signs = torch.randint(0, 2, (columns,), generator=generator).float().mul_(2).sub_(1)
    buckets = buckets.to(flat.device)
    signs = signs.to(flat.device)
    output = torch.zeros(
        (int(flat.shape[0]), width), device=flat.device, dtype=torch.float32
    )
    output.scatter_add_(1, buckets.unsqueeze(0).expand_as(flat), flat * signs)
    return output / max(columns / width, 1.0) ** 0.5


def fit_ridge(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float,
) -> RidgeModel:
    """Fit standardized multi-output ridge regression.

    The dual solve is used when samples are fewer than features, which is the
    common regime for expensive long-context model captures.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    x = _matrix(features, name="features")
    y = _matrix(targets, name="targets")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must have the same row count")
    x_mean = x.mean(dim=0)
    x_scale = x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    y_mean = y.mean(dim=0)
    xs = (x - x_mean) / x_scale
    yc = y - y_mean
    samples, feature_count = map(int, xs.shape)
    if samples <= feature_count:
        gram = xs @ xs.T
        gram.diagonal().add_(ridge)
        dual = torch.linalg.solve(gram, yc)
        weights = xs.T @ dual
    else:
        gram = xs.T @ xs
        gram.diagonal().add_(ridge)
        weights = torch.linalg.solve(gram, xs.T @ yc)
    return RidgeModel(x_mean, x_scale, y_mean, weights)


def predict_ridge(model: RidgeModel, features: torch.Tensor) -> torch.Tensor:
    x = _matrix(features, name="features")
    if x.shape[1] != model.feature_mean.numel():
        raise ValueError("feature width does not match the fitted model")
    return ((x - model.feature_mean) / model.feature_scale) @ model.weights + model.target_mean


def merge_selected_with_omitted_innovation(
    selected_output: torch.Tensor,
    log_mass_ratio: torch.Tensor,
    value_innovation: torch.Tensor,
) -> torch.Tensor:
    """Merge selected and omitted online-softmax states.

    With ``rho = log(Z_omitted / Z_selected)`` and
    ``innovation = o_omitted - o_selected``, exact attention factors as
    ``o = o_selected + sigmoid(rho) * innovation``. Predicting the innovation
    is essential: a stable omitted-output mean is not evidence that the
    correction itself is predictable.
    """

    if selected_output.shape != value_innovation.shape:
        raise ValueError("selected_output and value_innovation shapes must match")
    if log_mass_ratio.shape != selected_output.shape[:-1]:
        raise ValueError(
            "log_mass_ratio must match selected_output without its final dimension"
        )
    return selected_output.float() + torch.sigmoid(log_mass_ratio.float()).unsqueeze(
        -1
    ) * value_innovation.float()


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    candidate = candidate.float()
    reference = reference.float()
    return float(
        (
            torch.linalg.vector_norm(candidate - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
        ).item()
    )


def correction_report(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = _matrix(predicted, name="predicted")
    target = _matrix(target, name="target")
    if predicted.shape != target.shape:
        raise ValueError("predicted and target shapes must match")
    residual = target - predicted
    row_baseline = torch.linalg.vector_norm(target, dim=-1).clamp_min(1.0e-12)
    row_error = torch.linalg.vector_norm(residual, dim=-1)
    baseline_norm = torch.linalg.vector_norm(target).clamp_min(1.0e-12)
    error_norm = torch.linalg.vector_norm(residual)
    energy = target.square().sum().clamp_min(1.0e-12)
    return {
        "relative_l2": float((error_norm / baseline_norm).item()),
        "relative_l2_reduction": float((1.0 - error_norm / baseline_norm).item()),
        "predictable_energy_fraction": float((1.0 - residual.square().sum() / energy).item()),
        "row_relative_l2_p50": float(torch.quantile(row_error / row_baseline, 0.50).item()),
        "row_relative_l2_p95": float(torch.quantile(row_error / row_baseline, 0.95).item()),
        "row_relative_l2_p99": float(torch.quantile(row_error / row_baseline, 0.99).item()),
    }


def calibrate_canary_threshold(
    risk_scores: torch.Tensor,
    error_ratios: torch.Tensor,
    *,
    unsafe_limit: float = 1.0,
) -> dict[str, float | int]:
    """Choose the largest monotone acceptance region with no known unsafe row.

    Lower scores must indicate lower risk. Rows are accepted only when their
    score is strictly below the least-risky calibration failure. Exact error
    labels are required for offline calibration, never for applying the gate.
    """

    scores = risk_scores.float().flatten()
    ratios = error_ratios.float().flatten()
    if scores.shape != ratios.shape or scores.numel() == 0:
        raise ValueError("risk_scores and error_ratios must be non-empty and aligned")
    if not bool(torch.isfinite(scores).all() and torch.isfinite(ratios).all()):
        raise ValueError("risk_scores and error_ratios must be finite")
    unsafe = ratios > unsafe_limit
    if bool(unsafe.any()):
        threshold = float(scores[unsafe].min().item())
        accepted = scores < threshold
    else:
        threshold = float("inf")
        accepted = torch.ones_like(unsafe)
    accepted_unsafe = accepted & unsafe
    return {
        "threshold": threshold,
        "unsafe_limit": float(unsafe_limit),
        "calibration_rows": int(scores.numel()),
        "calibration_unsafe_rows": int(unsafe.sum().item()),
        "accepted_rows": int(accepted.sum().item()),
        "accepted_unsafe_rows": int(accepted_unsafe.sum().item()),
        "coverage": float(accepted.float().mean().item()),
    }


def canary_gate_report(
    risk_scores: torch.Tensor,
    error_ratios: torch.Tensor,
    *,
    threshold: float,
    unsafe_limit: float = 1.0,
) -> dict[str, float | int | None]:
    scores = risk_scores.float().flatten()
    ratios = error_ratios.float().flatten()
    if scores.shape != ratios.shape or scores.numel() == 0:
        raise ValueError("risk_scores and error_ratios must be non-empty and aligned")
    accepted = scores < threshold
    accepted_ratios = ratios[accepted]
    accepted_unsafe = accepted_ratios > unsafe_limit
    return {
        "rows": int(scores.numel()),
        "accepted_rows": int(accepted.sum().item()),
        "accepted_unsafe_rows": int(accepted_unsafe.sum().item()),
        "coverage": float(accepted.float().mean().item()),
        "mean_accepted_error_ratio": (
            float(accepted_ratios.mean().item()) if accepted_ratios.numel() else None
        ),
        "worst_accepted_error_ratio": (
            float(accepted_ratios.max().item()) if accepted_ratios.numel() else None
        ),
        "mean_accepted_error_reduction": (
            float((1.0 - accepted_ratios.mean()).item())
            if accepted_ratios.numel()
            else None
        ),
    }


def binary_roc_auc(risk_scores: torch.Tensor, unsafe: torch.Tensor) -> float | None:
    """Compute pairwise ROC AUC without an external statistics dependency."""

    scores = risk_scores.float().flatten()
    labels = unsafe.bool().flatten()
    if scores.shape != labels.shape or scores.numel() == 0:
        raise ValueError("risk_scores and unsafe must be non-empty and aligned")
    positive = scores[labels]
    negative = scores[~labels]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(
        ((comparisons > 0).float().mean() + 0.5 * (comparisons == 0).float().mean()).item()
    )


def _energy_rank(singular_values: torch.Tensor, fraction: float) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("energy fractions must lie in (0, 1]")
    energy = singular_values.square()
    if float(energy.sum().item()) == 0.0:
        return 0
    cumulative = energy.cumsum(dim=0) / energy.sum()
    return int(torch.searchsorted(cumulative, torch.tensor(fraction, device=cumulative.device)).item()) + 1


def rank_report(
    train: torch.Tensor,
    test: torch.Tensor,
    *,
    ranks: Iterable[int] = (1, 2, 4, 8, 16, 32),
    energy_fractions: Iterable[float] = (0.8, 0.9, 0.95, 0.99),
) -> dict[str, object]:
    """Report train spectrum and held-out oracle projection quality."""

    train = _matrix(train, name="train")
    test = _matrix(test, name="test")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test widths must match")
    mean = train.mean(dim=0)
    centered_train = train - mean
    centered_test = test - mean
    _u, singular, vh = torch.linalg.svd(centered_train, full_matrices=False)
    eigen = singular.square()
    total = eigen.sum().clamp_min(1.0e-20)
    stable_rank = total / singular.square().max().clamp_min(1.0e-20)
    participation = total.square() / eigen.square().sum().clamp_min(1.0e-20)
    baseline = centered_test.square().sum().clamp_min(1.0e-20)
    projections = []
    max_rank = int(vh.shape[0])
    for requested in ranks:
        rank = min(int(requested), max_rank)
        basis = vh[:rank]
        projected = (centered_test @ basis.T) @ basis
        error = centered_test - projected
        projections.append(
            {
                "rank": int(requested),
                "effective_rank": rank,
                "heldout_representable_energy": float(
                    (1.0 - error.square().sum() / baseline).item()
                ),
                "heldout_relative_projection_error": float(
                    (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(centered_test).clamp_min(1.0e-12)).item()
                ),
            }
        )
    return {
        "samples_train": int(train.shape[0]),
        "samples_test": int(test.shape[0]),
        "target_width": int(train.shape[1]),
        "stable_rank": float(stable_rank.item()),
        "participation_ratio": float(participation.item()),
        "energy_rank": {
            str(fraction): _energy_rank(singular, float(fraction))
            for fraction in energy_fractions
        },
        "heldout_projection": projections,
    }


def canonical_correlations(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    count: int = 16,
) -> list[float]:
    """Return unregularized train-set canonical correlations via QR bases.

    These values are descriptive only. Held-out ridge prediction is the actual
    promotion signal because high-dimensional CCA can overfit small captures.
    """

    x = _matrix(features, name="features")
    y = _matrix(targets, name="targets")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must have the same row count")
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    qx, _ = torch.linalg.qr(x, mode="reduced")
    qy, _ = torch.linalg.qr(y, mode="reduced")
    values = torch.linalg.svdvals(qx.T @ qy)[:count].clamp(0.0, 1.0)
    return [float(value.item()) for value in values]
