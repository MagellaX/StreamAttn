import torch

from stream_attention.residual_predictability import (
    binary_roc_auc,
    calibrate_canary_threshold,
    canary_gate_report,
    canonical_correlations,
    correction_report,
    fit_ridge,
    merge_selected_with_omitted_innovation,
    predict_ridge,
    rank_report,
    signed_hash_projection,
)


def test_online_softmax_innovation_merge_matches_weighted_outputs() -> None:
    selected = torch.tensor([[[1.0, 3.0], [4.0, -2.0]]])
    omitted = torch.tensor([[[5.0, -1.0], [0.0, 6.0]]])
    selected_mass = torch.tensor([[2.0, 7.0]])
    omitted_mass = torch.tensor([[6.0, 1.0]])
    rho = torch.log(omitted_mass / selected_mass)
    merged = merge_selected_with_omitted_innovation(
        selected,
        rho,
        omitted - selected,
    )
    reference = (
        selected_mass[..., None] * selected + omitted_mass[..., None] * omitted
    ) / (selected_mass + omitted_mass)[..., None]
    torch.testing.assert_close(merged, reference)


def test_canary_threshold_rejects_calibration_failures_monotonically() -> None:
    scores = torch.tensor([-2.0, -1.0, 0.5, 1.5, 3.0])
    ratios = torch.tensor([0.4, 0.8, 0.7, 1.1, 1.3])
    calibration = calibrate_canary_threshold(scores, ratios)
    assert calibration["threshold"] == 1.5
    assert calibration["accepted_rows"] == 3
    assert calibration["accepted_unsafe_rows"] == 0
    report = canary_gate_report(scores, ratios, threshold=calibration["threshold"])
    assert abs(float(report["coverage"]) - 0.6) < 1.0e-6
    assert report["accepted_unsafe_rows"] == 0


def test_binary_roc_auc_handles_ties() -> None:
    scores = torch.tensor([0.0, 1.0, 1.0, 2.0])
    unsafe = torch.tensor([False, False, True, True])
    assert binary_roc_auc(scores, unsafe) == 0.875


def test_signed_hash_projection_is_deterministic() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    first = signed_hash_projection(value, width=5, seed=7)
    second = signed_hash_projection(value, width=5, seed=7)
    assert first.shape == (3, 5)
    torch.testing.assert_close(first, second)


def test_ridge_recovers_heldout_linear_mapping() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(80, 6, generator=generator)
    weight = torch.randn(6, 4, generator=generator)
    y = x @ weight + 0.25
    model = fit_ridge(x[:60], y[:60], ridge=1.0e-5)
    predicted = predict_ridge(model, x[60:])
    assert correction_report(predicted, y[60:])["relative_l2"] < 1.0e-4


def test_rank_report_recognizes_shared_rank_two_subspace() -> None:
    generator = torch.Generator().manual_seed(19)
    basis = torch.randn(2, 12, generator=generator)
    train = torch.randn(40, 2, generator=generator) @ basis
    test = torch.randn(20, 2, generator=generator) @ basis
    report = rank_report(train, test, ranks=(1, 2, 4))
    rank_two = next(row for row in report["heldout_projection"] if row["rank"] == 2)
    assert report["energy_rank"]["0.99"] <= 2
    assert rank_two["heldout_representable_energy"] > 0.9999


def test_rank_report_rejects_unseen_orthogonal_direction() -> None:
    train = torch.zeros(8, 4)
    train[:, 0] = torch.linspace(-1, 1, 8)
    test = torch.zeros(4, 4)
    test[:, 1] = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    report = rank_report(train, test, ranks=(1,))
    assert report["heldout_projection"][0]["heldout_representable_energy"] < 0.01


def test_canonical_correlations_detect_shared_signal() -> None:
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(32, 3, generator=generator)
    y = torch.cat([x[:, :2], x[:, :1] - x[:, 1:2]], dim=1)
    correlations = canonical_correlations(x, y, count=3)
    assert correlations[0] > 0.999
    assert correlations[1] > 0.999
