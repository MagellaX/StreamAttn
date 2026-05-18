import torch

from benchmarks.profile_gate0_candidate_filters import (
    _candidate_metrics,
    _parse_str_values,
    _projection_matrix,
    _scan_blocks,
)


def test_parse_str_values_supports_commas_and_spaces():
    assert _parse_str_values(["projection_random,projection_hadamard", " certified_centroid "]) == [
        "projection_random",
        "projection_hadamard",
        "certified_centroid",
    ]


def test_scan_blocks_middle_only_excludes_sink_and_recent():
    blocks = _scan_blocks("middle_only", num_blocks=8, sink_blocks=2, recent_blocks=2)
    assert blocks == [2, 3, 4, 5]


def test_projection_matrix_shapes():
    random_projection = _projection_matrix(
        "random",
        dim=16,
        rank=4,
        seed=0,
        device=torch.device("cpu"),
    )
    hadamard_projection = _projection_matrix(
        "hadamard",
        dim=16,
        rank=4,
        seed=0,
        device=torch.device("cpu"),
    )

    assert random_projection.shape == (4, 16)
    assert hadamard_projection.shape == (4, 16)


def test_candidate_metrics_reports_false_skip_rate():
    scores = torch.tensor([[[[-10.0, -10.0, 10.0, 10.0]]]])
    thresholds = torch.zeros_like(scores)
    has_state = torch.ones_like(scores, dtype=torch.bool)
    actual_skip = torch.tensor([[[[True, False, True, False]]]])
    block_lengths = torch.ones(4, dtype=torch.long)

    metrics = _candidate_metrics(
        scores=scores,
        actual_skip=actual_skip,
        has_state=has_state,
        thresholds=thresholds,
        block_lengths=block_lengths,
        filter_margin=0.0,
        sink_blocks=1,
        recent_blocks=1,
    )

    assert metrics["predicted_skip_count"] == 2
    assert metrics["recovered_skip_count"] == 1
    assert metrics["false_skip_count"] == 1
    assert metrics["actual_skip_recovery"] == 0.5
    assert metrics["false_skip_rate"] == 0.5
