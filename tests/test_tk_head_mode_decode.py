import torch

from benchmarks.profile_tk_head_mode_decode import _seed_counts, _stats_summary


def test_seed_counts_recent_first_overlap_free() -> None:
    counts = _seed_counts(
        128,
        block_size=16,
        sink_blocks=1,
        recent_blocks=2,
        middle_seed_blocks=1,
        block_order="recent_first",
        device=torch.device("cpu"),
    )

    assert counts["seed_tokens"] == 64
    assert counts["seed_blocks"] == 4
    assert counts["skipped_tokens"] == 64
    assert counts["total_blocks"] == 8


def test_stats_summary_checks_exact_and_seed_ranges() -> None:
    stats = torch.tensor(
        [
            [
                [0, 0, 128, 0, 256, 1],
                [1, 0, 64, 64, 128, 1],
                [0, 1, 128, 0, 256, 1],
            ]
        ],
        dtype=torch.int64,
    )

    summary = _stats_summary(stats, seed_heads=[1], kv_len=128)

    assert summary["all_tk_helpers_reached"] is True
    assert summary["all_exact_heads_full_range"] is True
    assert summary["seed_scheduled_tokens_max"] == 64
    assert summary["seed_skipped_tokens_min"] == 64
    assert summary["exact_skipped_tokens_max"] == 0
