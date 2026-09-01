from benchmarks.profile_sm80_d128_phased_gate import _paired_summary


def test_paired_summary_tracks_strict_wins_and_tail():
    summary = _paired_summary(
        [
            {"speedup_vs_flashinfer": 1.03},
            {"speedup_vs_flashinfer": 1.01},
            {"speedup_vs_flashinfer": 0.99},
        ]
    )

    assert summary["trial_count"] == 3
    assert summary["wins"] == 2
    assert summary["speedup_min"] == 0.99
    assert summary["speedup_median"] == 1.01
