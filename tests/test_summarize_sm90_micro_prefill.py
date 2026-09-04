import json

from benchmarks.summarize_sm90_micro_prefill import summarize


def test_summary_preserves_losses_and_family_boundaries(tmp_path):
    rows = [
        {
            "batch": 1,
            "query_len": 4,
            "kv_len": 4096,
            "group_size": 8,
            "head_dim": 128,
            "winner": "transposed",
            "strict_correct": True,
            "promotion_pass": True,
            "median_speedup_vs_flash": 1.5,
        },
        {
            "batch": 1,
            "query_len": 64,
            "kv_len": 4096,
            "group_size": 8,
            "head_dim": 128,
            "winner": "natural",
            "strict_correct": True,
            "promotion_pass": False,
            "median_speedup_vs_flash": 0.8,
        },
    ]
    artifact = tmp_path / "matrix.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "streamattn.sm90_micro_prefill_canary.v2",
                "provider": "test",
                "device": "H100",
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    result = summarize(artifact)

    assert result["overall"]["cells"] == 2
    assert result["overall"]["paired_flash_gate_cells"] == 1
    assert result["by_query_length"]["4"]["transposed_winners"] == 1
    assert result["by_query_length"]["64"]["natural_winners"] == 1
    assert result["unresolved_cells"] == [
        {
            "batch": 1,
            "query_len": 64,
            "kv_len": 4096,
            "group_size": 8,
            "head_dim": 128,
            "winner": "natural",
            "median_speedup_vs_flash": 0.8,
        }
    ]
