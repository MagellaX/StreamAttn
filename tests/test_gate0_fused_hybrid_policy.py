from benchmarks.build_gate0_fused_hybrid_policy import build_policy
from benchmarks.summarize_gate1_inline_projection_splitk_robustness import parse_budgets


def _row(
    *,
    prompt_type: str = "needle",
    layer_id: int = 8,
    trusted=None,
    speedup: float = 1.4,
    max_error: float = 0.005,
    mean_error: float = 0.0005,
    trusted_skip: float = 1.0,
):
    trusted = [7] if trusted is None else trusted
    dense_ms = 1.4
    fused_ms = dense_ms / speedup
    all_heads = list(range(14))
    exact_heads = [head for head in all_heads if head not in trusted]
    per_head_stats = []
    per_head_error = []
    for head in all_heads:
        is_trusted = head in trusted
        per_head_stats.append(
            {
                "head": head,
                "projection_skip_fraction": trusted_skip if is_trusted else 0.0,
                "pv_executed_fraction": 0.0 if is_trusted else 1.0,
            }
        )
        per_head_error.append(
            {
                "head": head,
                "max_abs_error": max_error if is_trusted else 0.0001,
                "mean_abs_error": mean_error if is_trusted else 0.00001,
            }
        )
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt_type": prompt_type,
        "layer_id": layer_id,
        "kv_len": 32768,
        "shape": {"heads": 14, "kv_len": 32768},
        "policy": {
            "aggressive_sparse_heads": [2, 3, 4, 6, 7, 8, 9, 11],
            "trusted_sparse_heads": trusted,
            "exact_heads": exact_heads,
        },
        "runtime": {
            "block_size": 32,
            "sink_blocks": 2,
            "recent_blocks": 2,
            "middle_seed_blocks": 8,
            "chunk_anchor_blocks": 0,
            "block_order": "recent_first",
            "num_chunks": 32,
            "seed_strategy": "recompute_seed",
            "filter_margin": 64.0,
            "error_budget": 0.01,
            "projection_kind": "random",
            "projection_dim": 8,
            "projection_seed": 1,
            "projection_metadata_dtype": "fp16",
            "splitk_workspace": "reuse",
        },
        "timing": {
            "dense_all_ms": dense_ms,
            "fused_hybrid_ms": fused_ms,
            "fused_hybrid_speedup_vs_dense_all": speedup,
        },
        "quality": {
            "fused_hybrid_error_vs_dense_all": {
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
            },
            "fused_hybrid_error_vs_dense_all_per_head": {"per_head": per_head_error},
            "fused_hybrid_stats": {
                "projection_skip_fraction": len(trusted) / 14,
                "pv_executed_fraction": 1.0 - (len(trusted) / 14),
            },
            "fused_hybrid_per_head_stats": {"per_head": per_head_stats},
        },
    }


def test_fused_hybrid_policy_picks_fastest_safe_group_per_prompt_layer():
    rows = [
        _row(trusted=[6], speedup=1.35, max_error=0.001, mean_error=0.0001),
        _row(trusted=[7], speedup=1.46, max_error=0.006, mean_error=0.0002),
        _row(trusted=[6, 7], speedup=1.5, max_error=0.012, mean_error=0.0003),
    ]

    payload = build_policy(rows, parse_budgets("moderate:1e-2:1e-3:1.15"))

    assert payload["summary"]["entries"] == 1
    entry = payload["entries"][0]
    assert entry["runtime"]["trusted_sparse_heads"] == [7]
    assert entry["quality"]["speedup_vs_dense_all"] == 1.46
    assert entry["runtime"]["head_modes"][7] == 0
    assert entry["runtime"]["head_modes"][6] == 1


def test_fused_hybrid_policy_reports_prompt_stable_groups():
    rows = [
        _row(prompt_type="needle", trusted=[7], speedup=1.45, max_error=0.006),
        _row(prompt_type="code", trusted=[7], speedup=1.42, max_error=0.005),
        _row(prompt_type="long_doc", trusted=[7], speedup=1.4, max_error=0.004),
        _row(prompt_type="code", trusted=[6], speedup=1.5, max_error=0.02),
    ]

    payload = build_policy(
        rows,
        parse_budgets("moderate:1e-2:1e-3:1.15"),
        min_stable_prompts=3,
    )

    assert payload["summary"]["entries"] == 3
    assert payload["summary"]["stable_entries"] == 1
    stable = payload["stable_entries"][0]
    assert stable["runtime"]["trusted_sparse_heads"] == [7]
    assert stable["robustness"]["prompt_count"] == 3
    assert stable["robustness"]["min_speedup_vs_dense_all"] == 1.4
    assert stable["robustness"]["max_abs_error_seen"] == 0.006


def test_fused_hybrid_policy_rejects_low_skip_trusted_heads():
    rows = [_row(trusted=[7], speedup=1.5, max_error=0.001, trusted_skip=0.1)]

    payload = build_policy(
        rows,
        parse_budgets("moderate:1e-2:1e-3:1.15"),
        min_trusted_skip_fraction=0.5,
    )

    assert payload["summary"]["entries"] == 0
    assert payload["summary"]["frontier_candidates"] == 0
