import argparse

import torch

from benchmarks.profile_gate0_summary_bounds import _real_tensor_cases
from benchmarks.summarize_gate0_summary_bounds import _add_ordering_metrics


def test_real_tensor_cases_support_per_head_pt_inputs(tmp_path):
    q = torch.randn(1, 1, 3, 8)
    k = torch.randn(1, 64, 3, 8)
    v = torch.randn(1, 64, 3, 8)
    q_path = tmp_path / "q.pt"
    k_path = tmp_path / "k.pt"
    v_path = tmp_path / "v.pt"
    torch.save({"post_rope_q": q}, q_path)
    torch.save({"post_rope_k": k}, k_path)
    torch.save({"v": v}, v_path)

    args = argparse.Namespace(
        q_path=str(q_path),
        k_path=str(k_path),
        v_path=str(v_path),
        tensor_format="pt",
        per_head=True,
        head_indices=[],
    )

    cases = list(_real_tensor_cases(args, device=torch.device("cpu"), dtype=torch.float32))

    assert len(cases) == 3
    for head_id, (q_h, k_h, v_h, extra) in enumerate(cases):
        assert q_h.shape == (1, 1, 1, 8)
        assert k_h.shape == (1, 64, 1, 8)
        assert v_h is not None and v_h.shape == (1, 64, 1, 8)
        assert extra == {"head_id": head_id, "real_case": "single_head"}


def test_ordering_metrics_compare_against_sequential_baseline():
    rows = [
        {
            "tensor_source": "synthetic",
            "tensor_space": "synthetic",
            "query_len": 1,
            "kv_len": 128,
            "heads": 2,
            "dim": 16,
            "block_size": 32,
            "pattern": "sliding_recent",
            "requested_active_fraction": 0.25,
            "num_summary_outliers": 0,
            "scan_backend": "torch",
            "blocks_per_program": 32,
            "block_order": "sequential",
            "predicted_skip_fraction": 0.0,
            "estimated_gate0_speedup_vs_gate1": 0.5,
        },
        {
            "tensor_source": "synthetic",
            "tensor_space": "synthetic",
            "query_len": 1,
            "kv_len": 128,
            "heads": 2,
            "dim": 16,
            "block_size": 32,
            "pattern": "sliding_recent",
            "requested_active_fraction": 0.25,
            "num_summary_outliers": 0,
            "scan_backend": "torch",
            "blocks_per_program": 32,
            "block_order": "recent_first",
            "predicted_skip_fraction": 0.75,
            "estimated_gate0_speedup_vs_gate1": 1.5,
        },
    ]

    _add_ordering_metrics(rows)

    assert rows[0]["ordering_gain"] == 0.0
    assert rows[0]["speedup_gain"] == 1.0
    assert rows[1]["ordering_gain"] == 0.75
    assert rows[1]["speedup_gain"] == 3.0
