import sys
import types

import pytest
import torch


class _DummyFused:
    def __init__(self, num_heads, head_dim, dtype=None):
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, q, k, v, causal=True):  # pragma: no cover
        return q + k + v


def _create_dummy_fused(*args, **kwargs):  # pragma: no cover
    return _DummyFused(*args, **kwargs)


# Stub out the heavy Triton-dependent module before importing benchmark_suite
stub = types.ModuleType("fused_online_attention")
stub.FusedOnlineAttention = _DummyFused
stub.create_fused_online_attention = _create_dummy_fused
sys.modules.setdefault("stream_attention.core.fused_online_attention", stub)

from stream_attention.benchmarks.benchmark_suite import run_bench


def test_benchmark_output_keys_and_speedup():
    res = run_bench([4], batch_size=1, num_heads=2, head_dim=4, warmup=0, iters=1)
    r = res[4]
    assert set(r.keys()) == {
        "fused_time_ms",
        "fused_tflops",
        "fa3_time_ms",
        "fa3_tflops",
        "speedup_vs_fa3",
    }
    expected = r["fa3_time_ms"] / r["fused_time_ms"] if r["fused_time_ms"] > 0 else float("inf")
    assert pytest.approx(expected) == r["speedup_vs_fa3"]
