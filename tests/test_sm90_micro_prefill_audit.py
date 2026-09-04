import math
from types import SimpleNamespace

import pytest
import torch

from benchmarks.profile_sm90_micro_prefill_audit import (
    cases,
    errors,
    fp32_reference,
    natural_lse,
)
from stream_attention.backends.sm90.micro_prefill import micro_prefill_shape_reasons
from stream_attention.backends.sm90.transposed_gqa_exact_sources import CUDA_SOURCE


def test_provider_cohorts_share_replay_and_cover_disjoint_expansion():
    modal, lightning = cases("modal"), cases("lightning")
    assert len(modal) == len(lightning) == 52
    assert modal[:20] == lightning[:20]
    assert all(c["batch"] == 1 for c in modal[20:])
    assert all(c["batch"] == 2 and c["hq"] == 32 for c in lightning[20:])


@pytest.mark.parametrize("tiles", [1, 5, 7, 11, 19, 65, 512])
def test_balanced_split_partition_is_exhaustive_nonempty_and_balanced(tiles):
    for splits in range(1, tiles + 1):
        bounds = [
            (s * tiles // splits, (s + 1) * tiles // splits) for s in range(splits)
        ]
        assert all(lo < hi for lo, hi in bounds)
        assert [t for lo, hi in bounds for t in range(lo, hi)] == list(range(tiles))
        sizes = [hi - lo for lo, hi in bounds]
        assert max(sizes) - min(sizes) <= 1
    kernel = CUDA_SOURCE.split(
        "void streamattn_natural_wgmma_micro_prefill_partial_kernel(", 1
    )[1]
    kernel = kernel.split(
        "void streamattn_natural_wgmma_micro_prefill_merge_kernel(", 1
    )[0]
    assert "split * num_kv_tiles / num_splits" in kernel
    assert "tiles_per_split" not in kernel


def test_dense_reference_respects_batch_and_grouped_head_mapping():
    q = torch.zeros(2, 3, 8, 4)
    k = torch.randn(2, 2, 5, 4)
    v = torch.arange(2 * 2 * 5 * 4).float().view(2, 2, 5, 4)
    o, lse = fp32_reference(q, k, v)
    for b in range(2):
        for h in range(8):
            torch.testing.assert_close(o[b, :, h], v[b, h // 4].mean(0).expand(3, -1))
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(5)))


@pytest.mark.parametrize("m,g", [(3, 4), (9, 8), (17, 4), (63, 8)])
def test_natural_lse_reconstruction_handles_padded_query_tiles(m, g):
    b, hk, s, positions = 2, 3, 4, 64 // g
    tiles = (m + positions - 1) // positions
    state = torch.arange(b * hk * tiles * 64).float().view(b * hk * tiles, 64) / 100
    partial = state[:, None].expand(-1, s, -1).clone()
    plan = SimpleNamespace(
        query=torch.empty(b, m, hk * g, 64),
        key_cache=torch.empty(b, hk, 64, 64),
        query_tiles=tiles,
        partial_lse=partial,
    )
    actual = natural_lse(plan)
    for bi in range(b):
        for qi in range(m):
            for h in range(hk * g):
                work = (bi * hk + h // g) * tiles + qi // positions
                row = qi % positions * g + h % g
                assert float(actual[bi, qi, h]) == pytest.approx(
                    float(state[work, row]) * math.log(2) + math.log(s), abs=1e-5
                )


def test_error_gate_rejects_nonfinite_output():
    assert not errors(torch.tensor([float("nan")]), torch.ones(1))["passed"]
    assert errors(torch.ones(2), torch.ones(2))["passed"]


def test_contract_rejects_zero_batches_and_mixed_devices():
    q = torch.empty(0, 4, 8, 64, dtype=torch.bfloat16)
    k = torch.empty(0, 2, 64, 64, dtype=torch.bfloat16)
    assert "batch" in micro_prefill_shape_reasons(q, k, k)
    assert "device" in micro_prefill_shape_reasons(q.to("meta"), k, k)
