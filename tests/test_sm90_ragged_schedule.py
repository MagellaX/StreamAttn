import math

import pytest

from stream_attention.backends.sm90.ragged_schedule import plan_ragged_schedule
from stream_attention.backends.sm90.micro_prefill_ragged_sources import ragged_cuda_source


@pytest.mark.parametrize("g", [4, 8])
@pytest.mark.parametrize("budget", [1, 64, 256, 1024])
def test_tasks_partition_all_live_tiles(g, budget):
    qs, ns = [0, 1, 4, 16, 64], [0, 63, 257, 8191, 16381]
    s = plan_ragged_schedule(qs, ns, capacity=64, kv_heads=4, group_size=g, target_ctas=budget)
    groups = {}
    for group, split, begin, end in s.tasks:
        assert 0 <= begin < end
        groups.setdefault(group, []).append((split, begin, end))
    qt = math.ceil(64 / (64 // g))
    for b, (q, n) in enumerate(zip(qs, ns)):
        for h in range(4):
            for tile in range(qt):
                group = (b * 4 + h) * qt + tile
                if tile * (64 // g) >= q or n == 0:
                    assert group not in groups
                    continue
                parts = groups[group]
                assert len(parts) == s.splits[b]
                assert parts[0][1] == 0 and parts[-1][2] == math.ceil(n / 64)
                assert all(a[2] == z[1] for a, z in zip(parts, parts[1:]))
    minimum = sum(math.ceil(q / (64 // g)) * 4 for q, n in zip(qs, ns) if n)
    assert len(s.tasks) <= max(budget, minimum)


def test_known_heterogeneous_work_is_rebalanced():
    s = plan_ragged_schedule([1, 2, 4, 8, 16, 32, 48, 64],
        [63, 257, 1023, 2047, 4093, 8191, 12287, 16381],
        capacity=64, kv_heads=4, group_size=4)
    assert len(s.tasks) <= 256
    assert s.max_tiles_per_task < 128
    assert s.splits[-1] > 2


def test_empty():
    assert not plan_ragged_schedule([0, 2], [7, 0], capacity=2, kv_heads=2, group_size=8).tasks


@pytest.mark.parametrize("qs,ns", [([], []), ([1], []), ([-1], [64]),
                                   ([65], [64]), ([1], [-1]), ([True], [64])])
def test_invalid_lengths(qs, ns):
    with pytest.raises(ValueError):
        plan_ragged_schedule(qs, ns, capacity=64, kv_heads=2, group_size=8)


def test_split_cap_and_single_tile_are_respected():
    s = plan_ragged_schedule([2, 2], [1, 65536], capacity=2, kv_heads=2,
                             group_size=8, target_ctas=100000)
    assert s.splits == (1, 512)
    assert s.max_tiles_per_task == 2


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("causal", [True, False])
def test_compact_source_contract(dim, dtype, causal):
    source = ragged_cuda_source(dim, dtype, causal)
    assert "const int tile_begin = task[2]" in source
    assert "split < active_splits" in source
    assert "if (tasks.size(0)>0)" in source
    assert "query_position >= query_lengths[batch] || active_splits == 0" in source
    assert "work_group * num_splits + split" in source
