import math

import pytest

from stream_attention.backends.sm90.ragged_schedule import plan_ragged_schedule
from stream_attention.backends.sm90.ragged_schedule import validate_affine_append_positions
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


@pytest.mark.parametrize("mode", ["index", "interior"])
@pytest.mark.parametrize("dim,dtype", [(64, "bf16"), (128, "bf16"), (64, "fp16"), (128, "fp16")])
def test_affine_source_removes_position_loads_from_natural_producer(mode, dim, dtype):
    source = ragged_cuda_source(dim, dtype, True, mode)
    producer = source.split("void streamattn_natural_wgmma_micro_prefill_partial_kernel(")[1].split(
        "void streamattn_natural_wgmma_micro_prefill_merge_kernel(")[0]
    assert "key_positions[" not in producer and "query_positions[" not in producer
    assert "ki > sequence_length - valid_queries + qi" in producer
    assert ("query_begin + query_positions_per_tile <= valid_queries" in producer) == (mode == "interior")


def test_affine_position_validation_preserves_large_and_negative_origins():
    for origin in (-(1 << 40), 1 << 40):
        validate_affine_append_positions([2], [5], [[origin + 3, origin + 4]],
                                         [[origin + i for i in range(5)]])
    with pytest.raises(ValueError):
        validate_affine_append_positions([2], [3], [[1, 2]], [[0, 2, 1]])
    with pytest.raises(ValueError):
        validate_affine_append_positions([2], [3], [[0, 1]], [[0, 1, 2]])
    with pytest.raises(ValueError):
        validate_affine_append_positions([1], [2], [[-(1 << 63)]], [[(1 << 63)-1, -(1 << 63)]])


@pytest.mark.parametrize("g", [4, 8])
def test_interior_predicate_only_skips_fully_visible_tiles(g):
    qpt = 64 // g
    for q in (1, 7, 8, 9, 16, 31, 64):
        for n in (1, 63, 64, 65, 127, 257):
            for begin in range(0, q, qpt):
                for tile in range(math.ceil(n / 64)):
                    if begin + qpt <= q and tile * 64 + 63 <= n - q + begin:
                        assert all(k < n and k <= n - q + i
                                   for i in range(begin, begin + qpt)
                                   for k in range(tile * 64, tile * 64 + 64))
