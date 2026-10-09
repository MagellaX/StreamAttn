"""Fixed-support C1/C4 parity, including record transitions and neutral states."""

import math

import pytest
import torch

from stream_attention.paged import PagedKVCache, PagedSelectedDecodePlan
from stream_attention.planning import AttentionProblem, AttentionTilePlan
from stream_attention.selected_routes import prepare_paged_routes64


def _ready():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return False
    from stream_attention.backends.sm90.transposed_gqa_exact import resolve_cutlass_root
    try:
        resolve_cutlass_root()
    except FileNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _ready(), reason="requires SM90 and CUTLASS")
@pytest.mark.parametrize("support", ["full", "holes_and_head_masks"])
def test_grouped_selected_preserves_support_and_neutral_states(support):
    torch.manual_seed(711)
    length, pages, hk, dim = 1021, 64, 2, 128
    q = torch.randn(1, 1, 16, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(pages, 16, hk, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    v[..., 0] = torch.arange(pages * 16, device="cuda").view(pages, 16, 1) / 256
    cache = PagedKVCache(k, v, torch.randperm(pages, device="cuda", dtype=torch.int32)[None],
                         torch.tensor([length], device="cuda", dtype=torch.int32), "NHD")
    if support == "full":
        rows = [tuple(range(math.ceil(length / 32)))] * 16
    else:
        rows = [tuple(range(0, 30, 2)) + (31,) if h % 2 == 0 else (18, 22, 31)
                for h in range(8)]
        rows += [(1, 3, 31) if h % 2 == 0 else (3, 31) for h in range(8)]
    problem = AttentionProblem.from_paged(q, cache, guarantee="schedule_exact")
    logical = AttentionTilePlan.selected(problem, logical_tile_size=32, tile_ids_per_row=rows,
        route_granularity="q_head", policy_id="grouped-equivalence", reason=support, schedule_epoch=7)
    routes = prepare_paged_routes64(logical, cache)
    expected = []
    for h, blocks in enumerate(rows):
        tokens = torch.tensor([j for b in blocks for j in range(b * 32, min(length, (b + 1) * 32))],
                              device="cuda")
        physical = cache.page_table[0, tokens // 16].long()
        keys = k[physical, tokens % 16, h // 8].double()
        values = v[physical, tokens % 16, h // 8].double()
        expected.append(((keys @ q[0, 0, h].double() / math.sqrt(dim)).softmax(0) @ values))
    expected = torch.stack(expected)
    plans = [PagedSelectedDecodePlan.build(q, cache, routes, schedule_epoch=7, records_per_cta=c)
             for c in (1, 4)]
    for plan in plans:
        actual = plan.run().reshape(16, dim).double()
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        assert float((actual - expected).abs().max()) <= 0.02
        assert torch.isfinite(plan.workspace["partial_o"]).all()
        lse = plan.workspace["partial_lse"]
        assert not torch.isnan(lse).any() and not torch.isposinf(lse).any()
    old, grouped = plans
    assert grouped.workspace_bytes <= old.workspace_bytes / 3
    assert grouped.producer_ctas == routes.row_count * math.ceil(old.max_routes_per_row / 4)
    assert grouped.routes is old.routes
    if support != "full":
        # Odd heads first contribute after the first partition; the other KV row
        # has fewer records and must also emit neutral padded partitions.
        assert torch.isneginf(grouped.workspace["partial_lse"][0, 0, 1::2]).all()
        assert torch.isneginf(grouped.workspace["partial_lse"][1, 1:]).all()
        assert torch.count_nonzero(grouped.workspace["partial_o"][1, 1:]) == 0
