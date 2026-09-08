import pytest
import torch

from benchmarks import profile_sm90_micro_prefill_mixed as mixed
from stream_attention.baseline_resolver import resolve_direct_exact_baseline


def test_mixed_matrix_covers_both_query_contracts_and_storage_orders():
    cases = mixed.experiment_cases("full")
    assert len(cases) == 48
    assert len(mixed.experiment_cases("replay")) == 24
    assert {c["layout"] for c in cases} == {"HND", "NHD"}
    assert {c["dtype"] for c in cases} == {"fp16", "bf16"}
    assert {c["d"] for c in cases} == {64, 128}
    assert {len(c["query_lengths"]) for c in cases} == {4, 8}
    assert {c["causal"] for c in cases} == {True, False}
    assert mixed.INTERFACES == ("padded", "packed")


@pytest.mark.parametrize("c", mixed.experiment_cases("full"))
def test_mixed_metadata_exactly_matches_the_workload(c):
    meta = mixed.metadata(c)
    table = mixed.page_table(c, 17)
    wl = mixed.workload(c, table)
    assert wl.is_ragged and wl.has_shared_prefixes
    assert meta["qi"][-1] == sum(c["query_lengths"])
    assert meta["pi"][-1] == sum((n+15)//16 for n in c["kv_lengths"])
    for i, request in enumerate(wl.requests):
        assert request.cache_page_ids[0] == table[0][0]
        assert all(x == -1 for x in table[i][len(request.cache_page_ids):])
        assert meta["qslots"][meta["qi"][i]:meta["qi"][i+1]] == [
            i*meta["m"]+j for j in range(request.query_len)]
    assert mixed.page_table(c, 17) == table
    assert mixed.workload(c, mixed.page_table(c, 18)).fingerprint != wl.fingerprint
    assert resolve_direct_exact_baseline(wl, mixed.descriptor("flashinfer_fa3", "test")).eligible


def test_packing_and_unpacking_preserve_only_live_query_rows():
    c = mixed.experiment_cases("smoke")[0]
    meta = mixed.metadata(c)
    slots = torch.tensor(meta["qslots"])
    padded = torch.arange(4*meta["m"]*2).reshape(4*meta["m"], 2)
    packed = torch.empty(len(slots), 2, dtype=padded.dtype)
    torch.index_select(padded, 0, slots, out=packed)
    restored = torch.zeros_like(padded)
    restored.index_copy_(0, slots, packed)
    assert torch.equal(restored[slots], padded[slots])
    inactive = torch.ones(len(padded), dtype=torch.bool)
    inactive[slots] = False
    assert not restored[inactive].any()


def test_lse_check_does_not_overwrite_replayed_flashinfer_output(monkeypatch):
    import sys
    from types import SimpleNamespace

    class Wrapper:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self, *args, **kwargs):
            pass

        def run(self, q, kv, *, out, lse=None, return_lse=False):
            out.fill_(7 if return_lse else 3)
            if lse is not None:
                lse.fill_(1)
            return out

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=Wrapper))
    monkeypatch.setattr(mixed, "FLASHINFER_WORKSPACE_BYTES", 4)
    c = mixed.experiment_cases("smoke")[0]
    meta = mixed.metadata(c)
    q = torch.empty(4, meta["m"], 16, 64)
    cache = SimpleNamespace(page_table=torch.tensor(mixed.page_table(c, 2)),
                            kv_heads=4, key=None, value=None)
    fi = mixed.prepare_flashinfer(c, q, cache, meta, torch.tensor(meta["pslots"]),
                                 torch.empty(meta["qi"][-1], 16, 64), "flashinfer_fa2")
    fi.run()
    assert torch.allclose(fi.lse(), torch.full((meta["qi"][-1], 16), 0.6931471805599453))
    assert bool((fi.output == 3).all())


@pytest.mark.parametrize("missing", [None, "baseline", "native", "correctness"])
def test_summary_preserves_interface_boundary_and_requires_evidence(missing):
    from benchmarks.summarize_sm90_micro_prefill_mixed import summarize

    graphs = {}
    for interface, time in (("padded", 2), ("packed", 8)):
        times = {"natural/" + interface: time, "transposed/" + interface: time*2,
                 "flashinfer_fa3/" + interface: 4}
        graphs[interface] = dict(median_us=times, paired_trials=[dict(us=times)]*3,
            fastest_tested_baseline=None if missing == "baseline" else
                dict(baseline_id="flashinfer_fa3", correctness_passed=True))
    row = dict(case={}, graphs=graphs, passed=missing != "correctness",
               loaded_binary_provenance={n: dict(resolved=missing != "native")
                                         for n in ("natural", "transposed")})
    result = summarize(dict(schema=mixed.SCHEMA, complete=True, environment={}, rows=[row]))
    for interface, ratio in (("padded", 2), ("packed", 0.5)):
        report = result["interfaces"][interface]
        assert report["comparable_cases"] == (1 if missing is None else 0)
        assert report["oracle_geomean"] == (ratio if missing is None else None)
    assert not result["promotion"]


def test_rectangular_ragged_grid_counts_inactive_work_not_occupancy():
    from benchmarks.summarize_sm90_micro_prefill_mixed import schedule_geometry

    c = next(c for c in mixed.experiment_cases("full") if c["trace"] == "heterogeneous")
    geometry = schedule_geometry(c, "natural", 2)
    assert geometry["launched_ctas"] == 256
    assert geometry["nonempty_ctas"] == 108
    assert geometry["empty_cta_fraction"] == 148/256
    assert geometry["nonempty_ctas_per_request"][-1] == 32
    assert geometry["maximum_kv_tiles_per_cta"][-1] == 128
