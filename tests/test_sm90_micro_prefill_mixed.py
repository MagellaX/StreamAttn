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
    assert len(mixed.experiment_cases("causal")) == 24
    assert all(c["causal"] for c in mixed.experiment_cases("causal"))


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


def test_compact_summary_separates_control_improvement_from_baseline_victory():
    from benchmarks.summarize_sm90_micro_prefill_mixed import summarize

    graphs, families = {}, {}
    for interface in mixed.INTERFACES:
        times = {n + "/" + interface: t for n, t in
                 (("natural", 10), ("transposed", 12), ("natural_compact", 5), ("flashinfer_fa2", 2))}
        graphs[interface] = dict(median_us=times, paired_trials=[dict(us=times)],
            fastest_tested_baseline=dict(baseline_id="flashinfer_fa2", correctness_passed=True))
        families["natural_compact/" + interface] = dict(producer_ctas=256)
    row = dict(case={}, passed=True, graphs=graphs, families=families,
        loaded_binary_provenance={n: dict(resolved=True) for n in ("natural", "transposed", "natural_compact")})
    result = summarize(dict(schema=mixed.SCHEMA, complete=True, environment={}, rows=[row]))
    candidate = result["interfaces"]["packed"]["compact_candidate"]
    assert candidate["control_geomean"] == pytest.approx(2)
    assert candidate["baseline_geomean"] == pytest.approx(0.4)
    assert candidate["control_all_pair_wins"] == 1
    assert candidate["baseline_all_pair_wins"] == 0
    assert result["compact_diagnostics"]["matched_mask_pairs"] == 0
    assert result["compact_diagnostics"]["copy_free_baseline_speedup_proxy"] == pytest.approx(0.4)


def test_mask_diagnostic_does_not_confuse_layout_with_causality():
    from benchmarks.summarize_sm90_micro_prefill_mixed import compact_diagnostics

    def row(layout, causal, time):
        return dict(case=dict(layout=layout, causal=causal), passed=True,
            loaded_binary_provenance=dict(natural_compact=dict(resolved=True)),
            graphs=dict(padded=dict(median_us={"natural_compact/padded": time})))

    unmatched = [row("HND", False, 2), row("NHD", True, 4)]
    assert compact_diagnostics(unmatched)["matched_mask_pairs"] == 0
    matched = compact_diagnostics(unmatched + [row("HND", True, 6)])
    assert matched["matched_mask_pairs"] == 1
    assert matched["causal_over_noncausal_latency"] == pytest.approx(3)


def test_affine_ablations_keep_external_and_internal_wins_separate():
    from benchmarks.summarize_sm90_micro_prefill_mixed import affine_ablations

    families = ("natural_compact", "natural_compact_affine", "natural_compact_interior")
    times = dict(zip([f + "/packed" for f in families], [10, 5, 4]))
    times["flashinfer_fa2/packed"] = 2
    row = dict(case={}, passed=True,
        loaded_binary_provenance={f: dict(resolved=True) for f in families},
        graphs=dict(packed=dict(median_us=times, paired_trials=[dict(us=times)],
            fastest_tested_baseline=dict(baseline_id="flashinfer_fa2", correctness_passed=True))))
    result = affine_ablations([row], "packed")
    assert result["natural_compact_affine"]["control_geomean"] == pytest.approx(2)
    assert result["natural_compact_interior"]["control_geomean"] == pytest.approx(2.5)
    assert result["natural_compact_interior"]["baseline_all_pair_wins"] == 0
    assert result["natural_compact_interior"]["index_geomean"] == pytest.approx(1.25)
    assert result["natural_compact_interior"]["index_all_pair_wins"] == 1


def test_affine_mask_geometry_does_not_count_partial_query_tiles_as_interior():
    from benchmarks.summarize_sm90_micro_prefill_mixed import affine_mask_geometry

    c = dict(g=8, hq=16, query_lengths=[4, 16], kv_lengths=[128, 128])
    result = affine_mask_geometry(c)
    assert result["total_tiles"] == 12
    assert result["interior_tiles"] == 4
    assert result["mask_free_fraction"] == pytest.approx(1/3)
    assert affine_mask_geometry(dict(g=4, hq=16, query_lengths=[0], kv_lengths=[0]))["total_tiles"] == 0
