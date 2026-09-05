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


@pytest.mark.parametrize("baseline", ["torch_flash", "flashattention3", "cutlass_xformers"])
def test_single_baseline_measurement_always_keeps_both_native_controls(monkeypatch, baseline):
    from benchmarks import profile_sm90_micro_prefill_audit as audit
    from benchmarks.micro_prefill_baselines import BASELINE_IDS
    from tests.test_sm90_micro_prefill_isolated_audit import loaded_evidence

    original_randn, original_generator = torch.randn, torch.Generator
    monkeypatch.setattr(torch, "Generator", lambda **kwargs: original_generator(device="cpu"))
    monkeypatch.setattr(torch, "randn", lambda *shape, **kwargs:
                        original_randn(*shape, **dict(kwargs, device="cpu")))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(audit, "fp32_reference", lambda q, k, v: (q.float().clone(), torch.zeros(q.shape[:-1])))
    monkeypatch.setattr(audit, "natural_lse", lambda plan: torch.zeros(plan.query.shape[:-1]))
    seen, plans = [], []

    class Plan:
        @classmethod
        def build(cls, q, k, v, **kwargs):
            plan = cls()
            plan.query, plan.output = q, q.clone()
            plan.num_splits, plan.workspace_bytes = 1, 16
            plan.partial_output = q.clone()
            plan.extension = SimpleNamespace(__file__="/native.so")
            plans.append(plan)
            return plan

        def run(self):
            self.output.copy_(self.query)
            return self.output

    def prepare(q, k, v, requested):
        seen.extend(requested)
        return {baseline: lambda: q.clone()}, {}

    def capture(run, **kwargs):
        run()
        return SimpleNamespace(replay=run)

    monkeypatch.setattr(audit, "NaturalMicroPrefillPlan", Plan)
    monkeypatch.setattr(audit, "MicroPrefillPlan", Plan)
    monkeypatch.setattr(audit, "prepare_baselines", prepare)
    monkeypatch.setattr(audit, "component", lambda plan, which: plan.run())
    monkeypatch.setattr(audit, "_capture", capture)
    monkeypatch.setattr(audit, "_elapsed_graph_ms", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(audit, "loaded_binary_provenance", lambda name, **kwargs: loaded_evidence([name])[name])
    args = SimpleNamespace(cutlass_root=None, build_dir=None, warmup=1, iterations=1, repeats=2,
                           requested_baselines=[baseline], baseline_versions=dict.fromkeys(BASELINE_IDS, "v1"),
                           environment_sha256="a" * 64, binary_hash_cache={})
    row = audit.measure_case(dict(batch=1, m=2, n=64, hq=8, g=4, d=64, splits=1), args)
    assert seen == [baseline]
    assert len(plans) == 2
    assert set(row["median_ms"]) == {baseline, "natural", "transposed", "natural_producer", "natural_merge"}
    assert set(row["mutation"]) == {baseline, "natural", "transposed"}
    assert row["exact_pass"] and row["requested_baseline_set_complete"]
    assert not row["baseline_set_complete"]


def test_worker_refuses_to_overwrite_before_gpu_access(monkeypatch, tmp_path):
    from benchmarks import profile_sm90_micro_prefill_audit as audit
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: pytest.fail("GPU must not be touched"))
    with pytest.raises(FileExistsError):
        audit.main(["--provider", "test", "--cohort", "smoke", "--baseline", "torch_flash",
                    "--cutlass-root", str(tmp_path), "--build-dir", str(tmp_path),
                    "--output-json", str(output)])
    assert output.read_text() == "preserve"
