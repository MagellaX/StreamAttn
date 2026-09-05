import pytest
import torch

from stream_attention.backends.sm90 import micro_prefill as backend
from stream_attention.backends.sm90.micro_prefill_semantics import validate_positions
from stream_attention.backends.sm90.micro_prefill_semantics_sources import (
    semantic_cuda_source,
)
from stream_attention.backends.sm90.micro_prefill_temporal import temporal_shape_reasons


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("causal", [False, True])
def test_source_specialization_reuses_both_families(dim, dtype, causal):
    source = semantic_cuda_source(dim, dtype, causal)
    assert f"kHeadDim = {dim};" in source
    assert f"kPositionCausal = {str(causal).lower()};" in source
    assert "GroupedRSPrefillSharedStorage" in source
    assert "query_positions_per_batch" in source
    assert "row_max == -INFINITY ? 0.0f" in source
    assert "next_max == -INFINITY ? 0.0f" in source
    assert "normalizer > 0.0f" in source
    assert "tile_begin = split * num_kv_tiles / num_splits" in source
    assert "at::assert_no_overlap" in source
    assert "CUDAGuard" in source
    if dtype == "fp16":
        assert "cutlass::half_t" in source
        assert "F32F16F16_SS" in source
        assert "BF16" not in source and "BFloat16" not in source
    else:
        assert "F32BF16BF16_SS" in source


def tensors(dtype=torch.float16):
    q = torch.empty(2, 3, 16, 64, dtype=dtype)
    k = torch.empty(2, 2, 320, 64, dtype=dtype)
    qp = torch.empty(2, 3, dtype=torch.int64)
    kp = torch.empty(2, 320, dtype=torch.int64)
    return q, k, qp, kp


def test_native_fp16_contract_does_not_enable_rejected_temporal_family():
    q, k, _, _ = tensors()
    assert backend.micro_prefill_shape_reasons(q, k, k) == []
    assert temporal_shape_reasons(q, k, k) == ["dtype"]
    assert "dtype" in backend.micro_prefill_shape_reasons(q, k.bfloat16(), k)


def test_explicit_positions_cannot_be_silently_inferred_or_ignored():
    q, k, qp, kp = tensors()
    validate_positions(q, k, causal=True, query_positions=qp, key_positions=kp)
    with pytest.raises(ValueError, match="explicit query_positions"):
        validate_positions(q, k, causal=True, query_positions=None, key_positions=kp)
    with pytest.raises(ValueError, match="positions require"):
        validate_positions(q, k, causal=False, query_positions=qp, key_positions=kp)
    for bad in (qp.float(), qp[:, :2], qp.to("meta"), torch.empty(3, 2).T.long()):
        with pytest.raises(ValueError, match="query_positions"):
            validate_positions(q, k, causal=True, query_positions=bad, key_positions=kp)


@pytest.mark.parametrize("plan_type", [backend.MicroPrefillPlan, backend.NaturalMicroPrefillPlan])
def test_planned_replay_keeps_mutable_device_position_buffers(monkeypatch, plan_type):
    from types import SimpleNamespace

    calls = []
    ext = SimpleNamespace(out=lambda *args: calls.append(args))
    monkeypatch.setattr(backend, "supports_sm90_micro_prefill", lambda *a, **kw: True)
    monkeypatch.setattr(backend, "compile_semantic_extension", lambda **kw: ext)
    q, k, qp, kp = tensors()
    plan = plan_type.build(q, k, k, num_splits=4, causal=True,
                           query_positions=qp, key_positions=kp)
    assert plan.positions[0] is qp and plan.positions[1] is kp
    assert plan.run() is plan.output
    assert calls[0][0] is q and calls[0][6] is qp and calls[0][7] is kp
    assert calls[0][8] == 4
    assert calls[0][9] == (plan_type is backend.NaturalMicroPrefillPlan)
    qp.fill_(1 << 40)
    plan.run()
    assert calls[1][6].data_ptr() == qp.data_ptr()


def test_split_limit_matches_static_merge_storage(monkeypatch):
    monkeypatch.setattr(backend, "supports_sm90_micro_prefill", lambda *a, **kw: True)
    q = torch.empty(1, 2, 16, 64, dtype=torch.bfloat16, device="meta")
    k = torch.empty(1, 2, 65536, 64, dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="512"):
        backend.MicroPrefillPlan.build(q, k, k, num_splits=513)


def test_reference_uses_logical_not_physical_causal_alignment():
    from benchmarks.profile_sm90_micro_prefill_semantics import fp32_reference

    q = torch.zeros(1, 2, 4, 2)
    k = torch.zeros(1, 1, 3, 2)
    v = torch.tensor([[[[3.0, 4.0], [7.0, 8.0], [11.0, 12.0]]]])
    origin = 1 << 40
    qp = torch.tensor([[origin - 1, origin]])
    kp = torch.tensor([[origin + 2, origin + 1, origin]])
    out, lse = fp32_reference(q, k, v, qp, kp)
    assert (out[:, 0] == 0).all() and torch.isneginf(lse[:, 0]).all()
    assert torch.equal(out[0, 1], torch.tensor([[11.0, 12.0]]).expand(4, 2))
    assert (lse[:, 1] == 0).all()


def test_matrix_spans_dtypes_masks_tile_tails_and_batch_heads():
    from benchmarks.profile_sm90_micro_prefill_semantics import experiment_cases

    cases = experiment_cases("full")
    assert len(cases) == 84
    assert {c["dtype"] for c in cases} == {"bf16", "fp16"}
    assert {c["mask"] for c in cases} == {"append", "permuted", "noncausal"}
    assert {c["batch"] for c in cases} == {1, 2}
    assert {c["hq"] for c in cases} == {16, 32}
    assert any(c["n"] // 64 % c["splits"] for c in cases)


def test_deferred_sum_changes_only_denominator_reduction_placement():
    from benchmarks.profile_sm90_micro_prefill_deferred_sum import deferred_source

    original = semantic_cuda_source(128, "bf16", False)
    candidate = deferred_source()
    restored = candidate.replace(
        "  CUTE_UNROLL\n  for (int row = 0; row < kRowsPerThread; ++row) {\n"
        "    row_sum[row] = streamattn_quad_sum(row_sum[row]);\n  }\n", "", 1
    ).replace("row_sum[row] += local_sum;", "row_sum[row] += streamattn_quad_sum(local_sum);", 1)
    assert restored == original
