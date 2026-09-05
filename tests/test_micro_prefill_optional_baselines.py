"""CPU contract tests; mocked CUDA backends do not establish GPU availability."""

import importlib.util
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch

from benchmarks import micro_prefill_optional_baselines as baselines


@pytest.fixture
def inputs(monkeypatch):
    # Only the adapter's CUDA guard is mocked; all view/storage tests use real CPU tensors.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    return (
        torch.randn(2, 3, 8, 16, dtype=torch.float16),
        torch.randn(2, 2, 7, 16, dtype=torch.float16),
        torch.randn(2, 2, 7, 16, dtype=torch.float16),
    )


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _same_storage(view, original):
    assert view.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert view.untyped_storage().nbytes() == original.untyped_storage().nbytes()


def _reference(q, k, v):
    # Independent per-query-head reference; no production adapter uses this loop.
    group_size = q.shape[2] // k.shape[1]
    heads = []
    for head in range(q.shape[2]):
        kv_head = head // group_size
        weights = q[:, :, head].float() @ k[:, kv_head].float().transpose(-1, -2)
        heads.append(
            (weights * q.shape[-1] ** -0.5).softmax(-1) @ v[:, kv_head].float()
        )
    return torch.stack(heads, dim=2).to(q.dtype)


def test_import_does_not_load_optional_backends(monkeypatch):
    for name in ("flash_attn_interface", "flashinfer", "xformers", "xformers.ops.fmha"):
        monkeypatch.setitem(sys.modules, name, None)
    spec = importlib.util.spec_from_file_location(
        "optional_baselines_import_test", baselines.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.fa3_runner)
    assert callable(module.cutlass_runner)
    assert callable(module.cudnn_runner)


def test_fa3_uses_live_hnd_views_and_preallocated_output(monkeypatch, inputs):
    q, k, v = inputs
    calls = []

    def forward(**kwargs):
        calls.append(kwargs)
        assert kwargs["q"] is q
        for name, original in (("k", k), ("v", v)):
            view = kwargs[name]
            _same_storage(view, original)
            assert view.shape == (2, 7, 2, 16)
            assert view.stride() == original.transpose(1, 2).stride()
            assert not view.is_contiguous() and view.stride(-1) == 1
        assert kwargs["causal"] is False
        assert kwargs["num_splits"] == 0 and kwargs["pack_gqa"] is None
        assert kwargs["softmax_scale"] == 16**-0.5
        kwargs["out"].copy_(
            _reference(
                kwargs["q"], kwargs["k"].transpose(1, 2), kwargs["v"].transpose(1, 2)
            )
        )
        return kwargs["out"], None, None, None

    _module(monkeypatch, "flash_attn_interface", _flash_attn_forward=forward)
    run = baselines.fa3_runner(*inputs)
    assert calls == []
    out = run()
    torch.testing.assert_close(out, _reference(*inputs))
    v.add_(1)
    assert run() is out
    torch.testing.assert_close(out, _reference(*inputs))


def test_fa3_rejects_changed_api_without_execution(monkeypatch, inputs):
    def forward(q, k, v, *, out_=None):
        pytest.fail("Incompatible API must not execute")

    _module(monkeypatch, "flash_attn_interface", _flash_attn_forward=forward)
    with pytest.raises(RuntimeError, match="v2.8.3/hopper"):
        baselines.fa3_runner(*inputs)


def _cutlass_mock(monkeypatch, forward, reasons=(), available=True):
    class FwOp:
        OPERATOR = object() if available else None

        @staticmethod
        def not_supported_reasons(inp):
            assert inp.query.ndim == 5
            return list(reasons)

    _module(monkeypatch, "xformers")
    _module(monkeypatch, "xformers.ops")
    _module(
        monkeypatch, "xformers.ops.fmha", memory_efficient_attention_forward=forward
    )
    _module(monkeypatch, "xformers.ops.fmha.common", Inputs=SimpleNamespace)
    _module(monkeypatch, "xformers.ops.fmha.cutlass", FwOp=FwOp)
    return FwOp


@pytest.mark.parametrize("hkv", [1, 2, 8])
def test_cutlass_forces_fwop_with_gqa5d_views(monkeypatch, inputs, hkv):
    q, _, _ = inputs
    k = torch.randn(2, hkv, 7, 16, dtype=q.dtype)
    v = torch.randn_like(k)
    calls = []

    def forward(q5, k5, v5, *, p, scale, op):
        calls.append((q5, k5, v5))
        assert op is fwop and p == 0.0 and scale == 16**-0.5
        assert q5.shape == (2, 3, hkv, 8 // hkv, 16)
        assert k5.shape == v5.shape == (2, 7, hkv, 8 // hkv, 16)
        for view, original in ((q5, q), (k5, k), (v5, v)):
            _same_storage(view, original)
        if hkv < 8:
            assert k5.stride(3) == v5.stride(3) == 0
        scores = torch.einsum("bmgrd,bngrd->bgrmn", q5.float(), k5.float()) * scale
        return (
            torch.einsum("bgrmn,bngrd->bmgrd", scores.softmax(-1), v5.float())
            .contiguous()
            .to(q.dtype)
        )

    fwop = _cutlass_mock(monkeypatch, forward)
    run = baselines.cutlass_runner(q, k, v)
    assert calls == []
    torch.testing.assert_close(run(), _reference(q, k, v))
    k.zero_()
    torch.testing.assert_close(run(), _reference(q, k, v))


@pytest.mark.parametrize(
    "available,reasons", [(False, ()), (True, ("unsupported stride",))]
)
def test_cutlass_rejects_unavailable_operator(monkeypatch, inputs, available, reasons):
    _cutlass_mock(
        monkeypatch,
        lambda *a, **kw: pytest.fail("Must not execute"),
        reasons,
        available,
    )
    with pytest.raises(RuntimeError, match="CUTLASS"):
        baselines.cutlass_runner(*inputs)


def _cudnn_mock(monkeypatch, forward, supported=True):
    import torch.backends.cuda
    import torch.nn.attention
    import torch.nn.functional

    states = []
    backend = torch.nn.attention.SDPBackend.CUDNN_ATTENTION

    @contextmanager
    def sdpa_kernel(selected):
        assert selected is backend
        states.append(selected)
        try:
            yield
        finally:
            states.pop()

    def eligible(params, debug=False):
        assert states == [backend] and debug
        return supported

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", sdpa_kernel)
    monkeypatch.setattr(torch.backends.cuda, "can_use_cudnn_attention", eligible)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forward)
    return states


@pytest.mark.parametrize("hkv", [1, 2, 8])
def test_cudnn_forces_backend_and_keeps_true_gqa(monkeypatch, inputs, hkv):
    q, _, _ = inputs
    k = torch.randn(2, hkv, 7, 16, dtype=q.dtype)
    v = torch.randn_like(k)
    calls = []

    def forward(qh, kh, vh, **kwargs):
        calls.append(kwargs)
        assert len(states) == 1
        _same_storage(qh, q)
        assert qh.stride() == q.transpose(1, 2).stride()
        assert kh is k and vh is v
        assert kwargs == dict(
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=16**-0.5,
            enable_gqa=hkv != 8,
        )
        return _reference(qh.transpose(1, 2), kh, vh).transpose(1, 2)

    states = _cudnn_mock(monkeypatch, forward)
    run = baselines.cudnn_runner(q, k, v)
    assert states == [] and calls == []
    torch.testing.assert_close(run(), _reference(q, k, v))
    v.zero_()
    torch.testing.assert_close(run(), _reference(q, k, v))
    assert states == []


def test_cudnn_preflight_failure_never_calls_sdpa(monkeypatch, inputs):
    states = _cudnn_mock(
        monkeypatch, lambda *a, **kw: pytest.fail("Must not execute"), supported=False
    )
    with pytest.raises(RuntimeError, match="no fallback"):
        baselines.cudnn_runner(*inputs)
    assert states == []


@pytest.mark.parametrize("name", ["fa3_runner", "cutlass_runner", "cudnn_runner"])
def test_runtime_errors_propagate_without_retry(monkeypatch, inputs, name):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("backend execution failed")

    _module(monkeypatch, "flash_attn_interface", _flash_attn_forward=fail)
    _cutlass_mock(monkeypatch, fail)
    states = _cudnn_mock(monkeypatch, fail)
    run = getattr(baselines, name)(*inputs)
    with pytest.raises(RuntimeError, match="backend execution failed"):
        run()
    assert calls == [1] and states == []


@pytest.mark.parametrize("name", ["fa3_runner", "cutlass_runner", "cudnn_runner"])
@pytest.mark.parametrize("bad", ["layout", "gqa", "dtype", "shape", "empty", "grad"])
def test_invalid_inputs_fail_before_importing_backends(monkeypatch, inputs, name, bad):
    q, k, v = inputs
    if bad == "layout":
        k = k.transpose(2, 3).contiguous().transpose(2, 3)
    elif bad == "gqa":
        q = q[:, :, :7].contiguous()
    elif bad == "dtype":
        v = v.float()
    elif bad == "shape":
        v = v[:, :, :-1].contiguous()
    elif bad == "empty":
        q = q[:, :0]
    else:
        q.requires_grad_(True)
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    monkeypatch.setitem(sys.modules, "xformers.ops.fmha", None)
    with pytest.raises(ValueError):
        getattr(baselines, name)(q, k, v)


def test_cpu_inputs_are_rejected_without_backend_imports():
    q = torch.empty(1, 3, 8, 16, dtype=torch.float16)
    k = torch.empty(1, 2, 7, 16, dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA device"):
        baselines.fa3_runner(q, k, k)
