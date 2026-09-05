import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.micro_prefill_baselines import (
    BASELINE_IDS,
    descriptors,
    flashinfer_runner,
    resolve_measurements,
    workload_for_case,
)
from stream_attention.baseline_resolver import resolve_direct_exact_baselines
from benchmarks import micro_prefill_baselines as baselines


def case(batch=2):
    return dict(
        batch=batch, m=17, n=320, hq=16, g=8, d=64, splits=4, purpose="boundary"
    )


def test_runtime_descriptors_resolve_original_hnd_contract():
    workload = workload_for_case(case())
    assert workload.cache_layout.value == "hnd"
    assert workload.query_lengths == (17, 17)
    assert workload.kv_lengths == (320, 320)
    assert all(
        row.eligible
        for row in resolve_direct_exact_baselines(
            workload, descriptors(dict.fromkeys(BASELINE_IDS, "test_revision"))
        )
    )


def test_baseline_resolution_requires_mutation_correctness_and_complete_set():
    versions = dict.fromkeys(BASELINE_IDS, "v1")
    medians = dict(torch_flash=0.1, flashinfer_fa3=0.05)
    accuracy = {x: {"passed": True} for x in medians}
    mutation = {"torch_flash": {"passed": True}, "flashinfer_fa3": {"passed": False}}
    result = resolve_measurements(
        case(), versions, "a" * 64, medians, accuracy, mutation
    )
    assert result["winner"]["baseline_id"] == "torch_flash"
    assert not result["complete"]
    mutation["torch_flash"]["passed"] = False
    result = resolve_measurements(
        case(), versions, "a" * 64, medians, accuracy, mutation
    )
    assert result["winner"] is None


@pytest.mark.parametrize("revision", ["not_installed", "unresolved"])
def test_missing_implementation_identity_cannot_win(revision):
    versions = dict.fromkeys(BASELINE_IDS, revision)
    result = resolve_measurements(
        case(),
        versions,
        "a" * 64,
        {"flashattention3": 0.001},
        {"flashattention3": {"passed": True}},
        {"flashattention3": {"passed": True}},
    )
    assert result["winner"] is None
    assert not result["complete"]


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_flashinfer_direct_hnd_keeps_storage_live_without_gqa_expansion(
    monkeypatch, backend
):
    seen = []
    workspaces = []

    class Wrapper:
        def __init__(self, workspace, **kw):
            workspaces.append(workspace)
            assert kw["kv_layout"] == ("NHD" if backend == "fa3" else "HND")
            assert kw["backend"] == backend
            assert kw["use_cuda_graph"]

        def plan(self, qi, ki, hq, hk, d, **kw):
            assert hq == 16 and hk == 2

        def run(self, q, k, v, *, out):
            seen.append(
                (k.untyped_storage().data_ptr(), tuple(k.shape), tuple(k.stride()))
            )
            out.fill_(v.flatten()[0])

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(BatchPrefillWithRaggedKVCacheWrapper=Wrapper),
    )
    q = torch.zeros(2, 17, 16, 64)
    k = torch.zeros(2, 2, 320, 64)
    v = torch.zeros_like(k)
    run = flashinfer_runner(q, k, v, backend)
    assert len(workspaces) == 2 and workspaces[0] is workspaces[1]
    assert workspaces[0].dtype == torch.uint8
    assert workspaces[0].numel() * workspaces[0].element_size() == 256 * 1024 * 1024
    # FI2's M64/D128 plan requests 138412032 bytes for tmp_v alone.
    assert workspaces[0].numel() > 138412032
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("workspace allocation during run"))
    assert torch.count_nonzero(run()) == 0
    v.add_(2)
    assert torch.all(run() == 2)
    assert len(seen) == 4
    assert all(
        ptr == k.untyped_storage().data_ptr()
        and shape == ((320, 2, 64) if backend == "fa3" else (2, 320, 64))
        and stride == ((64, 320 * 64, 1) if backend == "fa3" else (320 * 64, 64, 1))
        for ptr, shape, stride in seen
    )


def test_flashinfer_workspace_allowance_is_recorded_in_adapter_provenance(monkeypatch):
    monkeypatch.setattr(baselines, "distribution_identity", lambda name: dict(
        version="1.0", record_sha256="a" * 64, interfaces={},
    ))
    monkeypatch.setattr(baselines, "baseline_versions", lambda: dict.fromkeys(BASELINE_IDS, "1.0"))
    settings = baselines.runtime_provenance()["adapter_settings"]
    assert settings["flashinfer_workspace_bytes"] == baselines.FLASHINFER_WORKSPACE_BYTES == 268435456
    assert settings["flashinfer_workspace_allocation"] == "prepare_only_shared_by_sequential_requests"


@pytest.mark.parametrize("selected", BASELINE_IDS)
def test_prepare_only_invokes_requested_factory(monkeypatch, selected):
    from benchmarks import micro_prefill_optional_baselines as optional
    from benchmarks import profile_sm90_micro_prefill as native

    seen = []

    def factory(name):
        seen.append(name)
        return lambda: name

    monkeypatch.setattr(baselines, "flashinfer_runner", lambda q, k, v, b: factory("flashinfer_" + b))
    monkeypatch.setattr(optional, "fa3_runner", lambda *args: factory("flashattention3"))
    monkeypatch.setattr(optional, "cutlass_runner", lambda *args: factory("cutlass_xformers"))
    monkeypatch.setattr(optional, "cudnn_runner", lambda *args: factory("cudnn"))
    monkeypatch.setattr(native, "_flash_sdpa", lambda *args: factory("torch_flash")())
    runners, unavailable = baselines.prepare_baselines(None, None, None, [selected])
    assert list(runners) == [selected]
    assert runners[selected]() == selected
    assert seen == [selected]
    assert unavailable == {}


@pytest.mark.parametrize("requested", [[], ["unknown"], ["torch_flash", "torch_flash"]])
def test_invalid_requested_subset_is_rejected(requested):
    with pytest.raises(ValueError, match="subset"):
        baselines.prepare_baselines(None, None, None, requested)


def test_subset_never_claims_complete_global_baseline_set():
    result = resolve_measurements(case(), dict.fromkeys(BASELINE_IDS, "v1"), "a" * 64,
                                  {"torch_flash": 1, "flashattention3": 0.5},
                                  {n: {"passed": True} for n in BASELINE_IDS},
                                  {n: {"passed": True} for n in BASELINE_IDS},
                                  requested=["torch_flash"])
    assert result["winner"]["baseline_id"] == "torch_flash"
    assert [r["baseline_id"] for r in result["measurements"]] == ["torch_flash"]
    assert [r["baseline_id"] for r in result["eligibility"]] == ["torch_flash"]
    assert not result["complete"]


def test_loaded_evidence_hashes_actual_files_without_importing_other_backends(monkeypatch, tmp_path):
    import builtins
    import hashlib

    interface = tmp_path / "flash_attn_interface.py"
    binary = tmp_path / "flash_attn_3_cuda.so"
    interface.write_bytes(b"interface")
    binary.write_bytes(b"actual binary bytes")
    monkeypatch.setitem(sys.modules, "flash_attn_interface", SimpleNamespace(__file__=str(interface)))
    monkeypatch.setitem(sys.modules, "flash_attn_3_cuda", SimpleNamespace(__file__=str(binary)))
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("flash_attn", "xformers", "flashinfer"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    cache = {}
    result = baselines.loaded_binary_provenance("flashattention3", cache=cache)
    assert result["resolved"]
    assert result["binaries"] == [dict(path=str(binary.resolve()), size_bytes=19,
                                      sha256=hashlib.sha256(binary.read_bytes()).hexdigest())]
    assert result == baselines.loaded_binary_provenance("flashattention3", cache=cache)
    binary.write_bytes(b"replaced binary")
    assert result != baselines.loaded_binary_provenance("flashattention3", cache=cache)


def test_missing_loaded_binary_excludes_even_known_installation(monkeypatch, tmp_path):
    interface = tmp_path / "flash_attn_interface.py"
    interface.write_bytes(b"interface")
    monkeypatch.setitem(sys.modules, "flash_attn_interface", SimpleNamespace(__file__=str(interface)))
    monkeypatch.delitem(sys.modules, "flash_attn_3_cuda", raising=False)
    evidence = baselines.loaded_binary_provenance("flashattention3")
    assert not evidence["resolved"]
    assert not evidence["binaries"]
    versions = baselines.measurement_versions(dict.fromkeys(BASELINE_IDS, "known"), {
        "flashattention3": evidence, "natural": {"resolved": True}, "transposed": {"resolved": True},
    })
    assert versions["flashattention3"] == "unresolved"


@pytest.mark.parametrize("backend", ["torch_flash", "cudnn"])
def test_real_torch_namespaces_do_not_poison_file_provenance(monkeypatch, backend):
    # Inspect the real module census, but avoid rehashing large Torch libraries
    # in every unit-test run. Actual byte hashing is tested below.
    def identity(path):
        return dict(path=str(path), sha256="a" * 64, size_bytes=path.stat().st_size)

    monkeypatch.setattr(baselines, "file_identity", identity)
    monkeypatch.setitem(sys.modules, "torch.ops.test_namespace", SimpleNamespace(__file__="torch.ops"))
    monkeypatch.setitem(sys.modules, "torch.classes.test_namespace", SimpleNamespace(__file__="torch.classes"))
    evidence = baselines.loaded_binary_provenance(backend)
    assert evidence["resolved"], evidence["errors"]
    assert not any(name in evidence["modules"] for name in (
        "torch.ops", "torch.classes", "torch.ops.test_namespace", "torch.classes.test_namespace",
    ))
    assert {"torch._ops", "torch._classes", "torch._C"} <= evidence["modules"].keys()
    versions = baselines.measurement_versions(dict.fromkeys(BASELINE_IDS, "known"), {
        backend: evidence, "natural": {"resolved": True}, "transposed": {"resolved": True},
    })
    assert ";loaded=" in versions[backend]


def test_missing_real_torch_module_still_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(baselines, "file_identity", lambda path: dict(
        path=str(path), sha256="a" * 64, size_bytes=path.stat().st_size,
    ))
    missing = tmp_path / "real_module.py"
    monkeypatch.setitem(sys.modules, "torch.real_missing_module", SimpleNamespace(__file__=str(missing)))
    evidence = baselines.loaded_binary_provenance("torch_flash")
    assert not evidence["resolved"]
    assert any("torch.real_missing_module: FileNotFoundError" in error for error in evidence["errors"])


def _xformers_loaded_files(monkeypatch, tmp_path, *, inventory, cuda_name="libtorch_cuda.so"):
    interface = tmp_path / "xformers.py"
    registration = tmp_path / "xformers_C.so"
    cuda_dir = tmp_path / "torch" / "lib"
    cuda_dir.mkdir(parents=True)
    cuda = cuda_dir / cuda_name
    interface.write_bytes(b"xformers interface")
    registration.write_bytes(b"xformers registration")
    cuda.write_bytes(b"actual ATen CUDA implementation")
    monkeypatch.setitem(sys.modules, "xformers", SimpleNamespace(__file__=str(interface)))
    monkeypatch.setitem(sys.modules, "xformers._C", SimpleNamespace(__file__=str(registration)))
    mapped = "/loaded/torch/lib/" + cuda_name

    def path(value):
        if str(value) == "/proc/self/maps":
            return SimpleNamespace(is_file=lambda: inventory == "maps", read_text=lambda:
                                   f"1000-2000 r-xp 0000 00:00 1 {mapped}\n")
        return cuda if str(value) == mapped else Path(value)

    monkeypatch.setattr(baselines, "Path", path)
    monkeypatch.setattr(torch.ops, "loaded_libraries", {str(cuda)} if inventory == "registry" else set())
    return cuda, registration


@pytest.mark.parametrize("inventory", ["registry", "maps"])
def test_xformers_hashes_loaded_aten_cuda_and_binds_revision_without_imports(monkeypatch, tmp_path, inventory):
    import builtins
    import hashlib

    cuda, registration = _xformers_loaded_files(monkeypatch, tmp_path, inventory=inventory)
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("xformers", "flashinfer", "flash_attn"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    cache = {}
    evidence = baselines.loaded_binary_provenance("cutlass_xformers", cache=cache)
    assert evidence["resolved"], evidence["errors"]
    binaries = {entry["path"]: entry for entry in evidence["binaries"]}
    assert set(binaries) == {str(cuda.resolve()), str(registration.resolve())}
    assert binaries[str(cuda.resolve())]["sha256"] == hashlib.sha256(cuda.read_bytes()).hexdigest()
    versions = dict.fromkeys(BASELINE_IDS, "known")
    native = {"natural": {"resolved": True}, "transposed": {"resolved": True}}
    before = baselines.measurement_versions(versions, dict(native, cutlass_xformers=evidence))
    cuda.write_bytes(b"different ATen CUDA implementation")
    changed = baselines.loaded_binary_provenance("cutlass_xformers", cache=cache)
    after = baselines.measurement_versions(versions, dict(native, cutlass_xformers=changed))
    assert before["cutlass_xformers"] != after["cutlass_xformers"]
    assert ";loaded=" in after["cutlass_xformers"]


@pytest.mark.parametrize("damage", ["unloaded", "missing", "cpu_only", "cuda_linalg_only"])
def test_xformers_registration_alone_cannot_resolve(monkeypatch, tmp_path, damage):
    filename = {"cpu_only": "libtorch_cpu.so", "cuda_linalg_only": "libtorch_cuda_linalg.so"}.get(damage, "libtorch_cuda.so")
    cuda, _ = _xformers_loaded_files(monkeypatch, tmp_path, inventory="registry", cuda_name=filename)
    if damage == "unloaded":
        # An installed file is not evidence that this process loaded it.
        monkeypatch.setattr(torch.ops, "loaded_libraries", set())
    elif damage == "missing":
        cuda.unlink()
    evidence = baselines.loaded_binary_provenance("cutlass_xformers")
    assert not evidence["resolved"]
    assert "executing ATen CUDA library is absent" in evidence["errors"][-1]
    result = resolve_measurements(
        case(), dict.fromkeys(BASELINE_IDS, "known"), "a" * 64,
        {"cutlass_xformers": 0.1}, {"cutlass_xformers": {"passed": True}},
        {"cutlass_xformers": {"passed": True}}, requested=["cutlass_xformers"],
        loaded_binary_provenance={"cutlass_xformers": evidence,
                                  "natural": {"resolved": True}, "transposed": {"resolved": True}},
    )
    assert result["measurements"] == [] and result["winner"] is None
