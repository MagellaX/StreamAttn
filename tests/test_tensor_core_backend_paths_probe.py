from pathlib import Path

from benchmarks.probe_tensor_core_backend_paths import (
    _decision,
    _inspect_cutlass,
    _inspect_thunderkittens,
)


def test_inspect_cutlass_finds_cute_include_root(tmp_path: Path) -> None:
    header = tmp_path / "cutlass" / "include" / "cute" / "tensor.hpp"
    header.parent.mkdir(parents=True)
    header.write_text("// cute smoke\n", encoding="utf-8")

    info = _inspect_cutlass(tmp_path / "cutlass")

    assert info["compile_candidate"] is True
    assert info["cute_tensor_header"] == str(header.resolve())
    assert info["include_root"] == str((tmp_path / "cutlass" / "include").resolve())


def test_inspect_thunderkittens_finds_kittens_header(tmp_path: Path) -> None:
    header = tmp_path / "ThunderKittens" / "include" / "kittens.cuh"
    header.parent.mkdir(parents=True)
    header.write_text("// tk smoke\n", encoding="utf-8")

    info = _inspect_thunderkittens(tmp_path / "ThunderKittens")

    assert info["compile_candidate"] is True
    assert info["kittens_header"] == str(header.resolve())
    assert info["include_root"] == str(header.parent.resolve())


def test_decision_prefers_thunderkittens_compile_success() -> None:
    result = {
        "compile_smokes": {
            "thunderkittens": {"available": True},
            "cutlass_cute": {"available": True},
        },
        "flashinfer_context": {"installed_wheel_has_cuda_sources": True},
    }

    decision = _decision(result)

    assert decision["recommended_next_path"] == "thunderkittens_head_mode_decode_spike"
    assert "SEED_ONLY" in decision["streamattn_unique_requirement"]


def test_decision_falls_back_to_fetch_sources() -> None:
    result = {
        "compile_smokes": {
            "thunderkittens": {"available": False},
            "cutlass_cute": {"available": False},
        },
        "flashinfer_context": {"installed_wheel_has_cuda_sources": False},
    }

    decision = _decision(result)

    assert decision["recommended_next_path"] == "fetch_tensor_core_backend_sources_then_retry"
