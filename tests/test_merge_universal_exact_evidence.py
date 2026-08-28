import json
from pathlib import Path

import pytest

from benchmarks.merge_universal_exact_evidence import merge_evidence_payloads


def _artifact(path: Path, *, cell_ids: tuple[str, ...], device_uuid: str) -> Path:
    environment = {
        "architecture": "sm80",
        "device_name": "NVIDIA A100 80GB PCIe",
        "device_uuid": device_uuid,
        "driver_version": "580.95.05",
        "cuda_version": "12.8",
        "torch_version": "2.7.1+cu128",
        "library_versions": {"cudnn": "90701"},
        "compiler_versions": {"triton": "3.3.1"},
    }
    payload = {
        "evidence": [
            {
                "evidence_id": f"{cell_id}:external:test",
                "cell_id": cell_id,
                "environment": environment,
                "provider": "external" if cell_id == "train" else "streamattn",
                "detail": None,
            }
            for cell_id in cell_ids
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_merge_filters_cells_and_accepts_compatible_gpu_uuids(tmp_path):
    first = _artifact(
        tmp_path / "first.json",
        cell_ids=("decode", "prefill"),
        device_uuid="GPU-first",
    )
    second = _artifact(
        tmp_path / "second.json",
        cell_ids=("train",),
        device_uuid="GPU-second",
    )

    payload = merge_evidence_payloads(
        (first, second),
        exclude_cells=frozenset({"decode"}),
    )

    assert [row["cell_id"] for row in payload["evidence"]] == ["prefill", "train"]
    assert all("source artifact:" in row["detail"] for row in payload["evidence"])

    external_only = merge_evidence_payloads(
        (first, second),
        include_providers=frozenset({"external"}),
    )
    assert [row["cell_id"] for row in external_only["evidence"]] == ["train"]


def test_merge_rejects_overlapping_include_and_exclude_filters(tmp_path):
    artifact = _artifact(
        tmp_path / "artifact.json",
        cell_ids=("decode",),
        device_uuid="GPU-first",
    )

    with pytest.raises(ValueError, match="both included and excluded"):
        merge_evidence_payloads(
            (artifact,),
            include_cells=frozenset({"decode"}),
            exclude_cells=frozenset({"decode"}),
        )
