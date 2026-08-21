import copy
import zipfile
from pathlib import Path

from benchmarks.check_seed_only_policy_route_smoke import (
    _check_policy_registry,
    _load_registry_json,
)
from benchmarks.check_wheel_contents import check_wheel


def _registry_failures(payload):
    return _check_policy_registry(payload)["registry_failures"]


def test_required_policy_must_remain_green():
    payload = copy.deepcopy(_load_registry_json())
    entries = {entry["name"]: entry for entry in payload["policies"]}
    entries["qwen25_3b_l2_s416_32k_seed_only_batched"]["status"] = "candidate"
    entries["qwen25_3b_l2_s640_32k_seed_only_batched"]["status"] = "green"

    assert (
        "registry_missing_green_cell:qwen25_3b_l2_s416_32k_seed_only_batched"
        in _registry_failures(payload)
    )


def test_registry_lookup_aliases_cannot_collide():
    payload = copy.deepcopy(_load_registry_json())
    first, second = payload["policies"][:2]
    first["aliases"] = [second["name"]]

    assert any(
        failure.startswith("registry_lookup_key_duplicate:")
        for failure in _registry_failures(payload)
    )


def test_boolean_min_batch_is_rejected():
    payload = copy.deepcopy(_load_registry_json())
    payload["policies"][0]["min_batch"] = True

    assert any(
        failure.startswith("registry_min_batch_invalid:")
        for failure in _registry_failures(payload)
    )


def test_wheel_checker_reports_malformed_registry(tmp_path: Path):
    wheel = tmp_path / "stream_attention-1.0.0-py3-none-any.whl"
    metadata = """Metadata-Version: 2.4
Name: stream-attention
Version: 1.0.0
Requires-Python: >=3.10
License-Expression: Apache-2.0
Project-URL: Repository, https://github.com/MagellaX/StreamAttn
"""
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("stream_attention/__init__.py", "")
        archive.writestr("stream_attention/policies/registry.json", "not-json")
        archive.writestr("stream_attention-1.0.0.dist-info/METADATA", metadata)
        archive.writestr(
            "stream_attention-1.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )

    result = check_wheel(wheel)

    assert "registry_json_invalid" in result["failures"]


def test_wheel_checker_rejects_platform_tag(tmp_path: Path):
    wheel = tmp_path / "stream_attention-1.0.0-cp311-cp311-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("stream_attention/__init__.py", "")

    result = check_wheel(wheel)

    assert any(
        failure.startswith("wheel_tag_not_portable:")
        for failure in result["failures"]
    )
