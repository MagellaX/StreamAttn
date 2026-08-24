"""Validate the portable StreamAttn wheel without importing PyTorch."""

from __future__ import annotations

import argparse
import json
import zipfile
from email.parser import BytesParser
from pathlib import Path


REQUIRED_FILES = {
    "stream_attention/__init__.py",
    "stream_attention/backends/sm100/csrc/common.cuh",
    "stream_attention/backends/sm100/csrc/tgv_gqa.cuh",
    "stream_attention/backends/sm100/csrc/tgv_gqa_paged.cuh",
    "stream_attention/backends/sm100/paged_gqa_exact.py",
    "stream_attention/backends/sm100/paged_gqa_exact_sources.py",
    "stream_attention/backends/sm90/transposed_gqa_exact.py",
    "stream_attention/backends/sm90/transposed_gqa_exact_sources.py",
    "stream_attention/kernels/gate0_seed_only_triton.py",
    "stream_attention/kernels/qwen_o_proj_triton.py",
    "stream_attention/policies/registry.json",
}
FORBIDDEN_PREFIXES = ("artifacts/", "benchmarks/", "tests/")
EXPECTED_METADATA = {
    "Name": "stream-attention",
    "Requires-Python": ">=3.10",
    "License-Expression": "Apache-2.0",
}
EXPECTED_REPOSITORY_URL = "https://github.com/MagellaX/StreamAttn"


def check_wheel(path: Path) -> dict[str, object]:
    failures: list[str] = []
    if not path.name.endswith("-py3-none-any.whl"):
        failures.append(f"wheel_tag_not_portable:{path.name}")
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        for required in sorted(REQUIRED_FILES):
            if required not in names:
                failures.append(f"missing:{required}")
        for name in sorted(names):
            if name.startswith(FORBIDDEN_PREFIXES):
                failures.append(f"forbidden:{name}")

        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_files = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata_files) != 1:
            failures.append(f"metadata_file_count:{len(metadata_files)}")
        else:
            metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))
            for key, expected in EXPECTED_METADATA.items():
                actual = metadata.get(key)
                if actual != expected:
                    failures.append(f"metadata_mismatch:{key}:{actual!r}")

            project_urls = metadata.get_all("Project-URL") or []
            if not any(
                value == f"Repository, {EXPECTED_REPOSITORY_URL}"
                for value in project_urls
            ):
                failures.append("metadata_repository_url_missing")

            requirements = metadata.get_all("Requires-Dist") or []
            mandatory_triton = [
                value
                for value in requirements
                if value.lower().startswith("triton") and 'extra == "triton"' not in value
            ]
            if mandatory_triton:
                failures.append("metadata_triton_is_mandatory")
        if len(wheel_files) != 1:
            failures.append(f"wheel_file_count:{len(wheel_files)}")
        else:
            wheel_metadata = BytesParser().parsebytes(wheel.read(wheel_files[0]))
            if "py3-none-any" not in (wheel_metadata.get_all("Tag") or []):
                failures.append("wheel_metadata_portable_tag_missing")

        registry_name = "stream_attention/policies/registry.json"
        if registry_name in names:
            try:
                registry = json.loads(wheel.read(registry_name))
            except (UnicodeDecodeError, json.JSONDecodeError):
                failures.append("registry_json_invalid")
            else:
                if not isinstance(registry, dict):
                    failures.append("registry_root_invalid")
                else:
                    policies = registry.get("policies") or []
                    if not isinstance(policies, list):
                        failures.append("registry_policies_invalid")
                        policies = []
                    for entry in policies:
                        if not isinstance(entry, dict):
                            failures.append("registry_entry_invalid")
                            continue
                        relative = entry.get("path")
                        if not isinstance(relative, str):
                            failures.append(
                                f"registry_path_invalid:{entry.get('name')}"
                            )
                            continue
                        packaged = f"stream_attention/{relative}"
                        if packaged not in names:
                            failures.append(f"registry_policy_missing:{packaged}")

    return {
        "schema": "streamattn.wheel_contents.v1",
        "wheel": str(path),
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    result = check_wheel(args.wheel)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
