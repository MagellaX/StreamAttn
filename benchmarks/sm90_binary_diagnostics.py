"""Offline inspection of the loaded CUDA binary, never a compile or GPU launch.

Call outside capture/timing. Kernel names are full demangled identities, including
every template argument but excluding return/parameter types. Missing, duplicate,
or mismatched symbols fail closed instead of attributing a neighboring kernel.
"""

from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _run(command: list[str], timeout: float) -> str:
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=timeout
    )
    return result.stdout


def _tool(name: str, cuda_home: str | None) -> str:
    if cuda_home:
        for suffix in ("", ".exe"):
            path = Path(cuda_home) / "bin" / (name + suffix)
            if path.is_file():
                return str(path)
        raise FileNotFoundError(f"{name} missing from build toolkit {cuda_home}")
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(f"{name} is required for CUDA binary diagnostics")
    return resolved


def _function_sections(text: str) -> list[tuple[str, str]]:
    headers = list(re.finditer(r"^\s*Function\s*:?\s+([^\s:]+):?\s*$", text, re.M))
    return [
        (
            match[1],
            text[
                match.end() : headers[i + 1].start()
                if i + 1 < len(headers)
                else len(text)
            ],
        )
        for i, match in enumerate(headers)
    ]


def _kernel_name(demangled: str) -> str:
    name = demangled.strip().removeprefix("void ")
    template_depth = 0
    for index, character in enumerate(name):
        if character == "<":
            template_depth += 1
        elif character == ">":
            template_depth -= 1
        elif character == "(" and template_depth == 0:
            return name[:index].strip()
    return name


def _kernel_identity(demangled: str) -> str:
    name = _kernel_name(demangled)
    # CUDA 12.8 cu++filt spells non-type arguments as (int)8 and (bool)1.
    # Keep bare integer 0/1 distinct from bool arguments when canonicalizing.
    name = re.sub(
        r"\(bool\)\s*([01])(?=\s*[,>])",
        lambda match: "true" if match[1] == "1" else "false",
        name,
    )
    name = re.sub(r"\(int\)\s*([+-]?\d+)(?=\s*[,>])", r"\1", name)
    return re.sub(r"\s+", "", name)


def inspect_cuda_binary(
    binary_path: str | Path,
    *,
    kernel_names: Mapping[str, str] | None = None,
    runtime_resources: Mapping[str, dict] | None = None,
    build_metadata: Mapping[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Return JSON-safe provenance and exact-symbol SASS/resources; no files written.

    ``kernel_names`` maps component names to exact demangled kernel identities.
    Retained PTX/cubins come from the original nvcc --keep build, never a rebuild.
    ``build_metadata`` uses the fields attached by the M128 compiler wrapper;
    other experimental compilers may supply the same fields directly.
    """
    if kernel_names == {} or timeout <= 0:
        raise ValueError("require nonempty kernel names and a positive timeout")
    metadata = dict(build_metadata or {})
    binary = file_identity(binary_path)
    loaded = metadata.get("loaded_binary")
    if loaded and any(binary[key] != loaded[key] for key in ("path", "sha256")):
        raise ValueError("binary no longer matches the extension recorded at load time")
    cuda_home = metadata.get("cuda_home")
    cuobjdump = _tool("cuobjdump", cuda_home)
    demangler = _tool("cu++filt", cuda_home)
    resource_dump = _run([cuobjdump, "--dump-resource-usage", binary["path"]], timeout)
    sections = _function_sections(resource_dump)
    if not sections:
        raise ValueError("no device function resources found in the loaded binary")
    symbols = list(dict.fromkeys(symbol for symbol, _ in sections))
    demangled = _run([demangler, *symbols], timeout).splitlines()
    if len(demangled) != len(symbols):
        raise ValueError("demangler did not return exactly one name per device symbol")
    names = dict(zip(symbols, demangled))
    explicit_selection = kernel_names is not None
    if kernel_names is None:
        kernel_names = {
            f"kernel_{i}": _kernel_name(name) for i, name in enumerate(demangled)
        }

    intermediate_dir = metadata.get("intermediates_dir")
    intermediate_files = (
        sorted(Path(intermediate_dir).rglob("*")) if intermediate_dir else []
    )
    ptx_files: list[dict] = []
    cubins: list[dict] = []
    for path in intermediate_files:
        if path.is_file() and path.suffix == ".ptx":
            entry_names = re.findall(r"\.entry\s+([\w$]+)\s*\(", path.read_text())
            ptx_files.append({**file_identity(path), "entry_symbols": entry_names})
        elif path.is_file() and path.suffix == ".cubin":
            cubins.append(file_identity(path))
    if metadata.get("keep_intermediates") and not ptx_files:
        raise ValueError("diagnostic build requested retained PTX, but none was found")

    records = {}
    for component, expected in kernel_names.items():
        matches = [
            symbol
            for symbol, name in names.items()
            if _kernel_identity(name) == _kernel_identity(expected)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one binary symbol for {expected!r}; got {matches}"
            )
        symbol = matches[0]
        blocks = [body for candidate, body in sections if candidate == symbol]
        if len(blocks) != 1:
            raise ValueError(
                f"ambiguous resource records (multiple device images) for {symbol}"
            )
        linked_ptx = [row for row in ptx_files if symbol in row["entry_symbols"]]
        if metadata.get("keep_intermediates") and not linked_ptx:
            raise ValueError(f"exact binary symbol {symbol} missing from retained PTX")
        sass_command = [cuobjdump, "--dump-sass", "--function", symbol, binary["path"]]
        sass = _run(sass_command, timeout)
        if [name for name, _ in _function_sections(sass)] != [symbol]:
            raise ValueError(
                f"SASS dump did not contain only the exact requested symbol {symbol}"
            )
        binary_resources = {
            key: int(value)
            for key, value in re.findall(
                r"([A-Z][A-Z0-9_]*(?:\[\d+\])?):(\d+)", blocks[0]
            )
        }
        if "REG" not in binary_resources:
            raise ValueError(f"no register resource record for exact symbol {symbol}")
        records[component] = {
            "mangled_symbol": symbol,
            "demangled_symbol": names[symbol],
            "requested_kernel": expected,
            "binary": binary,
            "ptx": linked_ptx,
            "binary_resources": binary_resources,
            "binary_resource_text": blocks[0].strip(),
            "runtime_resources": dict((runtime_resources or {}).get(component, {})),
            "sass_command": sass_command,
            "sass": sass,
        }

    tools = {}
    for name in ("nvcc", "ptxas", "cuobjdump", "cu++filt"):
        executable = _tool(name, cuda_home)
        version_flag = "-v" if name == "cu++filt" else "--version"
        tools[name] = {
            "path": executable,
            "version": _run([executable, version_flag], timeout).strip(),
        }
    directory = Path(metadata.get("build_directory") or Path(binary["path"]).parent)
    build_files = [
        file_identity(directory / name)
        for name in ("build.ninja", "cuda.cu", "main.cpp")
        if (directory / name).is_file()
    ]
    cutlass_git = None
    if metadata.get("cutlass_root"):
        root = str(metadata["cutlass_root"])
        try:
            cutlass_git = {
                "toplevel": _run(
                    ["git", "-C", root, "rev-parse", "--show-toplevel"], timeout
                ).strip(),
                "revision": _run(
                    ["git", "-C", root, "rev-parse", "HEAD"], timeout
                ).strip(),
                "status": _run(
                    ["git", "-C", root, "status", "--porcelain", "--", "."], timeout
                ).strip(),
            }
        except (OSError, subprocess.SubprocessError) as error:
            cutlass_git = {"unavailable": str(error)}
    if file_identity(binary["path"]) != binary:
        raise ValueError("loaded binary path changed during inspection")
    return {
        "schema": "streamattn.sm90_binary_diagnostics.v1",
        "selection": "exact_kernel_names"
        if explicit_selection
        else "all_device_symbols",
        "binary": binary,
        "build_metadata": metadata,
        "build_files": build_files,
        "cutlass_git": cutlass_git,
        "tools": tools,
        "retained_cubins": cubins,
        "kernels": records,
    }


def _ptx_entry(text: str, symbol: str) -> str:
    match = re.search(r"\.entry\s+" + re.escape(symbol) + r"\s*\(", text)
    if match is None:
        raise ValueError(f"PTX entry missing for {symbol}")
    # Count delimiters with comments masked, preserving offsets into the PTX.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", lambda m: " " * len(m[0]), text, flags=re.S)
    begin = code.find("{", match.end())
    if begin == -1:
        raise ValueError(f"PTX entry has no body for {symbol}")
    depth = 0
    for end in range(begin, len(code)):
        depth += (code[end] == "{") - (code[end] == "}")
        if depth == 0:
            return text[match.start() : end + 1]
    raise ValueError(f"PTX entry is unterminated for {symbol}")


def instruction_counts(sass: str, ptx: str = "") -> dict[str, Any]:
    """Static instruction counts, not dynamic execution counts or overlap proof."""
    opcodes = Counter(
        re.findall(
            r"/\*[0-9a-fA-F]+\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Za-z0-9_.]*)\b", sass
        )
    )
    ptx_code = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
    waits = Counter(
        re.findall(r"wgmma\.wait_group\.sync\.aligned\s+(\d+)\s*;", ptx_code)
    )
    return {
        "kind": "static_instructions",
        "sass_opcodes": dict(sorted(opcodes.items())),
        "sass_relevant": {
            key: count
            for key, count in sorted(opcodes.items())
            if key.startswith(
                (
                    "HGMMA",
                    "WGMMA",
                    "WARPGROUP",
                    "DEPBAR",
                    "BAR",
                    "LDGSTS",
                    "LDL",
                    "STL",
                    "MUFU",
                )
            )
        },
        "ptx_wgmma_mma_async": len(re.findall(r"\bwgmma\.mma_async\.", ptx_code)),
        "ptx_wgmma_commit_group": len(re.findall(r"\bwgmma\.commit_group\.", ptx_code)),
        "ptx_wgmma_wait_group": dict(sorted(waits.items())),
    }


def inspect_extension(
    extension: Any,
    output_dir: str | Path,
    *,
    kernel_names: Mapping[str, str] | None = None,
    runtime_resources: Mapping[str, dict] | None = None,
    build_metadata: Mapping[str, Any] | None = None,
    timeout: float = 60.0,
    include_archive: bool = True,
) -> dict[str, Any]:
    """Write SASS/PTX/resources/binaries/manifest and return a portable JSON report.

    The default report includes ``archive.data`` (base64 tar.gz) so a Modal return
    value can transport the artifacts, not merely remote paths. Set
    ``include_archive=False`` to return only archive path/hash/size. Omitting
    ``kernel_names`` inspects all symbols and makes no timed-symbol attribution.
    """
    metadata = (
        build_metadata
        if build_metadata is not None
        else getattr(extension, "_streamattn_build_metadata", {})
    )
    report = inspect_cuda_binary(
        extension.__file__,
        kernel_names=kernel_names,
        runtime_resources=runtime_resources,
        build_metadata=metadata,
        timeout=timeout,
    )
    selection_id = hashlib.sha256(
        json.dumps(kernel_names, sort_keys=True).encode()
    ).hexdigest()[:12]
    directory = (
        Path(output_dir).resolve()
        / f"binary_{report['binary']['sha256'][:16]}_{selection_id}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict] = {}

    def copy_artifact(identity: dict, label: str) -> dict:
        destination = directory / label
        if Path(identity["path"]).resolve() != destination:
            shutil.copy2(identity["path"], destination)
        actual = file_identity(destination)
        if actual["sha256"] != identity["sha256"]:
            raise ValueError(f"artifact changed after inspection: {identity['path']}")
        artifacts[label] = {**actual, "archive_member": f"{directory.name}/{label}"}
        return artifacts[label]

    def write_artifact(label: str, text: str) -> dict:
        destination = directory / label
        destination.write_text(text, encoding="utf-8")
        artifacts[label] = {
            **file_identity(destination),
            "archive_member": f"{directory.name}/{label}",
        }
        return artifacts[label]

    report["archived_binary"] = copy_artifact(
        report["binary"], Path(report["binary"]["path"]).name
    )
    for index, identity in enumerate(report["build_files"] + report["retained_cubins"]):
        copy_artifact(identity, f"build_{index}_{Path(identity['path']).name}")
    for index, row in enumerate(report["kernels"].values()):
        prefix = f"kernel_{index}"
        sass = row.pop("sass")
        row["sass_file"] = write_artifact(prefix + ".sass", sass)
        row["resource_file"] = write_artifact(
            prefix + ".resources.txt", row["binary_resource_text"]
        )
        entries = []
        row["ptx_files"] = []
        for ptx_index, identity in enumerate(row["ptx"]):
            full_ptx = copy_artifact(identity, f"{prefix}_{ptx_index}.ptx")
            row["ptx_files"].append(full_ptx)
            entry = _ptx_entry(
                Path(full_ptx["path"]).read_text(), row["mangled_symbol"]
            )
            entries.append(entry)
            write_artifact(f"{prefix}_{ptx_index}.entry.ptx", entry)
        if len(entries) > 1:
            raise ValueError(
                f"ambiguous retained PTX entries for {row['mangled_symbol']}"
            )
        row["instruction_counts"] = instruction_counts(
            sass, entries[0] if entries else ""
        )
        if not row["instruction_counts"]["sass_opcodes"]:
            raise ValueError(f"no SASS instructions found for {row['mangled_symbol']}")
        row["instruction_counts"]["ptx_available"] = bool(entries)
    report["artifacts"] = artifacts
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["manifest"] = file_identity(manifest)
    archive_path = shutil.make_archive(
        str(directory), "gztar", root_dir=directory.parent, base_dir=directory.name
    )
    report["archive"] = file_identity(archive_path)
    if include_archive:
        report["archive"].update(
            encoding="base64",
            data=base64.b64encode(Path(archive_path).read_bytes()).decode("ascii"),
        )
    return report


def inspect_plan_binary(
    plan: Any, output_dir: str | Path, **kwargs: Any
) -> dict[str, Any]:
    return inspect_extension(
        plan.extension,
        output_dir,
        kernel_names=plan.kernel_names,
        runtime_resources=plan.resources,
        **kwargs,
    )
