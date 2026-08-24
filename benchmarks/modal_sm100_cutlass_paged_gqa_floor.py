"""Measure the CUTLASS SM100 paged-GQA topology at StreamAttn's target cell.

This is a bounded architecture floor, not a StreamAttn runtime dependency.  It
pins CUTLASS example 93, changes only its compile-time shape constants, runs a
short correctness gate, and captures its L2-thrashed CUDA-graph latency on a
B200.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import modal


CUTLASS_COMMIT = "7107b05535f8977f5ecb9d01ee203205b1fd9bc4"
_ROOT = "/opt/cutlass"
_BINARY = f"{_ROOT}/build/examples/93_blackwell_low_latency_gqa/93_blackwell_low_latency_gqa"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("cmake", "git", "ninja-build")
    .run_commands(
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/NVIDIA/cutlass.git /opt/cutlass && "
        "cd /opt/cutlass && "
        f"git fetch --depth=1 origin {CUTLASS_COMMIT} && "
        f"git checkout {CUTLASS_COMMIT} && "
        "sed -i 's/static constexpr int CTA_dH = 64;/static constexpr int CTA_dH = 128;/' "
        "examples/93_blackwell_low_latency_gqa/tgv_gqa.cu && "
        "sed -i 's/static constexpr int Page_Size = 32;/static constexpr int Page_Size = 16;/' "
        "examples/93_blackwell_low_latency_gqa/tgv_gqa.cu && "
        "sed -i 's/int dH = 64;/int dH = 128;/' "
        "examples/93_blackwell_low_latency_gqa/tgv_gqa.cu && "
        "sed -i 's/, 100, 1000);/, 8, 1000);/' "
        "examples/93_blackwell_low_latency_gqa/tgv_gqa.cu && "
        "sed -i 's/bool success = tester.verify();/bool success = kvL <= 2048 ? tester.verify() : true;/' "
        "examples/93_blackwell_low_latency_gqa/tgv_gqa.cu && "
        "cmake -S . -B build -GNinja "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCUTLASS_NVCC_ARCHS=100a "
        "-DCUTLASS_ENABLE_TESTS=OFF "
        "-DCUTLASS_ENABLE_EXAMPLES=ON "
        "-DCUTLASS_ENABLE_LIBRARY=OFF && "
        "cmake --build build --target 93_blackwell_low_latency_gqa -j2",
    )
)

app = modal.App("streamattn-sm100-cutlass-paged-gqa-floor")

_TIME_RE = re.compile(r"Average time per iteration:\s+([0-9.eE+-]+)\s+ms")
_CORRECT_RE = re.compile(r"Correctness test mode=1 (PASSED|FAILED)")
_DEVICE_RE = re.compile(r"Device:\s*(.+)")


def _run_cell(*, batch: int, kv_len: int) -> dict[str, object]:
    command = [
        _BINARY,
        "--kvL",
        str(kv_len),
        "--kvH",
        "2",
        "--qH",
        "16",
        "--qL",
        "1",
        "--BS",
        str(batch),
        "--mode",
        "1",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30 * 60,
    )
    output = completed.stdout + completed.stderr
    timing = _TIME_RE.search(output)
    correctness = _CORRECT_RE.search(output)
    if timing is None or correctness is None:
        raise RuntimeError(f"could not parse CUTLASS output:\n{output}")
    correctness_executed = kv_len <= 2048
    return {
        "batch": batch,
        "kv_len": kv_len,
        "milliseconds": float(timing.group(1)),
        "correctness_executed": correctness_executed,
        "correctness": (
            correctness.group(1).lower() if correctness_executed else "not_run"
        ),
        "command": command,
    }


@app.function(image=image, gpu="B200", timeout=60 * 60)
def profile_b200(batches: list[int]) -> dict[str, object]:
    correctness = _run_cell(batch=1, kv_len=2048)
    cells = [_run_cell(batch=batch, kv_len=32768) for batch in batches]
    device = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0]
    return {
        "schema": "streamattn.sm100_cutlass_paged_gqa_floor.v1",
        "kind": "external_topology_floor",
        "cutlass_commit": CUTLASS_COMMIT,
        "device": device,
        "compute_capability": [10, 0],
        "shape": {
            "q_heads": 16,
            "kv_heads": 2,
            "group_size": 8,
            "head_dim": 128,
            "page_size": 16,
            "dtype": "bf16",
            "layout": "NHD-equivalent combined KV",
            "max_splits": 8,
            "reduction_ctas": 8,
        },
        "correctness": correctness,
        "cells": cells,
    }


@app.local_entrypoint()
def main(
    batches: str = "1,2,4,8",
    output_json: str = "",
) -> None:
    batch_values = [int(item) for item in batches.split(",") if item.strip()]
    result = profile_b200.remote(batch_values)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if output_json:
        output = Path(output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        print(json.dumps({"artifact": str(output), "cells": len(result["cells"])}, indent=2))
    else:
        print(payload)
