"""Map the SM90 exact-native D64/G8 decode region in one Modal H100 job."""

from __future__ import annotations

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ninja-build")
    .pip_install(
        "einops",
        "flashinfer-python==0.6.12",
        "flashinfer-cubin==0.6.12",
        "ninja",
    )
    .run_commands(
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/pengcuo/FlashMLA-ETAP.git /opt/flashmla-etap && "
        "cd /opt/flashmla-etap && "
        "git sparse-checkout init --cone && "
        "git sparse-checkout set csrc/cutlass/include && "
        "git fetch --depth=1 origin 39e616041ae6fb1243a0f6ac891e72d576b640e5 && "
        "git checkout 39e616041ae6fb1243a0f6ac891e72d576b640e5"
    )
    .add_local_dir("benchmarks", remote_path="/root/StreamAttn/benchmarks", copy=True)
    .add_local_dir(
        "stream_attention", remote_path="/root/StreamAttn/stream_attention", copy=True
    )
)

app = modal.App("streamattn-transposed-wgmma-exact-phase-diagram")
volume = modal.Volume.from_name("streamattn-artifacts", create_if_missing=True)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/artifacts": volume},
)
def run(
    *,
    batches: str = "2,4,8",
    kv_lens: str = "16384,32768,65536",
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 64,
    split_counts: str = "",
    warmup: int = 20,
    iters: int = 200,
    repeats: int = 9,
    output_json: str = (
        "/artifacts/gate0/"
        "transposed_wgmma_exact_phase_diagram_h100_20260818.json"
    ),
) -> str:
    import json
    import os
    import subprocess
    from pathlib import Path

    os.chdir("/root/StreamAttn")
    os.environ["PYTHONUNBUFFERED"] = "1"
    cutlass_root = "/opt/flashmla-etap/csrc/cutlass"
    output = Path(output_json)
    cell_dir = output.parent / f"{output.stem}_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, object]] = []
    explicit_splits = _parse_ints(split_counts) if split_counts else None

    for batch in _parse_ints(batches):
        target_splits = max(1, (256 + batch * kv_heads - 1) // (batch * kv_heads))
        split_candidates = (
            explicit_splits
            if explicit_splits is not None
            else sorted(
                {
                    max(1, target_splits // 2),
                    target_splits,
                    min(512, target_splits * 2),
                }
            )
        )
        for kv_len in _parse_ints(kv_lens):
            valid_splits = [value for value in split_candidates if value <= kv_len // 64]
            cell_name = f"b{batch}_n{kv_len}_h{q_heads}_kv{kv_heads}_d{head_dim}"
            cell_output = cell_dir / f"{cell_name}.json"
            cmd = [
                "python",
                "benchmarks/profile_transposed_wgmma_exact_qk.py",
                "--batch",
                str(batch),
                "--kv-len",
                str(kv_len),
                "--q-heads",
                str(q_heads),
                "--kv-heads",
                str(kv_heads),
                "--head-dim",
                str(head_dim),
                "--num-splits-list",
                ",".join(map(str, valid_splits)),
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
                "--repeats",
                str(repeats),
                "--cutlass-root",
                cutlass_root,
                "--build-dir",
                "/artifacts/torch_extensions/transposed_wgmma_exact_qk",
                "--output-json",
                str(cell_output),
            ]
            print(
                f"[phase] starting B={batch} N={kv_len} splits={valid_splits}",
                flush=True,
            )
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                if line.startswith("[transposed-qk]"):
                    print(line.rstrip(), flush=True)
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, cmd)

            result = json.loads(cell_output.read_text(encoding="utf-8"))
            paired = result["timing"]["paired_warp_merge_vs_flashinfer"]
            paired_serving = result["timing"]["paired_serving_vs_flashinfer"]
            quality = result["backend_plan_quality"]
            robust_win = (
                result["decision"]["exact_native_gate"] == "pass"
                and paired["wins"] == paired["trials"]
                and paired["ratio_min"] > 1.0
            )
            cell = {
                "batch": batch,
                "kv_len": kv_len,
                "split_candidates": valid_splits,
                "best_splits": result["timing"][
                    "best_exact_end_to_end_warp_splits"
                ],
                "producer_ctas": (
                    batch
                    * kv_heads
                    * result["timing"]["best_exact_end_to_end_warp_splits"]
                ),
                "streamattn_ms": result["timing"]["backend_plan_ms"],
                "serving_dispatch_ms": result["timing"]["serving_dispatch_ms"],
                "flashinfer_ms": result["timing"]["flashinfer_batched_exact_ms"],
                "paired_speedup_median": paired["ratio_median"],
                "paired_speedup_min": paired["ratio_min"],
                "paired_wins": paired["wins"],
                "paired_trials": paired["trials"],
                "max_abs_error": quality["max_abs_error"],
                "repeat_max_abs_diff": quality["repeat_max_abs_diff"],
                "nonfinite_count": quality["nonfinite_count"],
                "exact_native_gate": result["decision"]["exact_native_gate"],
                "serving_backend": (
                    result["serving_dispatch_quality"]["backend_variant"]
                    if result["serving_dispatch_quality"] is not None
                    else None
                ),
                "serving_num_splits": (
                    result["serving_dispatch_quality"]["num_splits"]
                    if result["serving_dispatch_quality"] is not None
                    else None
                ),
                "paired_serving_speedup_median": (
                    paired_serving["ratio_median"]
                    if paired_serving is not None
                    else None
                ),
                "paired_serving_speedup_min": (
                    paired_serving["ratio_min"]
                    if paired_serving is not None
                    else None
                ),
                "paired_serving_wins": (
                    paired_serving["wins"] if paired_serving is not None else None
                ),
                "serving_dispatch_gate": result["decision"][
                    "serving_dispatch_gate"
                ],
                "robust_win": robust_win,
                "cell_artifact": str(cell_output),
            }
            cells.append(cell)
            serving_suffix = (
                f" serving={paired_serving['ratio_median']:.6f}x"
                if paired_serving is not None
                else ""
            )
            print(
                f"[phase] B={batch} N={kv_len} C={cell['best_splits']} "
                f"speedup={paired['ratio_median']:.6f}x "
                f"min={paired['ratio_min']:.6f}x "
                f"wins={paired['wins']}/{paired['trials']} "
                f"robust={robust_win}{serving_suffix}",
                flush=True,
            )

    robust_cells = [cell for cell in cells if cell["robust_win"]]
    aggregate = {
        "schema": "streamattn.transposed_wgmma_exact_phase_diagram.v1",
        "device": "NVIDIA H100 80GB HBM3",
        "shape_family": {
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "group_size": q_heads // kv_heads,
            "head_dim": head_dim,
            "dtype": "bf16",
        },
        "benchmark": {
            "warmup": warmup,
            "iters": iters,
            "repeats": repeats,
            "target_producer_ctas": 256,
            "split_search": (
                split_counts if explicit_splits is not None else "half,target,double"
            ),
        },
        "summary": {
            "cells": len(cells),
            "robust_wins": len(robust_cells),
            "robust_win_cells": [
                {"batch": cell["batch"], "kv_len": cell["kv_len"]}
                for cell in robust_cells
            ],
        },
        "cells": cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print(
        f"[phase] complete robust_wins={len(robust_cells)}/{len(cells)} "
        f"artifact={output}",
        flush=True,
    )
    return str(output)


@app.local_entrypoint()
def main(
    batches: str = "2,4,8",
    kv_lens: str = "16384,32768,65536",
    q_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 64,
    split_counts: str = "",
    warmup: int = 20,
    iters: int = 200,
    repeats: int = 9,
    output_json: str = (
        "/artifacts/gate0/"
        "transposed_wgmma_exact_phase_diagram_h100_20260818.json"
    ),
) -> None:
    print(
        run.remote(
            batches=batches,
            kv_lens=kv_lens,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            split_counts=split_counts,
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            output_json=output_json,
        )
    )
