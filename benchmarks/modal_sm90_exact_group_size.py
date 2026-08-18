"""Map experimental exact-native GQA shape families on one Modal H100."""

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

app = modal.App("streamattn-sm90-exact-group-size")
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
    kv_heads: int = 4,
    head_dim: int = 64,
    split_counts: str = "",
    warmup: int = 20,
    iters: int = 200,
    repeats: int = 5,
    paired_repeats: int = 9,
    output_json: str = (
        "/artifacts/gate0/sm90_exact_g4_phase_diagram_h100_20260818.json"
    ),
) -> str:
    import json
    import os
    import subprocess
    from pathlib import Path

    os.chdir("/root/StreamAttn")
    os.environ["PYTHONUNBUFFERED"] = "1"
    output = Path(output_json)
    cell_dir = output.parent / f"{output.stem}_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, object]] = []
    explicit_splits = _parse_ints(split_counts) if split_counts else None

    for batch in _parse_ints(batches):
        groups = batch * kv_heads
        target = max(1, (256 + groups - 1) // groups)
        candidates = (
            explicit_splits
            if explicit_splits is not None
            else sorted({max(1, target // 2), target, target * 2})
        )
        for kv_len in _parse_ints(kv_lens):
            valid = [value for value in candidates if value <= kv_len // 64]
            group_size = q_heads // kv_heads
            cell_output = cell_dir / (
                f"b{batch}_n{kv_len}_g{group_size}_d{head_dim}.json"
            )
            cmd = [
                "python",
                "benchmarks/profile_sm90_exact_group_size.py",
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
                ",".join(map(str, valid)),
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
                "--repeats",
                str(repeats),
                "--paired-repeats",
                str(paired_repeats),
                "--cutlass-root",
                "/opt/flashmla-etap/csrc/cutlass",
                "--build-dir",
                "/artifacts/torch_extensions/sm90_exact_group_size",
                "--output-json",
                str(cell_output),
            ]
            print(
                f"[gqa-phase] starting B={batch} N={kv_len} C={valid}",
                flush=True,
            )
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            captured: list[str] = []
            for line in process.stdout:
                captured.append(line)
                if line.startswith("[gqa-exact]"):
                    print(line.rstrip(), flush=True)
            return_code = process.wait()
            if return_code:
                print("".join(captured[-80:]), flush=True)
                raise subprocess.CalledProcessError(return_code, cmd)

            result = json.loads(cell_output.read_text(encoding="utf-8"))
            paired = result["paired_vs_flashinfer"]
            serving_paired = result["serving_paired_vs_flashinfer"]
            best = result["best"]
            candidate = bool(result["decision"]["experimental_promotion_candidate"])
            cell = {
                "batch": batch,
                "kv_len": kv_len,
                "split_candidates": valid,
                "best_splits": best["num_splits"],
                "producer_ctas": best["producer_ctas"],
                "streamattn_ms": best["total_ms"],
                "partial_ms": best["partial_ms"],
                "merge_ms": best["merge_ms"],
                "paired_speedup_median": (
                    paired["ratio_median"] if paired is not None else None
                ),
                "paired_speedup_min": (
                    paired["ratio_min"] if paired is not None else None
                ),
                "paired_wins": paired["wins"] if paired is not None else 0,
                "paired_trials": paired["trials"] if paired is not None else 0,
                "max_abs_error": best["quality"]["max_abs_error"],
                "correctness": result["decision"]["correctness"],
                "promotion_candidate": candidate,
                "serving_backend": (
                    result["serving"]["backend_variant"]
                    if result["serving"] is not None
                    else None
                ),
                "serving_num_splits": (
                    result["serving"]["num_splits"]
                    if result["serving"] is not None
                    else None
                ),
                "serving_speedup_median": (
                    serving_paired["ratio_median"]
                    if serving_paired is not None
                    else None
                ),
                "serving_speedup_min": (
                    serving_paired["ratio_min"]
                    if serving_paired is not None
                    else None
                ),
                "serving_wins": (
                    serving_paired["wins"] if serving_paired is not None else 0
                ),
                "serving_trials": (
                    serving_paired["trials"] if serving_paired is not None else 0
                ),
                "serving_gate": result["decision"]["serving_gate"],
                "cell_artifact": str(cell_output),
            }
            cells.append(cell)
            print(
                f"[gqa-phase] B={batch} N={kv_len} C={best['num_splits']} "
                f"speedup={cell['paired_speedup_median']} "
                f"min={cell['paired_speedup_min']} "
                f"wins={cell['paired_wins']}/{cell['paired_trials']} "
                f"candidate={candidate}",
                flush=True,
            )

    winners = [cell for cell in cells if cell["promotion_candidate"]]
    aggregate = {
        "schema": "streamattn.sm90_exact_group_size_phase_diagram.v1",
        "device": "NVIDIA H100 80GB HBM3",
        "shape_family": {
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "group_size": q_heads // kv_heads,
            "head_dim": head_dim,
            "dtype": "bf16",
            "wgmma_column_utilization": (q_heads // kv_heads) / 8.0,
        },
        "summary": {
            "cells": len(cells),
            "promotion_candidates": len(winners),
            "candidate_cells": [
                {"batch": cell["batch"], "kv_len": cell["kv_len"]}
                for cell in winners
            ],
        },
        "cells": cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print(
        f"[gqa-phase] complete candidates={len(winners)}/{len(cells)} "
        f"artifact={output}",
        flush=True,
    )
    return str(output)


@app.local_entrypoint()
def main(
    batches: str = "2,4,8",
    kv_lens: str = "16384,32768,65536",
    q_heads: int = 16,
    kv_heads: int = 4,
    head_dim: int = 64,
    split_counts: str = "",
    warmup: int = 20,
    iters: int = 200,
    repeats: int = 5,
    paired_repeats: int = 9,
    output_json: str = (
        "/artifacts/gate0/sm90_exact_g4_phase_diagram_h100_20260818.json"
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
            paired_repeats=paired_repeats,
            output_json=output_json,
        )
    )
