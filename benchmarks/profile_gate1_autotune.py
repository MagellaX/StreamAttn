"""Run Gate-1 real-shape autotune configs in isolated subprocesses.

Some Triton tile/meta-parameter combinations can abort the compiler process.
This wrapper launches one config at a time so a bad config becomes one JSON
error row instead of killing the entire sweep.
"""

import argparse
import itertools
import json
import subprocess
import sys


def _parse_values(raw: str, cast):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def _command(args, *, seq, heads, dim, block_size, tile_size_q, num_warps, num_stages, active_fraction):
    cmd = [
        sys.executable,
        "benchmarks/profile_gate1_real_shapes.py",
        "--seq",
        str(seq),
        "--heads",
        str(heads),
        "--dim",
        str(dim),
        "--active-fraction",
        str(active_fraction),
        "--block-size",
        str(block_size),
        "--tile-size-q",
        str(tile_size_q),
        "--num-warps",
        str(num_warps),
        "--num-stages",
        str(num_stages),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--metadata-warmup",
        str(args.metadata_warmup),
        "--metadata-iters",
        str(args.metadata_iters),
    ]
    if args.causal:
        cmd.append("--causal")
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="4096")
    parser.add_argument("--heads", default="16")
    parser.add_argument("--dims", default="128")
    parser.add_argument("--active-fractions", default="0.0625,1.0")
    parser.add_argument("--block-sizes", default="64")
    parser.add_argument("--tile-sizes-q", default="32,64,128")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--num-stages", default="2,3,4")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--metadata-warmup", type=int, default=5)
    parser.add_argument("--metadata-iters", type=int, default=8)
    args = parser.parse_args()

    rows = []
    configs = itertools.product(
        _parse_values(args.seqs, int),
        _parse_values(args.heads, int),
        _parse_values(args.dims, int),
        _parse_values(args.block_sizes, int),
        _parse_values(args.tile_sizes_q, int),
        _parse_values(args.num_warps, int),
        _parse_values(args.num_stages, int),
        _parse_values(args.active_fractions, float),
    )
    for seq, heads, dim, block_size, tile_size_q, num_warps, num_stages, active_fraction in configs:
        cmd = _command(
            args,
            seq=seq,
            heads=heads,
            dim=dim,
            block_size=block_size,
            tile_size_q=tile_size_q,
            num_warps=num_warps,
            num_stages=num_stages,
            active_fraction=active_fraction,
        )
        try:
            output = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                text=True,
            )
            payload = json.loads(output)
            rows.extend(payload.get("rows", []))
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "shape": {
                        "seq": seq,
                        "heads": heads,
                        "dim": dim,
                        "dtype": args.dtype,
                    },
                    "block_size": block_size,
                    "tile_size_q": tile_size_q,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                    "active_fraction": active_fraction,
                    "error": f"subprocess failed with code {exc.returncode}",
                    "output_tail": (exc.output or "")[-4000:],
                    "command": cmd,
                }
            )
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
