"""Summarize Gate-1 real-shape autotune JSON output."""

import argparse
import json
from pathlib import Path


def _key(row):
    shape = row["shape"]
    return (
        shape["seq"],
        shape["heads"],
        shape["dim"],
        shape["dtype"],
        row["block_size"],
        row["requested_active_fraction"],
    )


def _config(row):
    return {
        "tile_size_q": row["tile_size_q"],
        "block_size": row["block_size"],
        "num_warps": row["num_warps"],
        "num_stages": row["num_stages"],
    }


def _best(rows, metric):
    valid = [row for row in rows if metric in row and row[metric] is not None]
    if not valid:
        return None
    row = min(valid, key=lambda item: item[metric])
    return {
        "metric": metric,
        "value": row[metric],
        "config": _config(row),
        "sdpa_dense_ms": row.get("sdpa_dense_ms"),
        "gate1_mass_ms": row.get("gate1_mass_ms"),
        "gate1_value_bound_ms": row.get("gate1_value_bound_ms"),
        "gate1_dense_equiv_ms": row.get("gate1_dense_equiv_ms"),
        "true_qk_scan_ms": row.get("gate1_true_qk_scan_ms"),
    }


def summarize(payload):
    groups = {}
    for row in payload.get("rows", []):
        if "error" in row:
            continue
        groups.setdefault(_key(row), []).append(row)

    summaries = []
    for key, rows in sorted(groups.items()):
        seq, heads, dim, dtype, block_size, active = key
        summaries.append(
            {
                "shape": {
                    "seq": seq,
                    "heads": heads,
                    "dim": dim,
                    "dtype": dtype,
                },
                "block_size": block_size,
                "requested_active_fraction": active,
                "num_configs": len(rows),
                "best_mass": _best(rows, "gate1_mass_ratio"),
                "best_value_bound": _best(rows, "gate1_value_bound_ratio"),
                "best_dense_equiv": _best(rows, "dense_equiv_ratio"),
                "best_true_qk": _best(rows, "true_qk_scan_ratio"),
            }
        )
    return {"summaries": summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    with Path(args.input_json).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = summarize(payload)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
