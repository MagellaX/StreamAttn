"""Head-sharing versus omission frontier. Offline decisions, not kernel timings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_adaptive_real_activations import capture_and_evaluate, evaluate

SCHEMA = "streamattn.adaptive_group_frontier.v1"
SOURCE_FILES = (
    "benchmarks/profile_adaptive_group_frontier.py",
    "benchmarks/profile_adaptive_real_activations.py",
    "benchmarks/profile_real_llm_gate1_heads.py",
)


def summarize(records):
    cells = {}
    for record in records:
        for row in record["results"]:
            key = (row["radius"], row["decision"], row["head_group_size"], row["query_tile_size"])
            cells.setdefault(key, []).append(row)
    summaries = []
    for (radius, decision, group, tile), rows in sorted(cells.items()):
        ratios = sorted(r["kv_read_ratio_vs_full_group"] for r in rows)
        summaries.append(dict(radius=radius, decision=decision, head_group_size=group,
                              query_tile_size=tile, captures=len(rows),
                              captures_below_full_group_reads=sum(r < 1 for r in ratios),
                              min_kv_read_ratio=ratios[0], max_kv_read_ratio=ratios[-1],
                              mean_kv_read_ratio=sum(ratios) / len(ratios),
                              max_pv_regions_saved_fraction=max(r["pv_regions_saved_fraction"] for r in rows),
                              max_error=max(r["max_error"] for r in rows),
                              max_bound=max(r["max_bound"] for r in rows)))
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 16, 24, 26, 27])
    p.add_argument("--head-groups", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--query-tiles", type=int, nargs="+", default=[1, 16])
    p.add_argument("--last-query-only", action="store_true")
    p.add_argument("--budget", type=float, default=1e-3)
    p.add_argument("--provider", default="local")
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    if args.output_json.exists() or args.output_json.with_suffix(".captures.pt").exists():
        raise FileExistsError(args.output_json)
    if args.last_query_only and not args.captures:
        p.error("--last-query-only requires --captures")
    torch.set_num_threads(4)
    options = dict(head_groups=args.head_groups, query_tiles=args.query_tiles, budget=args.budget)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.captures:
        captures = torch.load(args.captures, map_location="cpu", weights_only=True)
        if not captures:
            raise ValueError("empty capture archive")
        records = []
        for item in captures:
            q, positions = item["q"], item["query_positions"]
            if args.last_query_only:
                q, positions = q[:, -1:], positions[-1:]
            print(f"FRONTIER {item['prompt_id']} L{item['layer']}", flush=True)
            row = evaluate(q, item["k"], item["v"], positions, **options)
            row.update(prompt_id=item["prompt_id"], layer=item["layer"])
            records.append(row)
        result = dict(records=records, capture_sha256=hashlib.sha256(args.captures.read_bytes()).hexdigest(),
                      last_query_only=args.last_query_only, input_mode="saved_capture_replay")
    else:
        result, captures = capture_and_evaluate(args.model, args.max_seq, revision=args.revision,
                                               layers=args.layers, evaluate_options=options)
        torch.save(captures, args.output_json.with_suffix(".captures.pt"))
        result["input_mode"] = "fresh_dense_conditioned_model_capture"
    result.update(schema=SCHEMA, complete=True, provider=args.provider, diagnostic_only=True,
                  performance_promotion=False, budget=args.budget,
                  interpretation="Requested KV reads versus full-GQA execution at the same query tile size. Not measured bandwidth, speedup, or an optimal-subset frontier.",
                  sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
                  summary=summarize(result["records"]))
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
