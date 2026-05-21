"""Model row/tile work for true-GQA head-mode StreamAttn policies.

Once StreamAttn enters a fused true-GQA backend, savings are no longer simply
"seed heads skipped X tokens".  K/V tile loads are shared by Q heads that map to
the same KV head.  If a KV group still has exact rows, non-seed tiles must still
be loaded for those exact rows; seed-only rows save row compute, but not the K/V
tile load.

This utility estimates the backend-visible work:

* dense row-tile work: ``Hq * num_tiles``;
* hybrid row-tile work: exact rows on every tile, seed rows on seed tiles only;
* K/V tile loads skipped only when a KV group has no active rows for a tile.

It is a policy-economics tool, not a kernel benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _parse_heads(raw: str | Iterable[int]) -> List[int]:
    if isinstance(raw, str):
        return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))
    return sorted(set(int(item) for item in raw))


def _seed_tile_indices(
    *,
    kv_len: int,
    tile_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
) -> set[int]:
    num_tiles = math.ceil(kv_len / tile_size)
    seed = set(range(min(sink_blocks, num_tiles)))
    if recent_blocks > 0:
        seed.update(range(max(0, num_tiles - recent_blocks), num_tiles))
    if middle_seed_blocks > 0:
        sink_end = min(sink_blocks, num_tiles)
        recent_start = max(0, num_tiles - recent_blocks)
        if block_order == "sequential":
            start = sink_end
            end = min(start + middle_seed_blocks, recent_start)
        elif block_order == "recent_first":
            end = recent_start
            start = max(sink_end, end - middle_seed_blocks)
        else:
            raise ValueError("block_order must be sequential or recent_first")
        seed.update(range(start, end))
    return {idx for idx in seed if 0 <= idx < num_tiles}


def model_kv_group_work(
    *,
    q_heads: int,
    kv_heads: int,
    kv_len: int,
    tile_size: int,
    seed_heads: Iterable[int],
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    padded_group_rows: int | None = None,
) -> Dict[str, Any]:
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("q_heads must be a positive multiple of kv_heads")
    group_size = q_heads // kv_heads
    num_tiles = math.ceil(kv_len / tile_size)
    seed_set = set(_parse_heads(seed_heads))
    invalid = [head for head in seed_set if head < 0 or head >= q_heads]
    if invalid:
        raise ValueError(f"seed_heads out of range: {invalid}")

    seed_tiles = _seed_tile_indices(
        kv_len=kv_len,
        tile_size=tile_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
    )
    seed_tile_count = len(seed_tiles)
    dense_row_tile_work = q_heads * num_tiles
    hybrid_row_tile_work = 0
    dense_kv_tile_loads = kv_heads * num_tiles
    hybrid_kv_tile_loads = 0
    dense_padded_row_tile_work = 0
    hybrid_padded_row_tile_work = 0
    per_group = []

    padded_rows = int(padded_group_rows or group_size)
    if padded_rows < group_size:
        raise ValueError("padded_group_rows must be >= group_size")

    for kv_head in range(kv_heads):
        heads = list(range(kv_head * group_size, (kv_head + 1) * group_size))
        group_seed = [head for head in heads if head in seed_set]
        group_exact = [head for head in heads if head not in seed_set]
        seed_count = len(group_seed)
        exact_count = len(group_exact)

        group_hybrid_rows = exact_count * num_tiles + seed_count * seed_tile_count
        group_dense_rows = group_size * num_tiles
        # A tensor-core group kernel may pay padded row lanes whenever a tile is
        # active.  If no exact rows exist, only seed tiles are active.
        active_tiles = num_tiles if exact_count > 0 else seed_tile_count
        group_hybrid_kv_loads = active_tiles
        group_skipped_kv_loads = num_tiles - active_tiles
        group_dense_padded_rows = padded_rows * num_tiles
        group_hybrid_padded_rows = padded_rows * active_tiles

        hybrid_row_tile_work += group_hybrid_rows
        hybrid_kv_tile_loads += group_hybrid_kv_loads
        dense_padded_row_tile_work += group_dense_padded_rows
        hybrid_padded_row_tile_work += group_hybrid_padded_rows

        per_group.append(
            {
                "kv_head": kv_head,
                "q_heads": heads,
                "seed_heads": group_seed,
                "exact_heads": group_exact,
                "seed_fraction": seed_count / group_size,
                "exact_fraction": exact_count / group_size,
                "seed_only_whole_group": exact_count == 0,
                "dense_row_tile_work": group_dense_rows,
                "hybrid_row_tile_work": group_hybrid_rows,
                "row_work_reduction": 1.0 - (group_hybrid_rows / group_dense_rows),
                "dense_kv_tile_loads": num_tiles,
                "hybrid_kv_tile_loads": group_hybrid_kv_loads,
                "kv_tile_load_reduction": 1.0 - (group_hybrid_kv_loads / num_tiles),
                "dense_padded_row_tile_work": group_dense_padded_rows,
                "hybrid_padded_row_tile_work": group_hybrid_padded_rows,
                "padded_row_work_reduction": 1.0
                - (group_hybrid_padded_rows / group_dense_padded_rows),
            }
        )

    return {
        "schema": "streamattn.true_gqa_head_mode_work_model.v1",
        "shape": {
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "group_size": group_size,
            "kv_len": kv_len,
            "tile_size": tile_size,
            "num_tiles": num_tiles,
            "padded_group_rows": padded_rows,
        },
        "policy": {
            "seed_heads": sorted(seed_set),
            "exact_heads": [head for head in range(q_heads) if head not in seed_set],
            "sink_blocks": sink_blocks,
            "recent_blocks": recent_blocks,
            "middle_seed_blocks": middle_seed_blocks,
            "block_order": block_order,
            "seed_tile_count": seed_tile_count,
        },
        "totals": {
            "dense_row_tile_work": dense_row_tile_work,
            "hybrid_row_tile_work": hybrid_row_tile_work,
            "row_work_reduction": 1.0 - (hybrid_row_tile_work / dense_row_tile_work),
            "dense_kv_tile_loads": dense_kv_tile_loads,
            "hybrid_kv_tile_loads": hybrid_kv_tile_loads,
            "kv_tile_load_reduction": 1.0 - (hybrid_kv_tile_loads / dense_kv_tile_loads),
            "dense_padded_row_tile_work": dense_padded_row_tile_work,
            "hybrid_padded_row_tile_work": hybrid_padded_row_tile_work,
            "padded_row_work_reduction": 1.0
            - (hybrid_padded_row_tile_work / dense_padded_row_tile_work),
        },
        "per_kv_group": per_group,
        "interpretation": _interpret(
            row_reduction=1.0 - (hybrid_row_tile_work / dense_row_tile_work),
            kv_reduction=1.0 - (hybrid_kv_tile_loads / dense_kv_tile_loads),
            padded_reduction=1.0
            - (hybrid_padded_row_tile_work / dense_padded_row_tile_work),
        ),
    }


def _interpret(*, row_reduction: float, kv_reduction: float, padded_reduction: float) -> str:
    if kv_reduction > 0.0:
        return (
            "Policy has at least one all-seed KV group, so fused backend can skip "
            "some K/V tile loads before scheduling."
        )
    if row_reduction >= 0.25 and padded_reduction <= 0.05:
        return (
            "Policy saves logical Q-row work but not padded tensor-core tile work; "
            "a dense padded group kernel may not benefit unless it masks rows cheaply."
        )
    if row_reduction >= 0.25:
        return "Policy saves meaningful Q-row compute but K/V tile loads remain dense."
    return "Policy has weak backend-visible work reduction for fused true-GQA scheduling."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--seed-heads", default="2,3,4,6,7")
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=2)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first"])
    parser.add_argument("--padded-group-rows", type=int, default=8)
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()
    result = model_kv_group_work(
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        kv_len=args.kv_len,
        tile_size=args.tile_size,
        seed_heads=_parse_heads(args.seed_heads),
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
        padded_group_rows=args.padded_group_rows,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
