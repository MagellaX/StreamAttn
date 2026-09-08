"""Planning-time natural-R64 task geometry; no attention approximation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RaggedSchedule:
    splits: tuple[int, ...]
    tasks: tuple[tuple[int, int, int, int], ...]
    max_tiles_per_task: int


def plan_ragged_schedule(query_lengths, kv_lengths, *, capacity, kv_heads,
                         group_size, target_ctas=256):
    """Minimize the largest KV interval within a soft CTA budget.

    Each task is (rectangular work group, split, tile begin, tile end).
    A shared tile budget apportions more splits to longer requests. The budget
    is soft only when one unsplit task per live query tile already exceeds it.
    Zero-length requests have no producer tasks. Lengths are fixed by this plan.
    """
    qs, ns = tuple(query_lengths), tuple(kv_lengths)
    scalars = (capacity, kv_heads, group_size, target_ctas, *qs, *ns)
    if any(type(x) is not int for x in scalars):
        raise ValueError("schedule dimensions must be integers")
    if (not qs or len(qs) != len(ns) or not 2 <= capacity <= 64
            or kv_heads < 1 or group_size not in (4, 8) or target_ctas < 1
            or any(q < 0 or q > capacity for q in qs)
            or any(n < 0 or n > 2147483584 for n in ns)):
        raise ValueError("invalid ragged schedule geometry")
    qpt = 64 // group_size
    rectangular_tiles = math.ceil(capacity / qpt)
    tiles = [math.ceil(n / 64) for n in ns]
    groups = [math.ceil(q / qpt) * kv_heads if n else 0 for q, n in zip(qs, ns)]

    def counts(width):
        return [min(512, math.ceil(t / width)) if g else 0
                for t, g in zip(tiles, groups)]

    lo, hi = 1, max(tiles, default=1) or 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(g * s for g, s in zip(groups, counts(mid))) > target_ctas:
            lo = mid + 1
        else:
            hi = mid
    splits = counts(lo)
    tasks = []
    for b, (q, t, s) in enumerate(zip(qs, tiles, splits)):
        for head in range(kv_heads):
            for qt in range(math.ceil(q / qpt)):
                group = (b * kv_heads + head) * rectangular_tiles + qt
                for split in range(s):
                    tasks.append((group, split, split * t // s, (split + 1) * t // s))
    if len(tasks) > 2147483647:
        raise ValueError("task count exceeds int32")
    return RaggedSchedule(tuple(splits), tuple(tasks),
                          max((end - begin for _, _, begin, end in tasks), default=0))
