"""Planning-time natural-R64 task geometry; no attention approximation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RaggedSchedule:
    splits: tuple[int, ...]
    tasks: tuple[tuple[int, int, int, int], ...]
    max_tiles_per_task: int


def validate_affine_append_positions(query_lengths, kv_lengths, query_positions, key_positions):
    """Planning-only exact integer check; never assume a causal mask is affine."""
    if not (len(query_lengths) == len(kv_lengths) == len(query_positions) == len(key_positions)):
        raise ValueError("affine position batch mismatch")
    for q, n, qp, kp in zip(query_lengths, kv_lengths, query_positions, key_positions):
        if not kp or q < 0 or n < 0 or len(qp) < q or len(kp) < n:
            raise ValueError("invalid affine position lengths")
        origin = kp[0]
        if any(pos != origin + j for j, pos in enumerate(kp[:n])):
            raise ValueError("key positions are not affine")
        if any(pos != origin + n - q + i for i, pos in enumerate(qp[:q])):
            raise ValueError("query positions are not bottom-right affine")


def plan_ragged_schedule(query_lengths, kv_lengths, *, capacity, kv_heads,
                         group_size, target_ctas=256, min_kv_tiles=1,
                         task_order="query"):
    """Minimize the largest KV interval within a soft CTA budget.

    Each task is (rectangular work group, split, tile begin, tile end).
    A shared tile budget apportions more splits to longer requests. The budget
    is soft only when one unsplit task per live query tile already exceeds it.
    Zero-length requests have no producer tasks. Lengths are fixed by this plan.
    """
    qs, ns = tuple(query_lengths), tuple(kv_lengths)
    scalars = (capacity, kv_heads, group_size, target_ctas, min_kv_tiles, *qs, *ns)
    if any(type(x) is not int for x in scalars):
        raise ValueError("schedule dimensions must be integers")
    if (not qs or len(qs) != len(ns) or not 2 <= capacity <= 64
            or kv_heads < 1 or group_size not in (4, 8) or target_ctas < 1
            or min_kv_tiles < 1 or task_order not in ("query", "kv")
            or any(q < 0 or q > capacity for q in qs)
            or any(n < 0 or n > 2147483584 for n in ns)):
        raise ValueError("invalid ragged schedule geometry")
    qpt = 64 // group_size
    rectangular_tiles = math.ceil(capacity / qpt)
    tiles = [math.ceil(n / 64) for n in ns]
    groups = [math.ceil(q / qpt) * kv_heads if n else 0 for q, n in zip(qs, ns)]

    def counts(width):
        return [min(512, math.ceil(t / width), max(1, t // min_kv_tiles)) if g else 0
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
    if task_order == "kv":
        # Preserve output ownership and intervals; change only launch-list locality.
        tasks.sort(key=lambda task: (task[0] // rectangular_tiles, task[1],
                                    task[0] % rectangular_tiles))
    if len(tasks) > 2147483647:
        raise ValueError("task count exceeds int32")
    return RaggedSchedule(tuple(splits), tuple(tasks),
                          max((end - begin for _, _, begin, end in tasks), default=0))
