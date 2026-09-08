# Compact scheduling for mixed requests

The previous mixed-batch experiment found that a rectangular launch treated
padded query tiles as useful parallel work. This experiment changes the task
assignment, not the attention definition or the natural R64 producer's math.

## What changes

For request `i`, let `T_i = ceil(N_i / 64)` be its KV tile count and
`Q_i = ceil(M_i / (64 / G))` its natural query-tile count. A common interval
budget `W` gives `S_i = min(512, ceil(T_i / W))` splits. Empty requests have no
tasks. The planner binary-searches the smallest `W` whose
`sum_i Hkv * Q_i * S_i` fits 256 CTAs, unless even one task per live query tile
already exceeds that soft budget. This is a work-balance heuristic, not a
latency model: it does not yet price replicated Q loads or merge overhead.

Each task identifies a request/head/query tile and one contiguous interval
`[floor(s*T_i/S_i), floor((s+1)*T_i/S_i))`. These intervals partition every valid
KV tile exactly once per query tile. The producer still computes full exact
attention with online softmax; the merge reads only that request's active
split states. No token selection, KV gathering or repacking is introduced.

In the G4 heterogeneous trace, splits change from a global two to
`[1,1,1,1,2,4,6,8]`. The launch has 256 nonempty tasks instead of 108 nonempty
and 148 empty tasks. The largest interval drops from 128 KV tiles to 32.
These are derived task counts, not measured occupancy or a speedup prediction.

## Deliberate boundaries

- Experimental opt-in: `PagedMicroPrefillPlan.build(..., natural=True,
  compact_schedule=True)`. No public dispatch or phase registration changes.
- Lengths are snapshotted during planning. A different length batch requires
  rebuilding the plan; page IDs, Q/K/V and explicit positions remain mutable.
  The default rectangular plan retains its existing mutable-length contract.
- Task metadata is preallocated on device. Graph replay has no host readback,
  allocation or planning. Host planning time is reported separately from replay.
- Partial states retain a rectangular stride. Unused states are not read by
  the merge; unused LSE slots start at minus infinity for diagnostic reduction.
- Padded output is still explicitly zero for invalid query rows. The packed
  benchmark still includes native scatter/gather costs. Native packed addressing
  is a separate next experiment, not claimed by this one.
- The benchmark retains both rectangular families and FlashInfer FA2/FA3.
  Compact-vs-natural improvement and compact-vs-fastest-baseline performance
  are reported separately; a better control ratio is not an external victory.

## Reproduce

```bash
python benchmarks/profile_sm90_micro_prefill_mixed.py --suite full \
  --cutlass-root /path/to/cutlass --build-dir /tmp/compact-build \
  --output-json /tmp/compact.json
python benchmarks/summarize_sm90_micro_prefill_mixed.py /tmp/compact.json \
  --output-json /tmp/compact-summary.json
```

The full matrix remains 48 cases: three mixed traces, FP16/BF16, D64/D128,
G4/G8, HND/NHD and causal/noncausal. Independent replay uses 24 cases and a
different random seed. Every timed implementation is checked against an FP32
reference before and after changing page IDs and values. This matrix does not
establish changed-length replay or general-purpose scheduler promotion.
