# Removing unnecessary causal mask work

The compact mixed-request scheduler improved our natural producer substantially,
but did not close the FlashInfer gap. A matched-layout comparison found a 1.703x
causal/noncausal latency ratio for the compact producer, versus 1.004x for the
tested FlashInfer baseline. This motivates an execution ablation, not a change
to the attention definition.

## Three arms, one attention result

The causal matrix compares the existing compact explicit-position kernel with:

1. `affine_mode="index"`: remove global position-array loads from score masking.
2. `affine_mode="interior"`: also bypass score masking for fully visible tiles.

For a request with `N` keys and `M` appended queries, the validated positions are
`key_position(j) = origin + j` and
`query_position(i) = origin + N - M + i`. The common origin cancels. Visibility
is therefore exactly `j <= N - M + i`, including large or negative origins.
Planning validates these equalities with Python integers, avoiding signed
64-bit wraparound during validation. "Causal" alone is not this contract.

For a full query tile beginning at `i0`, every score in a full 64-key tile is
visible when its last key index is at most `N - M + i0`. This is the earliest
query's boundary; all later queries can see at least as much. The interior arm
uses that sufficient condition to remove the entire per-score mask loop.
Partial query tiles and boundary tiles retain explicit index masking.

Neither arm omits visible tokens. Both retain the same WGMMA producer, exact
online softmax, task partition, split-state merge and output allocation.
This is not sparse attention or approximate early termination.

## Scope and experiment

Use `PagedMicroPrefillPlan.build(..., natural=True, causal=True,
compact_schedule=True, affine_mode="index")`, or `"interior"`.
The default remains `"none"`. Affine plans snapshot lengths and positions;
changing either requires a new plan. Q/K/V and page mappings remain mutable.
Arbitrary/permuted positions retain the general explicit-position path.

The `causal` benchmark suite has 24 cases: three mixed-request traces, both
HND/NHD layouts and four D64/D128, FP16/BF16, G4/G8 configurations. Two independent
H100 runs use different random seeds. Each candidate is checked against FP32
output and LSE references before and after page/value mutation, then timed in
rotating paired CUDA graphs. Packed-interface timings include wrapper copies.
Both FlashInfer FA2 and FA3 are resolved; comparisons use the fastest correct
tested backend. This is not a claim against every available attention library.

```bash
python benchmarks/profile_sm90_micro_prefill_mixed.py --suite causal \
  --cutlass-root /path/to/cutlass --build-dir /tmp/affine-build \
  --output-json /tmp/affine.json
python benchmarks/summarize_sm90_micro_prefill_mixed.py /tmp/affine.json \
  --output-json /tmp/affine-summary.json
```

Report each arm versus explicit compact control and versus the external
baseline separately. A control improvement is not a FlashInfer victory.
Compare interior directly with index to determine whether the extra branch
earns its complexity. Until GPU results are collected, speedups are unknown.
No public dispatch or phase-database entry is changed by this experiment.
