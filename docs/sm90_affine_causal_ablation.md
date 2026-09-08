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

Derived geometry predicts a sharp difference between traces. The interior
condition covers 96.7% (G4) / 98.9% (G8) of heterogeneous producer tiles and
85.7% / 92.2% of long-tail tiles, but only 0% / 25.8% of short-trace tiles.
These are tile counts, not measured cycle savings, occupancy or attention mass.
The summary records them alongside timings so this hypothesis is testable.

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
earns its complexity.
No public dispatch or phase-database entry is changed by this experiment.

## First complete H100 result

All 24 cases passed both output and LSE checks before and after page/value
mutation. Both FlashInfer backends resolved; FA2 was fastest in every cell.
The captured source hashes match the tested implementation. Ratios below are
geometric means of per-cell paired median speedups, not full-model speedups.

| Arm | Padded vs compact | Padded vs FlashInfer | Packed vs compact | Packed vs FlashInfer |
| --- | ---: | ---: | ---: | ---: |
| Index mask | 1.705x | 0.644x | 1.659x | 0.433x |
| Interior fast path | 1.861x | 0.703x | 1.804x | 0.471x |

Both arms beat the compact control in every paired trial of all 24 cells, in
both interfaces. The interior arm beats index in all trials in 22/24 padded
and 23/24 packed cells, with 1.091x / 1.087x overall additional improvement.
There are only 4/24 all-pair external wins for padded queries and none for
packed queries. The general FlashInfer gap remains.

The interior/index gain follows the geometry prediction: padded improvement is
1.016x for short traces and 1.131x for both heterogeneous and long-tail traces.
The added branch earns its place as an experimental candidate, but not as a
universal default. Index arithmetic alone removes most of the observed cost.

The interior packed wrapper is 1.147x slower than its padded interface.
Comparing padded candidate time to the packed external baseline yields a
0.539x diagnostic proxy. Separate graphs and buffers can change cache behavior;
this is not a formal copy-removal bound. It still argues against expecting
wrapper deletion alone to close the gap. Isolate the remaining producer and
merge costs before changing the execution state machine or output scheduling.

Lightning's parallel attempt stopped with `USER_STOP_WORKLOAD_REASON_OUT_OF_FUNDS`
and returned no complete benchmark. The runner deleted that job; its reported
cost was 0.41653332 (provider-reported units). It contributes no performance
evidence. A second Modal H100 run with a different seed completed all 24 cases.

## Independent replay

Both runs used matching kernel/profile source hashes and seeds 9613 and 17071.
The replay again resolved both FlashInfer backends, selecting FA2 in every cell.

| Arm | Padded vs compact | Padded vs FlashInfer | Packed vs compact | Packed vs FlashInfer |
| --- | ---: | ---: | ---: | ---: |
| Index mask | 1.712x | 0.644x | 1.661x | 0.434x |
| Interior fast path | 1.867x | 0.701x | 1.807x | 0.472x |

Every candidate/control paired comparison won again. Interior/index improvement
was 1.090x padded and 1.087x packed, with 20/24 and 23/24 all-pair wins. Padded
trace ratios were 1.012x short, 1.132x heterogeneous and 1.131x long-tail. The
short-trace benefit is small and not consistently an all-pair win; do not
interpret it as proof that branching helps when no tile can take the fast path.
External all-pair wins remain four padded cells and zero packed cells.

The retained candidate is justified by a repeated reduction in avoidable mask
work, not a generalized external win. Keep the general explicit-position path
and experimental opt-in contract. Next, measure producer versus merge time with
the interior arm as the control, then choose the larger remaining cost for a
native execution change. Do not resume seed sweeps or assume copy removal is
sufficient. No broad literature search is required to obtain that measurement.

Final verification: 1,074 CPU tests passed, 71 skipped; CI and offline CUDA
compilation passed. Both GPU runs finished, and Lightning's failed job was
deleted. No public route or phase-database promotion was made.

- [First run](../artifacts/gate0/sm90_micro_affine_modal_h100_20260908.json)
- [First summary](../artifacts/gate0/sm90_micro_affine_modal_summary_h100_20260908.json)
- [Lightning interruption](../artifacts/gate0/sm90_micro_affine_lightning_h100_20260908.failure.json)
- [Independent replay](../artifacts/gate0/sm90_micro_affine_modal_h100_20260908_replay.json)
- [Combined summary](../artifacts/gate0/sm90_micro_affine_summary_h100_20260908.json)
