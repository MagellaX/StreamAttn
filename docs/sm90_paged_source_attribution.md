# Paged producer source attribution

The scheduling ablations at `3b98601` did not generalize. This experiment keeps
the R64 tile, 128 threads, split schedule, one output state, exact online
softmax, and merge unchanged. It profiles the current producer before choosing
one smaller intervention.

## What the H100 counters establish

Six checked launches compare native and the actual FlashInfer 0.6.13 FA2 paged
kernel on D128/G8/BF16/HND short, heterogeneous, and long-tail requests. Each
capture checks output/LSE and graph replay after page/value mutation. Native
PCs are deduplicated across inline source correlations, then checked against
kernel-wide instruction and sample totals. The exact generated CUDA is retained
with its hash.

For affine append, visible head-pairs are
`Hq * sum(M*N - M*(M-1)/2)`. Useful QK+PV work is four times that quantity times
D, counting FMA as two operations. Scheduled work instead counts complete
matrix tiles, including masked rows. Neither quantity predicts latency.

| Quantity | Short native / FA2 | Heterogeneous native / FA2 | Long tail native / FA2 |
|---|---:|---:|---:|
| Useful head-pairs | 392,320 / same | 31,728,160 / same | 34,452,640 / same |
| Scheduled head-pairs | 983,040 / 1,966,080 | 31,899,648 / 32,342,016 | 36,306,944 / 40,091,648 |
| Executed warp instructions | 6,490,768 / 1,760,936 | 52,885,548 / 17,955,424 | 59,798,168 / 22,106,146 |
| DRAM read bytes | 453,632 / 192,512 | 46,024,448 / 46,022,656 | 55,852,032 / 55,821,056 |
| L2 read sectors | 289,853 / 256,818 | 8,199,261 / 5,014,130 | 10,434,876 / 6,349,327 |

Native does not issue more tensor tile work in these comparisons. On the
heterogeneous trace it executes 2.95x the warp instructions and 9.34x the
predicated-on integer thread instructions, with almost identical DRAM reads.
The integer counter includes work other than addressing. The inference is
**execution overhead deserves attention**, not that all the gap is address math.
L2 sectors are not DRAM bytes, and cache hits do not make instructions free.

The FA2 work decoder uses its observed R128/64 kernel traits, live task list,
and actual KV chunk size. Its causal loop extent follows the pinned
[FA2 paged implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/include/flashinfer/attention/prefill.cuh).
This is scheduled-work accounting, not a hardware tensor-instruction count.

## Why test the Q copy first

The largest individual long-scoreboard sampled PC is the scalar Q store into
shared storage on all three native traces. In the heterogeneous case that PC
has 1,124 samples; its entire Q staging region has 1,850 of 15,696 samples and
3,075,552 of 52,885,548 executed warp instructions. In the short case Q staging
accounts for 2,976,712 of 6,490,768 warp instructions.

The paged loader is the larger aggregate instruction region on the long traces:
22,034,652 heterogeneous and 25,078,160 long-tail warp instructions. Repeated
page-load-dependent SASS address chains are another concrete suspect. They are
not being changed in this experiment.

The scalar Q path is a smaller, clearly isolated first test: replace eight
two-byte scalar copies with one aligned 16-byte load/store. Retain zero-fill
for invalid query rows. A compile-time assertion checks that the shared layout
preserves each aligned vector; planning rejects unaligned Q storage. No copy
barrier, V issue position, or arithmetic operation changes.

This is an experimental `q_vector_copy=True` option, not a new default. The
predicted machine change is fewer Q load/store and indexing instructions.
It does not predict an 8x kernel speedup, or enough gain to reach FA2 parity.

Earlier V staging is not selected by these captures. The existing joint copy
wait covers next K and current V; sampled locations do not isolate a late-V
dependency. Neither do they establish that moving V earlier could never help.

## Limits of this measurement

- Nsight 2025.1 kernel replay used unmanaged clocks and caches after warmup.
  Traffic from different replay passes need not add up as one execution's
  memory accounting. Instrumented durations are not the performance gate.
- Samples are not elapsed-time percentages. A stalled PC is not necessarily
  its dependency producer. The export did not provide producer-attributed
  scoreboard stalls; see [Nsight's source-view explanation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html).
- The loaded FA2 binary has no CUDA lineinfo. Its aggregate hardware counters
  and decoded plan are present, but CUDA/SASS correlation is unavailable in
  this export. It was not rebuilt with different flags and relabeled as the
  measured baseline.
- Some native PCs map only to inlined library code; those stay explicitly
  unmapped rather than being assigned to an attractive explanation.
- These are synthetic boundary traces, not production traffic or a universal
  public-dispatch promotion.

## Reproduce

```bash
python benchmarks/profile_sm90_mixed_attribution_counters.py --source-correlated \
  --cutlass-root /path/to/cutlass --build-dir /tmp/source \
  --output-json /tmp/source.json
python benchmarks/summarize_sm90_paged_source.py /tmp/source.json \
  --output-json /tmp/source-summary.json
python benchmarks/profile_sm90_micro_prefill_mixed.py --attribution --producer-copy \
  --suite causal --seed 39089 --cutlass-root /path/to/cutlass \
  --build-dir /tmp/copy --output-json /tmp/discovery.json
python benchmarks/profile_sm90_micro_prefill_mixed.py --attribution --producer-copy \
  --suite holdout --seed 49093 --cutlass-root /path/to/cutlass \
  --build-dir /tmp/copy --output-json /tmp/holdout.json
```

The source harness prebuilds and checks native and baseline modules before
Nsight injection. The first attempt used an invalid metric-query mode; the
second was stopped in a cold FA2 build under Nsight. Both failed artifacts are
retained and excluded from the six checked captures.

The first vector-copy run passed its six D64 discovery cases, then hit NVCC's
constant-expression evaluation limit in the exhaustive shared-layout assertion
while compiling D128. This was a compile failure, not an accuracy result. The
assertion now checks the eight-row repeating SW128 atom; runtime copy code is
unchanged. The corresponding first Lightning job was stopped and deleted when
that shared compile issue was identified (reported cost 0.5648444). Its finished
logs were unavailable, so it supplies no holdout evidence. Fresh discovery and
holdout runs use the corrected assertion and separate output paths.

Evidence:

- [Checked raw source capture](../artifacts/gate0/sm90_paged_source_counters_modal_h100_prebuilt_20260909.json)
- [Useful-work and PC summary](../artifacts/gate0/sm90_paged_source_summary_h100_final_20260909.json)
- [Initial compile-failure artifact, excluded](../artifacts/gate0/sm90_vector_q_copy_modal_h100_20260909.json)
