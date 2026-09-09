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

## Vector-copy mechanism and discovery

The corrected H100 discovery run passed all 24 cases: page-16 HND/NHD,
short/heterogeneous/long-tail traces, and four head/dtype configurations:
`D64/G4/Hq16/BF16`, `D128/G8/Hq16/BF16`, `D64/G8/Hq32/FP16`, and
`D128/G4/Hq32/FP16`. These are not the full Cartesian product. Discovery uses
B4/B8; the independent holdout changes both batch composition and lengths.

The heterogeneous D128/BF16/HND source replay confirms the predicted mechanism:

| Quantity | Scalar control | Vector Q |
|---|---:|---:|
| Q load / store SASS | `LDG.E.U16` / `ST.E.U16` | `LDG.E.128` / `ST.E.128` |
| Executed Q loads, warp instructions | 63,488 | 7,936 |
| Executed Q stores, warp instructions | 63,488 | 7,936 |
| Q-region warp instructions | 3,075,552 | 413,024 |
| Total producer warp instructions | 52,885,548 | 48,999,884 |
| Paged-address/copy warp instructions | 22,034,652 | 22,034,652 |
| Q-region long-scoreboard samples | 1,206 | 200 |
| Registers / local bytes per thread | 167 / 0 | 167 / 0 |
| Dynamic shared bytes / resource-limited CTAs per SM | 65,536 / 3 | 65,536 / 3 |

Q-region instructions fall 86.6%; total producer instructions fall 7.35%.
Some removed indexing instructions map to inlined code, not the Q source
region. Softmax, QK, and paged-copy region instruction counts are unchanged.
The new sample counts are supporting evidence, not a percentage of time saved.
The raw replay retains both generated source and machine-code correlations.

Uninstrumented complete-call speedups versus the paired native control are:

| Discovery interface | Warm geomean | Perturbed geomean | Warm all-pair winning cells | Perturbed all-pair winning cells |
|---|---:|---:|---:|---:|
| Padded | 1.1002x | 1.1065x | 23/24 | 24/24 |
| Packed | 1.0653x | 1.0517x | 23/24 | 24/24 |

Every discovery cell's median improves, but two long-tail FP16/D64 cells have
individual losing pairs: padded NHD reaches 0.9868x and packed HND reaches
0.9918x. They remain in the report, with no special-case routing added.
Warm padded short traces improve 1.2123x; heterogeneous and long-tail traces
improve only 1.0566x and 1.0397x. This fits a reduced setup cost, not an inner-loop
repair. The isolated heterogeneous producer improves from 138.94 to 126.00 us;
the isolated merge stays near 9.30 us. Isolated times are not additive.

FA2 remains the fastest tested external backend. Against its complete matching
interface, discovery vector-Q geomeans are 0.7734x padded and 0.5041x packed.
No packed discovery cell wins every pair against FA2. This is a useful native
improvement, **not** an exact mixed-request kernel victory or public promotion.

## Independent holdout and decision

The independent Lightning H100 run uses seed 49093, B5/B6, changed query/KV
lengths, and freshly compiled binaries. All 24 cases pass output/LSE,
poisoned-tail, graph page/value mutation and component ownership checks.
Recorded measured-source hashes from both providers match the committed code.
There is no discovery-fitted selector or holdout-specific schedule.

| Holdout interface | Warm geomean | Perturbed geomean | Warm all-pair winning cells | Perturbed all-pair winning cells |
|---|---:|---:|---:|---:|
| Padded | 1.1173x | 1.1198x | 24/24 | 24/24 |
| Packed | 1.0809x | 1.0657x | 23/24 | 24/24 |

All holdout cell medians improve in both cache modes and interfaces. One packed
warm long-tail FP16/D64/G8/NHD cell has a losing pair at 0.9943x; its median is
1.0172x. Each warm comparison uses nine randomized paired trials. Cache
perturbation does not guarantee a cold working set.

The main D128/G8/BF16/HND heterogeneous holdout still shows the remaining scale:

| Complete packed call, median | Time |
|---|---:|
| Scalar native control | 166.55 us |
| Vector Q candidate | 158.36 us |
| Fastest tested external, FA2 | 74.18 us |

Across the complete holdout, vector-Q speed relative to FA2 is 0.7387x padded
and 0.5569x packed. No packed cell wins every pair against FA2. Do not mistake
improvement over our control for baseline parity.

**Retain vector Q as a fixed experimental improvement; leave public dispatch
and the default schedule unchanged.** No cell exclusions are added to conceal
the individual losing pairs. The next producer test should isolate repeated
page-load-dependent address calculations, checking generated SASS before any
code change. The measured paged-loader instruction total is unchanged at
22,034,652 after the Q repair, while Q staging shrinks to 413,024. This supports
moving on from Q setup, not repeating Q/split tuning. Avoid persistent page
descriptors or another output state unless a separate measurement justifies
their added storage and instructions. Earlier V issue remains an unproven
alternative, not a demonstrated cause of the remaining gap.

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
python benchmarks/summarize_sm90_post_affine_attribution.py \
  /tmp/discovery.json /tmp/holdout.json --output-json /tmp/copy-summary.json
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

The successful Modal discovery/counter app is `ap-tcSvmQrfVjo7GSbBytlVGp`.
The successful Lightning holdout is `job_01m22n4ad21sr1rx78n3sw1p7m`, with
reported cost 1.7513334. Its runner deleted the job after retaining the result;
all experiment Modal apps are stopped with zero tasks. The earlier source-only
success is `ap-uGuxs2SRL1jwRJdMwJYkac`. No claim is made that the first cancelled
Lightning run supplied independent measurements.

Evidence:

- [Checked raw source capture](../artifacts/gate0/sm90_paged_source_counters_modal_h100_prebuilt_20260909.json)
- [Useful-work and PC summary](../artifacts/gate0/sm90_paged_source_summary_h100_final_20260909.json)
- [Initial compile-failure artifact, excluded](../artifacts/gate0/sm90_vector_q_copy_modal_h100_20260909.json)
- [Corrected discovery and machine-code replay](../artifacts/gate0/sm90_vector_q_copy_modal_h100_checked_20260909.json)
- [Independent holdout](../artifacts/gate0/sm90_vector_q_copy_lightning_h100_holdout_checked_20260909.json)
- [Combined paired timing and source summary](../artifacts/gate0/sm90_vector_q_copy_summary_h100_20260909.json)
