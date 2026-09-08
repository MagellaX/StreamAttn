# The remaining mixed-request cost is mostly in the producer

This experiment follows the [post-affine research review](sm90_post_affine_research_20260909.md).
It tests the current exact paged R64 producer, not an older contiguous kernel.
The question is whether task order, split-state overhead, or interface conversion
explains the remaining FlashInfer gap well enough to choose the next change.

## What changed

Four fixed variants retain the same attention definition and device kernels:

| Variant | Minimum KV tiles per task | Task-list order |
| --- | ---: | --- |
| `natural_compact_interior` | 1 | request/head/query/split |
| `interior_kv_order` | 1 | request/head/split/query |
| `interior_min2` | 2 | request/head/query/split |
| `interior_min2_kv_order` | 2 | request/head/split/query |

The floor leaves a request shorter than two tiles unsplit. Changing task order
does not change the task multiset, output ownership, or interval boundaries;
it also does not guarantee physical SM execution order. All visible tokens are
still processed with WGMMA, online softmax and exact split-state merging.

These are opt-in controls on the experimental compact plan. Defaults, public
dispatch and phase databases are unchanged. Plans freeze lengths and affine
positions; page IDs and Q/K/V values remain mutable.

Separate host entry points launch the existing producer or existing merge.
The merge replay consumes actual producer states at the plan's real strides.
Live partial slots are poisoned before validation; unused rectangular slots
retain the negative-infinity identity needed by the LSE reconstruction.
Producer-only execution must leave final output untouched.

## Measurement contract

The discovery matrix has 24 causal mixed-request cases: D64/D128, BF16/FP16,
G4/G8, Hq16/Hq32, HND/NHD page-16, and short/heterogeneous/long-tail traces.
The holdout changes lengths and compositions, including B5/B6 instead of B4/B8.
The four candidate definitions are fixed before either run; there is no fitted
selector or post-hoc winning-cell registration.

Each case checks FP32 output and LSE before and after page/value mutations,
in both padded and packed query interfaces. Complete-call paired CUDA graphs
include producer, merge, and required interface conversions. FlashInfer FA2
and FA3 are timed; the comparison uses the fastest correct tested backend.
JIT, host planning, changed-length replanning, KV append and request scheduling
are outside these steady-state timings.

Additional measurements separate producer, merge, packed adapters, page-ID
compaction and FlashInfer attention. **Isolated times are not additive.** They
may have different cache behavior from complete calls. A second complete-call
condition fills a 128 MiB device buffer before each timed replay; the fill is
excluded from timing. This perturbs the working set, but does not certify cold
caches. Its comparator remains the warm-selected baseline, not a separately
optimized cache-perturbed oracle.

## Discovery result

The completed Modal H100 80GB run passed 24/24 cases, including the separate
producer/merge checks. FA2 was the fastest tested baseline in every case.
These are geometric means of per-cell paired median speedup ratios:

| Variant | Padded vs control | Padded vs FA2 | Packed vs control | Packed vs FA2 |
| --- | ---: | ---: | ---: | ---: |
| Control | 1.000x | 0.700x | 1.000x | 0.471x |
| KV order | 0.994x | 0.696x | 0.995x | 0.469x |
| Two-tile floor | 1.013x | 0.709x | 1.009x | 0.475x |
| Both | 1.011x | 0.708x | 1.009x | 0.475x |

Reordering is not a useful general improvement here. The floor helps some
short cases, but barely changes the aggregate. It raises padded all-pair
external wins from 4/24 to 6/24; packed still has zero. Heterogeneous and
long-tail floor schedules are often unchanged, so tiny timing differences there
must not be explained as changed work. Cache-perturbed measurements also show
no broad recovery: the floor reaches 0.749x padded / 0.532x packed versus FA2.

## Independent holdout

The Lightning H100 80GB run passed all 24 new cases, including component checks
and mutable-buffer output/LSE validation. FA2 again won every baseline comparison.
Ratios are paired within this run, not across providers:

| Variant | Padded vs control | Padded vs FA2 | Packed vs control | Packed vs FA2 |
| --- | ---: | ---: | ---: | ---: |
| Control | 1.000x | 0.665x | 1.000x | 0.514x |
| KV order | 0.996x | 0.662x | 0.995x | 0.512x |
| Two-tile floor | 0.993x | 0.661x | 0.996x | 0.512x |
| Both | 0.991x | 0.659x | 0.992x | 0.510x |

Neither change generalizes. The floor's small discovery gain disappears; all
variants retain four padded all-pair external wins and zero packed wins.
In the short D128/G8/HND holdout, the floor only reduces CTAs from 194 to 186
and increases padded latency from 29.92 to 31.92 us. This is not the discovery
short case's 240-to-120 CTA intervention. A blanket minimum-work rule ignores
the different interval lengths and execution waves it creates.

## The decisive attribution

Representative BF16 D128/G8/Hq16/HND measurements, in microseconds:

| Discovery trace | Native producer alone | Native merge alone | Packed adapters alone | Native padded complete | FA2 packed complete |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 23.67 | 6.39 | 10.43 | 30.80 | 16.79 |
| Heterogeneous | 139.35 | 9.31 | 7.62 | 149.71 | 69.37 |
| Long-tail | 156.40 | 6.28 | 6.12 | 161.78 | 73.40 |

The last two columns intentionally have different interfaces and are not a
speedup claim. They show why a copy-free native proxy is still insufficient.
On the large traces, the producer alone is already about twice the complete
packed external call. Deleting every adapter and perfecting the merge cannot
plausibly be the sole solution under the observed cost structure. This is a
diagnosis to test with complete-call interventions, not a formal Amdahl bound
obtained by adding isolated times.

The short D128 case does benefit from the floor: native padded complete latency
falls from 30.80 to 25.56 us. That local result does not justify adopting the
rule universally or describing the aggregate as a breakthrough.

The independent D128/G8/HND holdout reproduces the producer diagnosis:

| Holdout trace | Native producer alone | Native merge alone | Native packed complete | FA2 packed complete |
| --- | ---: | ---: | ---: | ---: |
| Short | 25.38 | 4.77 | 35.09 | 17.61 |
| Heterogeneous | 153.67 | 8.49 | 166.01 | 73.84 |
| Long-tail | 115.66 | 5.29 | 123.44 | 55.72 |

This narrows the next question to why the producer is costly. It does not yet
identify a particular instruction or prove that memory traffic, dependency
latency, tensor-core utilization, or redundant work is the dominant cause.

## What the current counters do and do not say

A separate H100 Nsight run completed 12 checked captures, covering 27 actual
kernel launches across the three D128/G8/HND traces and four targets: control,
reordering, floor, and FA2. It records the actual native producer/merge and
FA2 compaction/attention/merge symbols. Instrumented durations are excluded
from the summary's speed comparisons.

The D128 producer uses 167 registers/thread and 64 KiB dynamic shared memory,
with no reported local allocation. The resource-only occupancy limit is three
CTAs/SM, not measured achieved occupancy. D64 uses 127 registers and 32 KiB,
with a resource limit of four. FA2's measured D128 attention launch uses 255
registers/thread, about 64 KiB dynamic shared memory, and 264 CTAs.

The pinned FA2 plan decoder records R128 query tiles and actual KV chunks of
128, 1024 and 1184 tokens for the three D128/G8 discovery traces. Live task-list
entries are 60, 125 and 131 before the KV-head grid dimension. These are runtime
values, not assumed from the automatic planner's minimum.

Two findings constrain the explanation:

1. The short floor halves native producer CTAs from 240 to 120 and lowers
   eligible warps/scheduler from 0.43 to 0.26. Less parallelism is real, yet the
   complete call improves in this case. Occupancy alone cannot rank the plans.
2. On the two larger traces, both native and FA2 producers have roughly 0.5
   eligible warps/scheduler despite the substantial runtime gap. That statistic
   alone is not a discriminative cause. Reordering does not improve large-trace
   L2 hit rates in these captures and does not deliver a timing win.

The native heterogeneous merge launches 8192 CTAs versus FA2's 1584, but the
measured isolated merge is much smaller than the producer. CTA counts exposed
waste; timing showed its limited leverage. Keep those statements separate.

The collected counter set contains resource, scheduler, hit-rate and throughput
metrics, not a complete byte-traffic or source-stall attribution. Clocks and
caches were unmanaged under kernel replay. Auxiliary Kineto traces also missed
some launches; use the dedicated checked Nsight captures for launch comparisons,
not an assumption that every auxiliary trace is complete.

## Next implementation decision

Keep the current default schedule. Do not add a learned scheduler, another
fixed split sweep, or a general KV-order rule from these results. Keep native
packed offsets and live-row merge ownership as useful secondary simplifications.

The primary next experiment belongs inside the **current exact paged producer**:
compare source-correlated instruction/load work with the actual FA2 winner,
separating Q staging, paged K/V staging, address work, and QK/softmax/PV progress.
Test the smallest supported reduction in staging/dependency cost while retaining
the single output state and complete-call contract. A compiler comparison can
help if generated scheduling is implicated; a new persistent or R128 state
machine needs evidence beyond low eligible-warps statistics.

This is where deeper kernel research is justified. A broad literature search,
a new model sweep, or replacing exact attention with a selected subset does
not answer the measured bottleneck.

## Reproduction and development failures

```bash
python benchmarks/profile_sm90_micro_prefill_mixed.py --attribution --suite causal \
  --seed 19073 --cutlass-root /path/to/cutlass --build-dir /tmp/attribution \
  --output-json /tmp/discovery.json
python benchmarks/profile_sm90_micro_prefill_mixed.py --attribution --suite holdout \
  --seed 29083 --cutlass-root /path/to/cutlass --build-dir /tmp/attribution \
  --output-json /tmp/holdout.json
python benchmarks/profile_sm90_mixed_attribution_counters.py \
  --cutlass-root /path/to/cutlass --build-dir /tmp/counters --output-json /tmp/counters.json
python benchmarks/summarize_sm90_post_affine_attribution.py /tmp/discovery.json /tmp/holdout.json \
  --counters /tmp/counters.json --output-json /tmp/summary.json
```

Two earlier Modal attempts caught diagnostic bugs: poisoning unused rectangular
LSE identities, then an incorrect field order in the FA2 plan decoder. Two
Lightning attempts rejected holdout inputs labelled mixed that lacked a decode
request. The corrected contract and decoder have CPU regression tests. Failed
artifacts are retained separately and never counted as successful measurements.

## Evidence and execution record

- [Raw discovery](../artifacts/gate0/sm90_post_affine_attribution_modal_h100_final_20260909.json)
- [Raw independent holdout](../artifacts/gate0/sm90_post_affine_attribution_lightning_h100_holdout_checked_20260909.json)
- [Checked Nsight captures](../artifacts/gate0/sm90_post_affine_counters_modal_h100_20260909.json)
- [Combined summary](../artifacts/gate0/sm90_post_affine_summary_h100_20260909.json)

Both timing runs used PyTorch 2.7.1 / CUDA 12.8 / FlashInfer 0.6.13 on H100 80GB.
Artifacts bind source hashes and loaded binary provenance to each run. The
holdout harness corrects the new trace definitions to include a decode request;
discovery cases and the native kernel are unchanged. The counter image predates
a host-source whitespace correction; no device math changed.

Lightning received 10 credits from the existing organization balance, with no
credit purchase. The successful holdout reported a job cost of 1.3600445; the
two failed setup attempts reported 0.17355555 and 0.14357778. These are provider
job reports, not a reconciled billing statement. All three jobs were deleted
after result collection. The discovery and counter compute apps also completed.

Local verification: 1123 passed, 71 skipped. GPU evidence here is functional
and performance attribution for synthetic mixed-request cases, not production
trace coverage or public phase-database promotion.
