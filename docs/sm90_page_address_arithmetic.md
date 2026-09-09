# D128 Page-Address Arithmetic Ablation

## Question

After page-pair reuse at `c273976`, the heterogeneous D128 producer still takes
more time than FlashInfer's complete packed-query call. Can unnecessary signed
address arithmetic be removed without changing the execution state machine?

This is a compiler representation experiment, not new attention mathematics.
The existing plan validates active page IDs against the physical allocation;
callers must preserve that contract when mutating IDs between graph replays.
Inactive page entries can be negative and must never be read by this loader.

## Single Change

For physical page `p`, KV head count `H`, head `h`, and within-half-page row `r`:

```text
page_heads = uint64(uint32(p)) * uint32(H)
HND row = (page_heads + h) * 16 + r
NHD row = page_heads * 16 + uint64(r) * uint32(H) + h
```

The multiply is widened BEFORE multiplication. Neither the row nor the final
byte offset is truncated to 32 bits. Both expressions equal the existing signed
expressions for valid active IDs and representable tensor allocations.

Only the D128 page-pair helper changes. D64, original/alias loaders, Q staging,
shared-memory destinations, zero-fill predicates, copy commit/wait order,
QK/softmax/PV arithmetic, split geometry, and merge are unchanged. The flag is
experimental and defaults off; no public dispatch entry is promoted.

The pinned [FlashInfer 0.6.13 page implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/include/flashinfer/page.cuh)
uses unsigned dimensions/strides and pointer-sized element offsets. That is a
useful implementation comparison, not evidence that this candidate is faster.
The motivation comes from StreamAttn's own source-correlated address chains in
the [preceding page-pair capture](sm90_page_pair_reuse.md).

## Predeclared Experiment

1. Lightning H100 canary: BF16 D128/G8, HND and NHD; the existing heterogeneous
   batch plus every tail from 1 through 64 tokens. Four cases, with live page-ID
   and value mutation, poisoned padding, output and reconstructed LSE checks.
2. Inspect actual compiled producers: full SASS, binary hashes, register/local/
   shared-memory resources, unchanged tensor and copy instruction sites.
   Static sites are NOT executed instruction counts or elapsed cycles.
3. Proceed only if both long-layout warm medians improve at least 1%, no tail
   median regresses more than 2%, correctness passes, and resources do not grow.
4. If warranted, run the existing 40-case regression and a fresh predeclared
   24-case holdout on Lightning. Holdout request shapes are fixed in
   `experiment_cases('address_holdout')` before the canary returns. Include
   D64 unchanged-binary controls, FP16/BF16, G4/G8, HND/NHD, and mixed requests.
5. Retention requires at least a 2% geometric-mean D128 complete-call gain on
   the independent non-edge cohort, positive warm and perturbed evidence,
   no material (>2%) per-cell median regression, unchanged geometry/resources,
   and all correctness checks passing. These are experiment decision thresholds,
   not error-tolerance or universal-promotion claims.

All calls include producer, merge, and interface conversion. Report the native
control and fastest correct tested FlashInfer backend separately. Isolated
component timings are not additive. No split retuning or layout-specific rescue
is allowed in this experiment.

## Result

The four-case Lightning canary passed output, reconstructed LSE, isolated
producer/merge checks, and live page/value graph mutation. Hardware was NVIDIA
H100 80GB HBM3, Torch 2.7.1+cu128, CUDA 12.8. The baseline resolver timed and
checked FlashInfer FA2 and FA3 at version 0.6.13; FA2 was the fastest here.

Complete-call speedup over the current page-pair control, expressed as the
ratio of median times (9 warm trials, 5 cache-perturbed trials):

| Case | Warm padded | Warm packed | Perturbed padded | Perturbed packed |
|---|---:|---:|---:|---:|
| Heterogeneous HND | 1.128399x | 1.109632x | 1.112248x | 1.100870x |
| Heterogeneous NHD | 0.999638x | 1.000606x | 0.999875x | 0.997487x |
| Tails 1-64 HND | 1.004558x | 0.999760x | 1.004909x | 1.009508x |
| Tails 1-64 NHD | 0.985292x | 1.000957x | 0.980690x | 0.996940x |

All 28 heterogeneous HND pairs improve across the two interfaces and two timing
modes; the worst is 1.092521x. HND producer-only time falls from 76.5002 us to
66.8848 us; merge remains approximately 9.242 us. These isolated times are not
additive. The complete packed HND call is 84.9126 us versus FA2's 69.5405 us,
only 0.818965x the baseline speed. This is a native-control gain, not a packed
FlashInfer victory. NHD's complete packed call is 0.793354x versus FA2.

### What The Compiler Changed

| Static producer property | HND control -> candidate | NHD control -> candidate |
|---|---|---|
| Instruction sites, including padding | 2,632 -> 2,528 | 2,544 -> 2,528 |
| Signed right-shift sites | 50 -> 32 | 34 -> 32 |
| KV `LDGSTS` sites | 32 -> 32 | 32 -> 32 |
| Registers/thread | 167 -> 167 | 167 -> 168 |
| Local bytes/thread | 0 -> 0 | 0 -> 0 |
| Shared bytes/CTA | 65,536 -> 65,536 | 65,536 -> 65,536 |
| Resource-limited CTAs/SM | 3 -> 3 | 3 -> 3 |

Each producer retains 16 `HGMMA.64x64x16.F32.BF16`, four
`HGMMA.64x128x16.F32.BF16`, and one `HGMMA.64x8x16.F16` static site. These are
STATIC counts, not an executed-work or memory-traffic capture. The larger HND
instruction reduction is consistent with the layout-specific timing gain;
this run does not identify instruction-level stall-cycle savings.

The first disassembly parser grouped matrix opcode names too broadly because
it omitted lowercase `x`. The complete raw SASS was captured, so the checked
artifact reparses that text with corrected width-preserving opcode names and
deduplicates identical binaries. No GPU timing or correctness value is changed.

### Decision

Do not advance the uniform change to the broad regression/holdout. NHD misses
the predeclared 1% long-row gain and grows by one register, while padded tail
calls regress. The additional register does not reduce the measured resource
CTA limit, but the cross-layout performance criterion independently fails.
The fresh 24-case holdout and 40-case regression were therefore NOT run.

Keep this explicit, default-off source ablation reproducible because the HND
signal is useful. It is not a selected production family, a new phase-database
entry, an FP16 GPU validation, or an independent confirmation of the HND gain.
There is no layout-specific rescue of this experiment and no public dispatch
change. The existing page-pair path remains the retained control.

The next single-change hypothesis is simpler: inside `valid0`, computing
`source1 = source0 + half_page_stride` does not need an additional `valid1`
branch. The first row is in [0,7], so its partner is in [8,15] of the SAME
allocated page. The second copy still uses `valid1` for zero fill. This can
remove address-control work without admitting invalid tokens, reading inactive
page IDs, or changing online softmax. Test it independently on the original
page-pair control; do not combine it with unsigned arithmetic yet.

## Evidence And Execution

- [Checked capture with full deduplicated SASS](../artifacts/gate0/sm90_page_address_lightning_h100_canary_checked_20260909.json)
- [Paired timing summary](../artifacts/gate0/sm90_page_address_summary_h100_20260909.json)
- [Execution and cleanup record](../artifacts/gate0/sm90_page_address_execution_lightning_20260909.json)

Lightning continued to report `running` after complete four-case output was
available. The result and logs were saved before requesting stop. The runner
then recorded platform state `stop`, saved the result again, requested deletion,
and exited nonzero because platform completion was not `completed`. GPU
benchmark completion and platform-job completion are deliberately not conflated.
Reported cost was 1.4894222; no Modal job ran. The local stop-status waiter was
terminated after the remote job entered `delete` state. A subsequent job listing
confirmed deletion completed and zero jobs remained in the teamspace.

Local verification: 1,173 passed, 71 skipped. The compile-only CI harness now
includes both BF16 and FP16 forms of the experimental address source; this is
not a claim of FP16 GPU runtime coverage.
