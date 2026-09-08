# Mixed-ragged H100 comparison and producer attribution

This experiment measures the retained exact kernels as complete mixed-batch
attention plans. It is not a model benchmark, a serving trace, or a new public
dispatcher route. Every request attends over its full visible logical KV cache.

## What is compared

The matrix combines three synthetic serving boundaries with BF16/FP16,
D64/D128, G4/G8, Hq16/Hq32, HND/NHD page-16 caches, and noncausal or
bottom-right causal attention. Batches contain four or eight requests, query
lengths from 1 to 64, and independent ragged KV lengths up to 32,765 tokens.
Physical pages are shuffled, the first page is shared across requests, and
invalid query rows and KV tails contain NaNs.

Two interfaces are timed separately:

| Interface | Native plan | FlashInfer paged plan |
|---|---|---|
| Padded queries | Producer and exact split merge | Query gather, page-ID compaction, attention, output scatter/zero |
| Packed queries | Query scatter, producer, exact split merge, output gather | Page-ID compaction and attention |

Neither side gathers or repacks K/V. The adapter copies are benchmark
integration costs, not a proposed PyTorch production backend. Timing includes
output only, using preallocated buffers and warm CUDA graphs. Host planning,
JIT, changed-length replanning, request scheduling, and KV append are excluded.
This measures a fixed-length batch with mutable data/pages, not a complete
continuous-batching runtime.

Both native families and forced FlashInfer FA2/FA3 run in the same worker.
Nine trials rotate execution order; each times 100 graph replays. The resolver
chooses the fastest correct tested baseline with matching workload/environment
fingerprints and recorded loaded binary identities. FlashInfer is pinned to
0.6.13 for continuity with the micro-prefill audit. This is not a claim against
every current exact backend or every later FlashInfer release.

Output and LSE are checked against an independent FP32 reference, before and
after mutating Q/K/V and page IDs. FlashInfer's base-2 LSE is converted to
natural logs outside the timed graph. LSE validation writes a separate output
buffer so it cannot repair or conceal a stale graph result. The initial smoke
artifact retains the failed unconverted-LSE comparison; that was a harness
convention error, not evidence of incorrect FlashInfer attention.

The summary labels the per-case fastest native family as an **oracle**. It is
not held-out routing and must not be presented as public dispatcher speed.

## What the mixed batch exposes

Both completed H100 runs passed all output/LSE comparisons with both external
backends available. The full Lightning run has 48 cases and 768 checks; the
independent Modal replay has 24 cases and 384 checks. The input seeds differ.
All recorded profiler/native source hashes match the published sources.

| Run | Correct cases | Padded best-family ratio | Padded all-pair wins | Packed best-family ratio | Packed all-pair wins |
|---|---:|---:|---:|---:|---:|
| Lightning full | 48/48 | 0.230x | 5/48 | 0.160x | 0/48 |
| Modal replay | 24/24 | 0.232x | 2/24 | 0.161x | 0/24 |

Ratios are geometric means of per-case median paired speedups: baseline time
divided by native time. Values below one are losses. FlashInfer FA2 was the
fastest tested baseline in every case on both workers. The full run's short
batch group reaches about 0.83x padded, but the heterogeneous and long-tail
groups are only about 0.12x. These results reject performance promotion of the
current rectangular mixed plan; previously promoted decode routes are unchanged.

Full-matrix padded ratios are 0.313x HND/noncausal, 0.307x NHD/noncausal,
0.170x HND/causal and 0.170x NHD/causal. Both layouts show the loss, and masking
is a further competitive gap. This does not isolate which masked instructions
cause it; an affine-mask ablation must still preserve exact visibility.

Functional coverage is not performance generalization. The retained plan
selects one split count from the maximum padded query extent and a target of
256 producer CTAs. This counts requests that have already exhausted their
queries or KV tiles as though they were useful parallel work.

For the D64/G4/Hq16 heterogeneous case, the natural family launches 256 CTAs
with two splits. Only 108 have nonempty work; 148 take the empty-state path.
The longest request gets 32 CTAs, each processing up to 128 KV tiles. This is
derived exactly from the recorded lengths and split count, not measured SM
occupancy. It identifies a scheduling defect worth testing, not a numerical
error or proof that all latency loss comes from empty CTAs.

The complete comparison includes both query interfaces. Removing query
conversion alone cannot fix a large loss that already exists on padded input,
where the native plan is timed without that conversion. The broader next step
is **ragged work-proportional assignment**, comparing a compact task list or
query-length cohorts against this rectangular control. Give long requests
enough parallel work without manufacturing useful occupancy from padded rows.
Add native packed-query addressing so callers need not round-trip through
padded buffers. Preserve the exact producer/merge state algebra.

The independent replay pairs HND/noncausal with NHD/causal; it cannot separate
layout from masking effects alone. The full factorial matrix is needed before
attributing a latency change to either one. An affine bottom-right causal
specialization is a further hypothesis, not permission to weaken arbitrary
logical-position semantics.

## Source-level producer evidence

Separate Nsight captures profile the retained contiguous natural R64 producer
at M64/G8/D128 with 16 splits. They do not time the mixed paged adapter above.
The exported CUDA/SASS source view identifies the stalled instruction, and
the summarizer deduplicates PCs repeated under inlined source files. CUDA
aggregate rows are never added to SASS rows.

| Anchor | Distinct-PC samples | Long-scoreboard samples | Samples at scalar Q staging store | Share of long-scoreboard samples |
|---|---:|---:|---:|---:|
| B1/N4K | 3,783 | 1,441 | 1,112 | 77.2% |
| B1/N16K | 5,633 | 2,094 | 1,112 | 53.1% |
| B2/N4K | 6,143 | 2,294 | 1,740 | 75.9% |

The hottest PC is `ST.E.U16`, mapped to `sQ(local_query_row, dim) = item;`
at generated CUDA line 312. A long-scoreboard stall at a consumer instruction
is consistent with waiting for a preceding global-load result; it is not proof
that shared-memory stores themselves are slow. The source loop copies Q one
16-bit element at a time through registers. At N16K, a softmax `MUFU.EX2` PC
also contributes 805 long-scoreboard samples, so Q staging is not the only
dependency worth investigating.

These are sampled stalls from three instrumented launches, not elapsed-time
fractions and not an expected speedup. Nsight replay/cache behavior differs
from uninstrumented warm graph timing. The result motivates an ablation; it
does not by itself justify promoting a changed kernel.

## Bounded producer experiment

Replace only the natural producer's scalar Q staging with aligned vector
global-to-shared copies, preferably `cp.async` with correct zero-fill and
completion barriers. Preserve its 64-row tile, split geometry, one FP32 output
state, online-softmax recurrence, and exact merge.
This is an independent local producer experiment, not a substitute for fixing
the complete mixed-batch schedule above.

The relevant distinction is dependency cost versus bytes. At B1/M64/Hq16/D128
in BF16, Q occupies 256 KiB. Sixteen splits stage it sixteen times, or 4 MiB in
total. A 16-byte copy moves eight elements per instruction instead of one, but
does not reduce those bytes or remove split replication. Whether the shorter
instruction/dependency path improves total latency must be measured. Do not
turn the sampling percentages into an Amdahl-law forecast.

Test retained versus vector staging on the same three anchors first, then
check D64/D128, FP16/BF16, partial query rows, causal masks and direct pages.
Keep uninstrumented paired producer-plus-merge timing separate from counters.
If no net improvement survives, retain the original path. R128 widening,
temporal overlap and denominator changes remain independent rejected/mixed
experiments, not ingredients to add to this test.

## Evidence and reproduction

- [Full H100 raw run](../artifacts/gate0/sm90_micro_mixed_lightning_h100_20260907.json)
- [Independent H100 raw replay](../artifacts/gate0/sm90_micro_mixed_modal_h100_20260907.json)
- [Combined timing and derived-grid summary](../artifacts/gate0/sm90_micro_mixed_summary_h100_20260908.json)
- [Source-correlated counter capture](../artifacts/gate0/sm90_micro_source_counters_modal_h100_20260907_v2.json)
- [Deduplicated source-PC summary](../artifacts/gate0/sm90_micro_source_summary_h100_20260907_v2.json)

Both compute jobs completed and were stopped/deleted after artifact retrieval.
The local suite passed 1,034 tests with 71 skipped. No native producer source
or public dispatch route was changed in this experiment.

```bash
python benchmarks/profile_sm90_micro_prefill_mixed.py --suite full \
  --cutlass-root /path/to/cutlass --build-dir /tmp/mixed-build \
  --output-json /tmp/mixed.json
python benchmarks/summarize_sm90_micro_prefill_mixed.py /tmp/mixed.json
python benchmarks/profile_sm90_micro_prefill_counters.py --source-correlated \
  --cutlass-root /path/to/cutlass --build-dir /tmp/source-build \
  --output-json /tmp/source.json
python benchmarks/summarize_sm90_micro_source_counters.py /tmp/source.json
```

Primary references: [FlashInfer paged wrapper implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py),
[FlashInfer LSE convention discussion](https://github.com/flashinfer-ai/flashinfer/issues/2113),
[Nsight profiling and sampled-stall semantics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
