# D128-local page-pair address reuse

This is an opt-in producer ablation, not a public dispatch promotion. Vector Q,
R64, 128 threads, the task/split schedule, buffers, exact online softmax, output
state and merge are fixed. D64 emits byte-identical source to its control.
The page-alias loader and noncompact kernel families are unchanged.

## Invariant and scope

For D128, thread `t` copies token `t/16 + 16*p + 8*u`, with `p` in `[0,4)` and
`u` in `{0,1}`. Both copies use page fragment `p` and feature offset
`8*(t%16)`. The second source follows from the first by eight token strides:
`8*D` elements in HND, `8*Hkv*D` in NHD. Physical pages can be arbitrarily
permuted; no cross-page contiguity is assumed.

For a valid-prefix cache, `valid1` implies `valid0`. Resolve a live page under
`valid0`, derive the second pointer only under `valid1`, and retain separate
zero-fill predicates. Both shared destinations still go through their existing
CUTE layouts. Reuse is expressed within one pair, with no state retained across
loader invocations; the compiler controls actual register lifetimes. Graph
replay still observes live page-table updates. The unchanged asynchronous copy and
completion contract follows the [PTX specification](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).

CPU tests cover every thread and local tail length from zero through 64,
multiple tile positions, both layouts, two KV-head counts and two physical-page
orders. Source-isolation tests check the untouched pipeline, alias helper and
D64 source. These tests do not replace GPU checks of the shared layout.

## Machine-code gate

The first H100 capture uses the existing heterogeneous D128/G8/BF16/HND case.
Both targets pass output/LSE and graph page/value mutation checks before
profiling. The source summary deduplicates PCs and checks instruction/sample
totals against kernel-wide counters.

| Measured quantity | Vector Q control | Page-pair candidate |
|---|---:|---:|
| Producer executed warp instructions | 48,999,884 | 24,679,192 |
| Paged-loader region warp instructions | 22,034,652 | 9,495,832 |
| Source-correlated page-load warp instructions | 498,268 | 249,156 |
| Executed 16-byte KV-copy warp instructions | 498,432 | 498,432 |
| Predicated-on integer thread instructions | 1,043,241,408 | 349,388,352 |
| Registers / local bytes per thread | 167 / 0 | 167 / 0 |
| Dynamic shared bytes / resource-limited CTAs per SM | 65,536 / 3 | 65,536 / 3 |

Each tensor opcode retains its executed count: `HGMMA.64x64x16.F32.BF16`
249,216; `HGMMA.64x128x16.F32.BF16` 124,608; and `HGMMA.64x8x16.F16`
31,152. Counts are compared within the same opcode, not treated as equal-sized
matrix operations.

The representation also changes unrolling and integer/index code generation.
For example, executed `SHF.R.S32.HI` instructions fall from 5,035,084 to 300,740.
There are more static KV-copy sites (12 to 32), but the same executed copy
count. Do not attribute the entire instruction reduction to eliminated page
loads, or interpret it as a predicted latency reduction.

DRAM reads are approximately 46 MB for both captures. Replay traffic is not a
single-execution accounting identity; clocks and caches are unmanaged in these
Nsight captures. Instrumented durations and stall samples are not timing gates.

The compiled-work gate passed, permitting the complete-call experiment.

## Predeclared latency experiment

- Regression: the existing 24 causal cases plus 16 tail-boundary cases.
- New holdout: 24 cases with B3/B7 request compositions, specified in
  `experiment_cases("pair_holdout")` before any page-pair latency results.
- Four configurations: D64/G4/Hq16/BF16, D128/G8/Hq16/BF16,
  D64/G8/Hq32/FP16, D128/G4/Hq32/FP16; HND and NHD, page-16.
- D64 is an unchanged-code control, not evidence of a D128 optimization gain.
- Both padded and packed complete graph interfaces include producer, merge
  and their interface costs. FA2/FA3 are resolved and checked as before.
- Output/LSE, poisoned tails, mutated pages/values and component ownership
  checks precede timing. Warm and cache-perturbed measurements are separate.

No timing-based selector, per-cell rescue rule, changed split heuristic or
pipeline variant is permitted in this ablation. Retain only a repeatable
complete-call improvement; otherwise reject the representation while keeping
the evidence. Even a win does not establish parity with the external baseline.

### Harness corrections before the complete replay

The first regression attempt passed all 24 existing cases, then stopped before
the tail cases: the workload metadata claimed a 16-token shared prefix even for
a one-token KV sequence. The prefix now describes the actual common extent,
`min(16, min(kv_lengths))`; existing cases are unchanged.

The first holdout attempt passed two D64 short cases, then stopped because two
new batches labeled `mixed` contained only micro-prefill requests. Their query
lists now include one decode request. No D128 holdout timing had been produced.
These are benchmark-input corrections, not kernel or performance-driven changes.
CPU tests now construct every workload in both matrices. Both failed artifacts
are retained, and complete reruns use separate `checked` artifact names and the
same seeds. Incomplete runs are excluded from the final performance gate.

## Complete-call result and decision

Retain the local candidate as an opt-in experiment. The complete Modal H100
regression run passes 40/40 cases, and the independent Lightning H100 holdout
passes 24/24. Both use PyTorch 2.7.1, CUDA 12.8, BF16/FP16, nine rotated warm
timing pairs and five cache-perturbed pairs per interface. Both FlashInfer
0.6.13 FA2 and FA3 were available and
checked in every cell; comparisons use the fastest tested correct backend.
Captured kernel/harness source hashes match the committed sources.

Geometric means of cell-median speedups over the fixed vector-Q control:

| D128 cohort | Cells | Warm padded | Warm packed | Perturbed padded | Perturbed packed |
|---|---:|---:|---:|---:|---:|
| Existing regression | 12 | 1.509x | 1.445x | 1.435x | 1.397x |
| Half-page/page/tile boundaries | 8 | 1.069x | 1.047x | 1.051x | 1.042x |
| Fresh request-mixture holdout | 12 | 1.554x | 1.503x | 1.498x | 1.461x |

All 896 D128 paired comparisons beat their control: 32 cells times two
interfaces times (nine warm plus five perturbed pairs). The worst ratio is 1.0076x
on a boundary case. For the combined 20-cell D128 regression/boundary set, warm
gains are 1.314x padded / 1.270x packed. The separate 32 unchanged-code D64 cells
have cohort/interface/mode geometric means between 0.9991x and 1.0019x; their
individual timing noise is not an optimization result. Cache perturbation is
not a guarantee of cold execution.

The external gap remains. D128 warm packed ratios versus FlashInfer are 0.673x
on the original regression set, 0.509x on boundary cases and 0.750x on holdout.
No packed cell wins against its external comparator. Padded ratios are 1.026x,
1.248x and 0.979x respectively, showing why the interfaces must stay separate.

Representative D128/G8/BF16/HND heterogeneous cases, complete packed calls:

| Request mixture | Vector Q | Page pair | FA2 | Native-to-FA2 gap removed |
|---|---:|---:|---:|---:|
| Existing regression | 144.57 us | 94.37 us | 69.47 us | 66.84% |
| Fresh holdout | 188.84 us | 117.22 us | 81.30 us | 66.60% |

Gap removed is `(control - candidate) / (control - FA2)` for these particular
matched measurements, not a global engine speedup. The isolated producer drops
126.25 to 75.44 us in the first case, and 170.46 to 100.17 us on holdout. Merge
stays approximately 9.28/9.25 us and 7.78/7.79 us respectively. These isolated
times cannot be summed into an exact complete-call decomposition.

The holdout producer alone remains slower than the external complete call.
Therefore packed-interface cleanup is not the sole remaining repair. The next
research question is which address/layout or arithmetic dependency still costs
more than FA2 after this reduction. The present source capture leaves 9.50M
paged-address/copy instructions, 5.63M softmax/rescale instructions and 5.83M
unmapped/external-helper instructions. These counts identify inspection targets,
not removable-time budgets or permission for a broader pipeline rewrite.

Local verification: 1,162 tests passed, 71 skipped. The offline CUDA build now
also compiles the pair variant for D128 BF16 and FP16, instantiating both layouts.
GPU correctness includes output/LSE, poisoned padding, live page/value mutations,
component ownership and unchanged schedule geometry. Public dispatch is unchanged.

## Evidence

- [Raw checked source capture](../artifacts/gate0/sm90_page_pair_counters_modal_h100_20260909.json)
- [Checked source and opcode summary](../artifacts/gate0/sm90_page_pair_source_summary_h100_checked_20260909.json)
- [Complete regression and boundary run](../artifacts/gate0/sm90_page_pair_modal_h100_regression_checked_20260909.json)
- [Complete independent holdout](../artifacts/gate0/sm90_page_pair_lightning_h100_holdout_checked_20260909.json)
- [Timing and component summary](../artifacts/gate0/sm90_page_pair_summary_h100_20260909.json)
- [Initial regression harness failure](../artifacts/gate0/sm90_page_pair_modal_h100_regression_20260909.json)
- [Initial holdout harness failure](../artifacts/gate0/sm90_page_pair_lightning_h100_holdout_20260909.json)
