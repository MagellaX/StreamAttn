# What to investigate after the exact affine-causal result

Research review, 2026-09-09. Code checkpoint: `29dde00`.

## Verdict

Both supplied notes identify the right next milestone: explain the remaining
complete-call cost before choosing another execution family. The broader search
adds useful mechanisms, especially from Germany and the Czech Republic. It does
not establish that scheduling is the dominant remaining cost, or that a published
optimization will close our FlashInfer gap.

The objective remains a general exact attention engine with hardware-appropriate
execution choices. This investigation does not replace visible-token attention
with seed selection, and it does not change public dispatch.

This review read both attachments, checked the current planner and result
documentation, reproduced planner arithmetic on CPU, and examined original
papers, author/company reports, and relevant source code. It did not rerun GPU
benchmarks. Country labels below identify institutions or publishing
organizations, not inferred author nationality. This is a targeted search, not
an exhaustive survey of every country's literature.

## What the repository actually establishes

The independent H100 replay reports these geometric means of per-cell paired
speedup ratios:

| Interior affine path | Relative to compact native control | Relative to fastest correct tested FlashInfer backend |
| --- | ---: | ---: |
| Padded interface | 1.867x | 0.701x |
| Packed interface | 1.807x | 0.472x |

Both completed runs selected FA2 in every cell. These are attention-plan results,
not model throughput measurements or a universal exact-kernel win. See the
[measured result](sm90_affine_causal_ablation.md)
and [combined artifact](../artifacts/gate0/sm90_micro_affine_summary_h100_20260908.json).

The affine identity is exact under the validated append contract:

```text
p_K(j) = a + j
p_Q(i) = a + N - M + i
p_K(j) <= p_Q(i)  <=>  j <= N - M + i
```

The implementation removes unnecessary position loads and masking, not visible
tokens. Arbitrary positions still need the general path.

I independently reproduced the attachment's Hq16/G8 schedule counts using
`plan_ragged_schedule` and the committed causal benchmark cases:

| Trace | Producer CTAs | Maximum KV tiles/task | Merge CTAs | Valid query/head output rows |
| --- | ---: | ---: | ---: | ---: |
| Short | 240 | 1 | 512 | 240 |
| Heterogeneous | 248 | 32 | 8192 | 2800 |
| Long-tail | 244 | 37 | 4096 | 1120 |

Short-trace splits are `(64, 16, 8, 32)`. Only 40% of its repeated R64
query/head row slots are useful, versus about 98.3% and 94.5% in the other two
traces. These are geometry calculations, not measured occupancy, transferred
bytes, or the fraction of runtime we can eliminate. Invalid merge rows return
early, so their launch count must not be converted directly into a time claim.

The relevant [planner](../stream_attention/backends/sm90/ragged_schedule.py)
targets 256 CTAs and minimizes the maximum KV interval. It does not model setup,
row utilization, memory reuse, or merge time.

## International findings that change the experiment

### Russia: Yandex's production attention experience

The original Russian-language Yandex decoder article distinguishes single-query
GEMV from GQA/speculative execution, where tensor cores can become useful. Its
introductory fusion comparison is not a comparison against current FlashInfer.
Do not borrow its simplified bandwidth accounting for our warm-cache graphs.
[Original article](https://habr.com/ru/companies/yandex/articles/1006526/).

The more useful source is the Alice AI technical report's inference section.
The team reports replacing a decode-oriented attention implementation for
multi-token speculative verification, then retuning FA3 tiles for that workload.
Its timings are their production configuration, not independent evidence about
StreamAttn. [Technical report, inference section](https://habr.com/ru/companies/yandex/articles/974594/).

**Application:** use useful packed query/head rows, including `M * G`, as a
family-selection feature. "Decode" alone is not an execution geometry. This
supports testing our existing families on realistic verification shapes before
inventing another one. I did not obtain or inspect Yandex's private engine code.

### Germany: model traffic at each memory level

FAU Erlangen's *Analytical Performance Estimation during Code Generation on
Modern GPUs* studies V100/A100 memory-intensive kernels. It models cache-level
traffic and overlapping working sets rather than equating tensor size with
DRAM traffic. Its own evaluation admits that the model cannot reliably rank
the very best configurations; scheduling-order assumptions also miss effects.
This is not a ready-made H100 attention predictor.
[Paper](https://arxiv.org/html/2204.14242v1).

The accompanying Warpspeed implementation explicitly separates L2 transfers,
new wave footprints, previous-wave overlap, and estimated cache evictions.
[Source: predict_metrics.py](https://github.com/te42kyfo/warpspeed/blob/master/warpspeed/predict_metrics.py).

**Application:** test identical tasks in a different order and measure L2 and
DRAM traffic separately. Distinguish logical bytes, generated load/store work,
and measured transactions. Borrow these features, not an entire estimator or
its fitted constants.

### Norway and the Netherlands: portability is about mechanisms, not constants

BAT 2.0 joins NTNU, the Netherlands eScience Center, and Masaryk University in
the Czech Republic. Its experiments show that transferred tuning configurations
can preserve between 58.5% and 99.9% of the target optimum on the tested GPUs.
The study also finds parameter interactions. These are older GPU/general-kernel
results, not a measured Hopper-to-Blackwell loss.
[Paper and institutional affiliations](https://arxiv.org/pdf/2303.08976).

**Application:** preserve interpretable features across architectures, but
recalibrate their costs. Use one-factor experiments to diagnose causes, then
test a small joint set of surviving changes. Holding out different random
values on the same shapes is reproducibility, not workload generalization.
Hold out lengths, mixtures, layouts, and head configurations.

### Czech Republic: predict work changes before predicting runtime

Masaryk's counter-guided tuning paper separates subsystem utilization from
operation counts. It learns how parameters affect work, then uses current
counters to direct a search. This is more transferable than assuming a direct
parameter-to-runtime mapping. Its evaluation covers five kernels on
Kepler-through-Turing GPUs, not WGMMA or Blackwell.
[Paper](https://arxiv.org/html/2102.05297v2),
[published KTT artifact](https://github.com/HiPerCoRe/KTT/tree/v1.3-profile-searcher).

**Application:** give every proposed optimization a predicted counter change:
fewer partial writes, fewer redundant key loads, fewer integer address
instructions, or less tail imbalance. Accept the explanation only when the
predicted change and complete-call improvement both appear. Do this manually
first; adding a tuning framework is unnecessary now.

### United Kingdom and China: compiler scheduling is a real experimental variable

Cambridge's CuAsmRL demonstrates improvements from instruction scheduling on
A100, using an older compiler stack. This establishes that generated schedules
can matter; it supplies neither H100 latency constants nor a promised speedup
for our producer. [Paper](https://arxiv.org/html/2501.08071v1).

DeepGEMM's own release history says post-compilation SASS optimization was
removed because NVCC 12.9 automatically interleaves the relevant FFMA work.
[Project release notes](https://github.com/deepseek-ai/DeepGEMM#news).

**Application:** if producer dependency stalls remain material, compare the same
source under two supported compiler versions. Record generated instructions,
registers, spills, and resource usage. A compiler change can affect more than
instruction ordering, so attribute only what the binary comparison supports.
Do not begin with SASS patching or an RL optimizer.

## Corrections and refinements to the supplied notes

1. **The FA2 chunk floor needs precise wording.** In the pinned automatic
   prefill planner, `max(128 / page_size, 1)` is the minimum searched chunk size
   in pages. With page size 16, that corresponds to 128 tokens. Tail chunks can
   be shorter; fixed-split and disabled-split branches also exist. It does not
   prove that every measured FA2 task processed at least 128 tokens.
   [Pinned scheduler source](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/include/flashinfer/attention/scheduler.cuh#L489).
2. **Recover the actual winning configuration.** Source-level possibilities
   are not launch telemetry. Record selected query/KV tiles, split sizes,
   producer and merge symbols, grids, registers, shared memory, and binary
   hashes from the FA2 path that actually ran. Do not explain an FA2 victory
   using a different FA3 persistent scheduler.
3. **Task costs depend on neighboring tasks.** Two schedules with identical
   task counts and intervals may have different cache reuse and completion
   tails. A sum of isolated task times is not a sufficient batch cost model.
   Task-list order influences locality; it does not guarantee physical SM order.
4. **Keep attribution and intervention distinct.** Isolated producer/merge
   timings help locate costs but need not sum to the complete CUDA graph time.
   Use real producer states for merge replay, with matching strides and split
   counts. Check conclusions through complete-call counterfactuals.
5. **Do not reuse old stall percentages as a diagnosis.** The earlier
   contiguous producer's counters do not establish the paged affine producer's
   bottleneck. Profile the current candidate and winner under matched replay
   and cache settings. Sample fractions are not wall-clock fractions.
6. **Bounded-state scheduling is conditional.** A linear work stream cut into
   P contiguous worker intervals crosses output boundaries at most P-1 times.
   That bounds extra logical fragments, not latency or bytes. State size,
   imbalance, synchronization and reduction costs remain. Associative exact
   softmax merging is an algebraic statement, not a bitwise-order guarantee.
7. **Separate useful leads from validated recipes.** The other personal blogs,
   patents, virtual-memory cache proposals, and mixed-serving designs in note
   two are leads, not established repairs for this kernel. This review does
   not claim to have independently reproduced their speedups. No patent is
   used as experimental performance evidence.

## A deletion budget before another redesign

This is an Amdahl-style decision test, not a new performance model.

For one measured cell, define:

```text
r = T_FlashInfer / T_native
f = removable fraction of native complete-call time
s = acceleration of that component
overall acceleration = 1 / (1 - f + f/s)
```

Under the assumption that unaffected work and interactions stay unchanged,
deleting a component entirely can reach parity only if `f >= 1 - r`.
The reported aggregate ratios illustrate the scale: roughly 29.9% native time
must disappear for padded parity, and 52.8% for packed parity. These are not
measured component fractions. Apply the test per cell; geometric means do not
describe one physical invocation.

For example, deleting a component that accounts for 10% of a call can provide
at most 1.111x under that assumption. It cannot by itself close a 1.43x or 2.12x
gap. If a rewrite changes cache behavior or producer work too, measure those
effects instead of crediting them to "merge removal."

## The next experiment I recommend

Use the current interior-affine implementation as the control. Keep exact
semantics, the pinned external baseline, and matched complete-call interfaces.

| Question | Small intervention | Expected evidence if the hypothesis is right |
| --- | --- | --- |
| What dominates now? | Attribute producer, real-state merge, adapters, and metadata updates | A component has enough complete-call leverage to matter |
| Is the short trace oversplit? | Compare its one-tile tasks with a two-tile floor | Less setup/partial-state work outweighs reduced parallelism |
| Are we losing KV reuse? | Reorder the same task multiset by request/head/KV split/query tile | Better locality or completion behavior without extra work |
| Are intermediate outputs avoidable? | Direct output for single-split work and native packed offsets | Fewer state transfers or adapter copies in the complete call |
| Is code generation limiting the producer? | Same source, controlled compiler comparison | Relevant instruction/resource changes plus repeatable latency improvement |

For the short trace, a two-tile candidate changes splits from
`(64,16,8,32)` to `(32,8,4,16)` and producer CTAs from 240 to 120. This is
arithmetic, not a predicted win: fewer CTAs can leave SMs idle.

Preserve a one-factor diagnostic stage. After identifying useful changes,
test their interactions on unseen request compositions and lengths. Include
both warm replay and a controlled cache-perturbed condition, with equivalent
conditions for both implementations. Keep compilation/planning cost separate
from steady-state latency and report update/replanning costs when contracts
change. New seeds alone do not constitute these holdouts.

Every result should retain output/LSE correctness, mutable page/value checks,
the exact tested GPU and toolchain, baseline configuration, paired timings,
and actual measured traffic/resource counters. Count failures and losing cells,
not only the geometric mean. Unprofiled graph latency remains the performance
decision metric; profiling is explanatory.

## Where deeper research is genuinely needed

The high-value open problem is a small planner that jointly accounts for
useful row density, split-state cost, cache reuse, and completion imbalance
across mixed requests. A universal fixed CTA target does not express these
tradeoffs. Neither does a lookup table of successful benchmark names.

Start with measured work and a few interpretable features. Add model complexity
only when held-out mispredictions expose a missing mechanism. A new persistent
state machine becomes justified if attribution shows that task distribution
and partial-state ownership dominate. Virtual-memory cache ownership becomes
justified only if page indirection is material enough to pay for that interface
change. Neither is established yet.

The international search sharpens the next experiment. It does not justify
pausing for an unbounded literature search, abandoning exact online softmax,
or calling the present result a generalized FlashInfer victory.

This document records the research pass before its proposed GPU experiment.
Follow-up measurements belong in a separate result record; hypotheses above
are not retroactively presented as experimental conclusions.
