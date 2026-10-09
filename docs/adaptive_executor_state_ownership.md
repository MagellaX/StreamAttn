# Selection is not a state boundary

The fixed-support experiment confirms a removable execution cost: exporting one
attention state per retained 64-token record was unnecessary. Carrying one
online-softmax state across four records brings full selected execution close
to full native execution. **It does not yet make the supplied omission schedule
faster than the strongest tested exact comparator.**

## What changed

The logical omission unit remains 32 tokens, physical pages remain 16 tokens,
and each packed record still identifies four physical page fragments. Only state
ownership changes: up to four records now share initialization, query staging,
normalization and one exported partial state. Independently scheduled partitions
still finish with the existing associative merge.

This uses established [attention-state algebra](https://docs.flashinfer.ai/tutorials/recursive_attention.html),
not a new softmax identity. A hole in retained support need not end an online
state. Current/next record IDs explicitly advance K/V loads and masks, while
the existing WGMMA and asynchronous storage lifetimes remain intact.
Grouping also activates the existing inter-record prefetch. This experiment
does not isolate the individual savings from exports, staging and pipelining.

The candidate is an opt-in `records_per_cta=4` on the static NHD/D128/G8 selected
plan. Public dispatch remains unchanged. No new selector, KV repacking, head-
private duplication, context increase or error-budget change was introduced.

## Frozen experiment

See the [predeclared protocol](adaptive_executor_attribution_protocol.md).
The first attribution run confirmed that the full native plan already uses four
records per partition, so C=4 was chosen before the candidate replay, not from a
grouping sweep. The initial isolated full producer/merge medians were 12.524 and
16.511 us warm. These timings are not additive.

The candidate replay uses one saved technical layer-0 Qwen2.5-3B capture:
B1/M1, 32K KV, BF16, two KV heads, G8, D128, NHD/page16, post-RoPE. Its two
unchanged schedules are full support and offline contribution triangle. The
latter omits 12.890625%, retaining 24,704/32,384 tokens per KV head. Its FP64
maximum head-L2 omission error is 0.0009544702 and bound 0.0009993859, below the
unchanged 1e-3 omission allowance. This is not a runtime-available decision.

Eight independently allocated copies are checked before replay. Each graph
performs eight calls; warm uses one repeated buffer and rotating cycles all
eight. Timing uses 100 replays and seven alternating exact/selected pairs.
Rotation is not guaranteed cold-cache measurement or an independent-worker
replay.

## Complete-call results

Selected times below are medians of complete selected calls from the paired
trials, in microseconds. Native full is its separately measured pilot median;
these are not the isolated diagnostic phase timings.

| Arm | Warm, us | Rotating, us |
| --- | ---: | ---: |
| A: full native | 16.288 | 24.433 |
| B: full selected, C1 | 34.987 | 40.710 |
| C: full selected, C4 | 16.564 | 25.057 |
| D: saved selected support, C1 | 36.864 | 43.769 |
| E: identical saved support, C4 | 18.473 | 26.179 |

C4 full actually executes the selected producer; it does not dispatch to A.
Ratios of the paired-call medians show C1/C4 improvements of 2.112x warm and
1.625x rotating for full support, and 1.996x/1.672x for the saved selected support.
These are improvements over the old executor, not over FlashInfer.

For the full-support representation penalty, compare the same pilot series:
warm C1/C4/native are 35.052/16.680/16.288 us, reducing the penalty from 18.764
to 0.391 us (97.9%). Rotating pilots reduce it from 16.469 to 0.632 us (96.2%).
Pilot and paired measurements are reported separately; their differences are
not paired confidence intervals.

FlashInfer 0.6.13 ragged FA2 wins the independently resolved correct exact
candidate comparison in both conditions. Its pilots are 14.873 us warm and
22.250 us rotating. Correct measured alternatives include paged FA2/FA3,
ragged FA3, forced Flash SDPA and both StreamAttn full controls. Standalone FA3
and xFormers were absent; cuDNN rejected M1. This is a fastest-*tested* claim,
not completeness across libraries or versions.

`H = paired exact latency - paired supplied-schedule latency`:

| Schedule/executor | Warm H, us | Rotating H, us | Positive pairs |
| --- | ---: | ---: | ---: |
| Full, C1 | -19.902 | -18.200 | 0/7 in each condition |
| Full, C4 | -1.185 | -2.578 | 0/7 in each condition |
| Triangle, C1 | -21.417 | -21.222 | 0/7 in each condition |
| Triangle, C4 | -3.018 | -3.645 | 0/7 in each condition |

There is still **no supplied-schedule headroom for decision/upkeep costs** in
this tested case. Do not pool related controls as independent evidence or call
the executor improvement an adaptive serving speedup.

## State geometry and attribution

| Support/executor | Retained records | Valid tasks | Padded grid | Partial bytes |
| --- | ---: | ---: | ---: | ---: |
| Full C1 | 1024 | 1024 | 1024 | 4,227,072 |
| Full C4 | 1024 | 256 | 256 | 1,056,768 |
| Triangle C1 | 892 | 892 | 1012 | 4,177,536 |
| Triangle C4 | 892 | 224 | 254 | 1,048,512 |

These are geometry and emitted-state counts, not measured hardware traffic.
The memo's 862-record example is hindsight, not the triangle schedule tested
here. Every retained token still contributes once, with G8 physical sharing.

| Isolated phase | Full C1 warm | Full C4 warm | Triangle C1 warm | Triangle C4 warm |
| --- | ---: | ---: | ---: | ---: |
| Producer, us | 12.645 | 10.286 | 11.320 | 9.494 |
| Merge, us | 16.373 | 5.669 | 18.414 | 7.316 |

Rotating C4 producer medians are 18.229 us full and 17.076 us triangle;
merge medians are 5.687 and 7.275 us. Merge-only consumes real completed
partials, and producer+merge composition matches the respective combined
entry point bitwise. Isolated sums are **not** complete-call latency.

Compiled producer registers per thread are 43 for C1, 74 for C4, and 68 for
native; all use 36,096 static shared bytes and report zero local bytes. The
merge uses 66 registers, 2,048 shared bytes and zero local bytes. No hardware
spill-load/store or memory-transaction counters were collected.

## Correctness and scope

All eight copies per schedule/executor pass the separate native component gate
0.02. Largest selected component error versus its FP64 reference is 0.003684607;
largest head-L2 arithmetic error is 0.005132271. The largest observed combined
native/full-FP64 head-L2 difference is also 0.005132271. These are **not** a
native 1e-3 total-error certificate or an autoregressive quality result.

Two additional H100 equivalence cases pass: full support, and nonadjacent
support with nonidentity physical pages, changing per-record head masks, a
partial final page, initially empty heads and neutral padded partitions.
The local suite reports 1294 passed/83 skipped; those skips are not GPU passes.

## The next discriminator

State fragmentation was a major removable cost. The grouped full control now
approaches native, but the selected schedule still costs more than grouped
full. Isolated producer timing improves with omissions while merge timing
regresses. That localizes the next question more usefully than adding a selector.

The selected C4 workspace has 127 partitions per row, versus 128 for full.
The next controlled assay should preserve the exact records and valid states,
but pad only the selected partial-state capacity to 128 using neutral entries.
Compare natural-127 versus padded-128 merge and complete calls. This changes
neither token support nor C=4; it tests whether partial-state geometry is
responsible for the merge regression. It does not by itself distinguish loop
tail, stride, alignment or cache effects, and is not a predeclared performance
fix. Do not sweep grouping factors or call alignment causal without that assay.

If padding removes the merge penalty but full selected execution still loses,
the remaining producer/native/external gap must be paid before runtime decisions
are useful. If it does not, investigate the merge dependency/address behavior
using actual states. Only robust positive complete-call headroom would reopen
the question of a certified, runtime-available decision cheap enough to fit it.
Pre-K decisions could avoid K and V; post-QK decisions have already paid for K.
Neither outcome authorizes relaxing the omission budget or reopening late-layer
L26/L27 schedules with almost no shared omission opportunity.

## Evidence

- [Initial attribution](../artifacts/gate0/adaptive_executor_attribution_lightning_h100_20261009_retry1.json)
- [C1/C4 equivalence ladder](../artifacts/gate0/adaptive_executor_grouped_lightning_h100_20261009.json)
- [Original 32K feasibility](adaptive_qwen32k_feasibility.md)

Candidate source is `ce68851`. The replay verifies the existing capture archive
and tensor hashes, then reuses saved schedules without another search or model
forward. The bounded job completed and was deleted. Platform cost at collection
was 0.6343 (billing can lag); observed teamspace balance changed from 2.22 to
1.40, with organization credits still 9.00. No purchase or credit transfer was
made. These platform details are provenance, not performance claims.
