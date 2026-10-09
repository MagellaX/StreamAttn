# Fixed-support state ownership experiment

Predeclared after `a400b1d`, before the attribution run. No new model forward,
selection search, context increase, changed support, budget or head grouping.

## First: attribution

Use the hashed Qwen 32K archive from the preceding feasibility run. Select only
technical layer 0 and its saved full/contribution-triangle schedules. Preserve
B1/M1/BF16, NHD/page16, G8, D128, the 32-token logical decision unit, cumulative
FP64 omission allowance 1e-3 and separate native component allowance 0.02.

Measure full native, current full selected and current supplied selected support
against the independently resolved fastest tested correct exact candidate in
each condition. Preserve eight calls per graph, eight independent prepared sets,
warm and rotating conditions, 100 graph replays, seven alternating paired trials.
No guaranteed cold cache, measured HBM transactions or independent-worker claim.

Expose diagnostic producer/merge phases of the existing selected launcher.
Merge-only must consume actual producer states; verify split phase composition
matches the unchanged combined entry point exactly. Isolated component timings
are not additive or a replacement for complete-call latency.

Record valid/padded tasks, query staging repetitions, state finalizations,
allocated partial state bytes, emitted/consumed state counts and compiled CUDA
register/shared/local resources. These are representation counts, not hardware
traffic counters. CUDA local storage is not a measurement of spill transactions.
Producer-heavy timing does not isolate initialization, epilogues or necessary
QK/PV work; a small merge alone does not falsify state grouping.

## Conditional candidate

Only after attribution, consider one grouped-record producer. Match the full
native split length: the current 32K native plan uses 128 partitions per KV head,
four 64-token records per partition. Confirm the actual plan geometry on device.
Do not sweep grouping factors or serialize the full head into one CTA.

Use the equivalence ladder: native full, old selected full, grouped selected full,
old fixed selected, grouped identical selected. The grouped full control must
execute the selected code, not a full-route dispatch shortcut. Preserve each
record identity in K/V staging, next-record prefetch and masks; preserve neutral
states and asynchronous lifetimes. Change state ownership, never token support.

Check numerical order changes against the same FP64 selected reference and
separate unchanged native gate. Candidate wins are supplied-schedule executor
headroom only; decision/upkeep costs and eventual model behavior remain untested.

If full grouping remains far behind native, identify remaining representation
cost. If full approaches native but holes lose, inspect irregular traversal and
mask costs. If selected gains headroom, identify whether runtime-available
certified decisions can fit it. No automatic selector or 64K escalation follows.

The [attention-state algebra](https://docs.flashinfer.ai/tutorials/recursive_attention.html)
permits accumulation across disjoint support. Selection granularity need not
equal state-finalization granularity; that algebra does not predict latency.

## Attribution result and frozen candidate

The complete attribution artifact is
`artifacts/gate0/adaptive_executor_attribution_lightning_h100_20261009_retry1.json`.
The observed full native plan confirms 128 partitions per KV head and four
64-token records per partition. Isolated warm producer/merge medians were
12.524/16.511 us for full support and 11.395/18.442 us for saved triangle support.
These are not additive: rotating isolated sums exceed complete-call latency.
State export and merge therefore have enough visible cost to justify the
conditional candidate, without attributing every producer cycle to fragmentation.

Freeze C=4 before its GPU replay. Full support keeps 1024 records but changes
the rectangular producer grid from 1024 to 256 CTAs and partial storage from
4,227,072 to 1,056,768 bytes. The saved triangle keeps 892 records (386/506),
changing valid tasks from 892 to 224, padded grid from 1012 to 254, and allocated
partials from 4,177,536 to 1,048,512 bytes. These are geometry, not speed forecasts.
The memo's 862-record example described hindsight; it is not substituted for
the saved contribution-triangle schedule in this experiment.

Run `--experiment executor_grouped` once with the same source report and archive.
No grouping sweep. Verify C1/C4 against FP64 with holes, nonidentity physical
pages, per-record head masks, partial final pages, initially empty heads and
fully padded partitions before collecting the unchanged timing protocol. Compare
complete selected calls, retain all candidates and errors, and resolve the
fastest correct exact control separately in each working-set condition.
