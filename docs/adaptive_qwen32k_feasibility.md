# Genuine 32K adaptive feasibility: opportunity without execution margin

**The length hypothesis has now been tested.** Longer context exposes more
removable shared work on these inputs, but every supplied schedule still loses
to the fastest tested correct exact implementation. Do not build a live selector
on this executor or increase context again under the same hypothesis.

The useful intuition is the no-omission control: full native attention costs
about 16 us, while the full selected-route executor costs about 35 us in warm
replay. Removing 16% of work cannot plausibly pay a roughly 20 us execution tax.
Even a generous proportional model gives `35 * 0.84 = 29.4 us`, still above
the roughly 15 us exact baseline. This is a heuristic cost argument, not an
architecture-independent lower bound or proof against adaptive attention.

The conditional upside is equally concrete: if supplied-support execution could
approach the existing roughly 16 us native full path, an idealized 16% reduction
would reach about 13.4 us, leaving perhaps 1-2 us for adaptation against the
15 us baseline. Neither that executor improvement nor proportional scaling has
been demonstrated. This is the cost target that makes the next attribution
experiment worth doing, rather than a predicted speedup.

## What was measured

The [predeclared protocol](adaptive_long_context_protocol.md) and
[raw report](../artifacts/gate0/adaptive_qwen32k_feasibility_lightning_h100_20261009.json)
cover ten fresh model captures, not repetitions of old activation tensors:

- Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Layers 0/16/24/26/27, the existing technical and instruction-holdout identities.
- B1/M1/BF16, final query at 32,767, post-RoPE, full G8 physical sharing.
- 32-token logical blocks; cumulative absolute output-L2 omission budget 1e-3
  per head. Native component allowance remains separately fixed at 0.02.
- Dense upstream. These are repeated-template controlled prompts, not broad
  long-document traffic or adaptive-conditioned generation.

Both conditions capture eight complete attention calls per graph and normalize
time per call. Warm replay repeats one prepared buffer set; rotating replay
uses eight independent valid sets containing identical captured values. The
rotating sets include Q/K/V, plan metadata, outputs and workspace, not just KV.
Full compact KV is 32 MiB per set, 256 MiB across eight sets. Selected active KV
payload is recorded separately. No copies, gather, eviction kernel, decisions,
route construction or summary updates are timed. There are no hardware memory
traffic counters, no guaranteed cold-L2 claim and no production-traffic claim.

Seven paired trials alternate exact/selected order, with 100 graph replays per
measurement. Conditions run warm then rotating in one H100 80GB HBM3 worker;
this is not independent-worker confirmation. Identical schedules reuse timings.

## All omission comparators

Fractions count tokens over physical KV heads, not independent Q-head masks.
Each schedule keeps the full group of eight Q consumers together.

| Comparator | Mean omitted | Maximum omitted |
| --- | ---: | ---: |
| Full control | 0% | 0% |
| Current centered two-gate support | 0.00977% | 0.04883% |
| Exact-mass radius | 0.86426% | 2.83203% |
| Contribution triangle | 3.50586% | 12.89063% |
| Hindsight best found | 4.63867% | 15.82031% |

For comparison, the earlier Qwen 8K triangle/hindsight means were 1.055%/1.562%.
This is increased semantic opportunity, not increased runtime margin. The old
8K and new 32K timing graphs have different call counts; their latency values
are not a matched length-scaling experiment.

| Layer | Technical triangle / hindsight | Instruction triangle / hindsight |
| --- | ---: | ---: |
| 0 | 12.891% / 15.820% | 6.982% / 14.697% |
| 16 | 3.516% / 3.809% | 2.930% / 3.076% |
| 24 | 5.127% / 5.225% | 2.344% / 2.441% |
| 26 | 0.391% / 0.439% | 0.244% / 0.244% |
| 27 | 0.391% / 0.391% | 0.244% / 0.244% |

The current pre-K rule skips nothing. Its only two-gate omissions are one
32-token block in each of two captures, after QK. The known-support native
replay optimistically omits K as well as V; a live post-QK decision has already
paid for K. Even that optimistic replay loses.

Hindsight and triangle results are different opportunities. For instruction
layer 0, hindsight retains accuracy with about twice as much removal as the
triangle certificate by exploiting vector cancellation. Both still use full
information; hindsight is a deterministic greedy best-found witness, not a
global optimum or a runtime certificate.

## Execution headroom

FlashInfer 0.6.13 **ragged FA2** wins the independent exact candidate resolution
for all ten captures in each condition. Paged FA2/FA3, ragged FA3, forced Flash
SDPA, StreamAttn full native, and StreamAttn full selected control were also
correctly measured. Standalone FA3 and xFormers were absent; cuDNN rejected M1.
This is not a latest-version or all-library completeness claim.

Pilot latency ranges across the ten captures:

| Exact candidate | Warm, us | Rotating, us |
| --- | ---: | ---: |
| Fastest tested: ragged FlashInfer FA2 | 15.168-15.404 | 22.258-22.421 |
| StreamAttn full native | 15.663-15.958 | 24.102-24.292 |
| StreamAttn full selected control | 34.523-34.810 | 40.582-40.753 |

`H = exact time - supplied-schedule time`. Ranges below are per-capture paired
median headroom, not confidence intervals.

| Supplied schedule | Warm H, us | Rotating H, us | Positive pairs, both conditions |
| --- | ---: | ---: | ---: |
| Full control | -19.541 to -19.383 | -17.983 to -17.755 | 0/140 |
| Current two-gate | -19.541 to -19.401 | -17.979 to -17.755 | 0/140 |
| Exact-mass radius | -21.843 to -18.834 | -20.477 to -17.695 | 0/140 |
| Contribution triangle | -21.931 to -19.956 | -20.864 to -19.375 | 0/140 |
| Hindsight best found | -21.913 to -19.388 | -20.550 to -18.915 | 0/140 |

Hindsight ratios are 0.408-0.437x warm and 0.522-0.545x rotating. Do not pool
all methods as independent trials: some share exactly the same schedules and
measurements. No selector cost is included, so cheaper decisions cannot rescue
these measured schedule/executor combinations.

## Numerical and provenance checks

- Largest FP64 omission error: 0.000999994059, within the unchanged 1e-3 budget.
- Largest selected native component error: 0.01344212, within the separate 0.02 gate.
- Largest selected native head-L2 arithmetic error: 0.01840645.
- Largest observed combined native/full-FP64 head-L2 difference: 0.01840647.

These are not native 1e-3 total-error certification or generation-quality results.
The arithmetic difference does not authorize relaxing the omission budget.
All eight replicas per candidate/schedule were checked before timing. The
338,222,461-byte capture archive was downloaded and every tensor hash verified;
archive SHA256 is `304482bd0d1a1ed319dcb6f6e1a312506f9d48ce6b3670756bb06eaeface6e20`.

Execution used the predeclared `29b6494` sources. Platform completion briefly
preceded the last log chunks; the complete ten-record report was recovered from
the same successful job before deletion. A bounded final-log drain and regression
coverage were added to the controller. This was not a GPU rerun or a scientific
failure. No GPU tasks remain active.

## The next research question

There are two boundaries, not one:

1. L26/L27 still expose very little shared hard-omission opportunity under the
   tested searches and unchanged budget. Stop tuning this formulation for those
   tested inputs unless a materially different hypothesis appears.
2. Layer 0 exposes nontrivial value/cancellation opportunity, but the existing
   adaptive executor cannot monetize even a free schedule. Attribute execution
   before building another bound or selector.

The technical layer-0 hindsight schedule contains 862 valid 64-token records,
yet launches 1,000 producer CTAs versus 1,024 full. Workspace falls only from
4,227,072 to 4,128,000 bytes despite 15.82% token removal. The instruction case
contains 874 records and launches 964 CTAs. These are representation costs to
investigate, not proof that padding or merging explains the roughly 20 us gap.

The intuitive candidate is to carry one online-softmax output state across
several retained records rather than emit a global partial state for every
64-token record. **This is a hypothesis, not the next implementation yet.**
First keep the retained schedule fixed and measure producer versus merge time,
partial-state traffic, route/address overhead and launch dependencies. Establish
how much cost can actually be removed. A merge-only saving cannot justify a
redesign if the producer dominates, and reducing padded CTAs does not create
additional useful parallelism.

Only a demonstrated cost lever justifies changing this adaptive executor.
Neither another fixed seed, another context increase, relaxed head sharing,
budget relaxation nor unrelated dense-kernel tuning answers the result.
