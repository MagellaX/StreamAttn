# Adaptive feasibility: value evidence before another kernel

This experiment asks whether a better omission certificate has useful work to
recover, and whether the existing native executor can turn that work into
latency headroom. It does not promote a selector or change public dispatch.

## Frozen Contract

- Saved, hashed dense-conditioned Qwen2.5-3B 8K and TinyLlama 2K activations:
  two prompts and five layers per model, 20 captures total.
- Only the final captured prefill query is replayed. Its position is `N-1`, so
  all supplied keys are visible. This is not autoregressive generation.
- B1, M1, BF16 inputs, 32-token logical blocks, complete G8 consumer sharing.
  One schedule per physical KV head; no head-private replication.
- The omission budget is cumulative `1e-3` L2 **per attention head output**.
  FP64 replay checks it against full-context attention on the same activations.
  Zero budget retains all blocks.
- The native BF16 arithmetic check is separate: maximum component error
  against the FP64 selected reference must be at most `0.02`. Actual L2
  arithmetic error is also recorded. This is an empirical diagnostic gate,
  not a floating-point certificate or a native `1e-3` accuracy promotion.

The existing two-gate certificate is replayed without changing its sequential
admission rule. Its pre-K and post-QK counters remain distinct. A post-QK
omission does not mean K/QK work was avoided.

## Full-Information Comparators

Let `p` be full normalized attention, `a_b = sum_b p`,
`c_b = sum_b p*v`, and `o` the full output. Define

```text
w_b = c_b - a_b * o

o_full - o_retained = sum_omitted w_b / (1 - sum_omitted a_b)
```

The direct selected-attention calculation independently checks this identity.

| Comparator | Admission test | Meaning |
|---|---|---|
| Full control | No omissions | Matched executor overhead without sparsity |
| Exact-mass radius | `2 R sum_omitted a_b <= epsilon` | Full-information mass bound using a global centered V radius |
| Contribution triangle | `sum_omitted norm(w_b) / (1 - sum_omitted a_b) <= epsilon` | Expensive offline certificate sensitive to value contributions |
| Hindsight best found | Actual cumulative output error within budget | Greedy witness, not a runtime certificate or global optimum |

The hindsight search tries contribution-norm order, mass order, and reverse
logical order, with up to three passes. It keeps the schedule with the fewest
blocks. A failed search is not an impossibility result. The contribution
certificate itself uses the full output; it is deliberately unavailable to a
cheap pre-K selector. Its role is to expose certificate slack.

## Native Headroom

Schedules lower to page-16 NHD `PackedRoute64` metadata and run through the
existing H100 selected WGMMA producer and state merge. Selected page atoms are
packed as metadata; K/V are not gathered inside execution. The complete native
call is graph-captured, including all producer and merge kernels. Route count,
maximum padded producer grid, metadata bytes, and workspace are recorded.

Available exact comparisons include forced Flash SDPA, FlashInfer FA2/FA3
contiguous and paged adapters, optional standalone FA3/CUTLASS/cuDNN adapters,
and StreamAttn's full native plan. Missing or rejected backends are explicit.
The fastest **tested correct** exact candidate is chosen by pilot median, then
alternating paired graph trials measure

```text
H = time_fastest_tested_correct_exact - time_precomputed_native_schedule
```

Route preparation, summary construction, decisions, and updates are excluded
on purpose. Positive `H` is only the available overhead budget. Nonpositive
`H` rejects this schedule/executor pair, not adaptive attention globally.
Warm repeated graph replay is not a measured HBM-traffic reduction. Full-route
controls distinguish kernel performance from benefits due to work avoidance.
The supplied K/V payload is only 8 MiB for Qwen 8K and 2 MiB for TinyLlama 2K,
before metadata/workspaces. Repeated isolated calls can therefore benefit from
H100's cache. A result here cannot substitute for model-interleaved or controlled
working-set measurements at the intended long-context operating point.

## Results, 2026-10-09

The [raw H100 replay](../artifacts/gate0/adaptive_feasibility_lightning_h100_native_20261009.json)
completed all 20 captures. All mathematical schedules stayed within the frozen
omission budget; the largest observed FP64 head-output error was `0.0009972391`.
No public routing policy or kernel was promoted.

### Removable Work With Full G8 Sharing

Each model contributes ten captures. Fractions count retained tokens over all
physical KV heads, not independent query-head masks.

| Model/context | Full-information comparator | Mean omitted | Maximum omitted |
|---|---|---:|---:|
| Qwen 8K | Exact-mass radius | 0.156% | 0.586% |
| Qwen 8K | Contribution triangle | 1.055% | 2.539% |
| Qwen 8K | Hindsight best found | 1.562% | 4.492% |
| TinyLlama 2K | Exact-mass radius | 0.469% | 2.344% |
| TinyLlama 2K | Contribution triangle | 8.828% | 29.688% |
| TinyLlama 2K | Hindsight best found | 16.445% | 69.922% |

The triangle bound exposes genuine certificate slack, especially at TinyLlama
layer 0. Hindsight admits further work removal by checking aggregate vector
error rather than summing residual norms. Neither comparator is cheap or
runtime-available. The same absolute `1e-3` budget permits different relative
errors at different output scales; omission percentages are not comparable
model-quality scores.

### Matched Native Execution

The fastest tested correct exact candidate was paged FlashInfer FA2 for every
Qwen capture and StreamAttn's **full, no-omission selected executor** for every
TinyLlama capture. FlashInfer FA3 was also measured. Standalone FA3 and xFormers
were not installed; the cuDNN adapter rejected M1. This is not an all-backend
or latest-release superiority claim.

- Qwen hindsight schedules lost all `70/70` paired trials across ten captures.
  Median speedup per capture was `0.725x-0.864x`, with headroom between
  `-4.208 us` and `-1.756 us`. The full selected controls also lost; selector
  cost cannot repair these measured schedule/executor combinations.
- TinyLlama's two layer-0 hindsight schedules had positive headroom in all
  seven paired trials each. Their large work reduction produced only a small
  latency margin:

| TinyLlama layer-0 prompt | Omitted tokens | Exact median | Selected median | Paired median H | Paired median speedup |
|---|---:|---:|---:|---:|---:|
| Technical | 69.922% | 7.187 us | 6.973 us | +0.215 us | 1.031x |
| Instruction holdout | 65.234% | 7.398 us | 7.085 us | +0.300 us | 1.042x |

The layer-0 contribution-triangle schedules omitted 29.688% but had **negative**
headroom: `-0.418 us` and `-0.204 us`. A separate TinyLlama layer-18 capture had
only `+0.032 us` hindsight/triangle headroom. The remaining hindsight cases
were negative or near parity; tiny positive medians in the no-omission controls
are measurement variation, not adaptive gains. Identical schedules reuse one
measurement and are not independent experiments.

The two layer-0 hindsight schedules contain 40/45 valid 64-token records but
still launch 80 producer CTAs because the grid uses the largest KV-head row;
full execution uses 128. These counts expose route padding and execution
granularity, but do not establish a hardware stall cause. Route preparation,
summary upkeep, and live decisions remain excluded from the quoted times.

The worst native component error versus its FP64 selected reference was
`0.01286024`, within the separately declared `0.02` gate. Worst native head-L2
arithmetic error was `0.01764049`, which is **larger than the omission budget**.
Do not interpret the FP64 omission check as a native `1e-3` total-error guarantee
or an autoregressive model-quality pass. This is one worker/run, not independent
performance confirmation.

## Decision Rule

- Positive conservative-comparator headroom: measure minimum plausible decision
  and update cost before researching a cheap replacement certificate.
- Positive headroom only with hindsight cancellation: investigate cheap
  cancellation identification; this does not establish selector readiness.
- Useful omissions but negative headroom: attribute executor costs before
  changing execution granularity or adding a selector.
- Little removal even under hindsight: do not keep tuning the current
  hard-omission formulation for that measured regime.
- A positive complete adaptive path still needs adaptive-conditioned generation
  and held-out validation before any public promotion.

The [predeclared genuine 32K follow-up](adaptive_qwen32k_feasibility.md) is now
complete. It preserved B1/M1/G8, the block size, budget, two prompt identities
and layers, while capturing fresh activations and separately testing warm and
rotating working sets. Qwen's triangle/hindsight omission means increased to
3.506%/4.639%, but every supplied schedule lost in both conditions. The 32K
result closes context escalation under this hypothesis, not adaptive attention
globally. Layer-0 schedules now justify fixed-schedule executor attribution;
L26/L27 remain low-opportunity under the tested searches. Do not build another
selector, loosen the contract or start an exact-only detour to avoid that
decision. Positive hindsight-only results would still require cancellation
identification research before selector integration.

## Focused Research Questions

The missing information is not another general attention survey. It is how to
replace full-information value evidence with a cheap live-query bound, and
whether the native schedule leaves enough time to evaluate that bound.

- [Value-aware Approximate Attention](https://aclanthology.org/2021.emnlp-main.753/)
  motivates measuring output error including V rather than ranking attention
  weights alone. Value awareness is established prior work, not a novelty claim
  for this experiment. Our contribution comparator tests whether that distinction
  changes the removable work under the frozen sharing and error contract.
- [BLASST, sections 3-4](https://arxiv.org/html/2512.12087v1) uses already-computed
  QK maxima and coordinated skip decisions. Its decode design can avoid V loads
  and PV; this does not establish pre-K avoidance or our cumulative output-L2
  guarantee. Its calibrated thresholds cannot replace that contract silently.
- [Runtime-Certified Bounded-Error Quantized Attention, section 4.2](https://arxiv.org/html/2605.20868v1)
  bounds value reconstruction error with attention mass times cached block error.
  The paper explicitly distinguishes its per-block escalation rule from total
  budget enforcement. The transferable question is which cached value-error
  annotations remain useful after charging their reads and updates; adopting a
  local threshold or host-tier cache would not answer the present experiment.
- [SVG-EAR, section 4.2](https://arxiv.org/html/2603.08982v1) amortizes an error
  probe across query clusters at cost `O(Cq * Nk * D)`. For M1 decode, our inference
  is that `Cq=1` still leaves a full-key `O(Nk * D)` scan, with no many-query
  amortization. Its attention-map bound assumes normalizer stability and its
  omitted regions receive mean compensation. Neither is a drop-in proof for
  strict token omission under our output-error contract.

None of these sources supplies a measured StreamAttn speedup. A candidate must
first identify runtime-available evidence, prove cumulative accounting, preserve
the shared physical bypass, and fit within measured execution headroom. A
summary-completion scheme would change the present hard-omission formulation
and needs an explicit decision rather than an unnoticed implementation change.
