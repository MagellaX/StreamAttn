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

## Decision Rule

- Useful shared omissions and positive headroom: research a cheap replacement
  for the full-information certificate, then charge its complete runtime cost.
- Useful omissions but negative headroom: fix execution granularity or overhead
  before adding a selector.
- Little removal even under hindsight: do not keep tuning the current
  hard-omission formulation for that measured regime.
- A positive complete adaptive path still needs adaptive-conditioned generation
  and held-out validation before any public promotion.

The present 2K/8K captures cannot decide the long-context ambition. A separate,
predeclared follow-up is Qwen at 32K with the same B1/M1/G8, block size, and
budget, keeping the two prompt identities and layers 0/16/24/26/27. New capture
costs and its native replay must be measured separately; do not tile or repeat
old activation tensors and call them genuine 32K model evidence.
