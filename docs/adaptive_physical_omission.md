# Adaptive Physical Omission: What the Controls Reveal

## Decision

Do not spend omission budget when the corresponding shared operation still
runs. The native two-gate prototype now proposes row omissions, commits them
only after all mathematically valid rows agree, and otherwise retains the
computed contributions. A failed pre-K proposal is not charged and remains
eligible for a separate post-QK vote. Masked and padded rows do not vote.

This is a contract repair, not a speed optimization. Keeping extra contributions
does not imply monotonically smaller observed error because cancellation and
rounding can change. The unchanged cumulative bound still applies.

The supplied note's division-free predicate is algebraically equivalent for
`0 < epsilon < 2R`, with `kappa = epsilon / (2R - epsilon)`. We did not change
that arithmetic or move the value-radius reduction in this ablation. Changing
several mechanisms at once would obscure what physical agreement costs.

## H100 Mechanism Run

[Raw canary](../artifacts/gate0/adaptive_physical_commit_modal_h100_20260910.json),
[summary](../artifacts/gate0/adaptive_physical_commit_summary_h100_20260910.json).

FP16/BF16, six synthetic cases each, graph-captured execution with seven
alternating-order timing pairs. Counters, omission-bound stores, and retained
support stores are excluded from timing. Ten focused GPU pytest checks also
passed on this H100 run.

Peaked FP16 median graph times:

| Execution | Microseconds | Interpretation |
|---|---:|---|
| Physical-commit adaptive | 130.603 | Decisions plus traversal and retained work |
| Legacy rowwise diagnostic | 120.724 | Old budget-spending semantics |
| Same-support masked-work control | 422.498 | Deliberately executes omitted regions |
| Zero-budget control | 221.986 | No omission decisions |
| Known support, retain traversal slots | 18.686 | Precomputed block IDs, no certificate evaluation |
| Known support, compact block IDs | 3.330 | No skipped traversal slots |
| Forced Flash SDPA | 18.856 | Full-context exact comparison |

The two oracle controls use the actual shared support `{block 0}` verified from
the adaptive kernel. Their output is checked against FP32 attention on that
support. Oracle discovery is not timed or supplied by a runtime selector.
Neither oracle is replayed after arbitrary Q mutation. The complete adaptive
graph is replayed after Q mutation and checked using freshly recorded support.

These controls establish a useful execution floor, not a 5.7x runtime win over
Flash SDPA. They implicate the decision/traversal path on the peaked case, but
are not an instruction-level additive cost decomposition: compiler scheduling
can change across kernels. The complete adaptive path loses to Flash SDPA in
every timed case. No route is promoted.

Mixed FP16 rows now report zero committed omissions, zero omission bound, and
bit-identical output to the zero-budget kernel. All 2,048 QK/PV regions still
execute. Latency is `472.339 us`, versus `438.819 us` for the rowwise diagnostic
and `217.818 us` without decisions. Physical voting fixed wasted approximation
budget and made this prototype slower. Keep both facts.

## Numerical Protocol

Version 2 records the actual retained support during an untimed launch. It
separates `||o_returned - o_A||_2` from `||o_A - o_full||_2`, using FP32 references.
The omission budget is unchanged. The predeclared execution allowance is:

```text
cast_only_L2 + 4 * dtype_epsilon * ||o_A||_2 + 1e-5 * sqrt(D)
```

The FP32 omission comparison allows `2e-5 * (1 + ||o_full||_2)` numerical slack.
These are diagnostic numerical tolerances, not formal interval certificates or
model-level guarantees. All 12 cases passed this new protocol. Each result also
retains `original_v1_limit_passed`; the two causal cases still fail the old
absolute limits. The earlier failed artifact is unchanged.

Key-only summaries now explicitly have `has_value_bounds=False`. Supplying
them for positive-budget value certification raises an error instead of
interpreting placeholder zeros as a bound on V. Existing cache-content validity
remains the caller's responsibility; this does not add full-cache rescans.

## Real Activations

[Raw diagnostic](../artifacts/gate0/adaptive_real_physical_modal_h100_20260910.json)
and [last-query replay](../artifacts/gate0/adaptive_real_physical_last_query_cpu_20260910.json).

Qwen/Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, BF16 H100 capture, 2,048 tokens,
layers 0/16/24/26/27, one technical prompt and a new instruction-conflict
holdout. Q is the last 16 prefill queries, with 16 query heads, two KV heads,
and D128. Visibility is explicitly `key_position <= query_position` in the
complete unpadded cache, not the prototype's top-left causal convention.

Analysis runs offline in FP64. It compares summary-only, two-gate, and exact
block-mass decisions under the same cumulative budget `1e-3`, sequential order,
and running-max admission condition. Exact block mass is an oracle comparator,
not an optimal subset solver or free runtime feature. Full-context LSE provides
only a common exponential scale in the offline analysis; all admission ratios
are invariant to that scale.

| Physical scope | Origin-radius result across ten captures |
|---|---|
| Summary-only, any scope | Zero omissions |
| Per-head 16-query tile, two-gate | Zero saved regions except L27: 3.03% / 3.71% PV regions |
| Per-head 16-query tile, exact block mass | Maximum 4.10% PV regions saved |
| Full G8 plus 16 queries | Zero removable regions, including exact block mass |
| Full G8 plus last query only | Zero removable regions in saved-capture replay |

The whole last-query analysis reuses the recorded tensors on CPU. It does not
constitute another model run. The 21.8 MB Q/K/V archive is retained locally
alongside the raw JSON, outside git; input hashes, model revision, prompt hashes,
positions, and analysis results are published. A `--captures` CLI supports
replay without GPU/model loading. The profiler gained this replay-only CLI after
the capture; the earlier source fingerprint is preserved.

Mean-centered value radii were tested offline, with no change to attention V
arithmetic. The centered/origin radius ratio ranged from `0.844` to `1.145`:
the mean is not always a better enclosing-ball center. Neither choice produced
a removable full-G8 region on this set. Translation-invariance and constant-V
properties have CPU regressions. There is no evidence here for integrating a
centered-radius runtime or starting a center-search campaign.

This is a small, dense-conditioned, single-family 2K diagnostic. It is not a
32K stress result, routed-state capture, generation-quality gate, or universal
impossibility proof. In particular, the fixed order, max gate, global value
envelope, error budget, and physical grouping constrain this negative result.

## Next Research Decision

The known-support execution floor is viable on the synthetic peak, but real
physical admissibility is the present blocker. A faster loop for the same gate
does not address that. Keep the adaptive objective; do not return to exact-only
optimization or weaken the omission budget to manufacture sparsity.

Next quantify smaller head groups against their extra KV reads on the saved
traces, then test whether a tighter value-contribution bound changes the limited
oracle opportunity. Any proposal must state the failure it changes and include
decision/metadata cost. Expand to longer contexts and another model family
before extrapolating this sample. Do not assume fewer grouped heads, a better
center, or more skipping must be faster.

## Execution Record

Lightning again rejected the submission before execution with `job
reconciliation failed`, reporting zero cost; the job was deleted. Two bounded
Modal H100 runs completed the mechanism and real-activation diagnostics and
stopped. No model/kernel result is attributed to Lightning. Local validation:
`1213 passed, 81 skipped` plus the ten on-H100 pytest checks.
