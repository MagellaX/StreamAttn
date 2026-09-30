# Adaptive Head Sharing Versus Omission

## What This Experiment Answers

The adaptive objective is unchanged: skip physical K/QK or V/PV work when a
cumulative contribution bound permits it, and spend less on the decision than
the work avoided. Exact attention is the control, not the research objective.

The preceding real-activation diagnostic found little removable work when
eight query heads shared a physical region. This experiment asks whether
smaller head groups recover enough omissions to justify independent KV reads.
It changes neither the `1e-3` omission budget nor the admission rule.

## Scope and Controls

- Saved Qwen2.5-3B 2K captures, including a separate last-query-only CPU replay.
- Fresh Lightning H100 captures: Qwen2.5-3B at 8K and TinyLlama-1.1B at 2K.
- Two prompts and five layers per model; BF16 Q/K/V, G8, dense upstream.
  Qwen uses Hq16/Hkv2/D128; TinyLlama uses Hq32/Hkv4/D64.
- Last 16 prefill queries with explicit append-position causal visibility.
- Head groups 1/2/4/8 and physical query tiles 1/16.
- Summary-only, two-gate, and exact-block-mass oracle admission; origin and
  mean-centered global value envelopes.
- FP64 offline replay checks actual output error against the cumulative bound.

These are not generation-quality gates, adaptive-conditioned captures, timed
subgroup kernels, or representative samples of every request. Exact block mass
is a comparator under the same order and max gate, not an optimal subset solver.
Mean-centered envelopes are offline diagnostics, not a runtime feature.

## The Read-Cost Constraint

Let `G` be the original query-head group, `g` the candidate subgroup, and
`r = G/g` the replication factor. For equal-sized K/V elements and full blocks,
let `a` be the fraction skipped before K and `b` the additional fraction skipped
after QK. With independent subgroup reads:

```text
requested_KV_ratio = r * (1 - a - b/2)
```

The denominator is full-GQA execution at the **same query tile size**. The
profiler counts actual tail-block lengths and ignores invisible/padded rows.

If `a = 0`, the ratio is at least `r/2`. Thus halving the head group cannot
reduce total requested KV bytes with V-only skipping, even if all V loads
could be avoided. Smaller groups face an even higher floor. This is why a
higher omission percentage alone is not a sufficient implementation signal.

This is a logical read-cost model, **not measured HBM traffic or a speed limit**.
Cache reuse, shared K staging, MMA utilization and execution scheduling can
change wall-clock performance. Summary reads, decision cost, and merge cost are
also excluded, so a ratio below one is not sufficient for a speedup.

## Measured Results

The following uses the origin envelope and the two-gate admission rule. Each
number is aggregated over ten captures; percentages are the largest saved-PV
fraction in any capture, not an average model speedup.

| Capture set | Physical region | Maximum PV regions avoided | Mean requested-KV ratio |
|---|---|---:|---:|
| Qwen 2K | 1 head, 16 queries | 3.711% | 7.973x |
| Qwen 8K | 1 head, 16 queries | 4.956% | 7.965x |
| TinyLlama 2K | 1 head, 16 queries | 2.295% | 7.979x |
| Qwen 8K | G8, 1 query | 0% | 1.000x |
| TinyLlama 2K | G8, 1 query | 0.0977% | 0.999927x |
| Both fresh sets | G8, 16 queries | 0% | 1.000x |

No pre-K region was omitted on any set. No smaller-head-group configuration
reduced requested KV bytes below full-GQA execution, including centered
envelopes and the exact-mass comparator. TinyLlama does admit a tiny amount of
G8 post-QK work avoidance at query-tile size one: two of ten captures for the
two-gate rule. That is not enough evidence to pay for a subgroup kernel.

Exact block mass improves TinyLlama's best G8/tile-1 logical read ratio to
`0.992920x` with the origin envelope (`0.992676x` centered), still before
decision costs. Qwen's corresponding origin-envelope minimum is `0.999512x`.
This is limited opportunity under the current contract, not impossibility of
adaptive attention.

A separate last-query-only replay of both new archives also finds zero G8
two-gate omissions and no subgroup read advantage. The exact-mass comparator
does admit some G8 omissions (two Qwen captures, four TinyLlama captures), but
that information is not available to the pre-K gate for free. A final prefill
query is not an autoregressive generation rollout.

The separate Lightning H100 mechanism canary completed all 12 cases and passed
47 focused tests. Physical skipping is real, but the complete adaptive
prototype still loses to Flash SDPA. The known-support compact floor remains
an oracle diagnostic, not a deployable selector.

## What We Do Next

Do not build a subgroup kernel from these omission counts alone. First test a
tighter **value-contribution bound** on the saved tensors at the same budget.
Separate looseness in the pre-K score bound from looseness in the global
value envelope. Record which physical groups become admissible and charge the
summary/decision work needed to obtain the tighter information.

The alternate systems hypothesis is shared K staging with independent V/PV
decisions, preserving K reuse across head subgroups. It changes the read-cost
assumption above, but must demonstrate enough saved V/PV work to cover extra
coordination. Neither direction is promoted by this experiment. Do not weaken
the budget, pivot to fixed seeds, or resume unrelated exact-only tuning.

## Evidence and Reproduction

- [Saved-capture sweep](../artifacts/gate0/adaptive_group_frontier_cpu_20260930.json)
- [Last-query replay](../artifacts/gate0/adaptive_group_frontier_last_query_cpu_20260930.json)
- [Qwen 8K initial diagnostic](../artifacts/gate0/adaptive_group_frontier_qwen8k_lightning_h100_20260930.json)
- [TinyLlama initial diagnostic](../artifacts/gate0/adaptive_group_frontier_tinyllama2k_lightning_h100_20260930.json)
- [Qwen completed replay](../artifacts/gate0/adaptive_group_frontier_qwen8k_lightning_h100_replay_20260930.json)
- [TinyLlama completed replay](../artifacts/gate0/adaptive_group_frontier_tinyllama2k_lightning_h100_replay_20260930.json)
- [Qwen 8K last-query replay](../artifacts/gate0/adaptive_group_frontier_qwen8k_last_query_cpu_20260930.json)
- [TinyLlama last-query replay](../artifacts/gate0/adaptive_group_frontier_tinyllama2k_last_query_cpu_20260930.json)
- [H100 physical canary](../artifacts/gate0/adaptive_physical_commit_lightning_h100_20260930.json)
- [Canary summary](../artifacts/gate0/adaptive_physical_commit_summary_lightning_h100_20260930.json)

The initial model calculations completed, but their subsequent capture upload
failed. Those artifacts retain `platform_state=failed`; they must not be
described as successful end-to-end jobs. The June Lightning SDK used an obsolete
upload endpoint. The runner now requires `lightning-sdk>=2026.9.18.post1`, uses
short-lived session authentication, verifies downloaded capture hashes, and
waits for terminal job state. It does not embed credentials in source or logs.

Both replacement single-H100 jobs completed on Lightning. Their per-capture
Q/K/V hashes and complete summary tables exactly match the initial attempts.
The 85.2 MB Qwen archive and 21.8 MB TinyLlama archive were downloaded and
SHA-256 checked before job deletion. Both are retained locally and in the
teamspace for follow-on offline research. All five GPU jobs from this session
were deleted; the teamspace job list was empty at closeout. The initial canary's
`platform_state=running` is a pre-cleanup snapshot, not a final billing record.

Verification: `1228 passed, 81 skipped` locally, 47 focused checks in the H100
mechanism run, and 14 diagnostic tests in each successful model job. No native
kernel or production dispatch policy was changed in this experiment.

```bash
python benchmarks/profile_adaptive_group_frontier.py \
  --captures /path/to/saved.captures.pt \
  --head-groups 1 2 4 8 --query-tiles 1 16 \
  --output-json /path/to/new-frontier.json
```

The Lightning runner uses `--experiment group_frontier`, with an explicit
`--model`, pinned `--revision`, `--max-seq`, and `--layers`. Jobs are bounded,
single-attempt, and deleted after evidence retrieval, including failure paths.
Tensor archives remain outside git; JSON retains input and source hashes.
