# Adaptive Output-Sufficiency Frontier

Date: 2026-08-27

## Question

StreamAttn's adaptive branch asks whether attention can avoid most exact KV work
while preserving the attention module output. The experiment in this note tests
that question before lowering another candidate into CUDA/Triton.

The reference path represents every KV block by an online-softmax state:

```text
state_b = (log Z_b, o_b)
Z_b     = sum_j exp(q k_j / sqrt(D))
o_b     = sum_j softmax_b(j) v_j
```

Exact block states merge through the same stable max/rescale arithmetic used by
streaming online softmax. The adaptive candidates compute selected blocks
exactly and attempt to account for the omitted tail through either block
moments or a sampled control-variate correction.

This is a semantic frontier, not a runtime promotion. It measures output error
on real model captures and does not claim lower latency.

## Methods

The H100 runs compare these methods at matched exact middle-block budgets:

- `qk_hard_drop`: select blocks by maximum QK score and drop the rest.
- `mass_hard_drop`: oracle selection by exact block partition mass.
- `post_wo_greedy_hard_drop`: oracle physical GQA route chosen greedily after
  the model output projection.
- `qk_exact_peaks_plus_moment_tail`: exact selected blocks plus a diagonal
  joint-Gaussian K/V moment estimate for every omitted block.
- `..._ht_mass_priority`: sampled Horvitz-Thompson correction using estimated
  block mass as the inclusion priority.
- `..._ht_oracle_residual_priority`: the same correction using the exact
  post-projection residual magnitude. This is deliberately undeployable and is
  a variance lower-bound for practical priority functions.

All physical selections are shared by the Q heads belonging to the same KV
head. The reported error is relative L2 after `o_proj`, because that is closer
to the model-visible contract than head-local attention error.

## H100 Evidence

### Qwen2.5-3B-Instruct, 32K

Prompt bucket: `chat_instruction`. Block size: 64. The table reports one real
capture with four query rows. `K` is the number of dynamic exact middle blocks
per KV group; sink and recent anchors remain present.

| Layer | K | QK hard drop | Post-WO greedy oracle | Moment-complete tail |
|---:|---:|---:|---:|---:|
| 14 | 4 | 0.1580 | 0.1538 | 0.1924 |
| 14 | 8 | 0.1492 | 0.1420 | 0.1698 |
| 26 | 4 | 0.2486 | 0.2154 | 0.2429 |
| 26 | 8 | 0.2189 | 0.1956 | 0.2192 |

Output-aware physical selection improves the hard-drop oracle, most clearly at
L26 K4, but the remaining error is still large. The diagonal moment tail is not
a reliable completion model for these Qwen captures.

Sampled correction is also not promotion-ready. At L26 K4, even the exact
residual-magnitude priority produced non-monotonic results and invalid
denominators on some head rows:

| Expected tail samples/KV group | Mean post-WO relative L2 | Worst repeat |
|---:|---:|---:|
| 4 | 0.5812 | 1.6464 |
| 8 | 0.1934 | 0.2069 |
| 16 | 0.2107 | 0.3999 |

This falsifies the narrow hypothesis that a better block priority alone is
enough to make ratio-form stochastic tail correction stable.

### Mistral-7B-Instruct-v0.3, 32K

Prompt bucket: `chat_instruction`. Block size: 64. The table reports one real
capture with two query rows.

| Layer | K | QK hard drop | Post-WO greedy oracle | Moment-complete tail |
|---:|---:|---:|---:|---:|
| 0 | 4 | 0.3574 | 0.3207 | 0.4277 |
| 0 | 8 | 0.3171 | 0.2746 | 0.4091 |
| 24 | 4 | 0.2626 | 0.2551 | 0.2146 |
| 24 | 8 | 0.2409 | 0.2324 | 0.1996 |

This cross-family capture establishes a useful boundary: moment completion can
help a later Mistral layer, but it fails badly at L0. The estimator is therefore
layer-dependent and cannot be promoted as a universal tail model.

## Residual Capacity Probes

Companion L26 Qwen probes tested whether a compact residual representation can
recover what hard-drop attention omits.

- A 128-feature positive linear summary improved held-out post-`o_proj` error
  from 0.3666 to 0.3643, only 0.64%.
- A learned static 32-token residual bank reduced held-out error to 0.2864,
  a 21.89% improvement, but its training error was 0.0380. The large
  train/held-out gap shows severe query-conditioned overfit.
- A query-specific one-token residual oracle reconstructed the output to about
  3.4e-7 relative error. Capacity exists, but the residual must be predicted
  from the live query rather than stored as a shared fixed bank.

## Decision

The current evidence rejects four simplistic promotion paths:

1. More exact blocks selected by QK score alone are not sufficient.
2. Diagonal Gaussian moment completion is not universal across layers.
3. Positive feature summaries do not recover the omitted Qwen L26 residual.
4. Horvitz-Thompson ratio correction remains high-variance even with oracle
   residual priority.

The positive signal is narrower and more useful: the omitted post-`o_proj`
residual has strong query-specific low-cardinality capacity. The next research
gate is therefore residual predictability, not another CUDA kernel.

## Next Experiment

Build an exact-canary residual predictability probe:

1. Capture `full_attention - adaptive_attention` after `o_proj` over multiple
   prompts, query rows, layers, and both Qwen and Mistral.
2. Compute singular-value curves for per-layer and cross-layer residuals.
3. Fit coefficients from cheap live features such as Q projections, selected
   block scores, block moments, and route metadata.
4. Separate same-prompt, later-row and cross-prompt held-out evaluation.
5. Promote only if the predicted correction materially reduces held-out error
   without scanning the omitted KV tail.

Interpretation:

- Low-rank and predictable: build a canary-calibrated adaptive mode.
- Low-rank but not predictable: exact canaries can only calibrate tightly
  coupled request cohorts; keep unrelated rows exact.
- High-rank or unstable: stop the training-free adaptive branch and retain
  exact-native or train a model/runtime jointly.

This research branch does not replace StreamAttn's main exact engine work.
Exact decode, prefill, paged KV, architecture coverage, and online-softmax
kernels remain independent engine milestones.

## Artifacts

- `artifacts/adaptive/qwen25_3b_32k_chat_l14_l26_output_sufficiency_frontier_h100.json`
- `artifacts/adaptive/qwen25_3b_32k_chat_l26_oracle_tail_priority_h100.json`
- `artifacts/adaptive/qwen25_3b_32k_chat_l26_positive_feature_residual_h100.json`
- `artifacts/adaptive/qwen25_3b_32k_chat_l26_shared_residual_capacity_h100.json`
- `artifacts/adaptive/mistral7b_32k_chat_l0_l24_output_sufficiency_frontier_h100.json`
