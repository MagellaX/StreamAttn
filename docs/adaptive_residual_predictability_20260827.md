# Adaptive Residual Predictability Gate

## Question

The adaptive frontier established that exact attention over a selected set of
KV blocks can leave a material output error. This gate asks whether the omitted
online-softmax state is sufficiently predictable from information that a
runtime already has, without reading the omitted KV tail.

This is a semantic upper-bound experiment. It is not a promoted backend and it
does not claim a runtime speedup. Middle blocks are selected with exact QK
block maxima, which is deliberately more favorable than the deployed support
summary selector.

## Exact target

Let `A` be the selected tokens and `U` the omitted tokens. Define their
partition masses and normalized value outputs as `Z_A`, `Z_U`, `o_A`, and
`o_U`. Exact attention can be written as:

```text
rho   = log(Z_U / Z_A)
Delta = o_U - o_A
alpha = sigmoid(rho)

o_exact = o_A + alpha * Delta
```

The predictor therefore estimates `rho` and `Delta`. Predicting `o_U`
directly is the wrong gate: a stable mean in `o_U` can look predictable while
the correction `o_U - o_A` remains prompt-specific. The implementation checks
the factorization against the exact merged online-softmax state.

Four progressively richer feature sets were evaluated:

- `F0`: query hash, query norms, and position.
- `F1`: `F0` plus selected attention state and selected QK metadata.
- `F2`: `F1` plus omitted block moments. This assumes persistent summaries.
- `F3`: `F2` plus current, previous, and delta temporal features.

The real gate is held-out ridge prediction. Train-set CCA is descriptive only,
because it saturates in this small-sample, high-dimensional setting.

## H100 experiment

Both models used FP16, 32K context, four exact QK-selected middle blocks per KV
group, and one prompt from each of `code`, `chat_instruction`, `needle_rag`,
and `json_tool`. Each layer contributed 64 rows: 16 decode queries from each
of four prompts. Results include future-token and leave-one-prompt-out splits.

### Qwen2.5-3B-Instruct

| layer | hard-drop post-WO error | rank-16 unseen representable energy | best unseen state reduction | minimum fold | worst p95 row ratio | decision |
|---:|---:|---:|---:|---:|---:|---|
| 14 | 0.1797 | 0.2847 | +12.44% (`F3`) | -12.50% | 1.436 | stop global predictor |
| 26 | 0.2807 | 0.2270 | +8.03% (`F3`) | -2.43% | 1.206 | stop global predictor |
| 27 | 0.2202 | 0.1798 | -0.49% (`F3`) | -2.97% | 1.254 | stop global predictor |

Future-token reductions were `33.92%`, `21.54%`, and `5.75%` for layers 14,
26, and 27. Their collapse under unseen-prompt evaluation is the important
result: local temporal smoothness is not cross-prompt sufficiency.

### Mistral-7B-Instruct-v0.3

| layer | hard-drop post-WO error | rank-16 unseen representable energy | `F3` unseen state reduction | minimum fold | worst p95 row ratio | decision |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.3268 | 0.3618 | +30.23% | +13.44% | 1.126 | canary only; not promotable |
| 24 | 0.2691 | 0.1293 | +0.89% | -13.95% | 1.396 | stop global predictor |

Mistral layer 0 is a real model-family-specific signal. `F3` reduced the mean
unseen-prompt hard-drop error by `30.23%`, and every fold improved in aggregate.
It still regressed on tail rows: the worst fold's p95 row error ratio was
`1.126`. That violates the monotonic safety requirement and prevents runtime
promotion. It justifies one bounded exact-canary study with an uncertainty
gate, not a general adaptive mode.

## The mathematical boundary

Omitted normalization mass is not the main obstacle. For the best feature in
each cell, unseen-prompt `rho` predictable-energy fractions were approximately:

```text
Qwen L14 / L26 / L27: 0.967 / 0.921 / 0.954
Mistral L0 / L24:     0.971 / 0.937  (F3)
```

The value innovation is harder and less stable. `F3` unseen-prompt predictable
energy was:

```text
Qwen L14 / L26 / L27: 0.494 / 0.432 / 0.317
Mistral L0 / L24:     0.257 / 0.275
```

High mass predictability does not translate directly into output correction,
because the merged error is controlled by the vector-valued innovation and by
tail outliers after `o_proj`. Rank-16 held-out residual energy never exceeded
`0.362` in these cells. A universal low-rank, training-free omitted-state
predictor is therefore not supported by this evidence.

## Decision

1. Freeze the global training-free residual predictor branch. Qwen fails the
   cross-prompt gate and Mistral L24 is effectively neutral.
2. Keep exact online-softmax attention as the engine invariant. No heuristic
   completion mode is promoted.
3. Permit one narrow Mistral-L0 exact-canary follow-up: test whether a cheap
   uncertainty score can reject every regressing row while retaining useful
   coverage. Stop if the canary requires omitted-tail reads or exact labels on
   normal decode steps.
4. Do not spend CUDA effort on adaptive completion before that semantic gate.
   The main engine work remains native exact attention and independently
   validated reduced-work routes.

## Reproduction

The reference benchmark and H100 launcher are:

```text
benchmarks/profile_adaptive_residual_predictability.py
benchmarks/modal_adaptive_residual_predictability.py
```

Raw local artifacts:

```text
artifacts/adaptive/qwen25_3b_32k_k4_residual_predictability_h100.json
artifacts/adaptive/mistral7b_32k_k4_residual_predictability_h100.json
```

The artifacts are evidence inputs, not packaged policy files.
