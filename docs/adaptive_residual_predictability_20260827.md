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
`1.126`. That violates the monotonic safety requirement and prevents direct
runtime promotion.

## Exact-canary follow-up

The Mistral-L0 cell was tested with a nested uncertainty gate. The state
predictor uses `F3`; a cheaper `F1` predictor provides disagreement features.
The risk vector contains only runtime-observable quantities: feature distance,
predicted omitted mass, predicted innovation magnitude, correction magnitude,
and `F1`/`F3` disagreement. Exact errors label offline calibration rows only.

The evaluation is deliberately nested:

1. An outer prompt is held out for the final test.
2. State predictions for calibration rows are generated while holding out each
   inner prompt.
3. Risk predictions for threshold calibration are also prompt-held-out.
4. The final state and risk models are evaluated on the unseen outer prompt.

This avoids fitting either the correction or its confidence score to the test
prompt.

| metric | result |
|---|---:|
| Candidate rows | 64 |
| Candidate regressions | 5 |
| Candidate mean error ratio vs hard drop | 0.6804 |
| Candidate worst error ratio | 1.1827 |
| Runtime-observable risk AUC | 0.9424 |
| Strict-gate accepted rows | 18 / 64 (28.125%) |
| Strict-gate accepted regressions | 0 |
| Strict-gate mean accepted error ratio | 0.7301 |
| Strict-gate worst accepted error ratio | 0.9750 |
| Five-percent-margin coverage | 6 / 64 (9.375%) |
| Five-percent-margin accepted regressions | 0 |

Strict accepted coverage by held-out prompt was uneven:

```text
chat_instruction:  0 / 16
code:             12 / 16
json_tool:         1 / 16
needle_rag:        5 / 16
```

This is a semantic canary pass, not a backend promotion. It proves that the
Mistral-L0 failures are strongly rankable without consulting exact attention
at inference time. It does not yet prove net speed: `F3` consumes persistent
block-moment summaries, two state predictors are evaluated, and the selected
blocks still come from the favorable exact-QK oracle rather than the deployed
support-summary selector.

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
3. The narrow Mistral-L0 canary passes the first semantic gate: it retains
   `28.125%` coverage while rejecting all observed regressions. Advance only
   this cell to deployed-selector and runtime-cost measurement.
4. Promotion requires the same zero-regression result with support-summary
   selection and positive end-to-end latency after summary maintenance,
   dual-predictor evaluation, and gating. Failure at either boundary freezes
   adaptive completion.
5. The main engine work remains native exact attention and independently
   validated reduced-work routes. This branch does not replace exact streaming
   online softmax.

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
artifacts/adaptive/mistral7b_32k_k4_residual_canary_h100.json
```

The artifacts are evidence inputs, not packaged policy files.
