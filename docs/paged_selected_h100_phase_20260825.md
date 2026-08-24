# H100 Selected Paged Decode Phase

## Scope

This phase tests the physical selected-paged executor, not policy safety:

```text
GPU: NVIDIA H100 / SM90
KV: NHD, page 16, randomized physical page order
shape: BF16, D128, Hq16, Hkv2, G8
capacity: 32K
baseline: fastest tested FlashInfer 0.6.17 exact backend (FA2 resolved)
```

StreamAttn consumes precomputed `PackedRoute64` metadata. K/V stays in the
original paged cache. Correctness is exact relative to an independent FP32
reference over the selected token set.

## Main Phase

| Batch | Selected tokens | StreamAttn ms | FlashInfer ms | Speedup | Min paired | Wins |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 384 | 0.01674 | 0.04541 | 2.713x | 2.235x | 7/7 |
| 1 | 2,048 | 0.01560 | 0.04166 | 2.671x | 2.432x | 7/7 |
| 1 | 8,192 | 0.02144 | 0.04426 | 2.064x | 2.035x | 7/7 |
| 1 | 16,384 | 0.02643 | 0.04478 | 1.694x | 1.663x | 7/7 |
| 1 | 32,768 | 0.04318 | 0.04426 | 1.025x | 0.951x | 5/7 |
| 4 | 384 | 0.01618 | 0.08317 | 5.141x | 4.735x | 7/7 |
| 4 | 2,048 | 0.01725 | 0.08405 | 4.873x | 4.944x | 7/7 |
| 4 | 8,192 | 0.02939 | 0.08346 | 2.839x | 2.807x | 7/7 |
| 4 | 16,384 | 0.06218 | 0.08474 | 1.363x | 1.341x | 7/7 |
| 4 | 32,768 | 0.10882 | 0.08488 | 0.780x | 0.780x | 0/7 |
| 8 | 384 | 0.01643 | 0.12851 | 7.821x | 7.945x | 7/7 |
| 8 | 2,048 | 0.01784 | 0.12816 | 7.184x | 6.569x | 7/7 |
| 8 | 8,192 | 0.05523 | 0.12832 | 2.323x | 2.325x | 7/7 |
| 8 | 16,384 | 0.09224 | 0.12586 | 1.364x | 1.359x | 7/7 |
| 8 | 32,768 | 0.17128 | 0.12774 | 0.746x | 0.758x | 0/7 |

All 15 cells passed the selected-reference gate. The full-route control also
matched FlashInfer within `2.45e-4`, isolating scheduling cost from semantics.

## Independent Confirmation

Fresh-container confirmation increased the paired count for the practical
384/2048-token region:

| Batch | Selected tokens | Speedup | Min paired | Wins |
|---:|---:|---:|---:|---:|
| 1 | 384 | 2.670x | 2.678x | 15/15 |
| 1 | 2,048 | 2.764x | 2.656x | 15/15 |
| 4 | 384 | 5.161x | 5.226x | 15/15 |
| 4 | 2,048 | 5.077x | 4.814x | 15/15 |
| 8 | 384 | 8.254x | 8.065x | 15/15 |
| 8 | 2,048 | 7.184x | 7.082x | 15/15 |

## Q-Head-Private Control

An alternating Q-head route deliberately reduced GQA union efficiency to
`E_group=0.545`. Per-atom head masks remained correct, and the 384-token route
still measured `2.441x`, `5.205x`, and `8.201x` at B1/B4/B8 with `27/27`
paired wins. This validates the ABI's head ownership, but low `E_group` remains
a planner signal because other selector geometries can inflate union work more
severely.

## Decision

Promote the static H100 executor for precomputed selected schedules only when
the semantic policy is independently verified and the measured route cell has
at least 10% margin. Keep near-full routes on the existing exact split
scheduler.

Do not place the current Python/Torch route lowering in a per-token dynamic
loop. Warm lowering is millisecond-scale while the kernel is microsecond-scale.
The next systems target is a device-side route-preparation kernel, followed by
compact scheduling for nonuniform row counts.

## Reproduction

```bash
modal run benchmarks/modal_paged_selected_decode.py \
  --batches 1,4,8 \
  --selected-tokens 384,2048,8192,16384,32768 \
  --kv-len 32768 \
  --q-heads 16 --kv-heads 2 --head-dim 128 \
  --layout NHD --route-mode all_heads
```

Local raw artifacts:

```text
artifacts/paged_selected_h100_d128_32k_phase_v1.json
artifacts/paged_selected_h100_d128_32k_confirm_v1.json
artifacts/paged_selected_h100_d128_32k_head_private_v1.json
```
