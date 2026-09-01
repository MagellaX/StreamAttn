# Exact-Native A100 D128 Register-Resident Pipeline

Date: 2026-09-01

## Result

StreamAttn now promotes one exact contiguous BF16 D128 decode cell on NVIDIA
A100:

```text
B=4, Hq=16, Hkv=2, GQA group=8, D=128, KV=16K
layout: Q [B,Hq,D], K/V [B,Hkv,N,D]
logical chunks: 64
```

The production route computes attention over every KV token with online
softmax. It does not gather, repack, select, or omit KV tokens. Unsupported
shapes continue through the exact fallback.

## Architecture

A direct D64-style double buffer is too large at D128 when four producer warps
also retain their partial outputs. The promoted schedule instead gives each
producer warp one shared K slot and one shared V slot:

1. Stage the first K/V pair with `cp.async`.
2. Load the current K and V tiles into registers.
3. Start asynchronous replacement of both shared slots with the next pair.
4. Execute QK, the exact online-softmax update, and PV from the register pair.
5. Wait only before the next register load.
6. After the final tile, reuse K shared storage for the partial output and merge
   the four exact online-softmax states.

This keeps four producer warps active without the 80 KiB footprint of four
shared K/V buffers. The production plan preallocates its output, BF16 partials,
and FP32 LSE workspace once; the measured cell uses 532,480 workspace bytes.

## Strict paired evidence

Each strict gate alternated benchmark order across 15 paired trials with 100
iterations per timing. FlashInfer 0.6.13 used its planned batched paged exact
decode path. Both outputs were checked against an FP32 dense true-GQA reference.

| Provider/device | Timed path | Wins | Median speedup | Worst trial |
|---|---|---:|---:|---:|
| Modal, A100 SXM4 80GB | extension `_out` | 15/15 | `1.01172x` | `1.00629x` |
| Lightning, A100 SXM4 40GB | extension `_out` | 15/15 | `1.01099x` | `1.00747x` |
| Modal, A100 SXM4 80GB | `ExactDecodePlan.run()` | 15/15 | `1.01001x` | `1.00592x` |
| Lightning, A100 SXM4 40GB | `ExactDecodePlan.run()` | 15/15 | `1.01377x` | `1.01133x` |

StreamAttn's maximum BF16 absolute error was `2.44e-4` in these gates.

## Falsified schedules and boundary

The promotion followed three rejected architecture variants, all measured on
the same strict A100 matrix:

| Schedule | B4/G8/16K median | B4/G8/32K median | Decision |
|---|---:|---:|---|
| Two producer warps | `0.977x` | `0.972x` | Reject: insufficient producer parallelism |
| Four producers, phased single-slot K/V | `0.956x` | `0.956x` | Reject: serialized K/V staging |
| Unified two-stage K/V ring | `1.006x` | `0.975x` | Reject except as an architectural lead; 16K tail crossed below parity |
| Register-resident K/V pair | `1.012x` | `0.975x` | Promote only B4/G8/16K |

The final strict matrix also rejected B4/G4/32K (`0.965x` median) and
B8/G8/32K (`0.949x` median). Those results show that the remaining boundary is
not fixed by another shared-buffer permutation or a blind split-count sweep.
Longer and higher-work cells need a different producer/work decomposition.

## Promotion contract

The native route is selected only when all of these are true:

```text
architecture = SM80 / A100
dtype = BF16
KV layout = contiguous head-major
B = 4
Hq = 16
Hkv = 2
GQA group = 8
D = 128
N = 16384
```

The phase table is discrete. It does not extrapolate to A100 D128 at 32K, G4,
B8, paged KV, ragged rows, FP16, or FP8.

## Reproduction

```text
python -m benchmarks.profile_sm80_d128_phased_gate \
  --batch 4 --q-heads 16 --kv-heads 2 --head-dim 128 \
  --kv-len 16384 --num-chunks 64 --production-plan
```

Evidence artifacts generated during promotion:

```text
artifacts/gate0/sm80_d128_register_pair_strict_gate_modal_20260901.json
artifacts/gate0/sm80_d128_register_pair_strict_gate_lightning_a100_40gb_20260901.json
artifacts/gate0/sm80_d128_register_pair_production_plan_gate_modal_20260901.json
artifacts/gate0/sm80_d128_register_pair_production_plan_gate_lightning_a100_40gb_20260901.json
```

## Next boundary

The next SM80 task is not another split or buffer sweep. It is to restructure
the 32K/high-work producer geometry so memory/MMA overlap scales without losing
CTA waves, then apply the same strict production-plan gate. Paged D128 remains a
separate exact-kernel family and continues to use its external fallback on A100.
