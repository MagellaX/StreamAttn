# H100 Dynamic Selected-Paged Route Compiler

## Result

StreamAttn can now consume mutable Q-head CSR atom IDs produced on the GPU and
execute selected paged decode without a CPU synchronization, K/V gather, or
physical-page repack.

The measured path is:

```text
GPU Q-head CSR atom IDs
  -> bounded on-chip membership map
  -> warp-parallel sorted union and page resolution
  -> row-local selected WGMMA producers
  -> exact online-softmax state merge
  -> [B,Hq,D] output
```

The selector that generates the CSR IDs is outside this measurement. Route
lowering, page-table resolution, attention, and merge are included.

## Why The Compiler Is Different

At 32K with 64-token atoms, the route universe contains only 512 possible
atoms. A generic sort or a serial G-way merge ignores that bound. The compiler
instead constructs a shared-memory membership function:

```text
M[atom] = bitwise OR of the local Q-head owners
```

Source CSR entries are inserted in parallel. Four warps then scan 128 logical
atoms per round and use ballots plus a block prefix to emit a deterministic,
sorted GQA-group union. The same pass resolves each 64-token atom to four live
page-16 physical IDs and emits head/token masks.

For `A=N/64` logical atoms and `R` source route entries per GQA group, the
parallel preparation work is approximately:

```text
T_prepare ~ launch + A / threads + R / threads + union / threads
```

The previous cursor merge scaled serially with route count and overlap. The
bounded compiler makes preparation nearly overlap-independent for this phase.

## Scope

```text
GPU: NVIDIA H100 / SM90
KV: NHD, page 16, randomized physical page order
shape: BF16, D128, Hq16, Hkv2, G8
capacity: 32K
selected tokens/head: 384 and 2,048
route overlap: shared, alternating, fully disjoint
baseline: fastest tested FlashInfer 0.6.17 exact backend (FA2 resolved)
```

`shared` has GQA union efficiency `E=1`. `alternating` measures about
`E=0.51-0.55`. `disjoint` is the adversarial `E=1/G=0.125` case.

## H100 Evidence

Per-call CUDA-event timing, including GPU preparation and selected execution:

| B | Tokens/head | Shared | Alternating | Disjoint |
|---:|---:|---:|---:|---:|
| 1 | 384 | `1.827x` | `1.816x` | `1.928x` |
| 4 | 384 | `3.608x` | `3.570x` | `3.500x` |
| 8 | 384 | `5.276x` | `5.261x` | `4.849x` |
| 1 | 2,048 | `1.719x` | `1.541x` | `1.264x` |
| 4 | 2,048 | `3.188x` | `2.794x` | `1.112x` |
| 8 | 2,048 | `4.638x` | `3.391x` | `1.238x` |

All 18 cells matched an independent FP32 selected-token reference. All 18
won every alternating-order paired trial; the minimum paired speedup was
`1.118x`.

A second timing method placed 400 calls inside one CUDA-event interval and
divided by call count. It won 15/18 cells. The losing corners were S2048 at
B1 alternating/disjoint and B4 disjoint. Every S384 cell won under both timing
methods, including the fully disjoint B1 route.

GPU preparation measured `0.01098-0.01362 ms` per isolated call and
`0.00508-0.00782 ms` amortized. The former Torch route lowering measured
`1.532-3.003 ms`, a `121.6x-253.0x` reduction in the measured preparation
path.

## Conservative Dispatch Boundary

The engine should require both timing regimes for automatic promotion:

```text
384 tokens/head:
  dynamic selected is eligible at B1/B4/B8 for all measured overlap patterns

2,048 tokens/head:
  B8 is eligible for all measured overlap patterns
  B1/B4 require an overlap guarantee or exact fallback
```

This rule needs no route-count readback. The fixed CSR row lengths provide a
worst-case union bound; a selector may optionally provide a stronger overlap
contract.

## Semantics

The kernel computes exact online-softmax attention over each Q head's selected
token set, within BF16 numerical tolerance. FlashInfer computes exact
attention over the full 32K KV. Therefore these speedups establish a dynamic
selected-route systems result, not universal full-context semantic
equivalence. Model-output preservation remains the policy compiler and
verifier's responsibility.

## Reproduction

```bash
python -m modal run benchmarks/modal_paged_dynamic_selected_decode.py \
  --batches 1,4,8 \
  --selected-tokens 384,2048 \
  --route-modes shared,alternating,disjoint \
  --kv-len 32768 \
  --q-heads 16 \
  --kv-heads 2 \
  --head-dim 128 \
  --layout NHD \
  --amortized-repeats 400
```

Raw local artifact:

```text
artifacts/paged_dynamic_h100_d128_32k_bitset_phase_v1.json
```

## Next

Connect a real GPU selector directly to the mutable CSR buffer and measure
selector plus preparation plus attention. Keep fixed/static routes on the
faster static executor. For high-variance large unions, add a compact work
queue only if end-to-end measurements beat the current fixed-capacity grid.
