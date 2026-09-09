"""Oracle-only execution floor with precomputed shared block IDs, no discovery.

Negative IDs retain a traversal slot without attention work. Compact positive
IDs remove those slots. Neither control is valid after arbitrary Q mutation.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _known_support(Q, K, V, Ids, Out, M: tl.constexpr, N: tl.constexpr,
                       HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                       SLOTS: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
        qb, b, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        kh = h // (HQ // HKV)
        rows = qb * TM + tl.arange(0, TM)
        ds = tl.arange(0, D)
        cs = tl.arange(0, TN)
        q = tl.load(Q + ((b * M + rows[:, None]) * HQ + h) * D + ds[None, :],
                    mask=(rows < M)[:, None], other=0.0)
        maximum = tl.full([TM], -float("inf"), tl.float32)
        den = tl.zeros([TM], tl.float32)
        num = tl.zeros([TM, D], tl.float32)
        for slot in range(SLOTS):
            block = tl.load(Ids + slot)
            if block >= 0:
                pos = block * TN + cs
                kt = tl.load(K + ((b * N + pos[:, None]) * HKV + kh) * D + ds[None, :],
                             mask=(pos < N)[:, None], other=0.0)
                scores = tl.dot(q, tl.trans(kt), out_dtype=tl.float32) * (D ** -0.5)
                scores = tl.where((pos < N)[None, :], scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                correction = tl.exp(maximum - new_max)
                p = tl.exp(scores - new_max[:, None])
                vt = tl.load(V + ((b * N + pos[:, None]) * HKV + kh) * D + ds[None, :],
                             mask=(pos < N)[:, None], other=0.0)
                num = num * correction[:, None] + tl.dot(p.to(vt.dtype), vt, out_dtype=tl.float32)
                den = den * correction + tl.sum(p, 1)
                maximum = new_max
        tl.store(Out + ((b * M + rows[:, None]) * HQ + h) * D + ds[None, :],
                 num / tl.maximum(den, 1e-30)[:, None], mask=(rows < M)[:, None])


def execute_known_support(q, k, v, ids, out, *, block_size=32, tile_size_q=16):
    """Benchmark-internal: caller validates nonempty unique legal shared IDs."""
    if triton is None:
        raise RuntimeError("Triton required")
    b, m, hq, d = q.shape
    _known_support[(triton.cdiv(m, tile_size_q), b, hq)](
        q, k, v, ids, out, M=m, N=k.shape[1], HQ=hq, HKV=k.shape[2], D=d,
        SLOTS=ids.numel(), TM=tile_size_q, TN=block_size, num_warps=4, num_stages=1)
    return out
