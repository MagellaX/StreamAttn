from types import SimpleNamespace

import pytest
import torch

import stream_attention as attention
from stream_attention import paged_append
from stream_attention.paged import PagedExactDecodePlan, PagedKVCache
from stream_attention.backends.sm90.transposed_gqa_exact_sources import CPP_SOURCE, CUDA_SOURCE


@pytest.fixture(scope='module')
def buffers():
    q = torch.empty(1, 1, 16, 128, dtype=torch.bfloat16)
    k = torch.empty(2048, 16, 2, 128, dtype=q.dtype)
    cache = PagedKVCache(k, torch.empty_like(k), torch.arange(2048, dtype=torch.int32)[None],
                         torch.tensor([65], dtype=torch.int32))
    return q, cache, torch.empty(1, 2, 128, dtype=q.dtype), torch.empty(1, 2, 128, dtype=q.dtype)


def test_requires_explicit_ownership(buffers):
    with pytest.raises(ValueError, match='exclusive_tail_ownership'):
        attention.PagedAppendDecodePlan.build(*buffers)


@pytest.mark.parametrize('splits', [True, 0, 32, 256, 128.0])
def test_rejects_unmeasured_geometry(buffers, splits):
    with pytest.raises(ValueError, match='splits'):
        attention.PagedAppendDecodePlan.build(*buffers, exclusive_tail_ownership=True, splits=splits)


def test_cpu_not_silently_used_as_append_backend(buffers):
    with pytest.raises(ValueError, match='SM90'):
        attention.PagedAppendDecodePlan.build(*buffers, exclusive_tail_ownership=True)


def test_validates_aliases_and_shape_without_cuda(buffers):
    q, cache, k, v = buffers
    paged_append._validate_buffers(q, cache, k, v)
    with pytest.raises(ValueError, match='overlap'):
        paged_append._validate_buffers(q, cache, k, k)
    with pytest.raises(ValueError, match='overlap'):
        paged_append._validate_buffers(q, cache, cache.key[0, 0][None], v)
    with pytest.raises(ValueError, match='overlap'):
        paged_append._validate_buffers(q, cache, k, v, q)
    with pytest.raises(ValueError, match='incoming'):
        paged_append._validate_buffers(q, cache, k.float(), v)
    with pytest.raises(ValueError, match='query'):
        paged_append._validate_buffers(q.reshape(1, 2, 8, 128), cache, k, v)


def test_tail_alias_rejected_but_immutable_prefix_alias_allowed(buffers):
    q, cache, k, v = buffers
    table = cache.page_table.clone()
    table[0, 1] = table[0, 0]
    shared = PagedKVCache(cache.key, cache.value, table, cache.sequence_lengths)
    paged_append._validate_buffers(q, shared, k, v)
    table[0, 0] = table[0, 4]
    with pytest.raises(ValueError, match='tail page aliases'):
        paged_append._validate_buffers(q, shared, k, v)


def test_rejects_inactive_or_out_of_range_tail(buffers):
    q, cache, k, v = buffers
    table = cache.page_table.clone()
    table[0, 4] = -1
    with pytest.raises(ValueError, match='physical pages'):
        paged_append._validate_buffers(q, PagedKVCache(cache.key, cache.value, table,
                                                       cache.sequence_lengths), k, v)


def test_run_reuses_existing_plan_and_never_revalidates(monkeypatch, buffers):
    q, cache, k, v = buffers
    output = torch.empty_like(q)
    workspace = {'partial_o': torch.empty(2, 128, 8, 128), 'partial_lse': torch.empty(2, 128, 8)}
    native = PagedExactDecodePlan(q, cache, output, 128, workspace,
                                  'streamattn_paged_sm90_wgmma_nhd_fragmented_ragged_exact',
                                  query_group=q.reshape(1, 2, 8, 128),
                                  output_group=output.reshape(2, 8, 128))
    seen = []
    monkeypatch.setattr(paged_append, '_is_sm90', lambda q: True)
    monkeypatch.setattr(PagedExactDecodePlan, 'build', lambda *a, **kw: native)
    from stream_attention.backends.sm90 import transposed_gqa_exact
    extension = SimpleNamespace(paged_append_decode_out=lambda *args: seen.append(args))
    monkeypatch.setattr(transposed_gqa_exact, 'compile_transposed_gqa_exact_extension', lambda **kw: extension)
    plan = attention.PagedAppendDecodePlan.build(q, cache, k, v, exclusive_tail_ownership=True)
    assert plan.plan is native and plan.output is output
    assert plan.workspace_bytes == native.workspace_bytes
    monkeypatch.setattr(paged_append, '_validate_buffers', lambda *a: pytest.fail('run() synchronized'))
    assert plan.run() is output and plan.run() is output
    assert seen[0][5] is k and seen[0][6] is v and seen[0][-1] == 128
    assert seen[0][7] is workspace['partial_o']


def test_source_keeps_normalized_merge_and_no_raw_state():
    assert 'paged_append_decode_out' in CPP_SOURCE
    owner = CUDA_SOURCE.split('if constexpr (kAppendOwner) {', 1)[1].split(
        '__shared__ AsyncQKPVSharedStorage storage;', 1)[0]
    assert 'new_tile >= tile_begin && new_tile < tile_end' in owner
    assert '__syncthreads();' in owner
    assert '!kSelectedPaged' in owner
    assert 'kRawPartials' not in CUDA_SOURCE and 'partial_sum' not in owner
    assert 'tOrORowCol(row, col) / storage.row_sum[head]' in CUDA_SOURCE
    signature = CUDA_SOURCE.split('void streamattn_transposed_wgmma_exact_partial_kernel(',1)[1].split(') {',1)[0]
    assert 'int query_positions_per_batch = 1' in signature
    assert 'new_k' not in signature and 'new_v' not in signature
