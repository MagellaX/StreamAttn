import torch

from benchmarks.profile_tk_tensor_core_exact_decode import (
    _pack_active_chunks_by_kv_group,
    _pack_kv_head_major,
    _pack_q_by_kv_group,
    _pack_row_modes_by_kv_group,
    _reference_from_packed,
    _unpack_q_by_kv_group,
)


def test_pack_and_unpack_q_by_kv_group() -> None:
    q = torch.arange(1 * 14 * 4, dtype=torch.float16).reshape(1, 14, 4)
    packed = _pack_q_by_kv_group(q, kv_heads=2, padded_rows=16)

    assert packed.shape == (1, 2, 16, 4)
    assert torch.equal(packed[:, 0, :7, :], q[:, :7, :])
    assert torch.equal(packed[:, 1, :7, :], q[:, 7:, :])
    assert torch.count_nonzero(packed[:, :, 7:, :]) == 0
    assert torch.equal(_unpack_q_by_kv_group(packed, q_heads=14), q)


def test_pack_kv_head_major_transposes_nhd() -> None:
    kv = torch.arange(1 * 3 * 2 * 4, dtype=torch.float16).reshape(1, 3, 2, 4)
    packed = _pack_kv_head_major(kv)

    assert packed.shape == (1, 2, 3, 4)
    assert torch.equal(packed[:, 0, :, :], kv[:, :, 0, :])
    assert torch.equal(packed[:, 1, :, :], kv[:, :, 1, :])


def test_pack_row_modes_by_kv_group_marks_seed_exact_and_padding() -> None:
    modes = _pack_row_modes_by_kv_group(
        q_heads=14,
        kv_heads=2,
        seed_heads=[2, 3, 4, 6, 7],
        padded_rows=16,
        device=torch.device("cpu"),
    )

    assert modes.shape == (2, 16)
    assert modes[0, :7].tolist() == [0, 0, 1, 1, 1, 0, 1]
    assert modes[1, :7].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert modes[:, 7:].eq(2).all()


def test_pack_active_chunks_uses_seed_only_kv_groups() -> None:
    active_chunks, active_counts, flat_active_chunks, active_offsets, active_by_kv = _pack_active_chunks_by_kv_group(
        q_heads=14,
        kv_heads=2,
        seed_heads=[0, 1, 2, 3, 4, 5, 6],
        kv_len=32768,
        num_chunks=256,
        block_size=32,
        sink_blocks=2,
        recent_blocks=2,
        middle_seed_blocks=2,
        block_order="recent_first",
        device=torch.device("cpu"),
    )

    assert active_counts.tolist() == [2, 256]
    assert active_by_kv[0] == [0, 255]
    assert active_by_kv[1][0] == 0
    assert active_by_kv[1][-1] == 255
    assert active_chunks.shape == (2, 256)
    assert flat_active_chunks.numel() == 258
    assert active_offsets.tolist() == [0, 2, 258]


def test_packed_reference_matches_manual_attention_for_actual_rows() -> None:
    torch.manual_seed(0)
    q = torch.randn((1, 14, 8), dtype=torch.float16)
    k = torch.randn((1, 5, 2, 8), dtype=torch.float16)
    v = torch.randn((1, 5, 2, 8), dtype=torch.float16)
    q_group = _pack_q_by_kv_group(q, kv_heads=2, padded_rows=16)
    k_group = _pack_kv_head_major(k)
    v_group = _pack_kv_head_major(v)

    ref_group = _reference_from_packed(q_group, k_group, v_group)
    ref = _unpack_q_by_kv_group(ref_group, q_heads=14)

    manual_heads = []
    scale = q.shape[-1] ** -0.5
    for head in range(14):
        kv_head = head // 7
        scores = torch.matmul(q[:, head : head + 1, :].float(), k[:, :, kv_head, :].float().transpose(-1, -2))
        probs = torch.softmax(scores * scale, dim=-1)
        manual_heads.append(torch.matmul(probs, v[:, :, kv_head, :].float()).to(q.dtype))
    manual = torch.cat(manual_heads, dim=1)

    assert torch.allclose(ref, manual, atol=1e-3, rtol=1e-3)
