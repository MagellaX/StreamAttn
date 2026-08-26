"""GPU query-aware atom selection for paged decode.

The selector consumes compact per-atom support keys and writes fixed-width
Q-head route rows directly into an existing ``AttentionRouteCSR.atom_ids``
buffer. Output order is score order; the downstream bounded-membership route
compiler canonicalizes the union, so the selector does not need a sort kernel.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _paged_support_score_kernel(
        Query,
        SupportKeys,
        SequenceLengths,
        Scores,
        H_Q: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_ATOMS: tl.constexpr,
        SUPPORT_WIDTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ATOMS_PER_PROGRAM: tl.constexpr,
    ):
        row = tl.program_id(0)
        atom_group = tl.program_id(1)
        batch = row // H_Q
        q_head = row - batch * H_Q
        kv_head = q_head // GROUP_SIZE

        atoms = atom_group * ATOMS_PER_PROGRAM + tl.arange(0, ATOMS_PER_PROGRAM)
        support = tl.arange(0, SUPPORT_WIDTH)
        dims = tl.arange(0, HEAD_DIM)
        valid_atoms = (tl.load(SequenceLengths + batch) + 63) // 64
        atom_mask = (atoms < NUM_ATOMS) & (atoms < valid_atoms)

        q = tl.load(
            Query + batch * H_Q * HEAD_DIM + q_head * HEAD_DIM + dims
        ).to(tl.float32)
        support_offsets = (
            (((batch * H_KV + kv_head) * NUM_ATOMS + atoms[:, None, None])
             * SUPPORT_WIDTH + support[None, :, None])
            * HEAD_DIM
            + dims[None, None, :]
        )
        keys = tl.load(
            SupportKeys + support_offsets,
            mask=atom_mask[:, None, None],
            other=0.0,
        ).to(tl.float32)
        dots = tl.sum(keys * q[None, None, :], axis=2)
        score = tl.max(dots, axis=1)
        score = tl.where(atom_mask, score, -float("inf"))
        tl.store(Scores + row * NUM_ATOMS + atoms, score, mask=atoms < NUM_ATOMS)

    @triton.jit
    def _paged_support_topk_kernel(
        Scores,
        SequenceLengths,
        AtomIds,
        H_Q: tl.constexpr,
        NUM_ATOMS: tl.constexpr,
        SINK_ATOMS: tl.constexpr,
        RECENT_ATOMS: tl.constexpr,
        MIDDLE_ATOMS: tl.constexpr,
        ROW_ATOMS: tl.constexpr,
        BLOCK_ATOMS: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // H_Q
        valid_atoms = (tl.load(SequenceLengths + batch) + 63) // 64
        atoms = tl.arange(0, BLOCK_ATOMS)
        valid = atoms < valid_atoms
        base = (atoms < SINK_ATOMS) | (atoms >= valid_atoms - RECENT_ATOMS)
        scores = tl.load(
            Scores + row * NUM_ATOMS + atoms,
            mask=atoms < NUM_ATOMS,
            other=-float("inf"),
        )
        scores = tl.where(valid & ~base, scores, -float("inf"))

        sink_slots = tl.arange(0, SINK_ATOMS)
        tl.store(
            AtomIds + row * ROW_ATOMS + sink_slots,
            sink_slots,
            mask=sink_slots < SINK_ATOMS,
        )
        recent_slots = tl.arange(0, RECENT_ATOMS)
        tl.store(
            AtomIds + row * ROW_ATOMS + SINK_ATOMS + recent_slots,
            valid_atoms - RECENT_ATOMS + recent_slots,
            mask=recent_slots < RECENT_ATOMS,
        )

        for selected in tl.static_range(0, MIDDLE_ATOMS):
            best = tl.max(scores, axis=0)
            winner = tl.min(
                tl.where(scores == best, atoms, BLOCK_ATOMS),
                axis=0,
            )
            tl.store(
                AtomIds + row * ROW_ATOMS + SINK_ATOMS + RECENT_ATOMS + selected,
                winner,
            )
            scores = tl.where(atoms == winner, -float("inf"), scores)

    @triton.jit
    def _paged_support_candidate_kernel(
        Scores,
        SequenceLengths,
        CandidateIds,
        H_Q: tl.constexpr,
        NUM_ATOMS: tl.constexpr,
        SINK_ATOMS: tl.constexpr,
        RECENT_ATOMS: tl.constexpr,
        CANDIDATE_ATOMS: tl.constexpr,
        BLOCK_ATOMS: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // H_Q
        valid_atoms = (tl.load(SequenceLengths + batch) + 63) // 64
        atoms = tl.arange(0, BLOCK_ATOMS)
        valid = atoms < valid_atoms
        base = (atoms < SINK_ATOMS) | (atoms >= valid_atoms - RECENT_ATOMS)
        scores = tl.load(
            Scores + row * NUM_ATOMS + atoms,
            mask=atoms < NUM_ATOMS,
            other=-float("inf"),
        )
        scores = tl.where(valid & ~base, scores, -float("inf"))

        for selected in tl.static_range(0, CANDIDATE_ATOMS):
            best = tl.max(scores, axis=0)
            winner = tl.min(
                tl.where(scores == best, atoms, BLOCK_ATOMS),
                axis=0,
            )
            tl.store(
                CandidateIds + row * CANDIDATE_ATOMS + selected,
                winner,
            )
            scores = tl.where(atoms == winner, -float("inf"), scores)

    @triton.jit
    def _paged_candidate_exact_score_kernel(
        Query,
        KeyPages,
        PageTable,
        CandidateIds,
        CandidateScores,
        H_Q: tl.constexpr,
        H_KV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        MAX_PAGES: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CANDIDATE_ATOMS: tl.constexpr,
        LAYOUT_NHD: tl.constexpr,
    ):
        row = tl.program_id(0)
        candidate_slot = tl.program_id(1)
        batch = row // H_Q
        q_head = row - batch * H_Q
        kv_head = q_head // GROUP_SIZE
        atom = tl.load(CandidateIds + row * CANDIDATE_ATOMS + candidate_slot)

        tokens = tl.arange(0, 64)
        dims = tl.arange(0, HEAD_DIM)
        q = tl.load(
            Query + batch * H_Q * HEAD_DIM + q_head * HEAD_DIM + dims
        ).to(tl.float32)
        logical_tokens = atom * 64 + tokens
        logical_pages = logical_tokens // PAGE_SIZE
        token_in_page = logical_tokens - logical_pages * PAGE_SIZE
        physical_pages = tl.load(
            PageTable + batch * MAX_PAGES + logical_pages
        )
        if LAYOUT_NHD:
            key_offsets = (
                (((physical_pages[:, None] * PAGE_SIZE + token_in_page[:, None])
                  * H_KV + kv_head)
                 * HEAD_DIM)
                + dims[None, :]
            )
        else:
            key_offsets = (
                (((physical_pages[:, None] * H_KV + kv_head) * PAGE_SIZE
                  + token_in_page[:, None])
                 * HEAD_DIM)
                + dims[None, :]
            )
        keys = tl.load(KeyPages + key_offsets).to(tl.float32)
        token_scores = tl.sum(keys * q[None, :], axis=1)
        exact_score = tl.max(token_scores, axis=0)
        tl.store(
            CandidateScores + row * CANDIDATE_ATOMS + candidate_slot,
            exact_score,
        )

    @triton.jit
    def _paged_candidate_final_topk_kernel(
        CandidateIds,
        CandidateScores,
        SequenceLengths,
        AtomIds,
        H_Q: tl.constexpr,
        SINK_ATOMS: tl.constexpr,
        RECENT_ATOMS: tl.constexpr,
        MIDDLE_ATOMS: tl.constexpr,
        ROW_ATOMS: tl.constexpr,
        CANDIDATE_ATOMS: tl.constexpr,
        BLOCK_CANDIDATES: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // H_Q
        valid_atoms = (tl.load(SequenceLengths + batch) + 63) // 64

        sink_slots = tl.arange(0, SINK_ATOMS)
        tl.store(
            AtomIds + row * ROW_ATOMS + sink_slots,
            sink_slots,
            mask=sink_slots < SINK_ATOMS,
        )
        recent_slots = tl.arange(0, RECENT_ATOMS)
        tl.store(
            AtomIds + row * ROW_ATOMS + SINK_ATOMS + recent_slots,
            valid_atoms - RECENT_ATOMS + recent_slots,
            mask=recent_slots < RECENT_ATOMS,
        )

        slots = tl.arange(0, BLOCK_CANDIDATES)
        candidate_ids = tl.load(
            CandidateIds + row * CANDIDATE_ATOMS + slots,
            mask=slots < CANDIDATE_ATOMS,
            other=-1,
        )
        scores = tl.load(
            CandidateScores + row * CANDIDATE_ATOMS + slots,
            mask=slots < CANDIDATE_ATOMS,
            other=-float("inf"),
        )
        for selected in tl.static_range(0, MIDDLE_ATOMS):
            best = tl.max(scores, axis=0)
            winner_slot = tl.min(
                tl.where(scores == best, slots, BLOCK_CANDIDATES),
                axis=0,
            )
            winner_atom = tl.sum(
                tl.where(slots == winner_slot, candidate_ids, 0),
                axis=0,
            )
            tl.store(
                AtomIds + row * ROW_ATOMS + SINK_ATOMS + RECENT_ATOMS + selected,
                winner_atom,
            )
            scores = tl.where(slots == winner_slot, -float("inf"), scores)


def paged_support_select_triton(
    query: torch.Tensor,
    support_keys: torch.Tensor,
    sequence_lengths: torch.Tensor,
    atom_ids: torch.Tensor,
    *,
    sink_atoms: int,
    recent_atoms: int,
    middle_atoms: int,
    scores: Optional[torch.Tensor] = None,
    atoms_per_program: Optional[int] = None,
) -> torch.Tensor:
    """Write query-selected Q-head atom rows without host synchronization.

    Args:
        query: Contiguous BF16 decode query ``[B,1,Hq,D]``.
        support_keys: Contiguous BF16 metadata ``[B,Hkv,A,P,D]``.
        sequence_lengths: CUDA int32 live lengths ``[B]``.
        atom_ids: Reusable CUDA int32 output with ``B*Hq*row_atoms`` entries.
        sink_atoms: Fixed prefix atoms included in every row.
        recent_atoms: Fixed live suffix atoms included in every row.
        middle_atoms: Query-selected atoms per row.
        scores: Optional reusable FP32 ``[B*Hq,A]`` workspace.

    Returns:
        The score workspace. ``atom_ids`` is updated in place in score order.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (query.is_cuda and support_keys.is_cuda and sequence_lengths.is_cuda):
        raise RuntimeError("paged support selection requires CUDA tensors")
    if not atom_ids.is_cuda:
        raise RuntimeError("atom_ids must be a CUDA tensor")
    if query.dtype != torch.bfloat16 or support_keys.dtype != torch.bfloat16:
        raise ValueError("query and support keys must use BF16")
    if sequence_lengths.dtype != torch.int32 or atom_ids.dtype != torch.int32:
        raise ValueError("sequence_lengths and atom_ids must use int32")
    if query.dim() != 4 or query.shape[1] != 1:
        raise ValueError("query must have shape [B,1,Hq,D]")
    if support_keys.dim() != 5:
        raise ValueError("support_keys must have shape [B,Hkv,A,P,D]")
    if not query.is_contiguous() or not support_keys.is_contiguous():
        raise ValueError("query and support_keys must be contiguous")
    if not sequence_lengths.is_contiguous() or not atom_ids.is_contiguous():
        raise ValueError("sequence_lengths and atom_ids must be contiguous")

    batch, _, q_heads, head_dim = (int(value) for value in query.shape)
    summary_batch, kv_heads, num_atoms, support_width, summary_dim = (
        int(value) for value in support_keys.shape
    )
    if summary_batch != batch or summary_dim != head_dim:
        raise ValueError("support metadata does not match query batch/head dimension")
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if head_dim not in {64, 128}:
        raise ValueError("paged support selector supports D64/D128")
    if support_width not in {1, 2, 4, 8}:
        raise ValueError("support width must be 1, 2, 4, or 8")
    if min(sink_atoms, recent_atoms, middle_atoms) < 0 or middle_atoms <= 0:
        raise ValueError("selector atom counts must be non-negative with middle > 0")
    row_atoms = sink_atoms + recent_atoms + middle_atoms
    if atom_ids.numel() != batch * q_heads * row_atoms:
        raise ValueError("atom_ids size must equal B*Hq*selected_atoms")
    if num_atoms > 12_288:
        raise ValueError("selector supports at most 12,288 logical atoms")

    if row_atoms > num_atoms:
        raise ValueError("selected atom count exceeds support metadata capacity")

    rows = batch * q_heads
    if scores is None:
        scores = torch.empty(
            rows,
            num_atoms,
            device=query.device,
            dtype=torch.float32,
        )
    elif scores.shape != (rows, num_atoms) or scores.dtype != torch.float32:
        raise ValueError("scores must be FP32 [B*Hq,num_atoms]")
    elif scores.device != query.device or not scores.is_contiguous():
        raise ValueError("scores must be contiguous on the query device")

    if atoms_per_program is None:
        atoms_per_program = 16 if support_width == 1 else 8
    if atoms_per_program not in {4, 8, 16, 32}:
        raise ValueError("atoms_per_program must be one of 4, 8, 16, 32")

    score_grid = (rows, triton.cdiv(num_atoms, atoms_per_program))
    _paged_support_score_kernel[score_grid](
        query,
        support_keys,
        sequence_lengths,
        scores,
        H_Q=q_heads,
        H_KV=kv_heads,
        GROUP_SIZE=q_heads // kv_heads,
        NUM_ATOMS=num_atoms,
        SUPPORT_WIDTH=support_width,
        HEAD_DIM=head_dim,
        ATOMS_PER_PROGRAM=atoms_per_program,
        num_warps=4,
    )
    _paged_support_topk_kernel[(rows,)](
        scores,
        sequence_lengths,
        atom_ids,
        H_Q=q_heads,
        NUM_ATOMS=num_atoms,
        SINK_ATOMS=sink_atoms,
        RECENT_ATOMS=recent_atoms,
        MIDDLE_ATOMS=middle_atoms,
        ROW_ATOMS=row_atoms,
        BLOCK_ATOMS=triton.next_power_of_2(num_atoms),
        num_warps=4,
    )
    return scores


def paged_support_refined_select_triton(
    query: torch.Tensor,
    support_keys: torch.Tensor,
    key_pages: torch.Tensor,
    page_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    atom_ids: torch.Tensor,
    *,
    layout: str,
    sink_atoms: int,
    recent_atoms: int,
    middle_atoms: int,
    candidate_atoms: int,
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    atoms_per_program: Optional[int] = None,
) -> None:
    """Select proxy candidates, refine them with exact block-max QK, and route.

    The full-context scan touches only compact support metadata. Exact token QK
    is restricted to ``candidate_atoms * 64`` tokens per Q head.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(
        tensor.is_cuda
        for tensor in (
            query,
            support_keys,
            key_pages,
            page_table,
            sequence_lengths,
            atom_ids,
            scores,
            candidate_ids,
            candidate_scores,
        )
    ):
        raise RuntimeError("refined paged support selection requires CUDA tensors")
    if query.dtype != torch.bfloat16 or support_keys.dtype != torch.bfloat16:
        raise ValueError("query and support keys must use BF16")
    if key_pages.dtype != query.dtype:
        raise ValueError("key pages must share the query dtype")
    if page_table.dtype != torch.int32 or sequence_lengths.dtype != torch.int32:
        raise ValueError("page table and sequence lengths must use int32")
    if atom_ids.dtype != torch.int32 or candidate_ids.dtype != torch.int32:
        raise ValueError("atom and candidate ids must use int32")
    if scores.dtype != torch.float32 or candidate_scores.dtype != torch.float32:
        raise ValueError("selector score workspaces must use FP32")
    if query.dim() != 4 or query.shape[1] != 1:
        raise ValueError("query must have shape [B,1,Hq,D]")
    if support_keys.dim() != 5 or key_pages.dim() != 4 or page_table.dim() != 2:
        raise ValueError("invalid support metadata or paged-cache rank")
    if not all(
        tensor.is_contiguous()
        for tensor in (
            query,
            support_keys,
            key_pages,
            page_table,
            sequence_lengths,
            atom_ids,
            scores,
            candidate_ids,
            candidate_scores,
        )
    ):
        raise ValueError("refined selector tensors must be contiguous")

    batch, _, q_heads, head_dim = (int(value) for value in query.shape)
    summary_batch, kv_heads, num_atoms, support_width, summary_dim = (
        int(value) for value in support_keys.shape
    )
    if summary_batch != batch or summary_dim != head_dim:
        raise ValueError("support metadata does not match the query")
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if head_dim not in {64, 128}:
        raise ValueError("refined selector supports D64/D128")
    if support_width not in {1, 2, 4, 8}:
        raise ValueError("support width must be 1, 2, 4, or 8")
    normalized_layout = str(layout).upper()
    if normalized_layout not in {"NHD", "HND"}:
        raise ValueError("layout must be NHD or HND")
    page_size = int(key_pages.shape[1] if normalized_layout == "NHD" else key_pages.shape[2])
    cache_kv_heads = int(key_pages.shape[2] if normalized_layout == "NHD" else key_pages.shape[1])
    if page_size != 16 or cache_kv_heads != kv_heads or int(key_pages.shape[3]) != head_dim:
        raise ValueError("refined selector requires matching page16 key pages")
    if candidate_atoms not in {8, 16, 32, 64}:
        raise ValueError("candidate_atoms must be one of 8, 16, 32, or 64")
    if candidate_atoms < middle_atoms:
        raise ValueError("candidate_atoms must cover the final middle atom count")
    if min(sink_atoms, recent_atoms, middle_atoms) <= 0:
        raise ValueError("sink, recent, and middle atom counts must be positive")

    rows = batch * q_heads
    row_atoms = sink_atoms + recent_atoms + middle_atoms
    if scores.shape != (rows, num_atoms):
        raise ValueError("scores must have shape [B*Hq,num_atoms]")
    if candidate_ids.shape != (rows, candidate_atoms):
        raise ValueError("candidate_ids must have shape [B*Hq,candidate_atoms]")
    if candidate_scores.shape != (rows, candidate_atoms):
        raise ValueError("candidate_scores must have shape [B*Hq,candidate_atoms]")
    if atom_ids.numel() != rows * row_atoms:
        raise ValueError("atom_ids size must equal B*Hq*selected_atoms")

    if atoms_per_program is None:
        atoms_per_program = 16 if support_width == 1 else 8
    if atoms_per_program not in {4, 8, 16, 32}:
        raise ValueError("atoms_per_program must be one of 4, 8, 16, 32")

    _paged_support_score_kernel[
        (rows, triton.cdiv(num_atoms, atoms_per_program))
    ](
        query,
        support_keys,
        sequence_lengths,
        scores,
        H_Q=q_heads,
        H_KV=kv_heads,
        GROUP_SIZE=q_heads // kv_heads,
        NUM_ATOMS=num_atoms,
        SUPPORT_WIDTH=support_width,
        HEAD_DIM=head_dim,
        ATOMS_PER_PROGRAM=atoms_per_program,
        num_warps=4,
    )
    _paged_support_candidate_kernel[(rows,)](
        scores,
        sequence_lengths,
        candidate_ids,
        H_Q=q_heads,
        NUM_ATOMS=num_atoms,
        SINK_ATOMS=sink_atoms,
        RECENT_ATOMS=recent_atoms,
        CANDIDATE_ATOMS=candidate_atoms,
        BLOCK_ATOMS=triton.next_power_of_2(num_atoms),
        num_warps=4,
    )
    _paged_candidate_exact_score_kernel[(rows, candidate_atoms)](
        query,
        key_pages,
        page_table,
        candidate_ids,
        candidate_scores,
        H_Q=q_heads,
        H_KV=kv_heads,
        GROUP_SIZE=q_heads // kv_heads,
        MAX_PAGES=int(page_table.shape[1]),
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        CANDIDATE_ATOMS=candidate_atoms,
        LAYOUT_NHD=normalized_layout == "NHD",
        num_warps=4,
    )
    _paged_candidate_final_topk_kernel[(rows,)](
        candidate_ids,
        candidate_scores,
        sequence_lengths,
        atom_ids,
        H_Q=q_heads,
        SINK_ATOMS=sink_atoms,
        RECENT_ATOMS=recent_atoms,
        MIDDLE_ATOMS=middle_atoms,
        ROW_ATOMS=row_atoms,
        CANDIDATE_ATOMS=candidate_atoms,
        BLOCK_CANDIDATES=triton.next_power_of_2(candidate_atoms),
        num_warps=4,
    )
