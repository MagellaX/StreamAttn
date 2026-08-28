"""Planned exact prefill and differentiable attention entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .core.fused_online_attention import FusedOnlineAttention, TRITON_AVAILABLE
from .planning import (
    ATTENTION_GUARANTEE_EXACT,
    ATTENTION_PHASE_PREFILL,
    ATTENTION_PHASE_TRAIN,
    AttentionBackendPlan,
    AttentionProblem,
    AttentionTilePlan,
    device_architecture,
)


@dataclass(frozen=True)
class StreamAttnAttentionInfo:
    """Execution evidence returned by prefill and training attention calls."""

    phase: str
    backend_used: str
    causal: bool
    dropout_p: float
    deterministic: bool
    attention_problem: AttentionProblem
    tile_plan: AttentionTilePlan
    backend_plan: AttentionBackendPlan

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "backend_used": self.backend_used,
            "guarantee": self.attention_problem.guarantee,
            "causal": self.causal,
            "dropout_p": self.dropout_p,
            "deterministic": self.deterministic,
            "tile_plan": self.tile_plan.as_dict(),
            "backend_plan": self.backend_plan.as_dict(),
        }


@dataclass
class StreamAttnAttentionPlan:
    """Exact all-tile attention plan bound to Q/K/V tensors."""

    phase: str
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    causal: bool
    attention_mask: Optional[torch.Tensor]
    dropout_p: float
    alibi_slopes: Optional[torch.Tensor]
    deterministic: bool
    module: Optional[FusedOnlineAttention]
    native_plan: Optional[Any]
    attention_problem: AttentionProblem
    tile_plan: AttentionTilePlan
    backend_plan: AttentionBackendPlan

    def summary(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "causal": self.causal,
            "dropout_p": self.dropout_p,
            "deterministic": self.deterministic,
            "tile_plan": self.tile_plan.as_dict(),
            "backend_plan": self.backend_plan.as_dict(),
        }

    def run(self, *, return_info: bool = False):
        if self.native_plan is not None:
            output = self.native_plan.run()
            backend_used = self.backend_plan.backend
        else:
            if self.module is None:  # pragma: no cover - construction invariant
                raise RuntimeError("functional attention plan has no executable backend")
            output = self.module(
                self.query,
                self.key,
                self.value,
                causal=self.causal,
                attention_mask=self.attention_mask,
                dropout_p=self.dropout_p,
                alibi_slopes=self.alibi_slopes,
                deterministic=self.deterministic,
            )
            backend_used = self.module.last_backend_used
        if not return_info:
            return output
        return output, StreamAttnAttentionInfo(
            phase=self.phase,
            backend_used=backend_used,
            causal=self.causal,
            dropout_p=self.dropout_p,
            deterministic=self.deterministic,
            attention_problem=self.attention_problem,
            tile_plan=self.tile_plan,
            backend_plan=self.backend_plan,
        )


def _validate_functional_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must use [B, S, H, D] layout")
    if not (
        query.is_floating_point()
        and key.is_floating_point()
        and value.is_floating_point()
    ):
        raise ValueError("query, key, and value must be floating-point tensors")


def _planned_backend(
    module: FusedOnlineAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout_p: float,
) -> tuple[str, str]:
    use_triton = bool(
        TRITON_AVAILABLE
        and query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and module.sm >= 70
        and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and int(query.shape[-1]) in {16, 32, 64, 128}
    )
    if use_triton and attention_mask is not None:
        use_triton = module._mask_supported_for_triton(
            attention_mask,
            int(query.shape[0]),
            int(query.shape[1]),
            int(key.shape[1]),
        ) and module.sm >= 80
    requires_grad = bool(
        torch.is_grad_enabled()
        and (query.requires_grad or key.requires_grad or value.requires_grad)
    )
    if use_triton and requires_grad and dropout_p != 0.0:
        use_triton = False
    if not use_triton:
        fallback = (
            "torch_sdpa_gqa_expanded"
            if module.group_size > 1
            else "torch_sdpa"
        )
        return fallback, "unsupported_native_shape_or_training_feature"
    if requires_grad:
        return "triton_online_softmax_autograd", "native_streaming_forward_backward"
    return "triton_online_softmax", "native_streaming_forward"


def plan_functional_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    phase: str,
    causal: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    seed: Optional[int] = None,
    scale: Optional[float] = None,
    tile_size_q: int = 128,
    tile_size_k: int = 64,
) -> StreamAttnAttentionPlan:
    """Lower exact prefill/training semantics through the shared planner."""

    if phase not in {ATTENTION_PHASE_PREFILL, ATTENTION_PHASE_TRAIN}:
        raise ValueError("functional attention phase must be prefill or train")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    if phase == ATTENTION_PHASE_PREFILL and dropout_p != 0.0:
        raise ValueError("prefill is inference-only and requires dropout_p == 0")
    if phase == ATTENTION_PHASE_PREFILL and torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    ):
        raise ValueError(
            "prefill is inference-only; use torch.no_grad(), detach Q/K/V, "
            "or call stream_attn.train(...)"
        )
    if seed is not None and not deterministic:
        raise ValueError("seed requires deterministic=True")
    _validate_functional_qkv(query, key, value)

    mask_kind = "causal" if causal else "none"
    if attention_mask is not None:
        mask_kind = "causal+custom" if causal else "custom"
    problem = AttentionProblem.from_qkv(
        query,
        key,
        value,
        phase=phase,
        guarantee=ATTENTION_GUARANTEE_EXACT,
        mask=mask_kind,
    )
    tile_plan = AttentionTilePlan.exact(
        problem,
        logical_tile_size=tile_size_k,
        reason=f"{phase}_exact_all_tiles",
    )
    requires_grad = bool(
        torch.is_grad_enabled()
        and (query.requires_grad or key.requires_grad or value.requires_grad)
    )
    native_plan = None
    if (
        phase == ATTENTION_PHASE_PREFILL
        and causal
        and attention_mask is None
        and alibi_slopes is None
        and scale is None
        and not requires_grad
    ):
        try:
            from .backends.sm100.gqa_prefill import (
                Sm100GqaPrefillPlan,
                is_promoted_sm100_gqa_prefill,
            )

            if is_promoted_sm100_gqa_prefill(query, key, value):
                native_plan = Sm100GqaPrefillPlan.build(
                    query,
                    key,
                    value,
                    tile="h8_q2",
                )
        except (ImportError, OSError, RuntimeError, ValueError):
            native_plan = None
    if native_plan is not None:
        backend_plan = AttentionBackendPlan(
            backend="sm100_tgv_gqa_causal_prefill",
            reason="promoted_exact_b200_prefill_cell",
            architecture=device_architecture(query.device),
        )
        return StreamAttnAttentionPlan(
            phase=phase,
            query=query,
            key=key,
            value=value,
            causal=causal,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            module=None,
            native_plan=native_plan,
            attention_problem=problem,
            tile_plan=tile_plan,
            backend_plan=backend_plan,
        )
    module = FusedOnlineAttention(
        num_heads=problem.q_heads,
        num_kv_heads=problem.kv_heads,
        head_dim=problem.head_dim,
        tile_size_q=tile_size_q,
        tile_size_k=tile_size_k,
        dropout=dropout_p,
        scale=scale,
        device=query.device,
        dtype=query.dtype,
    )
    module.train(phase == ATTENTION_PHASE_TRAIN)
    module.set_deterministic(deterministic, seed=seed)
    backend, reason = _planned_backend(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout_p,
    )
    backend_plan = AttentionBackendPlan(
        backend=backend,
        reason=reason,
        architecture=device_architecture(query.device),
    )
    return StreamAttnAttentionPlan(
        phase=phase,
        query=query,
        key=key,
        value=value,
        causal=causal,
        attention_mask=attention_mask,
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        module=module,
        native_plan=None,
        attention_problem=problem,
        tile_plan=tile_plan,
        backend_plan=backend_plan,
    )
