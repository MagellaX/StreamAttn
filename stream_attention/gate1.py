"""Runtime facade for Gate-1 attention with router telemetry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .certified import StreamAttnMetadataCache
from .kernels.gate1_fwd_triton import gate1_attention_triton_forward
from .router import AttentionRouteRequest, BackendDecision, StreamAttnRouter
from .telemetry import Prediction


@dataclass(frozen=True)
class Gate1Stats:
    row_skips: int
    row_computes: int
    cta_tiles_total: int
    cta_pv_skipped: int
    cta_pv_executed: int
    force_mode_sum: int

    @property
    def active_pv_fraction(self) -> float:
        if self.cta_tiles_total <= 0:
            return 0.0
        return self.cta_pv_executed / self.cta_tiles_total


@dataclass(frozen=True)
class Gate1RunInfo:
    decision: BackendDecision
    stats: Optional[Gate1Stats]
    request: AttentionRouteRequest
    per_head_stats: Optional[Tuple[Gate1Stats, ...]] = None

    @property
    def active_pv_fraction(self) -> Optional[float]:
        return None if self.stats is None else self.stats.active_pv_fraction


def summarize_gate1_raw_stats(raw_stats: torch.Tensor) -> Gate1Stats:
    """Summarize raw Gate-1 kernel counters."""

    if raw_stats.dim() != 4 or raw_stats.shape[-1] != 6:
        raise ValueError("raw_stats must have shape [batch, heads, q_blocks, 6]")
    totals = raw_stats.detach().sum(dim=(0, 1, 2)).cpu()
    return Gate1Stats(
        row_skips=int(totals[0].item()),
        row_computes=int(totals[1].item()),
        cta_tiles_total=int(totals[2].item()),
        cta_pv_skipped=int(totals[3].item()),
        cta_pv_executed=int(totals[4].item()),
        force_mode_sum=int(totals[5].item()),
    )


def summarize_gate1_raw_stats_per_head(raw_stats: torch.Tensor) -> Tuple[Gate1Stats, ...]:
    """Summarize raw Gate-1 counters independently for each head."""

    if raw_stats.dim() != 4 or raw_stats.shape[-1] != 6:
        raise ValueError("raw_stats must have shape [batch, heads, q_blocks, 6]")
    totals = raw_stats.detach().sum(dim=(0, 2)).cpu()
    return tuple(
        Gate1Stats(
            row_skips=int(row[0].item()),
            row_computes=int(row[1].item()),
            cta_tiles_total=int(row[2].item()),
            cta_pv_skipped=int(row[3].item()),
            cta_pv_executed=int(row[4].item()),
            force_mode_sum=int(row[5].item()),
        )
        for row in totals
    )


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")


def _device_name(tensor: torch.Tensor) -> str:
    if tensor.is_cuda:
        return torch.cuda.get_device_name(tensor.device)
    return tensor.device.type


def _device_class(tensor: torch.Tensor) -> str:
    if tensor.is_cuda:
        major, minor = torch.cuda.get_device_capability(tensor.device)
        name = _device_name(tensor).lower()
        if major == 9:
            return "sm90_h100" if "h100" in name else "sm90"
        if major == 8 and minor == 0:
            return "sm80_a100" if "a100" in name else "sm80"
        if major == 8 and minor == 6:
            return "sm86_a10g" if "a10" in name else "sm86"
        if major == 8 and minor == 9:
            return "sm89_l4" if "l4" in name else "sm89"
        return f"sm{major}{minor}"
    return tensor.device.type


def make_route_request(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    causal: bool,
    block_size: int,
    tile_size_q: int,
    model_id: str = "default",
    layer_id: int = 0,
    head_id: int = -1,
    kv_head_id: int = -1,
    q_group_id: int = -1,
    phase: str = "prefill",
    metadata_available: bool = False,
    metadata_build_allowed: bool = False,
    metadata_build_ms: Optional[float] = None,
) -> AttentionRouteRequest:
    """Build a router request from Q/K tensor metadata."""

    if query.dim() != 4 or key.dim() != 4:
        raise ValueError("query and key must have shape [batch, seq, heads, dim]")
    return AttentionRouteRequest(
        batch=query.shape[0],
        seq_q=query.shape[1],
        seq_k=key.shape[1],
        heads=query.shape[2],
        dim=query.shape[3],
        dtype=_dtype_name(query.dtype),
        device=_device_name(query),
        device_class=_device_class(query),
        tile_size_q=tile_size_q,
        block_size=block_size,
        causal=causal,
        model_id=model_id,
        layer_id=layer_id,
        head_id=head_id,
        kv_head_id=kv_head_id,
        q_group_id=q_group_id,
        phase=phase,
        metadata_available=metadata_available,
        metadata_build_allowed=metadata_build_allowed,
        metadata_build_ms=metadata_build_ms,
    )


def dense_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Reference dense SDPA path using StreamAttn tensor layout."""

    q_bh = query.permute(0, 2, 1, 3).contiguous()
    k_bh = key.permute(0, 2, 1, 3).contiguous()
    v_bh = value.permute(0, 2, 1, 3).contiguous()
    if q_bh.shape[1] != k_bh.shape[1]:
        if q_bh.shape[1] % k_bh.shape[1] != 0:
            raise ValueError("query heads must be a multiple of KV heads")
        group_size = q_bh.shape[1] // k_bh.shape[1]
        k_bh = k_bh.repeat_interleave(group_size, dim=1)
        v_bh = v_bh.repeat_interleave(group_size, dim=1)
    out = F.scaled_dot_product_attention(
        q_bh,
        k_bh,
        v_bh,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def _decision(
    *,
    backend: str,
    reason: str,
    prediction: Optional[Prediction] = None,
) -> BackendDecision:
    return BackendDecision(
        backend=backend,
        reason=reason,
        prediction=prediction
        or Prediction(active_frac_hat=1.0, confidence=1.0, source="forced"),
    )


def stream_attn_gate1(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    mode: str = "auto",
    router: Optional[StreamAttnRouter] = None,
    metadata: Optional[StreamAttnMetadataCache] = None,
    request: Optional[AttentionRouteRequest] = None,
    model_id: str = "default",
    layer_id: int = 0,
    head_id: int = -1,
    kv_head_id: int = -1,
    q_group_id: int = -1,
    phase: str = "prefill",
    error_budget: float = 1e-3,
    block_size: int = 64,
    tile_size_q: int = 64,
    skip_predicate: str = "value_bound",
    post_qk_threshold: float = 0.0,
    telemetry: bool = True,
    build_metadata_if_missing: bool = False,
    return_info: bool = False,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Gate1RunInfo]]:
    """Run dense, Gate-1, or router-selected attention.

    ``mode`` is ``"dense"``, ``"gate1"``, or ``"auto"``. In ``"auto"`` mode,
    value-bound Gate-1 requires cached ``metadata.value_norm_bounds`` unless
    ``build_metadata_if_missing`` is set. This keeps metadata construction out
    of the hot path by default.
    """

    if mode not in {"auto", "dense", "gate1"}:
        raise ValueError("mode must be 'auto', 'dense', or 'gate1'")
    if skip_predicate not in {"mass", "value_bound"}:
        raise ValueError("skip_predicate must be 'mass' or 'value_bound'")

    if request is None:
        request = make_route_request(
            query,
            key,
            causal=causal,
            block_size=block_size,
            tile_size_q=tile_size_q,
            model_id=model_id,
            layer_id=layer_id,
            head_id=head_id,
            kv_head_id=kv_head_id,
            q_group_id=q_group_id,
            phase=phase,
        )

    value_norm_bounds = None
    if metadata is not None:
        metadata.validate_for_value(value)
        value_norm_bounds = metadata.value_norm_bounds

    request = replace(
        request,
        metadata_available=value_norm_bounds is not None,
        metadata_build_allowed=build_metadata_if_missing,
    )

    if mode == "dense":
        decision = _decision(backend="dense", reason="forced_dense")
    elif mode == "gate1":
        decision = _decision(backend="gate1", reason="forced_gate1")
    elif router is None:
        decision = _decision(backend="dense", reason="missing_router")
    else:
        decision = router.choose(request)

    if (
        decision.backend == "gate1"
        and skip_predicate == "value_bound"
        and value_norm_bounds is None
        and not build_metadata_if_missing
    ):
        if mode == "gate1":
            raise ValueError(
                "value_bound Gate-1 requires metadata or build_metadata_if_missing=True"
            )
        decision = _decision(
            backend="dense",
            reason="missing_value_norm_bounds",
            prediction=decision.prediction,
        )
    elif (
        decision.backend == "gate1"
        and skip_predicate == "value_bound"
        and value_norm_bounds is None
        and build_metadata_if_missing
    ):
        metadata = StreamAttnMetadataCache.from_value(value, block_size=block_size)
        value_norm_bounds = metadata.value_norm_bounds
        request = replace(request, metadata_available=True)

    stats = None
    per_head_stats = None
    if decision.backend == "dense":
        output = dense_attention_forward(query, key, value, causal=causal)
    else:
        output, raw_stats = gate1_attention_triton_forward(
            query,
            key,
            value,
            causal=causal,
            error_budget=error_budget,
            block_size=block_size,
            tile_size_q=tile_size_q,
            value_norm_bounds=value_norm_bounds,
            skip_predicate=skip_predicate,
            post_qk_threshold=post_qk_threshold,
            force_mode=0,
            return_raw_stats=telemetry or return_info,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if raw_stats is not None:
            stats = summarize_gate1_raw_stats(raw_stats)
            per_head_stats = summarize_gate1_raw_stats_per_head(raw_stats)
            if telemetry and router is not None and stats.cta_tiles_total > 0:
                router.observe(
                    request,
                    cta_pv_executed=stats.cta_pv_executed,
                    cta_tiles_total=stats.cta_tiles_total,
                )
                for head_idx, head_stats in enumerate(per_head_stats):
                    if head_stats.cta_tiles_total <= 0:
                        continue
                    router.observe(
                        replace(request, head_id=head_idx),
                        cta_pv_executed=head_stats.cta_pv_executed,
                        cta_tiles_total=head_stats.cta_tiles_total,
                    )

    info = Gate1RunInfo(
        decision=decision,
        stats=stats,
        request=request,
        per_head_stats=per_head_stats,
    )
    return (output, info) if return_info else output
