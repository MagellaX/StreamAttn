"""Public fixed-buffer StreamAttn decode engine.

The engine is intentionally model-independent. Model adapters own projection,
RoPE, and cache layout; this module owns native mode selection and the planned
attention launch over stable Q/K/V buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .decode import (
    STREAMATTN_EXACT_NATIVE_BACKEND,
    STREAMATTN_MODE_EXACT_NATIVE,
    STREAMATTN_MODE_SEED_ONLY_NATIVE,
    STREAMATTN_MODE_VERIFIED_AUTO,
    DenseFallbackFn,
    Gate0SeedOnlyBatchedPolicy,
    StreamAttnDecodePolicy,
    StreamAttnExactNativeDirectRunner,
    StreamAttnSeedOnlyDecodeService,
    StreamAttnSeedOnlyDirectRunner,
    StreamAttnServingInfo,
    normalize_stream_attn_mode,
)
from .paged import (
    PAGED_EXACT_NATIVE_BACKEND,
    PagedExactDecodePlan,
    PagedExactDecodeRunner,
    PagedKVCache,
)


@dataclass(frozen=True)
class StreamAttnEnginePlan:
    """A decode plan bound to stable Q/K/V buffers."""

    mode: str
    backend: str
    reason: str
    model_id: Optional[str]
    layer_id: Optional[int]
    query: torch.Tensor
    key_cache: Union[torch.Tensor, PagedKVCache]
    value_cache: Optional[torch.Tensor]
    service: StreamAttnSeedOnlyDecodeService
    direct_runner: Optional[
        Union[
            StreamAttnSeedOnlyDirectRunner,
            StreamAttnExactNativeDirectRunner,
            PagedExactDecodeRunner,
        ]
    ] = None

    @property
    def uses_fixed_buffers(self) -> bool:
        return self.direct_runner is not None

    def run(self, *, return_info: bool = True):
        """Execute without replanning or policy parsing."""

        if self.direct_runner is not None:
            return (
                self.direct_runner.run_with_info()
                if return_info
                else self.direct_runner.run()
            )
        if self.value_cache is None or not isinstance(self.key_cache, torch.Tensor):
            raise RuntimeError("non-native plan is missing contiguous KV tensors")
        return self.service.run(
            self.query,
            self.key_cache,
            self.value_cache,
            model_id=self.model_id,
            layer_id=self.layer_id,
            mode=STREAMATTN_MODE_EXACT_NATIVE,
            return_info=return_info,
        )


class StreamAttnEngine:
    """Plan and run StreamAttn native decode modes.

    ``verified_auto`` means policy-verified, fail-closed routing. Runtime canary
    verification remains a model-adapter concern until a generic device-side
    verifier is available.
    """

    def __init__(
        self,
        *,
        policy: Optional[Gate0SeedOnlyBatchedPolicy] = None,
        policy_name: str = "qwen25_05b_l8_32k_seed_only_batched",
        decode_policy: Optional[StreamAttnDecodePolicy] = None,
        dense_fallback: Optional[DenseFallbackFn] = None,
        dense_fallback_backend: str = STREAMATTN_EXACT_NATIVE_BACKEND,
        model_id: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        self.service = StreamAttnSeedOnlyDecodeService(
            policy=policy,
            policy_name=policy_name,
            decode_policy=decode_policy,
            dense_fallback=dense_fallback,
            dense_fallback_backend=dense_fallback_backend,
            model_id=model_id,
            layer_id=layer_id,
            use_planned_direct_seed_only_path=True,
        )

    def _exact_plan(
        self,
        *,
        mode: str,
        reason: str,
        model_id: Optional[str],
        layer_id: Optional[int],
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> StreamAttnEnginePlan:
        direct_runner = None
        try:
            direct_runner = self.service.plan_exact_native(
                query,
                key_cache,
                value_cache,
                model_id=model_id,
                layer_id=layer_id,
            )
        except ValueError:
            pass
        return StreamAttnEnginePlan(
            mode=mode,
            backend=STREAMATTN_EXACT_NATIVE_BACKEND,
            reason=reason,
            model_id=model_id,
            layer_id=layer_id,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            service=self.service,
            direct_runner=direct_runner,
        )

    def _paged_exact_plan(
        self,
        *,
        mode: str,
        query: torch.Tensor,
        cache: PagedKVCache,
    ) -> StreamAttnEnginePlan:
        if mode == STREAMATTN_MODE_SEED_ONLY_NATIVE:
            raise ValueError(
                "seed_only_native does not yet support paged KV; "
                "use exact_native or verified_auto"
            )
        paged_plan = PagedExactDecodePlan.build(query, cache)
        reason = (
            "explicit_paged_exact_native"
            if mode == STREAMATTN_MODE_EXACT_NATIVE
            else "paged_kv_exact_only"
        )
        kv_len = int(cache.sequence_lengths.max().item())
        backend = paged_plan.backend
        info = StreamAttnServingInfo(
            backend_used=backend,
            policy_id=None,
            fallback_reason=None,
            batch_size=int(query.shape[0]),
            kv_len=kv_len,
            layer_id=None,
            model_id=None,
            dtype=str(query.dtype).removeprefix("torch."),
            device=str(query.device),
            plan_backend=backend,
            plan_reason=reason,
            seed_only_enabled=False,
            safety_policy_matched=False,
            runtime_counters={
                "backend_counts": {backend: 1},
                "fallback_reasons": {},
            },
            stats={
                "layout": cache.normalized_layout,
                "page_size": cache.page_size,
                "max_pages_per_request": cache.max_pages_per_request,
                "splits": paged_plan.splits,
                "workspace_bytes": paged_plan.workspace_bytes,
            },
        )
        runner = PagedExactDecodeRunner(plan=paged_plan, info=info)
        return StreamAttnEnginePlan(
            mode=mode,
            backend=backend,
            reason=reason,
            model_id=None,
            layer_id=None,
            query=query,
            key_cache=cache,
            value_cache=None,
            service=self.service,
            direct_runner=runner,
        )

    def plan(
        self,
        query: torch.Tensor,
        key_cache: Union[torch.Tensor, PagedKVCache],
        value_cache: Optional[torch.Tensor] = None,
        *,
        mode: str = STREAMATTN_MODE_VERIFIED_AUTO,
        model_id: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> StreamAttnEnginePlan:
        """Validate once and bind a native decode plan to stable buffers."""

        normalized_mode = normalize_stream_attn_mode(mode)
        if isinstance(key_cache, PagedKVCache):
            if value_cache is not None:
                raise ValueError(
                    "value_cache must be omitted when key_cache is PagedKVCache"
                )
            return self._paged_exact_plan(
                mode=normalized_mode,
                query=query,
                cache=key_cache,
            )
        if value_cache is None:
            raise ValueError("value_cache is required for contiguous KV decode")
        effective_model_id = model_id if model_id is not None else self.service.model_id
        effective_layer_id = layer_id if layer_id is not None else self.service.layer_id
        if normalized_mode == STREAMATTN_MODE_EXACT_NATIVE:
            return self._exact_plan(
                mode=normalized_mode,
                reason="explicit_exact_native",
                model_id=effective_model_id,
                layer_id=effective_layer_id,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
            )

        try:
            runner = self.service.plan_direct_seed_only(
                query,
                key_cache,
                value_cache,
                model_id=effective_model_id,
                layer_id=effective_layer_id,
                mode=normalized_mode,
            )
        except ValueError as exc:
            if normalized_mode == STREAMATTN_MODE_SEED_ONLY_NATIVE:
                raise
            return self._exact_plan(
                mode=normalized_mode,
                reason=str(exc),
                model_id=effective_model_id,
                layer_id=effective_layer_id,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
            )
        return StreamAttnEnginePlan(
            mode=normalized_mode,
            backend=STREAMATTN_MODE_SEED_ONLY_NATIVE,
            reason="verified_policy_match",
            model_id=effective_model_id,
            layer_id=effective_layer_id,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            service=self.service,
            direct_runner=runner,
        )

    def decode(
        self,
        query: torch.Tensor,
        key_cache: Union[torch.Tensor, PagedKVCache],
        value_cache: Optional[torch.Tensor] = None,
        *,
        mode: str = STREAMATTN_MODE_VERIFIED_AUTO,
        model_id: Optional[str] = None,
        layer_id: Optional[int] = None,
        return_info: bool = True,
    ):
        """Plan and execute one call; use :meth:`plan` for steady-state loops."""

        return self.plan(
            query,
            key_cache,
            value_cache,
            mode=mode,
            model_id=model_id,
            layer_id=layer_id,
        ).run(return_info=return_info)


def stream_attn_decode(
    query: torch.Tensor,
    key_cache: Union[torch.Tensor, PagedKVCache],
    value_cache: Optional[torch.Tensor] = None,
    *,
    mode: str = STREAMATTN_MODE_VERIFIED_AUTO,
    engine: Optional[StreamAttnEngine] = None,
    return_info: bool = True,
    **engine_kwargs,
):
    """Ergonomic one-shot decode entry point."""

    runtime = engine or StreamAttnEngine(**engine_kwargs)
    return runtime.decode(
        query,
        key_cache,
        value_cache,
        mode=mode,
        return_info=return_info,
    )
