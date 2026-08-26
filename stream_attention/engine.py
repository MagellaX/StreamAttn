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
    PagedDynamicSelectedDecodePlan,
    PagedDynamicSelectedDecodeRunner,
    PagedExactDecodePlan,
    PagedExactDecodeRunner,
    PagedKVCache,
    PagedQuerySelectedDecodePlan,
    PagedQuerySelectedDecodeRunner,
    PagedSelectedDecodePlan,
    PagedSelectedDecodeRunner,
)
from .planning import (
    ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    ATTENTION_GUARANTEE_EXACT,
    ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    AttentionBackendPlan,
    AttentionProblem,
    AttentionTilePlan,
    device_architecture,
    fixed_block_tile_ids,
)
from .selected_routes import prepare_paged_routes64


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
    attention_problem: AttentionProblem
    tile_plan: AttentionTilePlan
    backend_plan: AttentionBackendPlan
    direct_runner: Optional[
        Union[
            StreamAttnSeedOnlyDirectRunner,
            StreamAttnExactNativeDirectRunner,
            PagedExactDecodeRunner,
            PagedDynamicSelectedDecodeRunner,
            PagedQuerySelectedDecodeRunner,
            PagedSelectedDecodeRunner,
        ]
    ] = None

    @property
    def uses_fixed_buffers(self) -> bool:
        return self.direct_runner is not None

    def summary(self) -> dict[str, object]:
        """Return the semantic schedule and physical backend decision."""

        return {
            "mode": self.mode,
            "backend": self.backend,
            "reason": self.reason,
            "model_id": self.model_id,
            "layer_id": self.layer_id,
            "uses_fixed_buffers": self.uses_fixed_buffers,
            "tile_plan": self.tile_plan.as_dict(),
            "backend_plan": self.backend_plan.as_dict(),
        }

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
        native_plan = (
            getattr(direct_runner, "_sm90_plan", None)
            if direct_runner is not None
            else None
        )
        problem = AttentionProblem.from_contiguous(
            query,
            key_cache,
            value_cache,
            guarantee=ATTENTION_GUARANTEE_EXACT,
            cache_layout="HND" if native_plan is not None else "NHD",
        )
        tile_plan = AttentionTilePlan.exact(
            problem,
            logical_tile_size=64,
            reason=reason,
        )
        backend_variant = STREAMATTN_EXACT_NATIVE_BACKEND
        splits = 1
        workspace_bytes = 0
        if direct_runner is not None:
            backend_variant = str(direct_runner.backend_variant)
            if native_plan is not None:
                splits = int(native_plan.num_splits)
                workspace_bytes = int(native_plan.workspace_bytes)
        backend_plan = AttentionBackendPlan(
            backend=backend_variant,
            reason=reason,
            architecture=device_architecture(query.device),
            splits=splits,
            workspace_bytes=workspace_bytes,
        )
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
            attention_problem=problem,
            tile_plan=tile_plan,
            backend_plan=backend_plan,
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
        problem = AttentionProblem.from_paged(
            query,
            cache,
            guarantee=ATTENTION_GUARANTEE_EXACT,
        )
        tile_plan = AttentionTilePlan.exact(
            problem,
            logical_tile_size=paged_plan.tokens_per_tile,
            reason=reason,
        )
        backend_plan = AttentionBackendPlan(
            backend=backend,
            reason=reason,
            architecture=device_architecture(query.device),
            splits=int(paged_plan.splits),
            workspace_bytes=int(paged_plan.workspace_bytes),
        )
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
            attention_problem=problem,
            tile_plan=tile_plan,
            backend_plan=backend_plan,
            direct_runner=runner,
        )

    def plan_selected_paged(
        self,
        query: torch.Tensor,
        cache: PagedKVCache,
        tile_plan: AttentionTilePlan,
        *,
        output: Optional[torch.Tensor] = None,
    ) -> StreamAttnEnginePlan:
        """Bind a verified selected schedule to the native H100 paged executor."""

        routes = prepare_paged_routes64(tile_plan, cache)
        paged_plan = PagedSelectedDecodePlan.build(
            query,
            cache,
            routes,
            schedule_epoch=tile_plan.schedule.schedule_epoch,
            output=output,
        )
        backend = paged_plan.backend
        reason = "distribution_verified_selected_paged"
        info = StreamAttnServingInfo(
            backend_used=backend,
            policy_id=tile_plan.policy_id,
            fallback_reason=None,
            batch_size=int(query.shape[0]),
            kv_len=max(tile_plan.problem.kv_lengths),
            layer_id=None,
            model_id=None,
            dtype=str(query.dtype).removeprefix("torch."),
            device=str(query.device),
            plan_backend=backend,
            plan_reason=reason,
            seed_only_enabled=False,
            safety_policy_matched=True,
            runtime_counters={
                "backend_counts": {backend: 1},
                "fallback_reasons": {},
            },
            stats={
                "layout": cache.normalized_layout,
                "page_size": cache.page_size,
                "route_count": routes.route_count,
                "max_routes_per_row": paged_plan.max_routes_per_row,
                "group_route_efficiency": routes.group_route_efficiency,
                "scheduler_hint": routes.scheduler_hint,
                "metadata_bytes": routes.metadata_bytes,
                "workspace_bytes": paged_plan.workspace_bytes,
            },
        )
        runner = PagedSelectedDecodeRunner(plan=paged_plan, info=info)
        backend_plan = AttentionBackendPlan(
            backend=backend,
            reason=reason,
            architecture=device_architecture(query.device),
            splits=paged_plan.max_routes_per_row,
            workspace_bytes=paged_plan.workspace_bytes,
        )
        return StreamAttnEnginePlan(
            mode=STREAMATTN_MODE_VERIFIED_AUTO,
            backend=backend,
            reason=reason,
            model_id=None,
            layer_id=None,
            query=query,
            key_cache=cache,
            value_cache=None,
            service=self.service,
            attention_problem=tile_plan.problem,
            tile_plan=tile_plan,
            backend_plan=backend_plan,
            direct_runner=runner,
        )

    def plan_dynamic_selected_paged(
        self,
        query: torch.Tensor,
        cache: PagedKVCache,
        tile_plan: AttentionTilePlan,
        *,
        output: Optional[torch.Tensor] = None,
    ) -> StreamAttnEnginePlan:
        """Bind mutable GPU Q-head route atoms to the no-sync H100 executor."""

        routes = tile_plan.schedule.device_routes
        if routes is None:
            raise ValueError("dynamic selected paging requires device route CSR")
        paged_plan = PagedDynamicSelectedDecodePlan.build(
            query,
            cache,
            routes,
            output=output,
        )
        backend = paged_plan.backend
        reason = "distribution_verified_dynamic_qhead_paged"
        info = StreamAttnServingInfo(
            backend_used=backend,
            policy_id=tile_plan.policy_id,
            fallback_reason=None,
            batch_size=int(query.shape[0]),
            kv_len=max(tile_plan.problem.kv_lengths),
            layer_id=None,
            model_id=None,
            dtype=str(query.dtype).removeprefix("torch."),
            device=str(query.device),
            plan_backend=backend,
            plan_reason=reason,
            seed_only_enabled=False,
            safety_policy_matched=True,
            runtime_counters={
                "backend_counts": {backend: 1},
                "fallback_reasons": {},
            },
            stats={
                "layout": cache.normalized_layout,
                "page_size": cache.page_size,
                "source_route_nnz": routes.nnz,
                "max_routes_per_group": paged_plan.max_routes_per_group,
                "producer_ctas": paged_plan.producer_ctas,
                "metadata_bytes": paged_plan.metadata_bytes,
                "workspace_bytes": paged_plan.workspace_bytes,
                "route_preparation": (
                    "gpu_bounded_membership_warp_compaction_no_host_readback"
                ),
            },
        )
        runner = PagedDynamicSelectedDecodeRunner(plan=paged_plan, info=info)
        backend_plan = AttentionBackendPlan(
            backend=backend,
            reason=reason,
            architecture=device_architecture(query.device),
            splits=paged_plan.max_routes_per_group,
            workspace_bytes=paged_plan.workspace_bytes,
        )
        return StreamAttnEnginePlan(
            mode=STREAMATTN_MODE_VERIFIED_AUTO,
            backend=backend,
            reason=reason,
            model_id=None,
            layer_id=None,
            query=query,
            key_cache=cache,
            value_cache=None,
            service=self.service,
            attention_problem=tile_plan.problem,
            tile_plan=tile_plan,
            backend_plan=backend_plan,
            direct_runner=runner,
        )

    def plan_query_selected_paged(
        self,
        query: torch.Tensor,
        cache: PagedKVCache,
        *,
        selected_atoms: int = 6,
        sink_atoms: int = 1,
        recent_atoms: int = 1,
        support_width: int = 4,
        support_method: str = "centroid_extremes",
        refine_candidates: int = 0,
        policy_id: str = "query-selected-runtime-research",
        output: Optional[torch.Tensor] = None,
    ) -> StreamAttnEnginePlan:
        """Bind query-aware GPU selection to paged WGMMA selected decode.

        This is an explicit distribution-verified research entry point. The
        caller remains responsible for proving that its selector policy is
        admissible for the request distribution.
        """

        paged_plan = PagedQuerySelectedDecodePlan.build(
            query,
            cache,
            selected_atoms=selected_atoms,
            sink_atoms=sink_atoms,
            recent_atoms=recent_atoms,
            support_width=support_width,
            support_method=support_method,
            refine_candidates=refine_candidates,
            output=output,
        )
        problem = AttentionProblem.from_paged(
            query,
            cache,
            guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
        )
        rows = tuple(
            tuple(range(selected_atoms))
            for _ in range(cache.batch_size * int(query.shape[2]))
        )
        tile_plan = AttentionTilePlan.selected(
            problem,
            logical_tile_size=64,
            tile_ids_per_row=rows,
            policy_id=policy_id,
            reason="query_aware_support_function_proxy",
            route_granularity=ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
        )
        backend = paged_plan.backend
        reason = (
            "distribution_verified_query_refined_paged_research"
            if refine_candidates
            else "distribution_verified_query_selected_paged_research"
        )
        info = StreamAttnServingInfo(
            backend_used=backend,
            policy_id=policy_id,
            fallback_reason=None,
            batch_size=int(query.shape[0]),
            kv_len=max(problem.kv_lengths),
            layer_id=None,
            model_id=None,
            dtype=str(query.dtype).removeprefix("torch."),
            device=str(query.device),
            plan_backend=backend,
            plan_reason=reason,
            seed_only_enabled=False,
            safety_policy_matched=True,
            runtime_counters={
                "backend_counts": {backend: 1},
                "fallback_reasons": {},
            },
            stats={
                "layout": cache.normalized_layout,
                "page_size": cache.page_size,
                "selected_atoms_per_q_head": selected_atoms,
                "support_width": support_width,
                "support_method": support_method,
                "refine_candidates": refine_candidates,
                "support_metadata_bytes": paged_plan.support_metadata_bytes,
                "selector_workspace_bytes": paged_plan.selector_workspace_bytes,
                "route_preparation": (
                    "gpu_support_topk_exact_candidate_refine_membership_compaction_no_host_readback"
                    if refine_candidates
                    else "gpu_query_score_topk_membership_compaction_no_host_readback"
                ),
                "safety_scope": "caller_distribution_verified",
            },
        )
        runner = PagedQuerySelectedDecodeRunner(plan=paged_plan, info=info)
        workspace_bytes = (
            paged_plan.selector_workspace_bytes
            + paged_plan.selected_plan.workspace_bytes
        )
        backend_plan = AttentionBackendPlan(
            backend=backend,
            reason=reason,
            architecture=device_architecture(query.device),
            splits=paged_plan.selected_plan.max_routes_per_group,
            workspace_bytes=workspace_bytes,
        )
        return StreamAttnEnginePlan(
            mode=STREAMATTN_MODE_VERIFIED_AUTO,
            backend=backend,
            reason=reason,
            model_id=None,
            layer_id=None,
            query=query,
            key_cache=cache,
            value_cache=None,
            service=self.service,
            attention_problem=problem,
            tile_plan=tile_plan,
            backend_plan=backend_plan,
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
        policy = runner.policy
        problem = AttentionProblem.from_contiguous(
            query,
            key_cache,
            value_cache,
            guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
            cache_layout="NHD",
        )
        selected_blocks = fixed_block_tile_ids(
            kv_len=problem.max_kv_len,
            tile_size=int(policy.block_size),
            sink_tiles=int(policy.sink_blocks),
            recent_tiles=int(policy.recent_blocks),
            middle_tiles=int(policy.middle_seed_blocks),
            tile_order=str(policy.block_order),
        )
        tile_plan = AttentionTilePlan.selected(
            problem,
            logical_tile_size=int(policy.block_size),
            tile_ids_per_row=(selected_blocks,) * problem.batch_size,
            policy_id=str(policy.policy_id),
            reason="verified_policy_match",
        )
        backend_plan = AttentionBackendPlan(
            backend="gate0_seed_only_triton",
            reason="verified_policy_match",
            architecture=device_architecture(query.device),
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
            attention_problem=problem,
            tile_plan=tile_plan,
            backend_plan=backend_plan,
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
