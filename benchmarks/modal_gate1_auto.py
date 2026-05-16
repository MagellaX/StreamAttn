"""Modal smoke test for Gate-1 auto routing and telemetry updates."""

import modal


app = modal.App("streamattn-gate1-auto")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("triton==3.1.0")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


@app.function(image=image, gpu="A10G", timeout=600)
def smoke():
    import sys

    sys.path.insert(0, "/root/StreamAttn")

    import torch

    from stream_attention import StreamAttnMetadataCache
    from stream_attention.gate1 import dense_attention_forward, make_route_request, stream_attn_gate1
    from stream_attention.router import CostEntry, CostKey, Gate1CostModel, StreamAttnPolicy, StreamAttnRouter
    from stream_attention.telemetry import ActiveFractionTelemetry

    torch.manual_seed(0)
    device = torch.device("cuda")

    q = torch.zeros(1, 1024, 4, 64, device=device, dtype=torch.float16)
    k = torch.zeros(1, 1024, 4, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 1024, 4, 64, device=device, dtype=torch.float16)
    q[..., 0] = 8.0
    k[:, :64, :, 0] = 8.0
    k[:, 64:, :, 0] = -8.0

    block_size = 64
    tile_size_q = 64
    metadata = StreamAttnMetadataCache.from_value(v, block_size=block_size)
    request = make_route_request(
        q,
        k,
        causal=False,
        block_size=block_size,
        tile_size_q=tile_size_q,
        model_id="modal-smoke",
        layer_id=0,
        head_id=-1,
    )

    cost_model = Gate1CostModel()
    cost_model.update(
        CostKey.from_request(request),
        CostEntry(dense_ms=0.095, qk_only_ms=0.050),
    )
    router = StreamAttnRouter(
        policy=StreamAttnPolicy(min_confidence=0.7, history_min_observations=4),
        telemetry=ActiveFractionTelemetry(min_observations=4),
        cost_model=cost_model,
    )
    for _ in range(4):
        router.observe(request, cta_pv_executed=64, cta_tiles_total=1024)

    out, info = stream_attn_gate1(
        q,
        k,
        v,
        causal=False,
        mode="auto",
        router=router,
        metadata=metadata,
        request=request,
        error_budget=1e-3,
        block_size=block_size,
        tile_size_q=tile_size_q,
        telemetry=True,
        return_info=True,
    )
    ref = dense_attention_forward(q, k, v, causal=False)
    torch.cuda.synchronize()

    profile = router.telemetry.get(request.active_key())
    return {
        "device": torch.cuda.get_device_name(0),
        "decision": {
            "backend": info.decision.backend,
            "reason": info.decision.reason,
            "predicted_active": info.decision.prediction.active_frac_hat,
            "confidence": info.decision.prediction.confidence,
        },
        "stats": None
        if info.stats is None
        else {
            "cta_tiles_total": info.stats.cta_tiles_total,
            "cta_pv_executed": info.stats.cta_pv_executed,
            "cta_pv_skipped": info.stats.cta_pv_skipped,
            "active_pv_fraction": info.stats.active_pv_fraction,
        },
        "telemetry_count": None if profile is None else profile.count,
        "telemetry_p90": None if profile is None else profile.p90,
        "max_abs_error": torch.max(torch.abs(out - ref)).item(),
    }


@app.local_entrypoint()
def main():
    print(smoke.remote())
