"""Modal smoke test for the Gate-1 Triton forward kernel."""

import modal


app = modal.App("streamattn-gate1-smoke")

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

    from stream_attention.certified import certified_attention
    from stream_attention.kernels.gate1_fwd_triton import (
        build_value_norm_bounds,
        gate1_attention_triton_forward,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")

    def summarize_stats(raw_stats):
        totals = raw_stats.detach().sum(dim=(0, 1, 2)).cpu()
        return {
            "row_skips": int(totals[0].item()),
            "row_computes": int(totals[1].item()),
            "cta_tiles_total": int(totals[2].item()),
            "cta_pv_skipped": int(totals[3].item()),
            "cta_pv_executed": int(totals[4].item()),
            "force_mode_sum": int(totals[5].item()),
        }

    def run_case(name, q, k, v, budget, force_mode=0, compare=True):
        ref = None
        if compare:
            ref = certified_attention(
                q,
                k,
                v,
                causal=False,
                error_budget=budget,
                block_size=16,
                enable_summary_gate=False,
                enable_post_qk_gate=True,
                skip_predicate="value_bound",
            )
        bounds = build_value_norm_bounds(v, block_size=16)
        out, raw_stats = gate1_attention_triton_forward(
            q,
            k,
            v,
            causal=False,
            error_budget=budget,
            block_size=16,
            tile_size_q=16,
            value_norm_bounds=bounds,
            skip_predicate="value_bound",
            force_mode=force_mode,
            return_raw_stats=True,
        )
        torch.cuda.synchronize()
        result = {
            "name": name,
            "stats": summarize_stats(raw_stats),
            "output_l2": torch.linalg.vector_norm(out.float()).item(),
        }
        if ref is not None:
            result["max_abs_error"] = torch.max(torch.abs(out - ref)).item()
        return result

    q = torch.randn(1, 64, 2, 32, device=device, dtype=torch.float16)
    k = torch.randn(1, 64, 2, 32, device=device, dtype=torch.float16)
    v = torch.randn(1, 64, 2, 32, device=device, dtype=torch.float16)
    exact_case = run_case("random_exact", q, k, v, 0.0)

    q2 = torch.zeros(1, 64, 2, 32, device=device, dtype=torch.float16)
    k2 = torch.zeros(1, 64, 2, 32, device=device, dtype=torch.float16)
    v2 = torch.randn(1, 64, 2, 32, device=device, dtype=torch.float16)
    q2[..., 0] = 8.0
    k2[:, :16, :, 0] = 8.0
    k2[:, 16:, :, 0] = -8.0
    skip_case = run_case("peaked_gate1", q2, k2, v2, 1e-3)
    forced_skip_case = run_case(
        "peaked_late_force_all_skip_after_predicate",
        q2,
        k2,
        v2,
        1e-3,
        force_mode=2,
        compare=False,
    )
    early_skip_case = run_case(
        "peaked_early_force_all_skip_after_qk",
        q2,
        k2,
        v2,
        1e-3,
        force_mode=4,
        compare=False,
    )

    def time_bounds(v, block_size, iters=20):
        for _ in range(3):
            build_value_norm_bounds(v, block_size=block_size)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            build_value_norm_bounds(v, block_size=block_size)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def time_gate1(
        q,
        k,
        v,
        budget,
        force_mode=0,
        value_norm_bounds=None,
        skip_predicate="value_bound",
        iters=20,
    ):
        for _ in range(5):
            gate1_attention_triton_forward(
                q,
                k,
                v,
                causal=False,
                error_budget=budget,
                block_size=64,
                tile_size_q=64,
                value_norm_bounds=value_norm_bounds,
                skip_predicate=skip_predicate,
                force_mode=force_mode,
                return_raw_stats=False,
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            gate1_attention_triton_forward(
                q,
                k,
                v,
                causal=False,
                error_budget=budget,
                block_size=64,
                tile_size_q=64,
                value_norm_bounds=value_norm_bounds,
                skip_predicate=skip_predicate,
                force_mode=force_mode,
                return_raw_stats=False,
            )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    qb = torch.zeros(1, 1024, 4, 64, device=device, dtype=torch.float16)
    kb = torch.zeros(1, 1024, 4, 64, device=device, dtype=torch.float16)
    vb = torch.randn(1, 1024, 4, 64, device=device, dtype=torch.float16)
    qb[..., 0] = 8.0
    kb[:, :64, :, 0] = 8.0
    kb[:, 64:, :, 0] = -8.0
    bounds_build_ms = time_bounds(vb, block_size=64)
    bounds = build_value_norm_bounds(vb, block_size=64)
    torch.cuda.synchronize()

    contaminated_wrapper_ms = time_gate1(qb, kb, vb, 1e-3, force_mode=0)
    exact_ms = time_gate1(qb, kb, vb, 0.0, force_mode=0, value_norm_bounds=bounds)
    pred_no_skip_ms = time_gate1(
        qb, kb, vb, 1e-3, force_mode=1, value_norm_bounds=bounds
    )
    skip_ms = time_gate1(qb, kb, vb, 1e-3, force_mode=0, value_norm_bounds=bounds)
    late_all_skip_ms = time_gate1(
        qb, kb, vb, 1e-3, force_mode=2, value_norm_bounds=bounds
    )
    late_all_compute_ms = time_gate1(
        qb, kb, vb, 1e-3, force_mode=3, value_norm_bounds=bounds
    )
    early_all_skip_ms = time_gate1(qb, kb, vb, 1e-3, force_mode=4)
    dense_no_predicate_ms = time_gate1(qb, kb, vb, 1e-3, force_mode=5)
    qk_only_ms = time_gate1(
        qb,
        kb,
        vb,
        1e-3,
        force_mode=7,
        skip_predicate="mass",
    )
    mass_normal_ms = time_gate1(
        qb,
        kb,
        vb,
        1e-3,
        force_mode=0,
        skip_predicate="mass",
    )

    _, timing_stats = gate1_attention_triton_forward(
        qb,
        kb,
        vb,
        causal=False,
        error_budget=1e-3,
        block_size=64,
        tile_size_q=64,
        value_norm_bounds=bounds,
        skip_predicate="value_bound",
        force_mode=0,
        return_raw_stats=True,
    )
    torch.cuda.synchronize()

    return {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "cases": [exact_case, skip_case, forced_skip_case, early_skip_case],
        "timing": {
            "shape": "B1_S1024_H4_D64",
            "bounds_build_ms": bounds_build_ms,
            "contaminated_wrapper_ms": contaminated_wrapper_ms,
            "exact_ms": exact_ms,
            "predicate_no_skip_ms": pred_no_skip_ms,
            "gate1_ms": skip_ms,
            "late_force_all_skip_ms": late_all_skip_ms,
            "late_force_all_compute_ms": late_all_compute_ms,
            "early_force_all_skip_ms": early_all_skip_ms,
            "dense_no_predicate_ms": dense_no_predicate_ms,
            "qk_only_mass_ms": qk_only_ms,
            "mass_normal_ms": mass_normal_ms,
            "speedup": exact_ms / skip_ms,
            "early_force_all_skip_speedup": dense_no_predicate_ms / early_all_skip_ms,
            "qk_only_speedup": dense_no_predicate_ms / qk_only_ms,
            "normal_stats": summarize_stats(timing_stats),
        },
    }


@app.local_entrypoint()
def main():
    print(smoke.remote())
