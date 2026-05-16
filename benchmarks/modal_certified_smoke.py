"""Modal smoke test for the experimental Triton certified forward kernel."""

import modal


app = modal.App("streamattn-certified-smoke")

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
    from stream_attention.kernels.certified_fwd_triton import (
        certified_attention_triton_forward,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")

    def run_case(name, q, k, v, budget):
        ref = certified_attention(
            q,
            k,
            v,
            causal=False,
            error_budget=0.0,
            block_size=16,
        )
        out, raw_stats = certified_attention_triton_forward(
            q,
            k,
            v,
            causal=False,
            error_budget=budget,
            block_size=16,
            tile_size_q=16,
            skip_predicate="value_bound",
            return_raw_stats=True,
        )
        torch.cuda.synchronize()
        return {
            "name": name,
            "max_abs_error": torch.max(torch.abs(out - ref)).item(),
            "raw_stats": raw_stats.detach().cpu().tolist(),
        }

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
    skip_case = run_case("peaked_certified", q2, k2, v2, 1e-3)

    return {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "cases": [exact_case, skip_case],
    }


@app.local_entrypoint()
def main():
    print(smoke.remote())
