"""Modal runner for the Gate-1 profiler suite."""

import argparse
import modal


app = modal.App("streamattn-gate1-profile")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("triton==3.1.0")
    .add_local_dir(".", remote_path="/root/StreamAttn", copy=True)
)


def _profile_active_curve():
    import sys

    sys.path.insert(0, "/root/StreamAttn")

    from benchmarks.profile_gate1_kernel import _run_suite

    base = argparse.Namespace(
        batch=1,
        seq_q=1024,
        seq_k=1024,
        heads=4,
        dim=64,
        dtype="fp16",
        pattern="peaked",
        peak=8.0,
        active_fraction=None,
        active_blocks=None,
        causal=False,
        error_budget=1e-3,
        block_size=64,
        tile_size_q=64,
        skip_predicate="value_bound",
        bounds_builder="triton",
        force_mode=0,
        precompute_bounds=True,
        return_stats=True,
        suite=True,
        sweep_active_fracs=None,
        cost_json_out=None,
        warmup=20,
        iters=80,
    )

    results = []
    for fraction in [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]:
        args = argparse.Namespace(**vars(base))
        args.active_fraction = fraction
        results.append(_run_suite(args))
    return results


@app.function(image=image, gpu="A10G", timeout=900)
def profile_active_curve_a10g():
    return _profile_active_curve()


@app.function(image=image, gpu="A100", timeout=900)
def profile_active_curve_a100():
    return _profile_active_curve()


@app.function(image=image, gpu="H100", timeout=900)
def profile_active_curve_h100():
    return _profile_active_curve()


@app.local_entrypoint()
def main(target: str = "a10g"):
    if target == "a10g":
        print(profile_active_curve_a10g.remote())
    elif target == "a100":
        print(profile_active_curve_a100.remote())
    elif target == "h100":
        print(profile_active_curve_h100.remote())
    else:
        raise ValueError("target must be a10g, a100, or h100")
