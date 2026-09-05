"""Direct HND exact baseline adapters and v2 measurement resolution.

Backend imports are lazy. Missing binaries and rejected runtime configurations
remain explicit evidence, never a silent substitution with another backend.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
from dataclasses import asdict

import torch

from stream_attention.baseline_resolver import (
    ExactBaselineDescriptor,
    ExactBaselineMeasurement,
    fastest_measured_exact_baseline,
    resolve_direct_exact_baselines,
)
from stream_attention.inference_workload import AttentionBatchV2


BASELINE_IDS = (
    "torch_flash",
    "flashinfer_fa2",
    "flashinfer_fa3",
    "flashattention3",
    "cudnn",
    "cutlass_xformers",
)

FLASHINFER_WORKSPACE_BYTES = 256 * 1024 * 1024


def requested_baselines(names=None):
    names = tuple(BASELINE_IDS if names is None else names)
    if not names or len(set(names)) != len(names) or set(names) - set(BASELINE_IDS):
        raise ValueError("requested baselines must be a nonempty unique known subset")
    return names


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def baseline_versions():
    return {
        "torch_flash": torch.__version__,
        "flashinfer_fa2": package_version("flashinfer-python"),
        "flashinfer_fa3": package_version("flashinfer-python"),
        "flashattention3": package_version("flash_attn_3"),
        "cudnn": f"torch={torch.__version__};cudnn={torch.backends.cudnn.version()}",
        "cutlass_xformers": package_version("xformers"),
    }


def workload_for_case(c):
    phase = "verify" if c["m"] <= 8 else "micro_prefill"
    hk = c["hq"] // c["g"]
    return AttentionBatchV2.from_dict(
        {
            "batch_id": "micro_prefill_"
            + hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest(),
            "architecture": "sm90",
            "phase": phase,
            "requests": [
                dict(request_id=str(b), phase=phase, query_len=c["m"], kv_len=c["n"])
                for b in range(c["batch"])
            ],
            "attention_kind": "mqa" if hk == 1 else "gqa",
            "q_heads": c["hq"],
            "kv_heads": hk,
            "d_qk": c["d"],
            "d_v": c["d"],
            "q_dtype": "bf16",
            "kv_dtype": "bf16",
            "output_dtype": "bf16",
            "accumulator_dtype": "fp32",
            "scale_format": "scalar_fp32",
            "cache_kind": "contiguous",
            "cache_layout": "hnd",
            "mask_kind": "noncausal",
            "execution_mode": "cuda_graph",
            "maximum_captured_batch": c["batch"],
            "objective": "latency",
        }
    )


def descriptors(versions, requested=None):
    implementations = {
        "torch_flash": "torch.sdpa.forced_flash",
        "flashinfer_fa2": "flashinfer.BatchPrefillWithRaggedKVCacheWrapper.fa2.per_request_hnd",
        "flashinfer_fa3": "flashinfer.BatchPrefillWithRaggedKVCacheWrapper.fa3.per_request_strided_nhd_view",
        "flashattention3": "flash_attn_interface._flash_attn_forward.v2.8.3",
        "cudnn": "torch.sdpa.forced_cudnn",
        "cutlass_xformers": "xformers.ops.fmha.cutlass.FwOp",
    }
    return tuple(
        ExactBaselineDescriptor.from_dict(
            dict(
                baseline_id=name,
                implementation=implementations[name],
                revision=versions[name],
                architectures=["sm90"],
                phases=["verify", "micro_prefill"],
                attention_kinds=["gqa", "mqa"],
                q_dtypes=["bf16"],
                kv_dtypes=["bf16"],
                output_dtypes=["bf16"],
                d_qk=[64, 128],
                d_v=[64, 128],
                scale_formats=["scalar_fp32"],
                cache_kinds=["contiguous"],
                cache_layouts=["hnd"],
                mask_kinds=["noncausal"],
                execution_modes=["cuda_graph"],
                direct_layout=True,
            )
        )
        for name in requested_baselines(requested)
    )


def flashinfer_runner(q, k, v, backend):
    """Plan one direct HND request per batch element; time all launches."""
    import flashinfer

    b, m, hq, _ = q.shape
    hk, n, d = k.shape[1:]
    out = torch.empty_like(q)
    qi = torch.tensor([0, m], device=q.device, dtype=torch.int32)
    ki = torch.tensor([0, n], device=q.device, dtype=torch.int32)
    workspace = torch.empty(FLASHINFER_WORKSPACE_BYTES, device=q.device, dtype=torch.uint8)
    wrappers = []
    for _ in range(b):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            kv_layout="NHD" if backend == "fa3" else "HND",
            use_cuda_graph=True,
            qo_indptr_buf=qi.clone(),
            kv_indptr_buf=ki.clone(),
            backend=backend,
        )
        wrapper.plan(
            qi, ki, hq, hk, d, causal=False, q_data_type=q.dtype, kv_data_type=k.dtype
        )
        wrappers.append(wrapper)
    # FI 0.6.13's SM90 ragged C++ reads k.size(1) as the head count
    # even for HND. Logical NHD views fix that metadata without moving KV.
    # Its TMA path consumes the real strides, retaining physical HND storage.
    inputs = [
        (q[i], k[i].transpose(0, 1), v[i].transpose(0, 1), out[i])
        if backend == "fa3"
        else (q[i], k[i], v[i], out[i])
        for i in range(b)
    ]

    def run():
        for wrapper, (qb, kb, vb, ob) in zip(wrappers, inputs):
            wrapper.run(qb, kb, vb, out=ob)
        return out

    return run


def prepare_baselines(q, k, v, requested=None):
    from benchmarks.profile_sm90_micro_prefill import _flash_sdpa
    from benchmarks.micro_prefill_optional_baselines import (
        cudnn_runner,
        cutlass_runner,
        fa3_runner,
    )

    factories = {
        "torch_flash": lambda: lambda: _flash_sdpa(q, k, v),
        "flashinfer_fa2": lambda: flashinfer_runner(q, k, v, "fa2"),
        "flashinfer_fa3": lambda: flashinfer_runner(q, k, v, "fa3"),
        "flashattention3": lambda: fa3_runner(q, k, v),
        "cudnn": lambda: cudnn_runner(q, k, v),
        "cutlass_xformers": lambda: cutlass_runner(q, k, v),
    }
    runners, unavailable = {}, {}
    for name in requested_baselines(requested):
        factory = factories[name]
        print(json.dumps(dict(stage="baseline_prepare", runner=name)), flush=True)
        try:
            runners[name] = factory()
        except Exception as exc:
            unavailable[name] = f"prepare:{type(exc).__name__}: {exc}"
    return runners, unavailable


def resolve_measurements(
    c, versions, environment_sha256, medians, accuracy, mutation, requested=None,
    loaded_binary_provenance=None,
):
    requested = requested_baselines(requested)
    versions = measurement_versions(versions, loaded_binary_provenance)
    workload = workload_for_case(c)
    registry = descriptors(versions, requested)
    measurements = [
        ExactBaselineMeasurement(
            baseline_id=name,
            backend_revision=versions[name],
            workload_sha256=workload.fingerprint,
            environment_sha256=environment_sha256,
            latency_us=medians[name] * 1000,
            correctness_passed=accuracy.get(name, {}).get("passed", False)
            and mutation.get(name, {}).get("passed", False),
            graph_replay=True,
        )
        for name in requested
        if name in medians and versions[name] not in ("not_installed", "unresolved", "")
    ]
    winner = fastest_measured_exact_baseline(
        workload,
        registry,
        measurements,
        expected_environment_sha256=environment_sha256,
    )
    return dict(
        workload=workload.as_dict(),
        workload_sha256=workload.fingerprint,
        eligibility=[
            asdict(row) for row in resolve_direct_exact_baselines(workload, registry)
        ],
        measurements=[asdict(row) for row in measurements],
        winner=asdict(winner) if winner else None,
        complete=all(
            any(row.baseline_id == name and row.correctness_passed for row in measurements)
            for name in BASELINE_IDS
        ),
    )


def measurement_versions(versions, evidence=None):
    if evidence is None:
        return versions
    native_resolved = all(evidence.get(n, {}).get("resolved") is True
                          for n in ("natural", "transposed"))
    result = dict(versions)
    for name in BASELINE_IDS:
        if (not native_resolved or evidence.get(name, {}).get("resolved") is not True
                or versions[name] in ("not_installed", "unresolved", "")):
            result[name] = "unresolved"
        else:
            digest = hashlib.sha256(json.dumps(
                {n: evidence[n] for n in (name, "natural", "transposed")},
                sort_keys=True, allow_nan=False,
            ).encode()).hexdigest()
            result[name] = f"{versions[name]};loaded={digest}"
    return result


def file_identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed during hashing: {path}")
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def loaded_binary_provenance(name, *, extension=None, cache=None):
    """Inspect already-loaded modules/libraries only; never import a backend.

    /proc mappings include dlopen/JIT libraries absent from sys.modules. Hashing
    is outside timing and cached by file identity across cases in this worker.
    """
    prefixes = {
        "torch_flash": ("torch",), "cudnn": ("torch",),
        "flashinfer_fa2": ("flashinfer",), "flashinfer_fa3": ("flashinfer",),
        "flashattention3": ("flash_attn_interface", "flash_attn_3_cuda"),
        "cutlass_xformers": ("xformers",),
    }
    modules, binaries, failures = {}, {}, []
    cache = {} if cache is None else cache

    def identity(path):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = file_identity(path)
        return cache[key]

    def binary(path):
        return any(s in Path(path).name.lower() for s in (".so", ".pyd", ".dll", ".cubin"))

    selected = ({name: extension} if extension is not None else {
        key: mod for key, mod in tuple(sys.modules.items())
        if any(key == p or key.startswith(p + ".") for p in prefixes[name])
    })
    for key, module in selected.items():
        # These are synthetic operator/class namespaces, not file-backed modules.
        # torch._ops and torch._classes are real modules and remain in the census.
        if key in ("torch.ops", "torch.classes") or key.startswith(("torch.ops.", "torch.classes.")):
            continue
        path = getattr(module, "__file__", None)
        if path:
            try:
                entry = identity(path)
                modules[key] = entry
                if binary(path):
                    binaries[entry["path"]] = entry
            except (OSError, ValueError) as exc:
                failures.append(f"{key}: {type(exc).__name__}: {exc}")
    paths = set(torch.ops.loaded_libraries)
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                paths.add(fields[5])
    tokens = {
        "torch_flash": ("torch",), "cudnn": ("torch", "cudnn"),
        "flashinfer_fa2": ("flashinfer",), "flashinfer_fa3": ("flashinfer",),
        "flashattention3": ("flash_attn",), "cutlass_xformers": ("xformers", "torch"),
    }.get(name, ())
    for path in sorted(paths):
        if binary(path) and any(token in str(path).lower() for token in tokens):
            try:
                entry = identity(path)
                binaries[entry["path"]] = entry
            except (OSError, ValueError) as exc:
                failures.append(f"{path}: {type(exc).__name__}: {exc}")
    if name == "cutlass_xformers":
        # cutlassF-pt executes aten::_efficient_attention_forward in Torch's
        # CUDA library, not in xFormers' registration extension.
        cuda_libraries = {"libtorch_cuda.so", "libtorch_cuda_cu.so", "torch_cuda.dll"}
        if not any(
            Path(path).name.lower() in cuda_libraries
            or Path(path).name.lower().startswith("libtorch_cuda.so.")
            for path in binaries
        ):
            failures.append("executing ATen CUDA library is absent from loaded binary evidence")
    return dict(modules=modules, binaries=[binaries[p] for p in sorted(binaries)],
                errors=failures, resolved=bool(modules and binaries and not failures))


def distribution_identity(name):
    """Hash installed interfaces and the wheel's recorded file identities.

    RECORD identifies packaged binaries without importing CUDA extensions. It
    is not a re-hash or authenticity check of every installed binary.
    """
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return dict(version="not_installed", record_sha256=None, interfaces={})
    record = dist.read_text("RECORD")
    targets = {
        "flashinfer/prefill.py",
        "flash_attn_interface.py",
        "xformers/ops/fmha/cutlass.py",
        "torch/nn/attention/__init__.py",
    }
    interfaces = {}
    for path in dist.files or ():
        if str(path).replace("\\", "/") in targets:
            interfaces[str(path)] = hashlib.sha256(
                dist.locate_file(path).read_bytes()
            ).hexdigest()
    return dict(
        version=dist.version,
        record_sha256=hashlib.sha256(record.encode()).hexdigest() if record else None,
        interfaces=interfaces,
    )


def runtime_provenance():
    """Keep implementation identity separate from the benchmark adapter hash."""
    packages = {
        name: distribution_identity(name)
        for name in (
            "torch",
            "flashinfer-python",
            "flashinfer-cubin",
            "flash_attn_3",
            "xformers",
        )
    }
    package_versions = baseline_versions()
    dependencies = {
        "torch_flash": ("torch",),
        "cudnn": ("torch",),
        "flashinfer_fa2": ("flashinfer-python", "flashinfer-cubin"),
        "flashinfer_fa3": ("flashinfer-python", "flashinfer-cubin"),
        "flashattention3": ("flash_attn_3",),
        "cutlass_xformers": ("xformers", "torch"),
    }
    versions = {}
    for name, deps in dependencies.items():
        if package_versions[name] == "not_installed" or any(
            not packages[p]["record_sha256"] for p in deps
        ):
            versions[name] = "unresolved"
        else:
            digest = hashlib.sha256(
                json.dumps(
                    {p: packages[p] for p in deps},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            versions[name] = f"{package_versions[name]};installation={digest}"
    return dict(
        versions=versions,
        package_versions=package_versions,
        packages=packages,
        adapter_settings=dict(
            flashinfer_workspace_bytes=FLASHINFER_WORKSPACE_BYTES,
            flashinfer_workspace_allocation="prepare_only_shared_by_sequential_requests",
        ),
        adapter_sha256=hashlib.sha256(
            inspect.getsource(inspect.getmodule(runtime_provenance)).encode()
        ).hexdigest(),
    )
