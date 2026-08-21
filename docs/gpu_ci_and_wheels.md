# GPU CI and Wheel Strategy

StreamAttn separates package construction, CUDA compilation, GPU correctness,
and performance evidence. These are different claims and must not be collapsed
into a single green check.

## Wheel model

The current package contains Python, Triton JIT kernels, CUDA extension source,
and policy JSON. It does not embed a compiled CUDA extension in the wheel. The
build therefore produces one portable artifact:

```text
stream_attention-<version>-py3-none-any.whl
```

That wheel is appropriate for CPU installations and NVIDIA GPU installations.
The installed PyTorch/Triton/CUDA environment determines runtime support, and
the SM90 exact backend compiles its specialized extension during planning.
Separate A100, H100, or B200 wheels would not improve compatibility under this
source/JIT architecture.

If StreamAttn later ships precompiled native extensions, use Linux manylinux
wheels with explicit CUDA build variants and a deliberate architecture list.
Do not relabel a binary extension as `py3-none-any`.

## No-cost automatic checks

Public repositories can use standard GitHub-hosted CPU runners without paying
for execution. The required `CI` workflow and the path-triggered
`.github/workflows/gpu-source.yml` workflow use those runners for:

1. importing every shipped Triton module with Triton enabled;
2. checking the generated D64 and D128 SM90 source contracts;
3. compiling the SM90 exact and TMA extensions with `nvcc`/`ptxas` for
   `sm_90a` inside a CUDA development container;
4. never executing a kernel or claiming hardware correctness.

The offline build catches Python import failures, Triton import/API breakage,
C++/CUDA syntax errors, CUTLASS incompatibility, and SM90 code-generation
failures without requiring a GPU.

## Runtime correctness

Standard GitHub-hosted runners do not include an NVIDIA CUDA device. GitHub GPU
larger runners are billed even for public repositories. Consequently, a no-cost
central workflow cannot honestly execute StreamAttn CUDA kernels today.

Contributors changing GPU behavior must run the repository-owned harness on a
GPU they control:

```bash
python -m pip install -e .[dev,triton]
python benchmarks/run_gpu_correctness_ci.py
```

For the promoted H100 exact-native path:

```bash
python benchmarks/run_gpu_correctness_ci.py \
  --require-sm90 \
  --cutlass-root /path/to/FlashMLA-ETAP/csrc/cutlass
```

Attach the complete output to the pull request. The harness fails immediately
when CUDA or Triton is unavailable, preventing a skipped test suite from being
reported as a GPU pass.

## Evidence levels

| Evidence | Automatic | Requires GPU | Merge meaning |
| --- | --- | --- | --- |
| Universal wheel and metadata | Yes | No | Package is structurally installable |
| Triton source contract | Yes | No | GPU Python modules import with Triton |
| SM90 offline CUDA build | Yes | No | H100 source compiles for `sm_90a` |
| GPU correctness harness | Contributor evidence | Yes | Selected kernels match references |
| Performance gate | Maintainer/research evidence | Yes | A measured shape beats its baseline |

Runtime and performance evidence remain mandatory for GPU behavior and speed
claims. An offline compile pass is not a substitute for either.

## Future donated runner

If the project receives a donated ephemeral GPU runner, connect the same
`run_gpu_correctness_ci.py` harness rather than introducing a second test list.
Fork pull requests must remain approval-gated because they execute untrusted
code. A persistent self-hosted machine must not execute arbitrary public PRs.

References:

- GitHub Actions billing: https://docs.github.com/en/billing/concepts/product-billing/github-actions
- NVIDIA GPU runner guidance: https://docs.nvidia.com/datascience/deployment/stable/developer/ci/github-actions/
- PyTorch CUDA extension architecture targeting: https://docs.pytorch.org/docs/main/cpp_extension.html
- Python wheel compatibility tags: https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/
