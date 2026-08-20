# Contributing to StreamAttn

StreamAttn contains portable Python code, model-aware approximation policies,
and hardware-specific GPU kernels. A change is not ready to merge until its
claim type, correctness evidence, and performance evidence agree.

## Contribution workflow

1. Search existing issues before opening a new one.
2. Fork the repository and create a focused branch from `main`.
3. Add tests and update documentation for the behavior you change.
4. Run the portable checks locally.
5. Open a pull request using the repository template.
6. Resolve the required `CI / Required checks` result and maintainer review.

Do not include API keys, access tokens, private model artifacts, or credentials
in issues, logs, commits, or benchmark output.

## Required local checks

```bash
python -m pip install -e .[dev]
python benchmarks/check_seed_only_policy_route_smoke.py --allow-no-torch
python -m pytest -q
python -m build
python -m twine check dist/*
python benchmarks/check_wheel_contents.py dist/*.whl
```

The GitHub workflow repeats policy integrity, the CPU suite on Python 3.10 and
3.11, and source/wheel validation. GitHub-hosted CI does not claim GPU kernel
qualification.

## GPU and performance changes

CUDA, Triton, exact-native, and seed-only performance claims require an
owner-run GPU artifact in addition to portable CI. Include:

- GPU model, driver, CUDA, PyTorch, Triton, and baseline versions;
- `B`, `Hq`, `Hkv`, `N`, `D`, dtype, cache layout, and attention semantics;
- exact command, warmups, iterations, repeats, and paired-trial protocol;
- median, minimum paired ratio, and paired win count;
- numerical reference, tolerances, finite-output and mutation checks;
- the JSON artifact path or an attached artifact;
- confirmation that every paid GPU job was stopped after completion.

Never describe a seed-only result as an exact-attention victory. Never promote
one measured GPU cell into a universal device, model, or shape claim. New
serving cells must fail closed outside their measured policy/shape boundary.

## Policy changes

Every registry entry must reference a committed policy artifact. Its
`min_batch`, kernel-mode boundary, model/layer metadata, and policy ID must
match the artifact. Green policies require the safety and timing evidence
defined by their validation scope. Candidate policies must remain visibly
marked as candidates.

## Documentation

Update `README.md` whenever a contribution changes a promoted route, measured
speedup, supported shape, safety boundary, public API, or installation flow.
Detailed experiments should also have a dated document under `docs/` that
records negative controls and rejected paths.
