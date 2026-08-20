## Summary

Describe the problem and the smallest coherent change that solves it.

## Claim type (select exactly one)

- [ ] Portable code, tests, documentation, or packaging
- [ ] Exact attention kernel
- [ ] Approximate/model-validated attention policy
- [ ] Performance-only benchmark or research probe

## Evidence

- [ ] `python -m pytest -q` passes locally
- [ ] Policy registry smoke passes, or this change does not affect policies
- [ ] Package validation passes, or this change does not affect packaging
- [ ] New behavior has focused tests
- [ ] README/docs reflect changed public claims or boundaries
- [ ] No secrets, private artifacts, or credentials are included

For GPU/performance changes, provide the device, software versions, complete
shape, exact command, timing protocol, correctness tolerance, paired baseline,
and artifact path. State whether the result is exact or approximate.

## Risk and fallback

Describe unsupported shapes, expected failure behavior, and how the route fails
closed. List any paid GPU jobs used and confirm they were stopped.
