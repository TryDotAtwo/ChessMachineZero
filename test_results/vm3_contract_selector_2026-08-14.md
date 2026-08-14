# VM3 contract and selector evidence — 2026-08-14

## Passing evidence

- `python -m pytest -q`: 37 passed.
- `tests/test_vm3_source_purity.py`: 14 passed, including adversarial mutations
  for scalar reads, CPU transfer, detach, dynamic routing, semantic branches,
  unallowlisted operations, unregistered custom autograd and compiler linkage.
- Native `cmz_vm3_contract`, `cmz_vm3_program_loader` and
  `cmz_vm3_st_selector` build and pass in `cmz-native-dev:2026-05-26`.
- Loader negative cases reject forged selector bounds, overlapping enabled and
  disabled ranges, mismatched proof claims, SHA-256 corruption, proof cycles and
  invalid sentinels. The valid fixture replays `FrozenMatmul` and restores an
  ordered stage/block graph.
- Selector tests prove exact hard forward, lowest-index ties, sentinel routing,
  finite outputs and gradients equal to the explicit eligibility-weighted soft
  surrogate at `1e-12` absolute tolerance.

## Pending evidence

- `docker/vm3/Dockerfile` is pinned to the specified CUDA base and wheel hash and
  now checks non-empty TorchConfig, public headers and libtorch. A previous build
  was rejected after inspection found 1431 zero-length wheel files caused by a
  full Docker VHD. After cache cleanup, Docker VM access to Docker Hub repeatedly
  fails with TLS handshake timeout, while the same auth endpoint is reachable
  from the Windows host. Therefore the clean pinned-image CI gate is not yet
  accepted and no GPU-residency claim is made.
