# Native audit correction acceptance, 2026-08-31

Fresh current-source direct Windows build passed all15 native subprocesses.
This is position reconstruction and generic primitive evidence, **not** a
complete recurrent chess environment or learning result.

## Reproduce

From the repository root, with CUDA-enabled Torch installed:

```powershell
python tests/export_native_fixtures.py build/task2-fixtures
./scripts/build_native.ps1 -CudaArchitecture 86 -BuildDirectory build/task2-green -FixtureDirectory build/task2-fixtures -RunTests -AllowUnsupportedCompiler
python -m pytest -q
```

Use different new directory names if these exist. The build script was also
tested to reject existing output directories before compiler discovery.

Environment: RTX3070 Laptop, CC8.6, 8192MiB, driver572.70; Python3.11.5;
Torch2.8.0+cu128; MSVC19.44 / tools14.44.35207; Windows SDK10.0.26100.0;
CUDA Toolkit12.5 / NVCC12.5.82. Unsupported compiler opt-in was explicit.
Torch CUDA registration was retained through the Windows linker anchor
`/INCLUDE:?warp_size@cuda@at@@YAHXZ`. Existing Windows short paths bypassed
NVCC's ANSI corruption of Cyrillic source/output paths.

## Results

- Fresh build script exit0; all15 native process exits checked as0.
- Artifact and generic recurrent tensor/gradient invariant executables passed.
- Malformed graph rejection:18/18 cases, tested with impossible cuda:127 so
  only the expected metadata error before any device access counts as passing.
  Includes dynamic K with2 rows and candidate2, duplicates, Q/K/V shapes,
  row routes/stride, projections, broadcasts, FFN, masks, expansion and concat.
- Hardmax: exact masked tie/gradient; empty CPU/CUDA mask row rejected;
  zero/negative/NaN/infinite temperatures rejected.
- FP4: exact finite-extreme saturation and identity-surrogate unit upstream
  gradient, including saturated elements; invalid scales rejected.
- Attention: every Q/K/V gradient matched independent scalar-double formulas
  for unbatched, shared-key batched and dynamic batched selected-top3 softmax.
  Shuffled candidates exercise original-index ties. Batch-isolation variants
  assert exactly zero unused-batch gradients. FP32 compute, rtol2e-5/atol2e-6;
  hard winners exact. Non-default CUDA streams used with input uploads ordered
  on that stream.
- Four corrected-kernel invalid-index children (negative/large, ordinary/
  batched) observed device assertion before key dereference. Empty-mask and
  index children intentionally print assertion diagnostics; those are expected,
  not a pristine-stderr claim.
- Full native board:52/52 exact FP32 one-hot states, **TF32 explicitly disabled**.
  Includes seed1 400 legal nonterminal plies and sensitive prefixes, legal
  castling/en-passant, both-color short promotions and promotion prefixes in the
  long counterexample. The native executable reads test-only oracle fixtures;
  no production Python/chess calls are introduced.
- Python suite before independent site edits: `123 passed in 18.44s`, exit0.
- All captured native build input SHA256 values still matched after completion.
  No cmz test process remained at GPU handoff to controller.

## RED to GREEN

Before corrections, a fresh baseline `build/task2-red2` had17 graph rejection
failures (stride-zero already rejected), accepted empty masks on CPU and CUDA,
accepted positive infinity temperature/scale, and failed
`FP4 finite extremes must saturate`. Direct attention accepted all three bad
metadata cases (nonfinite temperature, rank-one values, FP64 values). All these
cases pass in GREEN. Existing QKV derivatives passed the stronger independent
formulas even on the baseline, so derivative arithmetic was not changed.

Old unsafe OOB kernels were deliberately **not executed** for RED: the unchecked
read was established by source audit. Only guarded kernels were run with bad
indices. This avoids invoking undefined device memory accesses for test ritual.

## Evidence identities

Generated files live under ignored build/; the script records commands and
source hashes and requires rebuilding rather than accepting stale binaries.

| File | SHA256 |
|---|---|
| build/task2-green/build.log | c214756fe3e26191b6411a9af525ff2cf4ce16319e692e427773c47e25f92d1b |
| build/task2-green/build-manifest.json | 65540d33b5515079553b5727984899b691228732db7c500aed55a4269593052c |
| build/task2-fixtures/position.cmz | b2405abab2073dbd54d4879495563c8570cc12a846a470fe59449124d48414c6 |
| build/task2-fixtures/positions.bin | c4cf40b28ee1996a8b91897add565621e5014d661be4a503592f4fdc35b92e33 |

## Boundaries

CMake/CTest registration was updated but this environment's CMake/Ninja
probe/child-exit behavior remains unverified; the direct PowerShell route is
the executed build evidence. No CMake success is inferred from it.

The old position test's manually constructed cases and nonzero-only
RESULT_PIECE-gradient assertion were replaced by exact legal full-board forward
tests under NoGradGuard. End-to-end position-gradient acceptance is therefore
**not covered**; primitive derivative and generic recurrent checks are separate.
The extra nonfinite-score kernel guard was not independently exercised in this
build. Final controller rerun/memcheck and review are separate evidence.

No native FP4/FP16/BF16/TF32 equivalence, allocation-free/sublinear performance,
throughput improvement, multi-GPU behavior, higher-order gradients, or useful
player learning is claimed.
