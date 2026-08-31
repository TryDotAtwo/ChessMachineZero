# Full-VM foundation and legal-set acceptance — 2026-08-31

Scope: inline implementation, no subagents. The full recurrent VM and its
website walkthrough are **not complete**. This report covers generic operators,
frozen circuit construction and a legal-set subgraph only.

## Implemented

- Wire11 MATRIX_TRANSPOSE,12 MATRIX_RESHAPE and13 MATRIX_MATMUL; native static
  schema/rank/product/contraction validation before CUDA; batch cannot move.
- Offline Circuit lowers Boolean gates and selection to projections, residuals,
  hardmax and generic layout. Constants preserve exact FP32 values using E2M1
  codes plus scales; unsupported scale capacities fail instead of quantizing.
- Stable compaction uses64-wide local/block prefix projections, hard rank
  routes and matrix multiplication. The generic helper reports overflow and
  occupied slots explicitly. No N-by-N prefix weight matrix is allocated.
- Legal-set artifact:549 operations,91 frozen records,7780 reusable candidate
  patterns; complete256-triple output with PAD rows. Includes piece/color,
  paths, king safety, castling rights/transit, en passant and promotions.

The production diff contains only three generic layout/GEMM dispatches and
validation. Piece-specific decisions are compiled offline into matrices and
connections; no C++/CUDA/JavaScript chess evaluator or procedural replay was added.

## Python/reference checks

Final command:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

**284 passed in103.35s**, exit0. Full output:
[pytest log](full_vm_foundation_pytest_2026-08-31.log).

Coverage includes literal2-batch matrix results and exact gradients20/40;
exhaustive Boolean truth tables and independent logistic derivatives; all64
six-bit compaction masks including overflow; block boundaries63/64 and127/128;
exact payload derivatives; whole legal sets for every prefix of10 fixed games;
additional seeded long histories through400 plies and promotions; explicit
named count/presence/overflow diagnostics and ignored request rows.

RED evidence: matrix tests failed for unknown wire11–13; circuit/compaction/legal
modules did not exist; graph inclusion failed for missing include_graph.
Intermediate full gate:280 passed,2 failures. One new test history illegally
tried f2f4 through its own knight onf3; corrected the fixture to g1h3. The other
failure was the expected stale reference source fingerprint in site JSON;
regenerated through the existing exporter. No expected legal set was altered
to fit implementation output. An initial Torch import failed with WinError1455
before test collection; later fresh checks passed without changing system memory
settings or terminating other processes.

## Fresh native builds and exact fixtures

Build directories (ignored generated outputs):

- build/full-vm-matrix-20260831: default native targets,16 checked process exits;
  exact two-batch matrix forward/gradients,7 new pre-device rejection cases,
  all prior attention derivatives and52 full-board fixtures passed.
- build/full-vm-legal-native-20260831: fresh generic tensor-fixture executable;
  first48 complete legal sets passed, followed by the expanded79-case set.

Toolchain: explicit sm86; installed CUDA12.5 and Torch2.8.0+cu128; MSVC19.44
with explicit AllowUnsupportedCompiler. FP32 arithmetic, TF32 disabled.
Native source/header/test/build-script hashes match the second build manifest
with **zero mismatches**. CMake targets were updated but CMake build execution
was not verified.

The expanded legal artifact and independent fixture are:

| File | SHA-256 |
| --- | --- |
| legal.cmz | 3a06b48ecc48d9abd8739ecc638d8afc4b7daf6577d9f39cfc957e09ed137e50 |
| legal.bin | 5399bb524f2228b0484d44a78c532264a97388b8631f76fd98a137c12fb24d25 |

Recompiling the current Python source produces the same artifact SHA.
The oracle replays each fixture independently using python-chess and supplies
every expected output element; native execution does not invoke that oracle.
[79-case native log](full_vm_legal_native_verified_2026-08-31.log).

Compute Sanitizer2024.2 memcheck on the same artifact/79-case fixture exited0
with **ERROR SUMMARY:0 errors**. Both check-exit-code and require-cuda-init were
enabled; [sanitizer log](full_vm_legal_memcheck_2026-08-31.log). Windows sanitizer
did not forward target stdout to the wrapper; the plain native run is logged
separately. The first tool-discovery command could not find sanitizer on PATH;
the verified run used its explicit CUDA12.5 compute-sanitizer path.

## Capacity and unverified boundaries

For batch1, logical FP32 frozen payload45,219,348 bytes; retained values
263,740,908 bytes, conservatively counting views. These are not peak allocator
measurements and exclude autograd/workspaces. No speedup, compactness, native
FP4 computation, full-graph gradient quality or useful learning is claimed.
Native legal-set acceptance is forward-only; primitive gradients are separate.

The requested-move transition, DRAW/win/illegal/overflow adjudication and full
next context are still missing. The initial design assumed oracle auto-claim
draw behavior. Whether claims remain automatic or belong to the player was
raised for user decision before changing the token protocol.

The site remains the45-operation position inspector. Only its local reference
source fingerprint and content-versioned JSON link were refreshed; exported
matrix values and the position artifact SHA are unchanged. It does not display
the549-operation legal graph or a full recurrent/native trace yet. No full-VM
release, new browser acceptance or deployment is claimed by this checkpoint.
