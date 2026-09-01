# Full VM Walkthrough Implementation Plan

> Execute inline with `superpowers:executing-plans` and TDD. The user explicitly
> prohibited subagents. No delegation or new user-owned tasks.

**Goal:** Show an actual complete recurrent frozen VM execution, down to scalar
matrix operations, on the existing RU/EN site.

**Architecture:** Offline circuit synthesis and the generic C++/CUDA executor
remain separate. Compose and validate executable subgraphs before presenting
the resulting complete transition. Export recorded execution, not JS chess.

**Tech Stack:** Python/NumPy/PyTorch compiler/test reference; C++17/LibTorch/CUDA;
static HTML/CSS/JavaScript and GitHub Pages.

**Spec:** `docs/superpowers/specs/2026-08-31-full-vm-walkthrough-design.md`.

## Global constraints

- Work only in the existing clean codex worktree; preserve the legacy checkout.
- No runtime chess branches, oracle, evaluation or external search.
- Input `[B,2048,128]`, output context `[B,2045,128]`, 400 plies, 256 legal triples.
- Hard one-hot forward and specified floating surrogate backward; FP32 compute.
- No subagents; self-review, exact tests and recorded evidence are mandatory.
- No full-VM publication before complete native transition acceptance.

## Task 1: Generic matrix-layout and dynamic-GEMM support

Files: `vm_compiler/graph.py`, `reference_executor.py`, `include/cmz/artifact.h`,
`src/artifact.cpp`, `src/module.cpp`, focused Python/native tests and build lists.

Wire operations: `MATRIX_TRANSPOSE=11`, `MATRIX_RESHAPE=12` with `(rows,cols)`,
`MATRIX_MATMUL=13` with two inputs. All act on rank-three `[B,M,N]` tensors;
transpose swaps only M/N, reshape preserves B and element count, matmul requires
equal batch symbols and matching contraction dimensions.

- [x] RED: execute a routed/projected `[[1,2],[3,4]]`, transpose and reshape to
  `[1,3,2,4]`, form its outer product, and assert all 16 literal results plus
  input gradients20 (second scaled batch40). Serialize/parse the same graph.
- [x] GREEN: add the three generic dispatches and pre-device shape validation.
- [x] Test zero/wrong reshape extents, moved batch/rank, mismatched matmul K,
  schemas and undefined SSA; malformed metadata must fail before CUDA access.
- [x] Fresh native build executes identical forward/gradient cases; keep all
  previous numerical and position acceptance passing.

## Task 2: Reusable frozen circuit construction

Files: new `vm_compiler/circuit.py`, `tests/test_circuit.py`.

API: `Circuit.input`, `constant(name,array)`, `project`, `transpose`, `reshape`,
`matmul`, `add`, `bias`, `concat_rows`, `threshold`, `select`, `artifact(output)`.
Track rank-three static shapes and semantic producer metadata at construction.
Boolean threshold uses two-class affine logits plus existing hardmax; selection
uses exact tensor gates, never a Python decision based on runtime values.

- [x] RED: exhaustive literal truth tables for AND/OR/NOT and tensor selection;
  test more than one batch, exact one-hot forward and specified derivatives.
- [x] GREEN: lower each construction to the generic wire operations; frozen
  constants are exact E2M1 values where possible, explicit FP32 scales otherwise.
- [x] Assert serialized graph behavior, reject incompatible static dimensions,
  and report decoded-weight/intermediate byte bounds before CUDA execution.

## Task 3: Position facts and legal candidate circuit

Files: new `vm_compiler/legal_circuit.py`; `tests/test_legal_circuit.py`;
development oracle fixture helpers only under `tests/`.

API: `build_legal_artifact()` returns an executable artifact plus named output
metadata. Inputs remain the fixed protocol; an explicitly named legal-set
subgraph is not represented as a full recurrent context.

- [x] RED: exact initial20 triples, both-color positions, blocked rays, captures,
  pawn doubles/promotions, pins/check evasions and forbidden king capture.
- [x] Synthesize fixed ordered geometry candidates; compute source/target/path
  eligibility through projections and threshold circuits.
- [x] Add full post-move king-safety checks and castling/en-passant regressions.
- [x] Compile ordered stable compaction with frozen prefix projections and
  attention routing. Compare every output triple and padding row with the
  independent oracle, not only move counts or a selected move.

## Task 4: Exact recurrent transition and terminal context

Decision confirmed 2026-09-01: use oracle-style automatic claims exactly as
`outcome(claim_draw=True)`. No `CLAIM_DRAW` token or request-language change.

Files: new `vm_compiler/recurrent_circuit.py`, `tests/test_recurrent_circuit.py`,
native fixtures/tests, compiler export entry point.

API: `build_recurrent_artifact()` returns the full artifact with `context_0`;
`FrozenVm::forward` returns its complete context without a procedural wrapper.

- [x] RED/GREEN: e2e4 initial transition, illegal e2e5 unchanged history, wrong result
  piece, black reply, exact sorted next legal set and hard-one-hot output.
- [x] Implement tensor request membership/conditional chronological append and
  invoke the legal circuit on the resulting position.
- [x] RED/GREEN: Fool's Mate absorbing BLACK_WIN, stalemate, material/repetition/
  halfmove draw-policy cases, terminal absorption and explicit history overflow.
- [x] Assert whole context equality across multi-step native runs and exact
  output-to-next-input feedback; run forward and gradient acceptance separately.

## Task 5: Native trace and artifact-driven inspector

Resolution recorded 2026-09-01: keep native inference uninstrumented. Native
acceptance compares complete contexts, sequential feedback and backward. The
static inspector uses compact exact Python-reference intermediates from the same
serialized artifact; it explicitly labels them as non-native. A native dump of
roughly3GiB retained SSA is not added to the production runtime or website.

Files: `vm_compiler/recurrent_site_trace.py`, `site/recurrent_inspector.js`,
site HTML/CSS/i18n, native output/feedback tests and focused browser tests.

Manifest interface: `schema_version`, artifact/source/precision provenance,
ordered `steps` with request/prior/output bindings and content-versioned data
paths; each step contains operation/tensor schemas and semantic producer links.

- [x] RED/GREEN: missing/corrupted step, stage gap, invalid producer/opcode,
  feedback mismatch and
  forged status/value are rejected; current45-op fixture remains inspectable.
- [x] Preserve native arithmetic/hot path unchanged; record the explicit
  `native_intermediate_capture=false` boundary and full-output native evidence.
- [x] Cover all2877 operations with declared schemas, bilingual semantics,
  generic scalar rules and compact exact windows. Retain full arbitrary-cell
  access in the nested45-op microscope; unknown opcodes fail closed.
- [x] Add nine-stage/operation navigation, decoded context/legal/status panels,
  exact feedback identity, frozen-matrix meanings and per-operation proofs.

## Task 6: Evidence, self-review and publication

Files: project docs, prompt/change history, `test_results/`.

- [x] Run full Python suite, fresh native transition/gradient tests and targeted
  CUDA memory checks. Record actual byte bounds and known unverified scope.
- [x] Inspect representative paths for every opcode family, full feedback loop,
  exact cells,
  producer jumps, RU/EN, mobile and console in a real browser.
- [x] Self-review all diffs against runtime purity and evidence boundaries.
- [x] Ordinary scoped fast-forward publication; verify deployed artifact/source
  identity and versioned resources before marking the task complete.
