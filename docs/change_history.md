# Change history

## 2026-08-10

- Replaced host early-exit on the `HALT` tensor with caller-fixed inference
  unrolling.
- Made post-HALT transitions byte-identical absorbing states.
- Replaced runtime elementwise write-mask routing with frozen batched matrix
  projections.
- Removed runtime tensor-to-host scalar extraction and input-dependent semantic
  branches.
- Expanded the purity gate with semantic-branch, scalar-read, routing,
  operation-allowlist, and compiler-link mutation tests.
- Added predicate-token `CMP_EQ` and `JUMP_IF` execution through frozen
  attention relations and branch-candidate tokens.
- Expanded the VM to sixteen instruction slots and eleven stages.
- Replaced large per-token write projections with compact row/feature matrix
  projections.
- Added exhaustive equality and conditional-target tests plus an exact
  token-native counter loop.
- Added an offline-only first chess compiler with 64 parallel white-pawn
  single-push candidates and 72 frozen legality relation tokens.
- Added attention-only source/target board writes, legal side flip, and illegal
  no-op behavior without changing the generic runtime.
- Extended purity mutations to reject chess vocabulary in runtime and any
  runtime linkage to the chess compiler.
- Generalized the pawn circuit to both colors with side-keyed geometry,
  complete four-piece relation products, token-routed pawn color, and
  token-routed next side.
- Closed the pawn board state recurrently: stage 10 routes output squares and
  side back into the next inference input without recompiling weights.
