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

## 2026-08-11

- Added a React/Vite GitHub Pages site with animated board, token stream,
  Q/K/V projections, QK^T heatmap, hardmax winner, AV, matrix-write, and
  recurrent-state stages.
- Added pause, single-step, stage selection, token-table and sparse-QK^T trace
  controls based on the implemented pawn circuit.
- Added an honest evidence section that distinguishes the proven recurrent
  pawn slice from unfinished full chess.
- Added inference-selected pawn self-play: hardmax chooses the lowest-index
  legal candidate and three recurrent transitions alternate both sides.
- Fixed a matrix-score bug where the legal gate could overpower square
  identity and erase unrelated occupied squares; exact square+legal now scores
  1.25, unmatched legal writes 0.25, and unchanged input 1.0.
- Expanded the chess trace to 128 single/double pawn candidates and 256
  side/kind/source-keyed geometry tokens.
- Added frozen predicate lookups for matching pawn, empty target, clear
  intermediate path, valid geometry and start rank, followed by an exact
  five-predicate hardmax conjunction.
- Increased the generic tensor width to 320 and fixed inference depth to 15;
  runtime sources remain unchanged and domain-agnostic.
- Added exact white/black double-push tests covering start ranks, occupied
  intermediate/target squares and source/target matrix writes.
- Published the interactive trace at
  `https://trydotatwo.github.io/ChessMachineZero/`; verified HTML, JS and CSS
  with independent HTTP 200 responses.
- Added both diagonal pawn-capture kinds, color-aware non-pawn tokens,
  opponent-target predicate lookup and file-a/file-h boundary geometry.
- Added a focused exact capture gate plus full CTest 5/5; the dense CPU
  regression took 1140.26 seconds and is not presented as a speed result.
- Split offline chess preparation into reusable immutable circuit compilation
  and board-token binding; bound tokens equal fresh compilation byte-for-byte
  while weights and masks share the same tensor storage.
- Added a Pages evidence job and claim gate that derives candidate/stage counts
  from C++ headers, enforces an explicit incomplete-full-chess label, and
  blocks deployment unless the mutation-tested runtime purity suite passes.
