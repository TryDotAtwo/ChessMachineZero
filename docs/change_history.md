# Change history

## 2026-08-13 — animated full native trace debugger

- Added a synchronized board/token/heatmap debugger for the complete 15-stage
  pawn native trace, with six microsteps per stage and playback controls.
- Added explanation, active-number, and all-matrix modes; the original exact
  coordinate explorer remains available below.

## 2026-08-13 — pair-synchronized knight trace

- The first board click now selects only a source and pauses the trace until a
  target is selected; it no longer evaluates a fake source-to-self pair.
- X, Q, scores and hardmax are visibly recomputed per selected pair, while Wq,
  Wk and class keys are explicitly labelled as frozen weights.
- Corrected visual board ordering to standard a8..h8 at the top.

## 2026-08-13 — clickable knight board

- Replaced numeric source/target controls with a clickable chessboard and
  matrix-derived reachable-square highlights.
- Reworked the matrix panel into four narrated steps from pair token to
  hardmax; retained ChessMachineZero as the product brand and Percepta only as
  concept attribution.

## 2026-08-13 — interactive knight matrix lab

- Added a Pages laboratory for selecting source/target squares and inspecting
  the exact pair token, Wq/Wk matrices, scores and hardmax class.
- Public copy explicitly limits the result to knight geometry; board legality
  and production move application remain pending.

## 2026-08-13 — factorized knight geometry gate

- Added a frozen matrix/hardmax circuit for the 64x64 knight relation.
- Exhaustively verified every source/target pair without expanding the main
  circuit by eight direction-specific candidate banks.

## 2026-08-13 — exact compact projection experiment

- Proved an offline-compiled, matrix-only compact Q/K path exactly equivalent.
- Rejected its production switch after two 900-second native gate timeouts.

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
- Specified an exact structurally sparse attention graph based only on frozen
  sparse selection matrices; rejected gather/index-select shortcuts that would
  reintroduce semantic host routing.
- Added a generic frozen block-attention primitive using only matrix
  selections, local QK^T/hardmax/AV and transpose-matrix routing; exact tests
  preserve dense outputs and global lowest-index ties.
- Replaced decorative site matrices with a native-exported 15-stage numeric
  trace, arbitrary cell navigation, scalar product expansion, hardmax winners,
  and three sequential matrix-write components.
- Added an exact mask-derived attention cost model. It proved 74.1694x fewer
  useful score pairs and exposed the need for padded block merging rather than
  thousands of exact-route microkernels.
- Added the generic offline mask-to-block compiler. It groups only immutable
  mask structure, freezes all plan tensors, and passes exact dense/block output
  and winner equivalence plus the 18-test purity/evidence gate.
- Rebuilt Pages as one synchronized native-trace laboratory with board, stage
  timeline, numeric matrices, cell arithmetic, hardmax, AV and matrix-write.
- Removed the synthetic knight classifier from the public UI; the frontend now
  visualizes only the exported pawn-rule trace.
