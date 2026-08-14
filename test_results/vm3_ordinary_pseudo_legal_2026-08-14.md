# VM3 ordinary pseudo-legal attention evidence — 2026-08-14

## Implemented boundary

- One universal 4272-row candidate batch now covers ordinary moves for pawns,
  knights, bishops, rooks, queens and kings, including all Q/R/B/N promotions.
- The compiler emits position-independent coordinate, piece-geometry and
  ordered-between-square relation tensors. Position-specific facts still enter
  only from the canonical board/side state at execution time.
- The executor assembles identical 53-token candidate packets and runs one
  immutable ordered list of 85 physical `d_head=2` Q/K/V stages. Source/target
  piece lookup, side match, six ray-empty predicates, Boolean conjunctions and
  the final OR reduction are all frozen attention stages.
- Every stage writes only the query token. Immutable K/V rows are retained once;
  exact fixed token routing uses compiler-serialized Int64 row IDs. V is stored
  as exact low-rank `Wv @ feature_router` factors rather than dense mostly-zero
  matrices. These are algebraic execution optimizations, not chess branches.
- Runtime contains no position-dependent C++ branch, scalar read, CPU transfer,
  detach, dynamic mask or compiler link. Castling, en-passant and own-king
  safety remain later tasks and are not claimed here.

## Exact gates

- Native exhaustive compiler-basis checks cover every source square for knight,
  king, bishop, rook and queen geometry. Runtime checks cover own/enemy ray
  blockers and exactly four promotion choices with no ordinary pawn-on-last-rank
  candidate.
- The GPU oracle compared exact UCI pseudo-LEGAL sets on four curated positions
  plus 512 seeded legal-random-walk positions against
  `python-chess.generate_pseudo_legal_moves()`. Castling and en-passant were
  explicitly filtered for this task; all 516 sets matched.
- Full native CTest: `10/10 passed`. Focused loader/candidate/ring/pseudo-legal
  gate: `4/4 passed`; recurrent ring includes exact board write and backward.
- Runtime purity audit: zero forbidden findings; executor has no compiler link.
  Source/public-evidence suite: `42 passed`.
- RTX 3070 Laptop GPU recurrent ring and gradient executable passed with the
  same 85-stage program. The 516-position oracle also ran on this GPU.

## Remaining scope

Task 9 proves ordinary pseudo-legality only. Exact special-move trial writes,
attack/own-king safety, complete LEGAL_SET, terminal/claim logic and full-game
BPTT remain open and public full-chess claims must not advance yet.
