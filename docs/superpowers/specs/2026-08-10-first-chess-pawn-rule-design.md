# First recurrent chess circuit: symmetric pawn single push

## Proven scope

The compiler emits a generic `ProgramImage` in which one frozen inference
transition evaluates all 64 one-square pawn candidates for the current side,
hardmax-selects a legal candidate when automatic mode is enabled, applies the
move, flips side, and feeds the emitted board back into recurrent input tokens.

This is a complete circuit for the stated pawn slice, not complete chess.
Pawn double push, captures, promotion, en passant, other pieces, check,
castling, draw rules and a strategic policy remain unimplemented.

## Runtime boundary

The unchanged domain-agnostic runtime executes only:

```text
Q = X Wq
K = X Wk
V = X Wv
S = Q K^T + mask
H = one_hot(argmax(S))
Y = H V
X_next = X + R @ (Y - X) @ C
```

It has no chess vocabulary, square arithmetic, move loops, piece dispatch,
occupancy branch, semantic scalar read, gather/scatter or board mutation. The
offline-only compiler constructs tokens, masks and frozen matrices and is not
linked into `cmz_vm2`.

## Token schema

- 64 recurrent square tokens with `EMPTY`, `WHITE_PAWN`, `BLACK_PAWN`, `OTHER`;
- one recurrent `SIDE_TO_MOVE` token;
- 64 candidate tokens, one per source square;
- 128 immutable side-keyed geometry tokens: `square × side`;
- 128 immutable legality relations:
  `source_piece × side × source_valid × target_onboard × target_piece`;
- one selected token, source/target write tokens, 64 output-square tokens and
  one output-side token.

Coordinates and chess predicates exist only as compiler-emitted token data.

## Fixed inference stages

1. Candidate attends to its source-square piece.
2. Candidate attends to recurrent side-to-move.
3. Candidate selects a side-keyed geometry token and receives target id,
   source-valid and target-onboard fields.
4. Candidate attends to the target square or offboard sentinel.
5. Candidate selects one exact legality relation using `QK^T + hardmax`.
6. Selected token either reads the compiled candidate or hardmax-selects the
   lowest-index `LEGAL=true` candidate in automatic-player mode.
7. Source-write token receives selected source and legal gate.
8. Target-write token receives selected target, moving piece and legal gate.
9. Every output square selects unchanged input, exact legal source write or
   exact legal target write.
10. Output side selects unchanged side or relation-emitted opposite side.
11. Output board and side attend back into recurrent input rows.

## Score separation invariant

At board-write stage, matching input-square identity scores `1.0`. A legal
gate contributes `0.25`. Therefore:

```text
matching legal write = 1.25 > unchanged input = 1.0
unrelated legal write = 0.25 < unchanged input = 1.0
```

This prevents a legal write token from erasing unrelated occupied squares.
The separation is encoded in fixed `Wq/Wk` coefficients, not host logic.

## Recurrent and automatic execution

`compile_chess1_auto` initializes the selected-token query for legal
candidates. Hardmax across all 64 candidate rows chooses the lowest-index legal
move deterministically. Stage 11 makes emitted board/side fields the next
transition input, so repeated calls use the same `ProgramImage` and weights.

The accepted three-ply trace is:

```text
WHITE: 8 -> 16
BLACK: 48 -> 40
WHITE: 16 -> 24
```

No host move generator or host legality check participates in these
transitions.

## Acceptance evidence

1. All 64 sources cover interior and offboard geometry for both sides.
2. Wrong side, wrong piece, occupied target and offboard target emit illegal.
3. Every legal white and black source changes exactly source/target and side.
4. Illegal selection preserves board and side.
5. A second exhausted fixed-selection inference preserves recurrent board/side
   bytes and changes its trace to illegal.
6. Three automatic transitions alternate both sides using unchanged weights.
7. An unrelated opponent pawn survives a legal move, proving score separation.
8. Full CTest, mutation purity and runtime audit remain green.

Exact command evidence is stored in
`test_results/first_chess_pawn_rule_2026-08-10.md`.
