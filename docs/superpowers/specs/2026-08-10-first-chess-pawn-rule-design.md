# First chess rule: white pawn single push

## Goal

Compile the first real chess rule into immutable tokens and frozen attention
weights: a white pawn may move one square forward into an empty square when
White is to move. The same fixed inference graph must emit legality tokens and
the resulting board state without chess-specific runtime code.

This slice is intentionally not full chess. Passing it proves one chess rule is
executed inside the token/attention machine; it does not justify a claim that
the branch already plays complete chess.

## Semantic boundary

Runtime remains the existing universal fixed-unroll graph:

```text
Q = X Wq
K = X Wk
V = X Wv
scores = Q K^T + mask
A = deterministic_hardmax(scores)
Y = A V
X_next = X + R @ (Y - X) @ C
```

The runtime may not contain chess vocabulary, square arithmetic, board loops,
piece dispatch, occupancy tests, move generation, or board mutation. It may not
read semantic tokens on the host. The offline compiler may generate immutable
chess relation tokens and frozen matrices.

## Token schema

The compiled context contains:

- 64 square-state tokens, one for each fixed square id `0..63`;
- piece-state fields `EMPTY`, `WHITE_PAWN`, and `OTHER`;
- a `SIDE_TO_MOVE` token with values `WHITE` and `BLACK`;
- 64 single-push candidate tokens, one per possible source square;
- immutable geometry relation tokens mapping each source square to its
  north-adjacent target or `OFFBOARD`;
- immutable source-domain tokens distinguishing pawn-valid ranks 2 through 7
  from ranks 1 and 8;
- legality result tokens `LEGAL` and `ILLEGAL`;
- one selected-move token used only after legality has been produced;
- board-output tokens for all 64 squares.

Square coordinates are data in relation tokens. Runtime code sees only token
indices and tensor shapes.

## Legality inference

Every candidate runs through the same fixed stages:

1. Attend from candidate to its source square.
2. Attend through the frozen geometry relation to obtain its target square or
   `OFFBOARD`.
3. Attend to the target square state.
4. Attend to `SIDE_TO_MOVE`.
5. Select a frozen legality relation keyed by source piece, side, target
  occupancy, on-board status, and pawn-valid source-rank status.
6. Write `LEGAL/ILLEGAL` into the candidate token.

The only relation yielding `LEGAL` is:

```text
source_piece=WHITE_PAWN
side_to_move=WHITE
source_status=PAWN_RANK
target_status=ONBOARD
target_piece=EMPTY
```

This condition is represented as attention-key matching, not a native boolean
expression.

## Move application inference

The selected-move token contains a candidate id. The compiler or caller may set
that token only to a candidate already emitted as `LEGAL`; selection policy is
outside this slice.

For every output-square token, attention selects exactly one value source:

- `EMPTY` when its square id matches the selected source;
- `WHITE_PAWN` when its square id matches the selected target;
- otherwise the corresponding input-square piece state.

The selection is performed by frozen relation tokens and hardmax attention.
Runtime C++ must not gather, scatter, index, copy, clear, or write board squares
according to the selected move.

The final side-to-move token becomes `BLACK` through the same frozen transition.

## Compiler boundary

The offline compiler may:

- encode square ids and north-neighbor relations;
- validate the three-state board input and selected candidate id;
- reject application of a candidate not marked legal by a tests-only result
  validator;
- construct immutable attention masks, relation tokens, and matrices.

The compiler must remain a separate library and must not be linked into the
runtime target. No compiled table may enumerate complete board positions or
complete program continuations.

## Acceptance tests

Acceptance requires:

1. Exhaustive empty-board geometry for all 64 source squares: ranks 1 through 7
   map north by eight; rank 8 maps to `OFFBOARD`.
2. For every source on ranks 2 through 7 with an on-board target, a white pawn
   with White to move and an empty target emits `LEGAL`.
3. The same candidates emit `ILLEGAL` when the source is empty or `OTHER`, the
   target is occupied, the side is Black, the source is on rank 1 or 8, or the
   target is off-board.
4. All 64 candidate legality tokens are produced in one fixed inference run;
   runtime does not loop over moves.
5. Applying every legal single push produces exactly two changed squares and
   flips side to move to Black.
6. Repeated runs are byte-identical.
7. Changing board, side, or selected-candidate tokens changes inference without
   rebuilding runtime code.
8. Mutation gates reject injected square arithmetic, chess vocabulary,
   occupancy branches, move loops, gather/scatter, board indexing, and host
   board writes in runtime sources.
9. The graph-operation allowlist and compiler/runtime linkage gate remain green.

## Failure behavior

Invalid board encodings and invalid selected candidate ids are rejected before
runtime. Runtime has no fallback or procedural interpreter. A fixed inference
run that does not emit the required one-hot result is invalid evidence and must
fail the tests rather than substitute a result.

## Deferred chess scope

Pawn double push, captures, en passant, promotion, Black movement, sliders,
knights, kings, check, castling, repetition, fifty-move rule, move choice, and
complete game play are separate reviewed slices. Full-chess claims remain
forbidden until every rule and state transition has exact attention-only
evidence.
