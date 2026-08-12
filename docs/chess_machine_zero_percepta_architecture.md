# Percepta Transformer VM architecture

## Hard boundary

The production transition is domain-agnostic. It receives a token matrix and
immutable tensors, then performs only fixed classical self-attention and
frozen matrix writes:

```text
Q = XWq
K = XWk
V = XWv
H = one_hot(lowest_index_argmax(QK^T + M))
Y = HV
X' = X + R(Y - X)C
```

The runtime does not inspect chess pieces, squares, predicates, legality,
opcodes, program counters or HALT values. It does not link either offline
compiler. The caller chooses a fixed inference unroll; token states implement
all semantic routing.

## Offline compilation

Ordinary C++ control flow is permitted only while compiling immutable token
rows, masks and weights. This is analogous to compiling a program: it may
describe chess rules, but it is absent from the runtime executable. Board
occupancy is encoded in input tokens; reusable rule structure is encoded in
frozen geometry/relation tokens and matrices.

## Current chess circuit

The current image contains 128 candidates: every source square paired with a
single or double pawn-push kind. Fifteen inference stages route:

1. source piece;
2. current side;
3. side/kind/source keyed geometry;
4. target occupancy;
5. intermediate occupancy;
6. `MATCHING_PAWN`;
7. `TARGET_EMPTY`;
8. `PATH_CLEAR`;
9. exact conjunction with `GEOMETRY_VALID` and `START_RANK`;
10. deterministic legal-candidate selection;
11. source write token;
12. target write token;
13. output board;
14. output side;
15. recurrent input state.

Every numbered item is executed by the tensor graph above. The final board is
not applied by a chess-aware host function.

## Claim boundary

This proves a recurrent, symmetric pawn single/double-push slice. It does not
yet prove captures, promotion, en passant, other pieces, attacked squares,
check legality, castling, terminal states or a complete game of chess. Public
claims must remain at this boundary until corresponding circuits and exact
oracle comparisons exist.

## Acceptance

- exhaustive functional CTest with exact board/side assertions;
- source purity audit over the four runtime files;
- mutation tests that must reject semantic branches, scalar reads, forbidden
  routing, chess vocabulary and compiler linkage;
- GitHub Pages text must match the latest evidence file under `test_results/`.
