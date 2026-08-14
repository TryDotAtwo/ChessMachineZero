# Percepta Transformer VM architecture

> **Status:** historical VM2 reference, superseded as the implementation target
> by
> `docs/superpowers/specs/2026-08-14-differentiable-recurrent-chess-machine-design.md`.
> The equations below describe VM2 hard-forward slice evidence only. They do
> not establish differentiable selection, one common full-chess ring,
> physically two-dimensional heads, whole-game BPTT or a host-free GPU rollout.

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

The current image contains 256 candidates: every source square paired with
single push, double push, capture-left or capture-right. Fifteen inference
stages route:

1. source piece;
2. current side;
3. side/kind/source keyed geometry;
4. target occupancy;
5. intermediate occupancy;
6. `MATCHING_PAWN`;
7. `TARGET_ALLOWED` (empty for pushes, opposite color for captures);
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

This proves recurrent symmetric ordinary pawn pushes and captures. It does not
yet prove promotion, en passant, other pieces, attacked squares,
check legality, castling, terminal states or a complete game of chess. Public
claims must remain at this boundary until corresponding circuits and exact
oracle comparisons exist.

## Acceptance

- exhaustive functional CTest with exact board/side assertions;
- source purity audit over the four runtime files;
- mutation tests that must reject semantic branches, scalar reads, forbidden
  routing, chess vocabulary and compiler linkage;
- GitHub Pages text must match the latest evidence file under `test_results/`.

The current implementation intentionally uses dense reference attention. The
19-minute CPU regression is semantic evidence, not a performance result;
geometry factorization or structural sparse attention is required before
scaling to complete chess.
