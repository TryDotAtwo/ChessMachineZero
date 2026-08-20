# VM3 attention king-safety evidence — 2026-08-20

## Implemented boundary

- Frozen physical `d_head=2` king-square lookup: `QK^T`, deterministic
  straight-through hardmax and `AV` relation routing.
- Frozen attack tensors for pawns, knights, kings, rook/queen rays,
  bishop/queen rays and intervening squares.
- Frozen two-class attention reducers compute path clear, attack OR, castle
  origin/transit/destination safety and final `LEGAL`.
- The executor uses fixed tensor operations and four-row HullKV castle transit
  routing. Chess relations and selector keys are compiler-produced, hashed
  program tensors. Source-purity audit rejects runtime `torch::tensor` creation.

## Completed gates

- Clean sequential rebuild of all 12 VM3 acceptance targets.
- Native CTest: `12/12` passed in `121.46 s` after the final purity fix.
- GPU `cmz_vm3_test_king_safety --cuda`: passed, including gradient from final
  legal selection to canonical state.
- Python suite: `52/52` passed, comprising state, source-purity, public-claim,
  520-position pseudo-legal and 520-position final-legal GPU oracles.
- Focused purity/public suite: `22/22` passed.
- `git diff --check`: clean apart from Windows line-ending notices.

## Exact cases

Native assertions cover a pinned rook, a legal along-pin move, king adjacency,
en-passant exposing a rook ray, castling from check, through check, into check,
and a legal castling control. The final oracle compares exact UCI move sets,
not only counts.

## Remaining acceptance gap

The approved Task-11 plan also requires start-position perft depths 1–4 and
committed Kiwipete/special-position perft. Those gates are not yet implemented
or claimed. The dense reference graph is exact but currently too expensive to
evaluate every node of depth 4 within the existing timeout; the next step is an
exact hard-forward block/HullKV execution path, followed by the perft gate.
