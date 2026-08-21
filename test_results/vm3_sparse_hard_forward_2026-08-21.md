# VM3 sparse hard-forward evidence — 2026-08-21

## Scope

The new inference-only backend uses the same frozen program and hard selector
as the dense reference, but gathers compiler-frozen between-square IDs and at
most six blocker values per target/attacker pair. The recurrent dense/ST path
and its soft-surrogate gradients are unchanged.

This is **not** a HullKV claim. King-square selection still computes dense QK
over 64 keys; there is no convex-hull support query or certificate yet.

## Exact gates

- `audit_runtime.py`: `compiler_linked=false`, no forbidden or unclassified
  runtime sources, `lookup_head_dim=2`.
- GPU `cmz_vm3_test_king_safety --cuda`: PASS, including dense-vs-sparse pin
  parity and the existing differentiability assertions for the dense/ST path.
- Filtered VM3 CTest: 12/12 PASS; complete Python suite: 53/53 PASS.
- `test_sparse_hard_forward_matches_dense_on_full_oracle_corpus`: PASS for all
  520 deterministic positions, including curated pin, en-passant, promotion
  and castling cases.
- Independent read-only invariant audit: PASS; the test-only CLI backend flag
  is outside the runtime and does not choose matrices from chess state.

## Open boundary

Terminal/repetition rules and a genuine certified HullKV backend remain
unproved and unclaimed.

## Batched perft follow-up

- Added a bounded batched sparse-hard transition entry point and frozen batched
  attack offsets for up to 16 states.
- Added a tensor-only selected-trial materializer for test harness recursion.
  It consumes one-hot move selections and writes board, side, castling, EP,
  halfmove and fullmove fields back into the same state ABI through fixed
  tensor arithmetic and frozen row routers.
- Added exact CUDA perft oracle tests. Accepted counts: start depths 1-3
  (20/400/8902), canonical Kiwipete depths 1-2 (48/2039), and depth-1
  promotion, en-passant and castling cases. The perft recursion is test-only
  host orchestration over VM-produced legal/trial tensors; it is not runtime
  chess semantics.
