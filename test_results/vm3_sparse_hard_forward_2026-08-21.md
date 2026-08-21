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

Perft depth gates, batched sparse execution, terminal/repetition rules and a
genuine certified HullKV backend remain unproved and unclaimed.
