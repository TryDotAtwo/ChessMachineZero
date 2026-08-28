# Frozen rule-relation evidence

- RED: `vm_compiler.relations` was absent.
- GREEN literal geometry: b1 knight, a1 king, c1 bishop ray, a1 rook ray, a1-a8 between, a1-h8 between, and a non-aligned between query matched hand-derived square sets.
- GREEN pawn geometry: white and black single, starting-rank double, and diagonal attacks matched literal native squares; non-starting double steps were empty.
- All relation tensors are float32 binary and have fixed exhaustive shapes; slider and leaper diagonals exclude self edges.
- RED artifact compiler: `build_rule_relation_records` and `build_rule_image_artifact` were absent.
- GREEN artifact compiler: eight named relation records, including `[64,64,64]` between, packed as exact E2M1 and survived strict combined-image serialization.
- Exact E2M1 packing was refactored from a per-value Python search to vectorized `searchsorted`; existing byte fixtures remained green and relation packing measured 0.06 s versus 12.46 s before refactor.
- Full Python suite: 41 passed.
- The rule image has no operations yet and is not claimed executable.
