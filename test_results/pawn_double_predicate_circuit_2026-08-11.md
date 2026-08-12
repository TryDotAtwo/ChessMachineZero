# Pawn double-push predicate circuit — 2026-08-11

## Scope

The frozen chess image now evaluates 128 candidates: single and double pawn
pushes for all 64 source squares. Full chess is still not implemented.

Double-push legality is emitted by attention lookup tokens for
`MATCHING_PAWN`, `TARGET_EMPTY`, `PATH_CLEAR`, `GEOMETRY_VALID`, and
`START_RANK`, followed by an exact hardmax conjunction. Source/target board
writes and the side flip remain matrix-routed.

## TDD evidence

RED failed with:

```text
what(): chess image must contain single and double pawn candidates
```

After implementation, the focused chess executable passed. It asserts white
and black start ranks, wrong ranks, occupied intermediate and target squares,
and exact source/target writes for both double-push directions.

## Full functional gate

```text
cmz_vm2_compiler   Passed    0.66 sec
cmz_vm2_attention  Passed    0.49 sec
cmz_vm2_machine    Passed  250.89 sec
cmz_vm2_chess1     Passed  135.17 sec
100% tests passed, 0 tests failed out of 4
Total Test time (real) = 387.22 sec
```

## Runtime purity gate

```text
python -m pytest tests/test_vm2_source_purity.py -q
.............. [100%]
14 passed
```

```json
{"allowed_tensor_graph": true, "attention_matmul": true, "chess_runtime": false, "compiler_linked": false, "forbidden_routing": false, "legacy_link": false, "opcode_runtime": false, "runtime_sources": 4, "semantic_branching": false, "semantic_scalar_read": false}
```

The audited runtime files are unchanged. Chess-specific construction remains
in the offline compiler library only.
