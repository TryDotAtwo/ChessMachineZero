# Pawn capture predicate circuit — 2026-08-12

## Scope

The frozen image evaluates 256 candidate tokens: single push, double push,
capture-left and capture-right for every source square. Piece vocabulary now
distinguishes white and black non-pawn occupancy so opponent detection is a
token relation rather than a host condition.

`TARGET_ALLOWED` is selected by exact attention lookup over target piece,
side-to-move and move kind. Pushes require empty targets; captures require an
opposite-color occupied target. Geometry tokens reject file-a/file-h wrapping.

## Exact acceptance

The focused capture test asserts both colors, both diagonal directions, empty
targets, friendly targets, file boundaries, and exact source/target matrix
writes. It passed in 13.14 seconds.

Full clean-container CTest:

```text
cmz_vm2_compiler          Passed    0.95 sec
cmz_vm2_attention         Passed    0.55 sec
cmz_vm2_machine           Passed  497.06 sec
cmz_vm2_chess1            Passed  628.55 sec
cmz_vm2_chess1_capture    Passed   13.14 sec
100% tests passed, 0 tests failed out of 5
Total Test time (real) = 1140.26 sec
```

Purity:

```text
14 passed
```

```json
{"allowed_tensor_graph": true, "attention_matmul": true, "chess_runtime": false, "compiler_linked": false, "forbidden_routing": false, "legacy_link": false, "opcode_runtime": false, "runtime_sources": 4, "semantic_branching": false, "semantic_scalar_read": false}
```

Runtime source diff was empty. The 19-minute CPU regression demonstrates that
the current dense reference graph is not yet a performance implementation.
