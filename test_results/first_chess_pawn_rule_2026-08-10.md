# First chess pawn rule evidence — 2026-08-10

Scope: side-relative white/black pawn one-square forward candidates only. This
is not full chess.

## Exact functional gate

Clean Docker build with LibTorch followed by full CTest:

```text
cmz_vm2_compiler   Passed
cmz_vm2_attention  Passed
cmz_vm2_machine    Passed
cmz_vm2_chess1     Passed
100% tests passed, 0 tests failed out of 4
Total Test time (real) = 357.97 sec
```

`cmz_vm2_chess1` asserts all 64 white source squares, offboard ranks, wrong
source piece, wrong side-to-move, occupied targets, every legal white and black
source/target delta, both side flips, and illegal-selection no-op.

The recurrent-state assertion runs a second transition with the exact same
ProgramImage. Board and side bytes remain unchanged after the exhausted move,
while the candidate trace changes to `ILLEGAL` as expected.

The automatic-player assertion uses one frozen ProgramImage for three
transitions: white `8→16`, black `48→40`, white `16→24`. Selection is the
hardmax winner among all 64 candidate rows. This test also preserves the
opponent pawn and guards against unrelated-square write leakage.

## Runtime purity gate

```text
python -m pytest tests/test_vm2_source_purity.py -q
.............. [100%]
14 passed
```

Audit report:

```json
{"allowed_tensor_graph": true, "attention_matmul": true, "chess_runtime": false, "compiler_linked": false, "forbidden_routing": false, "legacy_link": false, "opcode_runtime": false, "runtime_sources": 4, "semantic_branching": false, "semantic_scalar_read": false}
```

The audited runtime remains exactly `attention.h`, `machine.h`,
`attention.cpp`, and `machine.cpp`. The offline chess compiler is a separate
library and mutation tests reject linking it into `cmz_vm2`.
