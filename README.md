# Percepta-style Transformer VM

> **Public correction:** the previous ChessMachineZero implementation was
> incorrectly presented as fully frozen-attention-only. Its production C++/CUDA
> path still contained chess-specific procedural control flow. Function names
> and metadata that called those kernels "attention" did not prove that chess
> semantics were executed solely by self-attention. See
> [CORRECTION.md](CORRECTION.md) for the exact reasons.

This orphan branch contains a clean, standalone implementation of a minimal
token-native virtual machine.

Program instructions, registers, constants, ALU relations, the program counter,
predicate slots, and the halt state are tokens. A VM step consists of eleven frozen classical
self-attention stages:

```text
Q = XWq
K = XWk
V = XWv
scores = QK^T + mask
A = deterministic_hardmax(scores)
output = AV
```

The stages decode an instruction, read operands, select ALU/equality relations,
write registers or predicates, build branch candidates, select the next PC,
and update PC/HALT. Production runtime code has no opcode, predicate, PC, or
chess-specific execution branches.

Runtime execution is a caller-fixed unroll of the same transition. It never
reads `HALT`, PC, opcodes, registers, or predicates on the host. `HALT` is an
absorbing token state: remaining inference steps reproduce the same state.
State writes use frozen batched matrix projections rather than elementwise
write masks.

The offline compiler may inspect source instructions only to produce immutable
program tokens, relation tokens, attention masks, and frozen matrices. It is a
separate library and is not linked into the inference runtime. Runtime tensor
selection is limited to `QK^T`, deterministic hardmax, `AV`, fixed matrix
projections, and tensor addition.

The first accepted program is:

```text
LOAD_CONST r0, 2
LOAD_CONST r1, 3
ADD r2, r0, r1
HALT
```

It produces `r2=5` with PC trace `0,1,2,3,3`.

The second accepted program uses `CMP_EQ` and `JUMP_IF` to count from zero to
three. Equality becomes a predicate token; target versus fallthrough is chosen
by hardmax attention. An immutable `TRUE` token expresses the loop back-edge.
The final state is `r0=3`, `p0=TRUE`, and absorbing `HALT=TRUE`.

The first chess-rule slice evaluates all 64 white-pawn one-square candidates
in parallel. Source piece, north geometry, target occupancy and side-to-move
are routed into candidate tokens. A frozen 72-token relation table produces
`LEGAL/ILLEGAL` through `QK^T` plus hardmax. A selected legal candidate then
changes exactly two output-square tokens and flips the side; an illegal
selection is an attention-computed no-op.

This is deliberately **not yet full chess**. Castling, checks, captures,
promotion, en passant and every other piece rule remain future rule circuits.
The accepted claim is only that this pawn slice executes in the unchanged,
domain-agnostic inference runtime.

## Build and test

LibTorch is required.

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/pytorch/share/cmake
cmake --build build
ctest --test-dir build --output-on-failure
python -m pytest tests/test_vm2_source_purity.py -q
```
