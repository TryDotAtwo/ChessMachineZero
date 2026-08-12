# Percepta-style Transformer VM

Interactive trace: <https://trydotatwo.github.io/ChessMachineZero/>

> **Public correction:** the previous ChessMachineZero implementation was
> incorrectly presented as fully frozen-attention-only. Its production C++/CUDA
> path still contained chess-specific procedural control flow. Function names
> and metadata that called those kernels "attention" did not prove that chess
> semantics were executed solely by self-attention. See
> [CORRECTION.md](CORRECTION.md) for the exact reasons.

This orphan branch contains a clean, standalone implementation of a minimal
token-native virtual machine.

Program instructions, registers, constants, ALU relations, the program counter,
predicate slots, and the halt state are tokens. A VM step consists of fifteen frozen classical
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

The current chess-rule slice evaluates 256 pawn candidates in parallel: one-
and two-square pushes plus both diagonal captures for every source square and
either side. Source piece,
side-keyed geometry, target occupancy, intermediate occupancy, move kind,
start rank and side-to-move are routed into candidate tokens. Frozen relation
tokens produce `MATCHING_PAWN`, `TARGET_ALLOWED`, `PATH_CLEAR`,
`GEOMETRY_VALID`, `START_RANK`, and finally `LEGAL/ILLEGAL` through exact
`QK^T` lookups plus hardmax. A selected legal candidate then
changes exactly two output-square tokens and flips the side; an illegal
selection is an attention-computed no-op.

The final attention stage feeds emitted square and side tokens back into the
next transition's input rows. Repeating an exhausted selected move preserves
board/side bytes while its new trace correctly changes to `ILLEGAL`.

An optional player circuit hardmax-selects the lowest-index `LEGAL=true`
candidate. A three-transition test alternates white, black, then white again
using the same recurrent token state and frozen weights.

This is deliberately **not yet full chess**. The current player policy is only
deterministic legal-first selection. Castling, checks, promotion, en passant
and every non-pawn piece rule remain future rule circuits. The current dense
reference implementation proves semantics; it is not a speed claim.
The accepted claim is only that this recurrent symmetric pawn slice executes
in the unchanged, domain-agnostic inference runtime.

## Build and test

LibTorch is required.

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/pytorch/share/cmake
cmake --build build
ctest --test-dir build --output-on-failure
python -m pytest tests/test_vm2_source_purity.py -q
```
