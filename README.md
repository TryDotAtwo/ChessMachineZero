# Percepta-style Transformer VM

This orphan branch contains a clean, standalone implementation of a minimal
token-native virtual machine.

Program instructions, registers, constants, ALU relations, the program counter,
and the halt state are tokens. A VM step consists of six frozen classical
self-attention stages:

```text
Q = XWq
K = XWk
V = XWv
scores = QK^T + mask
A = deterministic_hardmax(scores)
output = AV
```

The stages decode an instruction, read two operands, select an ALU relation,
write a register, and update PC/HALT. Production runtime code has no opcode or
chess-specific execution branches.

The first accepted program is:

```text
LOAD_CONST r0, 2
LOAD_CONST r1, 3
ADD r2, r0, r1
HALT
```

It produces `r2=5` with PC trace `0,1,2,3,3`.

## Build and test

LibTorch is required.

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/pytorch/share/cmake
cmake --build build
ctest --test-dir build --output-on-failure
python -m pytest tests/test_vm2_source_purity.py -q
```
