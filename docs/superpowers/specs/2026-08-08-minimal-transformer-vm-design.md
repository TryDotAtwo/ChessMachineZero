# Minimal Transformer VM Design

## Goal

Build a new, isolated `native/vm2` execution core in which program semantics are
implemented by a frozen classical self-attention transition. The first accepted
program computes `2 + 3`, halts, and exposes `5` in a result register.

This design follows the public Percepta concept of executing programs inside a
transformer. It does not claim to reproduce unpublished Percepta implementation
details.

## Semantic boundary

One VM step has this semantic form:

```text
Q = X Wq
K = X Wk
V = X Wv
S = Q K^T + M
A = hardmax_rows(S)
Y = A V
X_next = fixed_linear_projection(X, Y)
```

The frozen matrices and program/state tokens determine the transition. The host
may repeat the same transition until a HALT token is active.

Allowed runtime operations are limited to:

- matrix multiplication;
- tensor addition and elementwise multiplication;
- fixed reshape, transpose, concatenation, and slicing without data-dependent
  indices;
- fixed attention masks;
- deterministic row-wise hardmax with a specified lowest-index tie break;
- fixed linear residual/projection operations;
- allocation, device transfer, shape validation, iteration counting, and tensor
  serialization in host code.

The first version uses six attention stages: instruction decode, left operand
read, right operand read, ALU relation lookup, register write, and PC/HALT
write. It must not use softmax, an MLP, an activation function, host
gather/scatter, or data-dependent host indexing. They can be considered only by
a later reviewed design change.

## Forbidden shortcuts

Neither production host code nor tensor kernels may contain instruction-specific
or domain-specific execution branches. In particular, production execution must
not contain:

- `switch(opcode)` or `if (opcode == ...)`;
- host arithmetic implementing ADD;
- host-side register or memory lookup;
- a table enumerating complete VM states or program continuations;
- chess pieces, squares, moves, castling, checks, or other chess semantics;
- calls into the existing ChessMachineZero rule engine;
- a fallback interpreter that substitutes for the attention transition.

Instruction behavior is compiled into frozen matrices and token layouts. A
reference interpreter is allowed only under tests and must never be linked into
the production VM target.

## First instruction set

The initial program representation supports exactly four opcodes:

- `LOAD_CONST destination, immediate`
- `MOVE destination, source`
- `ADD destination, left, right`
- `HALT`

Opcode values are data tokens, not native control-flow selectors. The frozen
attention weights select the matching instruction relation and route operands
and results.

Integer values in the first slice use a finite one-hot domain `0..15`. Addition
is represented compositionally by attention relations over the two operand
tokens. The accepted program uses only sums that remain in this domain. Overflow
is rejected by the compiler before execution; production inference has no
overflow fallback.

## Token layout

The input `X` is a fixed-size token matrix with typed slots:

- one control token containing the active program counter and halt state;
- four immutable instruction tokens;
- four register tokens `r0..r3`;
- constant tokens for `0..15`;
- fixed relation tokens used by instruction decode and addition.

Each slot has a stable positional/type encoding. Program tokens and state tokens
share one self-attention context. No token is appended during the first slice;
each step produces the next fixed-size state matrix.

## Components

The new `native/vm2` directory contains independent components:

- `schema`: token types, dimensions, and deterministic encodings;
- `compiler`: converts the four-instruction source representation into program
  tokens and frozen matrices;
- `attention`: the generic self-attention primitive;
- `machine`: repeats the attention transition with a host-visible maximum-step
  safety bound;
- `audit`: source and graph checks enforcing the semantic boundary;
- `tests/reference`: tests-only scalar interpreter.

The new target has no dependency on the legacy native engine. Integration with
the dashboard, self-play, chess traces, CUDA, or Rust FFI is outside this slice.
The first implementation may use CPU LibTorch so semantic correctness is
established before GPU optimization.

## Execution and failure behavior

The host stops only when the fixed halt-state output is active or the configured
step limit is reached. Reaching the limit is an explicit error. Invalid shapes,
non-finite tensors, ambiguous hardmax behavior, invalid bytecode, and arithmetic
overflow are explicit errors; none may trigger a weaker execution path.

Hardmax is deterministic. Equal maximum scores select the lowest key-slot index.
Tests must exercise the tie rule directly.

## Verification

Acceptance requires all of the following:

1. An attention-level unit test checks exact Q, K, score, hardmax, and value
   routing tensors on a small fixture.
2. Each opcode is tested against the tests-only reference interpreter across all
   valid registers and values in its finite domain.
3. The program `LOAD_CONST r0,2; LOAD_CONST r1,3; ADD r2,r0,r1; HALT` produces
   `r2=5` and the exact expected PC/halt trace.
4. Repeated runs are byte-identical.
5. Changing program tokens changes execution without rebuilding native code.
6. A negative test proves an uncompiled/invalid program is rejected.
7. A source audit rejects opcode branches, host ADD, data-dependent host
   indexing, chess vocabulary, legacy-engine calls, and runtime fallback paths.
8. A graph audit asserts every semantic output depends only on the declared
   attention transition and frozen tensors.
9. The full existing `pytest` suite remains green; it is regression evidence,
   not proof of VM purity.

## Completion criterion

The slice is complete only when the `2 + 3` program executes through the frozen
self-attention transition and the audits demonstrate that native code did not
perform the instruction semantics. Merely naming helpers "attention", reporting
metadata, or matching the reference output is insufficient.

## Deferred work

Memory addressing, comparisons, conditional jumps, loops, wider integers,
append-only traces, CUDA/CUTLASS acceleration, a C-like frontend, and chess are
separate later slices. Each must preserve the same semantic boundary and receive
its own reviewed design extension.
