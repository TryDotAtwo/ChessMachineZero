# Predicate-token Control Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `CMP_EQ` and `JUMP_IF` whose comparison and PC selection occur only through frozen attention inference.

**Architecture:** Expand the fixed token schema to sixteen instructions and four predicate slots. Equality is a frozen relation-token lookup; branching builds target/fallthrough candidate tokens and uses predicate-keyed attention to select the next PC. Runtime remains unchanged except for iterating a larger fixed stage stack.

**Tech Stack:** C++17, LibTorch, CMake/CTest, Python/pytest purity audit.

## Global Constraints

- No runtime opcode, predicate, PC, register, value, or HALT scalar reads or branches.
- Every stage runs for every instruction.
- Runtime is a caller-fixed inference unroll.
- Data-dependent selection occurs only through hardmax self-attention.
- Compiler branches are offline-only and compiler linkage into runtime remains forbidden.

---

### Task 1: Replace non-scalable write projections

**Files:** `schema.h`, `compiler.cpp`, `machine.cpp`, `test_machine.cpp`

**Interfaces:** Replace per-token `[N,D,D]` projections with fixed row `[N,N]` and feature `[D,D]` projections. Runtime update is `X + R @ (Y-X) @ C` for a fixed number of projection components.

- [ ] Add tests requiring exact legacy program behavior under compact projections.
- [ ] Run native tests and observe RED after changing the expected schema/API.
- [ ] Compile row/feature projections offline and apply them only with matrix multiplication and addition.
- [ ] Run all native and purity tests GREEN.

### Task 2: Add predicate and branch tokens

**Files:** `schema.h`, `compiler.cpp`, `test_compiler.cpp`

**Interfaces:** Add opcodes `CmpEq` and `JumpIf`; predicate slots `p0..p3`; literal predicate tokens; two branch-candidate scratch tokens; sixteen program slots; eleven fixed stages.

- [ ] Add compiler tests for valid/invalid predicate indices and branch targets.
- [ ] Run compiler build and observe RED because the new opcodes/schema do not exist.
- [ ] Implement token encoding and offline validation without changing runtime dispatch.
- [ ] Run compiler tests GREEN.

### Task 3: Implement equality through attention

**Files:** `compiler.cpp`, `test_machine.cpp`

**Interfaces:** Stage 3 equality relation lookup emits `FALSE/TRUE`; stage 5 writes it into the selected predicate token.

- [ ] Add exhaustive tests for all `16 x 16` operand pairs.
- [ ] Run machine tests and observe RED because predicate results are absent.
- [ ] Add frozen equality relation tokens, weights, masks, and projections.
- [ ] Run exhaustive equality tests GREEN.

### Task 4: Implement predicate-selected PC

**Files:** `compiler.cpp`, `test_machine.cpp`

**Interfaces:** Stages 6-9 read a predicate, form target/fallthrough scratch candidates, and select next-PC through hardmax attention; stage 10 writes control state.

- [ ] Add exhaustive true/false branch tests across all valid targets.
- [ ] Add the exact counter-loop test and post-HALT absorbing-state test.
- [ ] Run machine tests and observe RED because conditional PC selection is absent.
- [ ] Compile predicate-read, candidate, and selection attention stages.
- [ ] Run branch, loop, repeatability, and absorbing-HALT tests GREEN.

### Task 5: Purity evidence and publication

**Files:** `audit_runtime.py`, `test_vm2_source_purity.py`, `README.md`, `docs/*.md`, `test_results/predicate_control_flow_2026-08-10.md`

**Interfaces:** Existing runtime audit must continue to report only allowed attention/projection operations and reject predicate/PC host dispatch mutations.

- [ ] Add/confirm predicate and PC mutation gates.
- [ ] Run clean Docker build, full CTest, purity pytest, and source audit.
- [ ] Record exact results and update durable project documentation.
- [ ] Commit and push only the orphan branch.
