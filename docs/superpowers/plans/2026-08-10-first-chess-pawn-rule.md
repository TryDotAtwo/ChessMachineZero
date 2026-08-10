# First Chess Pawn Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit all white-pawn single-push legality tokens and apply one selected legal push using only the existing fixed attention runtime.

**Architecture:** Add an offline-only `chess1` compiler that produces a normal `ProgramImage` with a chess-specific token schema, frozen relation tokens, masks, and matrices. The existing `attention.cpp` and `machine.cpp` remain domain-agnostic and unchanged.

**Tech Stack:** C++17, LibTorch, CMake/CTest, Python/pytest purity mutation gates.

## Global Constraints

- Runtime must not contain chess vocabulary, coordinates, board access, move loops, occupancy checks, or semantic scalar reads.
- All 64 candidates are evaluated as token rows in one fixed inference run.
- Board application is attention routing, not host gather/scatter or indexing.
- Compiler is offline-only and must not link into `cmz_vm2`.
- Claims remain limited to the white-pawn single-push slice.

---

### Task 1: Define chess token image and compiler boundary

**Files:**
- Create: `native/vm2/include/cmz_vm2/chess1_compiler.h`
- Create: `native/vm2/src/chess1_compiler.cpp`
- Create: `native/vm2/tests/test_chess1.cpp`
- Modify: `native/vm2/CMakeLists.txt`

**Interfaces:** `Chess1Board` contains 64 three-state piece values and side-to-move; `compile_chess1(board, selected_source)` returns the generic `ProgramImage`; tests decode outputs only outside runtime.

- [x] Write a compiler-shape test requiring 64 square, candidate, and output token rows.
- [x] Build and observe RED because the chess compiler API does not exist.
- [x] Add the offline compiler target and fixed schema constants without editing runtime sources.
- [x] Run the compiler-shape test GREEN.

### Task 2: Emit pawn single-push legality

**Files:** `chess1_compiler.cpp`, `test_chess1.cpp`

**Interfaces:** Candidates receive source piece, geometry target, target piece, side, source-domain status, then `LEGAL/ILLEGAL` from a frozen relation lookup.

- [x] Add exhaustive geometry and legality assertions for all 64 candidates and every negative condition.
- [x] Run the focused test and observe RED before legality stages exist.
- [x] Compile the legality attention stages and row/feature write projections.
- [x] Run exhaustive legality tests GREEN.

### Task 3: Apply a selected legal move through attention

**Files:** `chess1_compiler.cpp`, `test_chess1.cpp`

**Interfaces:** Selected candidate feeds source/target scratch tokens; all 64 output-square tokens select source-empty, target-pawn, or unchanged input; side output becomes Black.

- [x] Add exact two-square-delta and side-flip tests for every legal source.
- [x] Run the focused test and observe RED before board-output stages exist.
- [x] Add selected-candidate, source/target scratch, output-square, and side-update attention stages.
- [x] Run application and illegal-no-op tests GREEN.

### Task 4: Purity evidence and publication

**Files:**
- Modify: `tests/test_vm2_source_purity.py`
- Modify: `README.md`, `docs/project_memory.md`, `docs/change_history.md`
- Create: `test_results/first_chess_pawn_rule_2026-08-10.md`

**Interfaces:** Mutation gates prove runtime rejects chess/board semantics while the compiler remains outside runtime linkage.

- [x] Add runtime mutation fixtures for chess vocabulary and chess-compiler linkage.
- [x] Extend the gate and run mutations GREEN.
- [x] Run clean Docker build, full CTest, purity pytest, and audit report.
- [ ] Record exact evidence, commit, and push the orphan branch.
