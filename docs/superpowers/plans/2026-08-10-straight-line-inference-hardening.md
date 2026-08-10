# Straight-line Inference Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the VM runtime a fixed-unroll, semantic-branch-free tensor graph whose state transitions are attention plus fixed matrix projections.

**Architecture:** The offline compiler remains allowed to inspect opcodes and construct frozen tokens and matrices. The runtime receives a compiled `ProgramImage`, repeats the same transition a caller-fixed number of times, and never reads semantic tensor values on the host. `HALT` is absorbing inside the frozen attention transition.

**Tech Stack:** C++17, LibTorch, CMake/CTest, Python 3/pytest source-audit tests.

## Global Constraints

- Runtime semantic selection occurs only through `QK^T`, deterministic hardmax, `AV`, and fixed matrix projections.
- No runtime branch or scalar extraction may inspect opcode, PC, register, predicate, or HALT tokens.
- The inference step count is fixed before execution and independent of all VM outputs.
- Compiler control flow is offline-only and the compiler library must not be linked into the runtime library.
- Tests must demonstrate both functional behavior and rejection of injected forbidden runtime semantics.

---

### Task 1: Make the purity gate catch semantic host execution

**Files:**
- Modify: `native/vm2/tools/audit_runtime.py`
- Modify: `tests/test_vm2_source_purity.py`

**Interfaces:**
- Consumes: the four declared runtime source files and `native/vm2/CMakeLists.txt`.
- Produces: a JSON report with independent gates for semantic scalar extraction, semantic branching, forbidden tensor routing, compiler linkage, and the required attention path.

- [ ] Add mutation fixtures containing `if (state[...HALT...].item<double>())`, predicate dispatch, PC dispatch, host arithmetic, `index_put_`, `gather`, `scatter`, `where`, and elementwise write-mask routing.
- [ ] Run `python -m pytest tests/test_vm2_source_purity.py -q` and verify failure because the current auditor accepts at least the HALT mutation/current HALT branch.
- [ ] Extend the auditor to reject semantic `.item`, semantic conditional branches, runtime mutation/indexing, data-dependent routing, elementwise write masks, and compiler linkage.
- [ ] Run the focused pytest and verify that it still fails on the current runtime's real HALT branch.

### Task 2: Replace early HALT with fixed unroll and absorbing state

**Files:**
- Modify: `native/vm2/include/cmz_vm2/machine.h`
- Modify: `native/vm2/src/machine.cpp`
- Modify: `native/vm2/tests/test_machine.cpp`

**Interfaces:**
- Consumes: `ProgramImage` and caller-fixed `std::int64_t inference_steps`.
- Produces: `RunResult run_fixed(const ProgramImage&, std::int64_t)` with exactly `inference_steps + 1` trace states and `steps == inference_steps`.

- [ ] Change machine tests to request eight inference steps and require that states 4 through 8 are byte-identical absorbing HALT states.
- [ ] Build/run `cmz_vm2_test_machine` and verify failure because the current runtime returns after four steps.
- [ ] Implement fixed unroll without reading HALT or any semantic field in host code.
- [ ] Build/run the focused machine test and verify the fixed trace behavior passes.

### Task 3: Compile state writes into fixed matrix projections

**Files:**
- Modify: `native/vm2/include/cmz_vm2/schema.h`
- Modify: `native/vm2/src/compiler.cpp`
- Modify: `native/vm2/src/machine.cpp`
- Modify: `native/vm2/tests/test_machine.cpp`

**Interfaces:**
- Consumes: per-stage row/feature write layout during offline compilation.
- Produces: frozen per-token keep/take projection matrices in `ProgramImage`; runtime combines old and attended state only with matrix multiplication and tensor addition.

- [ ] Add a test assertion/audit mutation proving the old `state * (1-write) + projected * write` expression is forbidden.
- [ ] Run the focused purity test and verify failure on the current elementwise write path.
- [ ] Replace `write_masks` with batched diagonal `keep_projection` and `take_projection` tensors built offline.
- [ ] Update `transition` to apply those projections with batched `torch::matmul` and addition, without runtime indexing or elementwise multiplication.
- [ ] Build/run attention, compiler, and machine tests and verify exact prior VM behavior plus absorbing HALT behavior.

### Task 4: Add structural graph and linkage evidence

**Files:**
- Modify: `native/vm2/tools/audit_runtime.py`
- Modify: `tests/test_vm2_source_purity.py`
- Modify: `native/vm2/CMakeLists.txt`
- Modify: `README.md`
- Create: `test_results/straight_line_inference_hardening_2026-08-10.md`

**Interfaces:**
- Consumes: runtime sources, CMake target graph, and mutation fixtures.
- Produces: an auditable manifest of allowed runtime semantic operations and documented verification evidence.

- [ ] Require the exact attention operations and fixed projection operations while rejecting unrecognized semantic tensor APIs in runtime sources.
- [ ] Assert from CMake text that `cmz_vm2` links only `cmz_vm2_attention`, never `cmz_vm2_compiler` or a legacy chess engine.
- [ ] Run every mutation fixture and verify each fails for its intended reason.
- [ ] Update README to describe fixed unroll, absorbing HALT, and the offline compiler/runtime boundary without claiming chess support.
- [ ] Run a clean build, full CTest, and purity pytest; record exact commands and results under `test_results/`.
- [ ] Commit only the scoped branch files.
