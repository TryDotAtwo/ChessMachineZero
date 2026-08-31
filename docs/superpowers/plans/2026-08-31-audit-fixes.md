# Audit corrections and honest matrix inspector implementation plan

> **For agentic workers:** use `superpowers:subagent-driven-development`; implement each task with failing behavioral regressions, focused green tests and independent review.

**Goal:** Correct the confirmed audit defects and make the site a faithful, bilingual inspection of one published tensor execution.

**Architecture:** Keep the existing frozen-rule compiler and generic tensor runtime. Correct their arithmetic and validation; do not implement a new chess engine. The static site consumes a generated reference trace and never pretends to run native CUDA.

**Tech Stack:** Python/NumPy/PyTorch development compiler; C++17/LibTorch/CUDA runtime; static HTML/CSS/JavaScript.

**Spec:** User approval on 2026-08-31 of the proposal in `test_results/audit_2026-08-31.md`, following the previous matrix-inspector requirements in `docs/prompt_history.md`.

## Global Constraints

- Work only in `codex/pure-frozen-transformer-vm-clean`; preserve the dirty legacy root and existing audit documents.
- Production C++/CUDA executes generic tensor primitives only; no chess-aware branches, replay, external search, evaluation, training labels or Python dependency.
- Current artifact consumes hard one-hot `[B,2048,128]`, ignores request rows 0–2, reconstructs `[B,64,128]` from up to 400 already valid plies. Full recurrent legal/status execution is not implemented.
- Current runtime arithmetic is FP32. FP4 codes plus FP32 scales are artifact storage, not a native FP4 performance claim.
- Hard forward must be exact; backward is a specified selected-top-k softmax surrogate, not the true derivative of discrete chess or evidence of useful learning.
- GPU execution must retain device/current-stream semantics and avoid CPU readback/synchronization in the hot path. No performance claim without measurement.
- Tests assert independently derived behavior/invariants, not source-string presence or nonzero-gradient-only acceptance. Preserve the audit as a historical snapshot.
- Use apply_patch for edits, focused commits with no unrelated files, no force-push/history rewrite. Implementers do not spawn agents; controller owns reviews and publication.

### Task 1: Correct compiler and development reference behavior

**Files:** `vm_compiler/compiler.py`, `reference_executor.py`, `ste.py`, `hullkv.py`, `oracle.py`; focused tests in `tests/`; generated `site/artifact_trace.json` and `site/numeric_trace.json` only to keep export consistency. No native, UI source, or architecture-document edits.

**Interfaces:** Keep existing entry points. Add reusable `attention_2d_ste(queries, keys, values, candidates, competitor_count, temperature)` in `vm_compiler/reference_executor.py` for dynamic batched Q/K/V; the artifact executor delegates to it. Candidates are original key-row indices; score ties use the lowest original index. Native task consumes the same specified gradients, not Python private implementation details.

- [x] Add and observe a failing long-history regression using the audit's seed-1 legal/nonterminal 400-ply history and an independent python-chess expected full one-hot board; include boundary prefixes and actual promotion/castling coverage. Keep chess usage inside development/test oracle wrappers.
- [x] Add a materialized-weight score invariant for all 64 queries, all 64 event addresses and all 401 timestamps: `score(exact,0) > score(other,400)` for other squares and strictly increasing scores for equal squares. Fix epsilon to `2**-21`; run full-board equality on legal histories, not just argmax equality.
- [x] Add failing selected-surrogate tests. For Q=`[[[1,0]]]`, K=`[[[2,0],[1,0],[0,0]]]`, V=`[[[0],[10],[100]]]`, candidates=`[0,1,2]`, k=2, T=1: hard output=0, dQx≈-1.9661194, dV≈(0.7310586,0.2689414,0). Assert Q/K/V against the independent selected-softmax expression; include unsorted candidate ties, batching and k=1. Implement identical selected-top-k backward, with exact hard output.
- [x] Add failing standalone STE stress tests: exact binary output over seed-0 random `[100,3]` logits; FP4 finite extreme saturation and identity gradient, including `1.1`, `6.1`, `+/-1e10`. Fix cancellation and saturate before distance computation. Define/reject nonfinite temperatures/scales; empty hardmax mask rows remain errors.
- [x] Add failing collinear/duplicate/zero-query hull tests using keys `(0,0),(1,0),(1,-1),(1,1),(-1,0)` and q=(1,0), expected original index=1. Retain all boundary candidates needed for stable dense top-k, including degenerate collinear layers.
- [x] Add failing oracle tests: fractional rows summing to 1 are rejected; completed draws/wins cannot resume. Terminal states reconstructed from history are absorbing (preserve history; keep canonical terminal status). This is development-oracle semantics, not production functionality.
- [x] Run focused tests per fix, regenerate JSON with `python -m vm_compiler.site_trace`, then full `python -B -m pytest -o addopts= -q -p no:cacheprovider --rootdir=. --confcutdir=. tests`. Write red/green evidence to the task report and commit only owned files.

### Task 2: Make native validation and acceptance reproducible

**Files:** `src/module.cpp`, `src/ste.cpp`, `src/hullkv.cu`, `include/cmz/` as needed; native tests; `CMakeLists.txt`; `scripts/build_native.ps1` and a narrowly scoped fixture exporter under `tests/` if needed; `pytest.ini`, `requirements-dev.txt`, `README.md`; native test notes under `test_results/`. Do not change Task 1 code or the website.

**Interfaces:** Preserve `FrozenVm` public execution shapes and generic opcodes. Reject malformed graph metadata before materialization/launch. Preserve native selected-top-k softmax backward for all Q/K/V. The FP4 primitive saturates finite extremes and retains identity surrogate. Fail closed on empty hardmax mask rows using device-side asynchronous validation, without `.item()`, CPU transfers or steady-state synchronization.

- [x] Write a native negative graph test for dynamic K with 2 rows and candidate 2; require constructor rejection before a kernel is launched. Include duplicate candidates and incompatible Q/K/V shapes. Propagate symbolic batch plus static non-batch shapes through every supported generic opcode; check row-route bounds/strides, broadcasts, projections, concatenation and attention capacities.
- [x] Test native hardmax invalid mask handling and FP4 saturation/gradient; retain exact valid-mask forward/backward checks. Use `at::_assert_async` or an equivalent generic device assertion for data-dependent GPU mask validity; invalid GPU cases run in an isolated child process if needed because device assertions poison a CUDA context. Add kernel index guards so direct generic attention APIs cannot dereference out-of-bounds indices.
- [x] Add explicit native Q/K/V gradient acceptance for unbatched, shared-key batched and dynamic batched attention against independently constructed selected-softmax derivatives. Include score/index ties and batch isolation. All comparisons state precision/tolerance.
- [x] Add exact full native-board acceptance for the 400-ply legal counterexample plus short special-move/promotion cases. Export fixture inputs and expected boards through a development/test oracle wrapper; tests read data, never call Python in production.
- [x] Diagnose/rebuild STE with CUDA registration correctly linked. Track a reproducible Windows build script that discovers the installed Torch, VS toolchain and CUDA paths, accepts an explicit CUDA architecture, builds from current sources into a new directory and checks every command exit. Do not conceal CMake failures by reporting cached binaries as current. Ensure CTest includes the main position test with generated artifact/fixture paths (or an explicit conditional when fixtures are absent, documented rather than silently omitted).
- [x] Add local pytest configuration and dependency instructions, distinguish optional CUDA runtime from the Python test tools; README must state current incomplete scope, storage/compute precision and exact commands.
- [x] Run fresh builds/tests on RTX 3070 Laptop sm86, recording toolchain versions, exact outputs and known unverified areas. No throughput claim. Commit owned changes and report red/green evidence.

### Task 3: Publish one truthful bilingual matrix execution

**Files:** `vm_compiler/site_trace.py`, new `vm_compiler/site_semantics.py`, `site/index.html`, `site/app.js`, `site/matrix_inspector.js`, new `site/trace_model.js` and `site/i18n.js` if useful, `site/styles.css`, generated JSON, `tests/test_site_contract.py` and focused Node behavioral tests under `tests/`. Do not change runtime/compiler arithmetic.

**Interfaces:** Export one trace schema with `fixture.moves`, a board tensor from the same reference run, `provenance` (executor, dtype, artifact hash, source identifier), per-tensor bilingual `semantics` (name, purpose, row/column meaning), and structured operation attributes. Browser consumes this one data source; no independent `from/to/piece` game replay. Keep complete raw numeric COO downloadable. Technical SSA identifiers remain accessible as secondary mappings, not the only explanation.

- [x] First test that fixture token decoding and the displayed board come from the same exported tensor; reject inconsistent/missing trace data instead of showing plausible fallback results. Use the legal e2e4,d7d5,e4d5 fixture already represented by `[[52,54,96],[47,45,102],[54,45,96]]`. If replay frames are offered, export each frame from the reference rather than computing chess effects in JavaScript; a static fixture display with explicit reload/reset is sufficient.
- [x] Add metadata coverage tests for every frozen tensor, every SSA value and both attention intermediates. Each has specific semantic identity and axes; e.g. padding is not an initial-square matrix, pattern columns are not vocabulary tokens, time biases are not trained coefficients. Route provenance consumes structured attributes, not equation regex. Include original IDs and producer links in optional technical details.
- [x] Correct visible capabilities: position reconstruction only; requested move prefix unused; no LEGAL_SET/status transition yet; matrices are a Python reference export, native evidence is separate; packed FP4 storage/current FP32 computation; selected-top-k surrogate not proven learning benefit; candidate scanning not demonstrated geometric speedup. Remove Run artifact/verified claims for browser animation and stale test-count marketing. Reference dated test evidence rather than inventing a live result.
- [x] Make all matrix elements addressable by tensor selector plus row/column coordinates. Input values can jump to their producer; frozen values show their definition; derived QK/argmax/AV each have true scalar provenance. For GEMM show exactly left row × right column, including a real transposed K view for QK, not two arbitrary rectangles. For argmax show a centered operation, the eligible source row and winning column, then one-hot output. Copy/add/concat highlight only contributing cells.
- [x] Provide whole-page RU/EN switching (including controls, statuses, titles and axis labels), no unexplained SSA/vN primary labels. Keep current restrained visual language, repair mobile grid specificity/overflow and keyboard coordinate navigation. Avoid cosmetic sections or new architecture.
- [x] Test exported model/provenance arithmetic in Node/Python using independent expected values, not exact source-text assertions. Regenerate trace; run full tests. Controller performs browser checks on every operation, selected arbitrary cells, language toggles, responsive layouts and console errors before publication.
- [x] Commit owned changes and report evidence/remaining limits. Native parity evidence must refer only to tests actually run, not imply every intermediate was captured natively.

### Task 4: Integrate evidence, documentation and release

**Files:** `docs/architecture.md`, `docs/project_memory.md`, `docs/change_history.md`, `docs/prompt_history.md`, `test_results/`; publication workflow only if a concrete deployment defect appears.

- [x] Review each task against its brief and resolve important findings. Run a final independent whole-diff review after integration; record any remaining limits.
- [x] Rewrite stale architecture/current-state claims so target recurrent VM and implemented position subgraph cannot be confused. Record exact epsilon bound, surrogate contract, FP32 execution, cache behavior, validation and test commands. Append correction evidence; retain the original audit as a dated snapshot with a link to remediation.
- [x] Rerun complete Python, newly built native, and browser acceptance on the final tree. Verify the board and numeric trace share the exact fixture and all 45 operation paths remain reachable. Record dated results rather than claim universal correctness.
- [ ] Commit scoped changes and publish with an ordinary fast-forward push to the already agreed remote main only after checks and remote-state verification. No force push, dirty root mutation or implicit unrelated merge. Verify Pages deployment status and live content; report failed/unverified deployment honestly if blocked.
