# Change History

## 2026-08-31 — Audit corrections (integration in progress)

- Corrected latest-event epsilon from1e-6 to2^-21; independently legal400-ply histories and full materialized FP32 score invariants now guard square/time separation.
- Aligned the Python selected-top-k Q/K/V surrogate with the native contract; added independent derivative assertions, stable original-index ties and exact hard-forward checks.
- Fixed hardmax/FP4 STE cancellation, finite FP4 saturation, collinear HullKV tie preservation and development-oracle binary-input/absorbing-terminal semantics.
- Rewrote architecture/project-memory claims to separate the executable45-op position subgraph from unfinished recurrent legality/status stages and distinguish FP4 storage from FP32 computation.
- Native validation now propagates static graph shapes before device access, rejects malformed candidate metadata and empty hardmax masks, and guards invalid kernel indices before dereference.
- Fresh Python/reference gate after independent task review: 123 passed. Fresh native build: 15 checked processes, 52 exact FP32 boards and independent Q/K/V derivatives. Controller memcheck: zero reported errors for attention and all board fixtures. Website integration remains in progress; [correction evidence](../test_results/audit_corrections_2026-08-31.md) records the actual gates and unverified boundaries.

## 2026-08-28 — Semantic tensor provenance and operation-specific highlighting

- Removed `SSA: vN/wN` identifiers and raw SSA equations from the primary matrix cards; the UI now explains what every tensor stores and what its rows and columns mean.
- Restricted full row-plus-column highlighting to real matrix multiplication. Routing, expansion, addition, and concatenation now mark only the source and result cells that participate in the selected value.
- Made hardmax explicit as a centered `ARGMAX ПО СТРОКЕ` / `ROW ARGMAX` operation, with the complete score row, actual winning cell, and selected output cell shown separately.
- Added responsive wide operator slots so semantic operation names remain readable on desktop and mobile.

## 2026-08-28 — Bilingual interactive matrix provenance

- Replaced COO tables as the primary UI with transformer-style matrix grids for every one of the 45 executed SSA operations.
- Added RU/EN switching, semantic matrix names with secondary SSA IDs, selectable output coordinates, and per-cell provenance.
- GEMM now highlights the exact input row and frozen-weight column and expands every visible dot-product term; attention exposes `Q×Kᵀ`, its hardmax winner, and `A×V`.
- Added operation-specific provenance for row routing, concatenation, residual/position addition, frozen expansion, and hardmax rather than presenting every opcode as multiplication.

## 2026-08-28 — Matrix-level inference proof on the VM site

- Added a six-stage executable tensor trace with the exact matrix shapes for history routing, castling and en-passant projections, 2064-event construction, 2D key projection, `QK^T` hardmax/STE, and `A×V` board reconstruction.
- Added a generic-runtime proof mapping every opcode used by the artifact to its slice, expand, GEMM, add, concat, hardmax/STE, or attention equation.
- Added `vm_compiler.site_trace`, which exports every real SSA operation from `build_position_reconstruction_artifact()` to the static site; tests compare all indices, opcodes, value IDs, and attributes exactly.
- Added a responsive raw 45-operation table loaded from the generated trace rather than maintained as website copy.

## 2026-08-28 — Current-artifact VM inspector site

- Added a responsive static VM inspector under `site/`, designed around the executable position reconstruction artifact rather than the retired dashboard architecture.
- Pinned the visible contract to `[B,2048,128]` input, `[B,64,128]` position output, 45 generic SSA operations, and one 2D latest-event attention over 2064 events.
- Made the browser/runtime boundary explicit: the UI replays an exact numeric-token fixture while the actual frozen artifact remains C++/CUDA.
- Added a GitHub Pages Actions workflow and exact site contract tests; the UI explicitly marks `LEGAL_SET` and terminal status execution as subsequent artifact stages.

## 2026-08-28 — Exact numerical matrix execution trace

- Replaced the shape-only proof with a real fixed-fixture execution export from the development reference executor.
- Published every frozen FP4-materialized matrix and every SSA input/output as exact zero-based COO values for all 45 operations.
- Exposed the full `QKᵀ` score matrix, hard attention matrix, and `A×V` output, with arithmetic expansion of a selected output cell.
- Added paginated matrix rendering so the complete 132,096-entry attention score matrix remains inspectable without freezing the browser.

## 2026-08-28

- Compiled the position rules into a 45-op executable frozen artifact: strided move routing, exact castling/en-passant pattern hardmax, parabolic 2D latest-event attention over 2064 events, generic C++ execution, and CUDA backward acceptance.
- Added latest-event reconstruction for castling rook movement and en-passant pawn removal; compiled all four castling transformations as an exact one-hot FP4 tensor.
- Replaced the promotion/padding third move slot with a mandatory color-specific `RESULT_PIECE` token, removed the recurrent workspace bias, and added one-hard-attention latest-event position reconstruction for ordinary moves and captures.
- Added the first position-reconstruction circuit constants: exact one-hot initial piece state, numeric square decoder, frozen service-row workspace bias, and a branch-free tensor reference transition for ordinary moves and captures.
- Extended 2D HullKV and the artifact opcode to dynamic batched Q/K/V self-attention with exact per-batch routing and floating Q/K/V backward.
- Compiled reusable chess geometry into exact binary FP4 relation tensors and combined them with bootstrap/address records in a strict non-executable rule image; vectorized exact E2M1 packing reduced relation compilation from seconds to milliseconds.
- Added exact FP4-scaled convex-ring 2D addressing for every vocabulary channel and input row, with exhaustive self-address selection tests.
- Added the exact board-free `context_0`, canonical FP4 bootstrap record, CUDA batch loader, and a development-only history-replay oracle for legal, illegal, and terminal transition acceptance.
- Bound `HULL_ATTN_2D` to frozen 2D keys and cached nested-hull indices with batched dynamic values, then verified a complete artifact-declared Transformer block and input gradient on CUDA.
- Replaced the placeholder unconditional recurrent slice with the first strict artifact-declared SSA executor: one-time FP4 CUDA materialization, generic row/GEMM/position/residual/gated-FFN/hardmax dispatch, and load-time graph validation.
- Added exact 2D HullKV CUDA support routing and differentiable hard-attention backward over exact nested-hull top-k competitors.
- Added deterministic dense-equivalence coverage for nested convex layers, including interior points, duplicate quantized keys, stable ties, and zero queries.
- Added fresh hardmax and FP4 custom-autograd STE primitives with exact Python references and real CUDA C++ tests.
- Added a fresh strict FP4 graph artifact writer/parser and independent C++ loader with explicit opcodes and no tensor-count inference.
- Added the fresh tensor-native LibTorch module and a real CUDA autograd test; no legacy executor or loader was copied.
- Created an empty orphan implementation branch after the user rejected migration of the legacy prototype.
- Added only the authoritative recurrent numeric protocol, exact tests, and architecture/project documentation.
