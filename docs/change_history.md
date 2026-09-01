# Change History

## 2026-09-01 — Complete recurrent frozen VM and full-graph inspector

- Added the complete 2,877-operation/192-record recurrent artifact: request membership in the prior `LEGAL_SET`, conditional history append, new position, opposing legal set, mate/stalemate/overflow status and an exact `[B,2045,128]` feedback context. Terminal win/draw states are absorbing.
- Implemented the confirmed automatic `outcome(claim_draw=True)` policy as frozen tensor circuits: insufficient material, current/future threefold with effective legal en-passant keys, current/future fifty-move, and the halfmove-99 legal-reply edge.
- Added generic grouped intermediate matrix GEMM as wire opcode14 across Python serialization/reference and C++/CUDA dispatch, with exact two-batch forward/gradient tests and malformed-metadata rejection. No chess opcode or procedural C++ move logic was introduced.
- Fresh native acceptance matched 47 complete FP32 contexts exactly: 40 output-to-next-input edges and 5 `context_0` bindings. Full `LEGAL_SET` backward reached the request with 144 nonzero finite components (`abs_sum=7056.8`). The hash-identical full workload passed Compute Sanitizer memcheck with exit 0 and `ERROR SUMMARY: 0 errors`.
- Rebuilt the static site around all 2,877 operations in nine stages. Every operation has compact exact matrix windows, scalar provenance and bilingual tensor/frozen meanings; GEMM distinguishes left row from right column, hardmax is centered, and attention/grouped GEMM/final feedback are navigable. The nested 45-op full-COO position microscope remains available.
- Strengthened browser validation to reject stage/producer/opcode/feedback/status/count/proof/value inconsistencies. Final real-browser acceptance passed desktop, 390×844 mobile, RU/EN and empty console assertions. Full integrated gate: 312 passed in 243.37s. [Evidence](../test_results/full_recurrent_vm_and_site_2026-09-01.md).
- Preserved explicit limits: site intermediates are exact Python-reference windows for the same artifact, not every native CUDA intermediate; current compute is FP32; no trained player, learning advantage or performance speedup is claimed.
- Published content commit `3b1dac6` by ordinary fast-forward to the clean branch and `main`. Pages run `33490289502` succeeded; the public HTML and all eight content-versioned resources matched local content. A fresh public Playwright session loaded the full trace, reached operations 9/13/2877, switched RU/EN, passed the 390×844 layout check and emitted no console/page errors.

## 2026-08-31 — Inline full-VM foundation and executable legal sets

- Added generic matrix transpose, batch-preserving reshape and intermediate GEMM wire ops, with pre-CUDA rank/shape/product validation and literal two-batch native forward/gradient acceptance.
- Added the offline Circuit builder, exact constant packing, Boolean/selection lowering and safe graph inclusion; added exhaustive truth tables and independent surrogate-derivative tests.
- Added block-prefix stable compaction with hard rank routing, exact payload GEMM, padding/presence and explicit capacity diagnostics; no runtime sort/filter or quadratic prefix weight matrix.
- Added a549-operation/91-record legal-set subgraph over7780 reusable geometry patterns, including side/source/path checks, post-move king safety, castling rights/transit, en passant and promotions. Its output is768 rows, not a full recurrent context.
- Final Python/reference/Node gate:284 passed in103.35s. Fresh native matrix/default gate preserved52 exact boards; new legal artifact matched79 full independent legal sets. Memcheck exited0 with zero errors on the legal fixture run. [Evidence](../test_results/full_vm_foundation_2026-08-31.md).
- Refreshed only the site's local reference-source fingerprint and JSON content version; UI remains the45-op position inspector. Full requested-move application, adjudication, feedback context and native full-VM website are pending. Draw-claim policy was raised for user choice before changing protocol. No subagents were used; no full-VM deployment claimed.

## 2026-08-31 — Audit corrections and exact-trace inspector

- Corrected latest-event epsilon from1e-6 to2^-21; independently legal400-ply histories and full materialized FP32 score invariants now guard square/time separation.
- Aligned the Python selected-top-k Q/K/V surrogate with the native contract; added independent derivative assertions, stable original-index ties and exact hard-forward checks.
- Fixed hardmax/FP4 STE cancellation, finite FP4 saturation, collinear HullKV tie preservation and development-oracle binary-input/absorbing-terminal semantics.
- Rewrote architecture/project-memory claims to separate the executable45-op position subgraph from unfinished recurrent legality/status stages and distinguish FP4 storage from FP32 computation.
- Native validation now propagates static graph shapes before device access, rejects malformed candidate metadata and empty hardmax masks, and guards invalid kernel indices before dereference.
- Replaced the website's independent seven-move JavaScript replay with one three-ply reference export; the board now decodes only exported v45. All24 frozen/46 SSA/2 derived tensors have bilingual semantic axes, scalar provenance and arbitrary-coordinate/producer navigation.
- Corrected GEMM row/column highlighting, centered ARGMAX, true K-transpose rendering and multi-stage attention/concatenation layout. Invalid exports fail closed; language controls remain usable on validation failure.
- Final integrated Python/reference/Node gate through721db71: 257 passed in20.92s; four JS files individually syntax checked. All45 operations checked in desktop/mobile browser layouts, including exact cells, RU/EN and a deliberately malformed export. Final review caught and resolved a ready→Loading label regression with explicit lifecycle ownership and browser verification. Fresh native build: 15 checked processes, 52 exact FP32 boards and independent Q/K/V derivatives. Controller memcheck: zero reported errors for attention and all board fixtures. [Correction evidence](../test_results/audit_corrections_2026-08-31.md) records actual gates, release status and unverified boundaries.
- Live acceptance caught mixed new HTML/old cached JavaScript despite a successful Pages workflow. Added generator-managed content versions for CSS, four JS files and JSON; fetch/download share the same versioned URL, with regression tests preventing stale published references.
- Published721db71 via ordinary fast-forward to main and the clean branch; Pages33356317808 succeeded. Public browser verification passed all45 operations, exact output/score cells, producer navigation, RU/EN and fresh-console checks; the previously broken cached tab recovered. Final documentation records the deployed source revision separately from report-only commits.

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
