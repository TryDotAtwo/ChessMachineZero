# VM inspector site verification

- Source: `site/`; deployment: `.github/workflows/pages.yml` uploads only `site/`.
- Contract gate: exact current tensor shapes, 45 generic ops, 2064 events, and the unfinished `LEGAL_SET`/terminal boundary are asserted by `tests/test_site_contract.py`.
- Browser behavior: verified a 64-cell reconstructed board, seven native three-token fixture moves, replay/reset controls, and zero console errors.
- Responsive viewports: 1536×1024 and 390×844 both reported `scrollWidth == clientWidth`.
- Browser runtime: current interactive matrix redesign was verified directly in the in-app Browser; no fallback was used.
- Visual design source: `site/design/vm-inspector-concept.png`.
- Matrix-proof extension: six explicit tensor stages, eight generic runtime primitive equations, and all 45 generated SSA operations rendered from `site/artifact_trace.json`.
- Synchronization gate: the published JSON is compared operation-for-operation and attribute-for-attribute with `build_position_reconstruction_artifact()`.
- Updated regression: 64 tests passed after the inference-proof extension.
- Numerical matrix gate: `site/numeric_trace.json` is compared exactly with a fresh real execution; it includes 45 operations, all frozen matrices and SSA values, `QKᵀ [1,64,2064]`, hard attention `[1,64,2064]`, and final board `[1,64,128]`.
- Browser QA: final operation rendered 45 selectable steps, 200-row pagination over 132,096 nonzero score cells, 64 hard-attention cells, no horizontal overflow, and no console errors.
- Responsive numerical inspector: verified at default desktop and 390×844 mobile viewport; the inspector is now the first page content and uses measured black-on-white contrast.
- Updated full regression: 66 tests passed.
- Interactive matrix QA: all 45 operation selections rendered nonempty matrices, equations, and cell provenance with zero browser console warnings/errors.
- Exact interaction checks: arbitrary GEMM output cell `(row=2,column=3)` updated the highlighted input row, weight column, and 128-term sum; hardmax distinguished the selected cell from the real winner; concat mapped output row 401 to its exact source/local row; final attention rendered six matrices across `QKᵀ → hardmax → A×V`.
- Language QA: RU/EN changed the document language, operation title, coordinate labels, and explanation title without reloading the trace.
- Responsive QA: after fixing intrinsic select sizing, 390×844 reported `scrollWidth == clientWidth`; desktop preserved the accepted three-matrix row×column composition.
- Current full regression after the interactive redesign: 68 tests passed.
