# VM inspector site verification

- Source: `site/`; deployment: `.github/workflows/pages.yml` uploads only `site/`.
- Contract gate: exact current tensor shapes, 45 generic ops, 2064 events, and the unfinished `LEGAL_SET`/terminal boundary are asserted by `tests/test_site_contract.py`.
- Browser behavior: verified a 64-cell reconstructed board, seven native three-token fixture moves, replay/reset controls, and zero console errors.
- Responsive viewports: 1536×1024 and 390×844 both reported `scrollWidth == clientWidth`.
- Browser tool fallback: the in-app browser runtime could not create its kernel assets, so Playwright Chromium was used.
- Visual design source: `site/design/vm-inspector-concept.png`.
- Matrix-proof extension: six explicit tensor stages, eight generic runtime primitive equations, and all 45 generated SSA operations rendered from `site/artifact_trace.json`.
- Synchronization gate: the published JSON is compared operation-for-operation and attribute-for-attribute with `build_position_reconstruction_artifact()`.
- Updated regression: 64 tests passed after the inference-proof extension.
- Numerical matrix gate: `site/numeric_trace.json` is compared exactly with a fresh real execution; it includes 45 operations, all frozen matrices and SSA values, `QKᵀ [1,64,2064]`, hard attention `[1,64,2064]`, and final board `[1,64,128]`.
- Browser QA: final operation rendered 45 selectable steps, 200-row pagination over 132,096 nonzero score cells, 64 hard-attention cells, no horizontal overflow, and no console errors.
- Responsive numerical inspector: verified at default desktop and 390×844 mobile viewport; the inspector is now the first page content and uses measured black-on-white contrast.
- Updated full regression: 66 tests passed.
