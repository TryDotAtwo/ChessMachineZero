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
