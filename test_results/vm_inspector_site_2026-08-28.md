# VM inspector site verification

- Source: `site/`; deployment: `.github/workflows/pages.yml` uploads only `site/`.
- Contract gate: exact current tensor shapes, 45 generic ops, 2064 events, and the unfinished `LEGAL_SET`/terminal boundary are asserted by `tests/test_site_contract.py`.
- Browser behavior: verified a 64-cell reconstructed board, seven native three-token fixture moves, replay/reset controls, and zero console errors.
- Responsive viewports: 1536×1024 and 390×844 both reported `scrollWidth == clientWidth`.
- Browser tool fallback: the in-app browser runtime could not create its kernel assets, so Playwright Chromium was used.
- Visual design source: `site/design/vm-inspector-concept.png`.
