# Site layout QA

- Viewports: 1253x861 desktop and 390x844 mobile.
- Desktop Q x K^T + M = S grid: all seven children fit in one 746.8 px row.
- Horizontal document overflow: false at both viewports.
- Mobile board: 354x354 px, square aspect retained.
- Interaction: Numbers -> Full trace exposes `Полная матрица S`.
- Browser console after reload: 0 errors, 0 warnings.
- Vitest: 16/16 passed.
- Vite production build: passed.
- Source purity/public evidence: 18/18 passed.
- In-app Browser bootstrap was unavailable due to missing kernel-assets path;
  QA used the approved Playwright CLI fallback.
