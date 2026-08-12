# Percepta Chess Pages evidence — 2026-08-11

## Build

```text
npm run build
vite v8.2.1
1795 modules transformed
dist/index.html                  0.75 kB
dist/assets/index-IoNQkSPf.css 11.74 kB
dist/assets/index-Bvba-d-8.js 207.57 kB
built in 767ms
```

## Browser verification

- Desktop render loaded without console errors.
- Pause changes to `Запустить ход`.
- `Один шаг` advances the active inference stage.
- `Sparse QKᵀ` reveals the sparse hardmax matrix trace.
- Mobile viewport: `clientWidth=375`, `scrollWidth=375`; no horizontal page
  overflow.
- Desktop implementation screenshot: `pages_desktop_viewport_2026-08-11.png`.

## Claim boundary

The site now visualizes the implemented 15-stage symmetric pawn single/double
circuit, including intermediate-square routing and the predicate-token
legality conjunction. It still labels complete chess as unfinished.
