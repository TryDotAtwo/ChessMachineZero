# Percepta Chess Pages evidence — 2026-08-11

Public URL: <https://trydotatwo.github.io/ChessMachineZero/>

## Deployment verification — 2026-08-12

- Workflow run `31568575699`: build and deploy succeeded.
- Deployment branch policy allows only `main` and
  `codex/percepta-transformer-vm`.
- Obsolete account-level `яндекс.рф` CNAME was removed from
  `TryDotAtwo.github.io` by commit `2367bbc` with explicit user approval.
- Public HTML returned HTTP 200 and the expected Percepta title.
- `assets/index-Bvba-d-8.js` returned HTTP 200, 207576 bytes.
- `assets/index-IoNQkSPf.css` returned HTTP 200, 11740 bytes.

## Build

```text
npm run build
vite v8.2.1
1795 modules transformed
dist/index.html                  0.75 kB
dist/assets/index-IoNQkSPf.css 11.74 kB
dist/assets/index-Bpkc4ZLs.js 207.69 kB
built in 1.41s
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

The site now visualizes the implemented 15-stage symmetric ordinary pawn
push/capture circuit, including color-aware target routing, intermediate-square
routing and the predicate-token legality conjunction. It still labels complete
chess as unfinished.
