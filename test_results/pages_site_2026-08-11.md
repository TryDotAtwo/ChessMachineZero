# Percepta Chess Pages evidence — 2026-08-11

## Build

```text
npm run build
vite v8.2.1
1795 modules transformed
dist/index.html                  0.75 kB
dist/assets/index-IoNQkSPf.css 11.74 kB
dist/assets/index-D3tgrqoh.js 205.07 kB
built in 954ms
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

The site visualizes the currently implemented symmetric single-pawn circuit,
recurrent board/side state, fixed attention stages and purity evidence. It
labels complete chess as unfinished.
