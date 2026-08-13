# Animated full-trace debugger — 2026-08-13

The Pages debugger consumes the existing 15-stage native float64 pawn trace.
Each stage exposes Q, K, S, A, V and XPrime as six ordered microsteps, with a
synchronized board, token lane, normalized heatmap, hardmax focus, playback,
speed and detail modes. The exact full matrix explorer remains below.

Vitest: 15/15 passed. Vite production build passed. This is a visualization of
the pawn-rule trace; it does not expand the full-chess implementation claim.
