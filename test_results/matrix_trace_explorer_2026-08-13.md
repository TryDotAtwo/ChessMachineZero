# Matrix trace explorer — 2026-08-13

- Native exact trace test: 15 stages; Q/K/V, QK^T+M, hardmax and sequential
  matrix-write identities pass.
- Artifact: `cmz.matrix-trace.v1`, float64, 17,758,655 bytes, 15 stages,
  score matrices 1004x1004.
- Site: 8/8 Vitest checks and production Vite build pass.
- Purity/public evidence: 18/18 pytest checks pass.
- Browser snapshot verified all 15 stage controls, all required matrices,
  arbitrary coordinate inputs, scalar arithmetic, hardmax winner and R0/C0,
  R1/C1, R2/C2 write equations.
- Claim boundary: this remains the implemented pawn-rule slice; full chess is
  explicitly not claimed.
