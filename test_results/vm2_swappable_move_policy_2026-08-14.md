# VM2 swappable move policy — 2026-08-14

- Knight rule circuit compilation is independent of board and desired move.
- Source, target, board and side are input-token data only.
- Batched dense attention equals independent inference exactly.
- Rule output exposes `[NOT_LEGAL, LEGAL]` predicate tokens.
- Frozen fallback policy uses two matrix projections plus hardmax; it does not
  inspect chess state on the host.
- Exact checks: attention, knight transition/policy integration, standalone
  policy replacement contract, runtime purity and public evidence gates.
- Docker test environment had no NVIDIA driver; CPU LibTorch exactness was
  verified, while real GPU residency/performance remains unverified.
