# Project memory

- The active implementation is the orphan branch `codex/percepta-transformer-vm`.
- The inference runtime is `native/vm2/src/attention.cpp` plus
  `native/vm2/src/machine.cpp`.
- Runtime executes a fixed number of identical transitions and never exits
  early on `HALT`.
- `HALT` is an absorbing token state implemented by frozen attention routing.
- State writes are frozen batched matrix projections; elementwise write masks
  are offline compiler intermediates only.
- Opcode-specific compiler control flow is permitted only offline. The runtime
  target must not link `cmz_vm2_compiler`.
- Purity acceptance requires functional CTest plus mutation-tested source and
  graph-operation audits.
- The VM now has sixteen instruction slots, four predicate slots, and fifteen
  fixed attention stages.
- `CMP_EQ` emits `FALSE/TRUE` through frozen equality relation tokens.
- `JUMP_IF` builds target/fallthrough scratch tokens and selects one through
  predicate-keyed hardmax attention.
- Runtime write routing uses compact row and feature matrices with
  `X + R @ (Y-X) @ C`; it does not use per-token 3D projection banks.
- The chess slice compiles 256 pawn push/capture candidates, 512 side-keyed
  geometry tokens, and small predicate relation tables into a
  generic ProgramImage.
- Double-push legality is the attention-computed conjunction of matching pawn,
  empty target, clear intermediate path, valid geometry, and start rank.
- White and black pawn color, direction, legality and next side are all routed
  from tokens; no color branch was added to the runtime.
- Legal board application changes source, target, and side only through
  attention routing; an illegal selected candidate is an inference no-op.
- Full chess is not yet implemented and public claims must remain slice-scoped.
- Stage 10 attention now copies emitted board/side fields into recurrent input
  tokens; repeated transitions reuse the same ProgramImage and frozen weights.
- `compile_chess1_auto` enables an inference player: selected-token hardmax
  chooses the lowest-index legal candidate, and a three-ply recurrent test
  alternates white/black/white with unchanged weights.
- Exact-square write score is 1.0 and the legal gate contributes 0.25; this
  makes matching legal writes win at 1.25 while preventing unrelated legal
  writes from erasing other squares.
- Ordinary diagonal pawn captures are implemented for both sides with
  color-aware target tokens and file-boundary geometry.
- The next architectural gate is compact geometry factorization, followed by
  non-pawn movement and attacked-square/check circuits.
- `compile_chess1_circuit` materializes immutable tensors once;
  `bind_chess1_board` clones only the token matrix and shares weight/mask
  storage byte-for-byte with the circuit template.
- Pages deployment now depends on a source-aware evidence job: runtime purity
  and public-claim/schema consistency must pass before deploy.
- Sparse-attention scaling must use offline-compiled sparse selection matrices;
  runtime gather/index-select/nonzero or state-dependent block construction is
  explicitly rejected as hidden host routing.
- The approved visual site lives in `site/` and deploys from the orphan branch
  through `.github/workflows/pages.yml` to the repository GitHub Pages URL.
- Site claims must be driven by implemented evidence: symmetric pawn circuit,
  recurrent board state, current CTest and purity counts; full chess remains
  explicitly pending.
