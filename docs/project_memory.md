# Project memory

- 2026-08-13: Pages presents the full pawn trace as 90 animated debugger
  microsteps over the real float64 artifact. It synchronizes board state,
  selected token labels, matrix heatmaps and hardmax focus. Knight geometry is
  still a separate geometry-only gate.

- 2026-08-13: Pages includes an interactive knight matrix lab backed by the
  same 3-to-2 frozen classification construction as the exhaustive native
  gate. It must remain labelled geometry-only until board writes are wired.

- 2026-08-13: knight geometry gate factorizes source/target pairs instead of
  adding eight direction-specific candidate banks. A frozen matrix/hardmax
  circuit exhaustively classifies all 4096 square pairs. It is not yet wired
  into board legality or production move application.

- 2026-08-13: compact Q/K projection is exact but not production-enabled; two
  CPU LibTorch full-gate trials exceeded 900 seconds. Production retains dense
  projections with frozen block-sparse score attention.

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
- `block_self_attention` now executes a frozen selection-matrix block plan and
  is byte-exact with dense attention for outputs and global hardmax ties; the
  production transition still uses the dense reference path.
- `compile_attention_blocks` now compiles a square frozen mask offline by
  grouping queries with identical finite key sets; compiled plans are frozen
  and byte-exact with dense output and global hardmax winners.
- The approved visual site lives in `site/` and deploys from the orphan branch
  through `.github/workflows/pages.yml` to the repository GitHub Pages URL.
- The Pages UI is now a numeric matrix-trace explorer backed by a 15-stage
  native-exported float64 artifact; it exposes arbitrary coordinates of
  X/Q/K/V/S/A/Y/XPrime and all three sequential R/C write components.
- Current chess masks contain 203,861 finite score pairs versus 15,120,240
  dense pairs (74.1694x theoretical reduction), but exact-route grouping yields
  748–1004 blocks per stage and is rejected due to launch overhead.
- A budgeted offline merge preserves masks exactly; stage 2 reaches 171,920
  padded pairs in 16 blocks and is byte-exact with dense attention. Dense
  selection matrices are still prototype-only and no speedup is claimed.
- Frozen sparse COO selection uses one nonzero per route row. Stage 2 remains
  byte-exact and measured 1.43941x faster than dense in a five-run warm
  single-thread CPU diagnostic; full-transition/GPU performance is unverified.
- Production transition now uses 1–16 frozen block-sparse plans on all 15
  stages and is byte-exact with the dense trace oracle. Three-run warm
  single-thread CPU median improved 1346.7 to 1007.3 ms (1.33694x); GPU remains
  unmeasured.
- Site claims must be driven by implemented evidence: symmetric pawn circuit,
  recurrent board state, current CTest and purity counts; full chess remains
  explicitly pending.
- The public site is one synchronized native-trace laboratory and must not
  derive chess legality in TypeScript.
- The former browser-side knight demo is withdrawn from public evidence because
  its input token already contained the classified geometry label.
