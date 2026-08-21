# Project memory

- 2026-08-21: VM3 adds the first minimal terminal/absorbing slice. The compiler
  emits frozen terminal/result row routers, a fixed legal-count reducer,
  halfmove-150 detector and an 8-row terminal lookup table whose hard
  `d_head=2` selector maps tensor facts to RUNNING, WHITE/BLACK_CHECKMATED,
  STALEMATE or SEVENTY_FIVE_MOVE plus result and commit gate. The executor no
  longer hand-computes mate/stalemate branches; it forms the terminal query via
  fixed matmul row projections, applies deterministic ST selection, writes
  terminal/result back into the same state ABI, and appends only inactive MOVE
  padding for terminal states. Exact native CPU gates cover black checkmate,
  stalemate, automatic 75-move draw and absorbing recurrence. Repetition,
  claim-draw modes, insufficient-material subsets and GPU evidence for this
  terminal slice remain open.

- 2026-08-21: VM3 now has the first accepted perft gate around VM tensor
  outputs. A bounded batched sparse-hard path evaluates up to 16 states with
  compiler-frozen attack offsets, and `materialize_selected_trial_state` writes
  a selected trial board/side/castling/EP/clocks/fullmove result back into the
  same canonical state ABI through fixed tensor arithmetic and frozen row
  routers. The perft recursion is explicitly a test oracle harness, not runtime
  chess semantics. CUDA pytest evidence passes start position depths 1-3
  (20/400/8902), canonical Kiwipete depths 1-2 (48/2039), and depth-1
  promotion, en-passant and castling cases. Source purity remains clean with no
  new runtime operation classes. Terminal/repetition and genuine HullKV remain
  open.

- 2026-08-21: VM3 adds an exact single-state sparse hard-forward attack
  backend beside the unchanged dense/ST training path. It selects frozen
  between-square IDs and gathers only six possible ray blockers per
  target/attacker pair, avoiding the dense 64-value attack payload. Dense and
  sparse final legal tensors agree on the complete 520-position GPU oracle
  corpus, including the curated pin, en-passant, promotion and castling cases.
  This is explicitly not yet HullKV: king-square selection still performs a
  dense 64-key QK operation, and no convex-hull support query or certificate is
  claimed. Runtime purity remains clean; perft and genuine HullKV stay open.

- 2026-08-20: VM3 now computes own-king safety from candidate-batched trial
  boards. A frozen 64-cell circular key bank performs physical `QK^T -> ST
  hardmax`; `AV` carries piece attack relations and the complete between-square
  mask; frozen two-class `d_head=2` selectors reduce blocker, piece-attack,
  castling-safety and final LEGAL predicates. Selector keys live in the hashed
  program image, not runtime-created tensors. Fixed four-row routing evaluates
  transit attacks only for the four universal castle candidates. Exact tests
  cover pins, king adjacency, en-passant discovered check, and castling from,
  through and into attack; final legal sets match `python-chess` on 520 GPU
  positions. Current gates: native 12/12, Python 52/52, purity/public 22/22 and
  GPU king-safety backward. This is not yet the complete Task-11 acceptance:
  start/Kiwipete perft depth gates remain pending because the dense reference
  graph is not yet fast enough for depth 4.

- 2026-08-15: VM3 Task 10 adds special pseudo-legal transitions to the same
  immutable rule image. Every one of 4272 candidates now runs an identical
  60-token packet through 112 frozen physical `d_head=2` attention stages;
  state-dependent en-passant and castling predicates are computed from board,
  side, rights and raw-EP tokens. Candidate-batched matrix writes produce the
  full trial board plus castling, EP and move-clock fields. `recurrent_step`
  selects that complete trial tensor, so promotion, en-passant capture and
  castling are not reconstructed by host chess branches. Native CTest is 11/11,
  both current special/backward and full-ring tests pass on RTX 3070, source and
  public gates are 51/51, and the exact GPU oracle covers 520 positions. The
  boundary remains pseudo-legal: king safety, attacked castling, check/mate and
  terminal adjudication are still pending Task 11+.

- 2026-08-14: VM3 ordinary pseudo-legality now covers pawns, knights, bishops,
  rooks, queens, kings and all four promotions in the same 4272-candidate rule
  image. Identical 53-token packets run 85 frozen physical `d_head=2` Q/K/V
  stages; source/target lookup, side match, six ray-empty predicates and all
  Boolean reductions are attention operations. Compiler relations contain only
  universal geometry/between-square facts; board-dependent results come from
  the canonical state. Exact fixed-index HullKV routing and low-rank V factors
  remove zero work while preserving hard output and gradients. Native CTest is
  10/10, purity has zero findings, a 516-position RTX 3070 pseudo-legal oracle
  matches python-chess exactly after the Task-9 castling/EP filter, and GPU
  recurrent-gradient evidence passes. This is not yet king-safe/full chess:
  special moves, trial transitions and attack filtering remain open.

- 2026-08-14: VM3 now closes a minimal unified pawn-and-knight recurrent ring.
  One serialized and SHA-256-validated frozen image runs 4272 candidates as a fixed batch of
  identical 47-token packets through 34 physical `d_head=2` attention stages.
  Compiler-only rule tables provide universal pawn/knight geometry; runtime
  receives only board/side state plus raw candidates and performs fixed
  matmul, attention, ST selection and matrix board/state writes. A step emits
  one common LEGAL tensor, exact selected descriptor, same-ABI next state and
  one attached MOVE/KV slot; its output feeds the same step directly. Exact
  CPU/GPU tests cover white/black moves, captures, own occupancy, blocked double
  pushes, two-ply recurrence and gradient flow from output board through legal
  eligibility into input state. Acceptance executes only after a binary
  `save_frozen_program -> load_frozen_program` round trip, including direct GPU
  loading; the compiler is absent from the executor link closure. This remains a pawn/knight slice, not full
  pseudo-legal or king-safe chess.

- 2026-08-14: VM3 has a swappable bounded attention-only policy ABI. Raw
  trainable scalars are converted by stable signed-binary softmax before every
  matmul; all Q/K/V, learned queries and reusable outputs are re-bounded, every
  Q/K head is physically width two, and local candidate/control decisions are
  two-class ACCEPT/REJECT probabilities. The absolute frozen gate admits only
  rule-emitted legal/witness/control routes and has deterministic move,
  claim-after, auto-claim and halt behavior. Both side modules execute and a
  tensor side route selects output and emitted K/V. Initialization preflight
  recomputes fan-in, includes summed projection fan-ins and fixed margins,
  checks dtype/raw finiteness and rejects incomplete operation manifests before
  rollout. CPU and RTX 3070 tests prove permutation equivariance, extreme-finite
  stability, terminal-zero gradients, inactive-padding gradient invariance and
  gradient reachability through every active history slot.

- 2026-08-14: the VM3 compiler emits one universal, deterministic candidate
  bank: all 4096 source-major/target-minor square pairs, followed by all 176
  promotion records (44 square pairs times Q/R/B/N), plus one zero identity
  selector sentinel. Its frozen tensors contain only raw source, target,
  promotion and coordinate one-hots. No piece, geometry, occupancy or legality
  fact is precomputed into the bank; those remain work for the frozen rule graph.

- 2026-08-14: VM3 training trajectory uses one separate functional MoveSlot per
  fixed structural step: `commit*MOVE + (1-commit)*PAD`. There is no history
  `cat`, tensor cursor indexing, prior-slot mutation, `copy_` or recurrent
  detach. CPU/GPU tests prove exact committed/PAD values, distinct O(d_model)
  storage, unchanged prior pointers and gradient reachability to every active
  move and the commit predicate. Policy K/V attachment remains Task 7.

- 2026-08-14: the canonical VM3 recurrent state ABI is a same-shape 2D token
  tensor with predicate activity in column zero and immutable identity/workspace
  features in the remaining columns. Its 1724 core rows encode 64x13 square
  states, side, exact castling state, raw EP, clocks, radix fullmove/cursor,
  claim mode/availability, terminal and result. An initialization-only compiler
  binder verifies a checksum-loaded layout tensor, preserves workspace/storage,
  and binds on CPU or GPU. Exact native boundary tests and eight python-chess
  FEN oracle cases pass; no FEN adapter is linked to runtime.

- 2026-08-14: VM3 now has true physical `d_head=2` dense attention and an
  immutable grouped key-block backend. Both use the same `Wq/Wk/Wv`; block
  routing is fixed by row routers, query-group offsets and global key IDs.
  Forward hard winners and outputs are exact, including lowest global-index
  ties. The block surrogate normalizes across the complete query group;
  scores/Q/K receive dense-soft gradients while V/Wv receive hard-selection
  gradients. Batched CPU and RTX 3070 tests prove dense/block parity for
  X/Q/K/V/Wq/Wk/Wv at `1e-12` tolerance.

- 2026-08-14: VM3 implementation has started with the frozen tensor contract,
  generic checksum/interval/certificate program loader, adversarial runtime
  purity and operation-manifest gate, and the canonical deterministic
  straight-through selector. The selector is exact lowest-index hard in the
  forward pass and sends gradients through the eligibility-weighted stable
  surrogate without `detach`. This is infrastructure evidence only, not yet a
  recurrent chess or GPU-residency claim. The pinned container build remains
  unaccepted until a clean image passes its internal LibTorch integrity checks
  and the unified CI entrypoint; Docker VM TLS failures are currently external
  to the repository tests.

- 2026-08-14: a full architecture/autograd audit supersedes VM2 as the active
  implementation target, while preserving it as historical slice evidence.
  VM2 hard `max -> one_hot` cuts gradients to Q/K and policy logits; pawn and
  knight have incompatible layouts; the legal-set test is synthetic; policy is
  not connected to one common recurrent board write; trajectory, complete
  state, terminal logic and whole-game BPTT are absent; current heads are not
  physically `d_head=2`; the block path is not candidate-batch safe; and the
  main purity audit covers only four generic files.
- 2026-08-14: the active target is the clean `native/vm3` ring specified in
  `docs/superpowers/specs/2026-08-14-differentiable-recurrent-chess-machine-design.md`
  and sequenced in
  `docs/superpowers/plans/2026-08-14-differentiable-recurrent-chess-machine.md`.
  Its fixed contract is: one immutable rule image; byte-exact lowest-index hard
  forward with custom stable eligibility-weighted softmax surrogate backward at
  every selector; machine-checkable frozen-selector interval proofs; bounded
  trainable policy parameters/activations with pre-launch numeric validation;
  frozen rule weights and trainable trace-policy weights; no recurrent detach;
  same-ABI output-to-input recurrence; one MOVE token per committed ply;
  complete move legality plus explicitly declared metadata, claim and terminal
  routing (without overclaiming general FIDE dead-position recognition); true
  `d_head=2` dense training/reference attention; and a separately proven exact
  hard-forward HullKV inference backend. Public claims remain at the VM2
  pawn/isolated-knight boundary until corresponding VM3 native gates pass.

- 2026-08-14: the preclassified knight circuit and browser mirror are removed.
  They encoded the final legal class in X and are invalid evidence. Correct
  knight work must derive geometry from coordinate tokens via reusable frozen
  stages before it may re-enter acceptance or public claims.

- 2026-08-14: the replacement knight geometry circuit is valid parameteric
  evidence. Binding changes only four coordinate one-hots. Three frozen
  attention/matrix-write stages derive absolute file/rank deltas and the final
  geometry predicate. It is not yet integrated with pieces, occupancy, check
  legality or board transitions, so public full-chess claims remain unchanged.

- 2026-08-14: selected knight moves now include piece/side/target-occupancy
  legality, legal-gated source/target matrix writes, 64 output tokens and side
  transition in the fixed attention runtime. Check legality and integration
  into automatic multi-piece candidate selection remain pending.

- 2026-08-13: Pages presents the full pawn trace as 90 animated debugger
  microsteps over the real float64 artifact. It synchronizes board state,
  selected token labels, matrix heatmaps and hardmax focus. Knight geometry is
  still a separate geometry-only gate.

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
- The knight circuit is now compiled once independently of source/target; board,
  side and desired move live only in input tokens, so every move uses identical
  frozen weights and masks.
- Dense hardmax attention accepts a leading candidate batch dimension, enabling
  candidate transitions to run in parallel without per-candidate host dispatch.
- The temporary move policy consumes rule-emitted `[NOT_LEGAL, LEGAL]` predicate
  tokens, applies frozen matrix projections and hardmax-selects the first legal
  trace candidate. Its candidate-score tensor contract is intentionally the
  replacement seam for a future trained transformer policy.
- Pawn requested-move selection no longer changes an attention mask. One
  compiled circuit accepts the requested `(source, move_kind)` in
  `SelectedToken`; fixed stage-9 Q/K weights match it against all 256 pawn
  candidates.
- Pawn and knight predicate streams now share a `LegalSetAssembler`: two frozen
  row projections form one ordered `[NOT_LEGAL, LEGAL]` tensor consumed by the
  same swappable move policy.
