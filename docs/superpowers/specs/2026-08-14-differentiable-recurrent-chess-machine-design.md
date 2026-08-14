# Differentiable Recurrent ChessMachineZero Design

## Status and correction

The current `native/vm2` branch is valid evidence for isolated pawn and knight
experiments, but it is not the requested machine ring:

- `attention.cpp` and `move_policy.cpp` use `max -> one_hot`; selection has no
  gradient to Q/K or policy logits;
- pawn and knight circuits have incompatible token counts, widths and depths;
- `LegalSetAssembler` is tested with synthetic predicates, not one shared board;
- policy selection is not routed into a common recurrent board write;
- there is no move trajectory, 2D KV cache, complete chess state, terminal
  circuit or full-game trainer;
- current Q/K tensors have physical width 320 or 208, so this code is not yet a
  strict `d_head=2` implementation;
- the production block-attention path is not candidate-batch safe;
- the main purity auditor covers only four generic runtime files.

Accordingly, `native/vm2` remains a frozen reference while the target is built
in a new `native/vm3` directory. Public claims remain at the current pawn-slice
boundary until each new milestone has native artifacts and exact tests.

## Objective

Build one immutable, GPU-resident, recurrent and end-to-end differentiable
ChessMachineZero graph. The graph receives a canonical chess state plus an
append-only sequence containing one logical token per committed move, emits the
complete rule-derived `LEGAL_SET`, scores it with a swappable trainable
attention-only policy, commits exactly one legal move with deterministic hard
forward semantics, returns the same state ABI, appends one move token, and
repeats until an internally computed absorbing terminal state.

The entire unrolled game is one autograd graph. Frozen rule tensors do not
receive gradients, but they remain in the path through which the terminal loss
reaches policy parameters and first-ply policy logits.

## Considered approaches

### A. New VM3 ring beside the accepted VM2 reference — selected

Create a clean ABI and differentiability contract under `native/vm3`. Port
accepted rule fragments only after exact comparison. This avoids weakening the
existing purity evidence and makes every migration gate reversible.

### B. Mutate VM2 in place

This minimizes filenames but mixes hardmax-only evidence, two incompatible
piece circuits, new autograd semantics and new runtime audit allowances. A
passing old test could easily be misread as proof of the new ring.

### C. Train policy outside the rule VM

This is conventional and simpler, but breaks the defining requirement: the
gradient would stop at move selection and board application instead of flowing
through the executed chess program.

## Non-negotiable invariants

### Runtime and compilation

1. One immutable rule image handles every position, side, ply, candidate and
   policy checkpoint.
2. Only initialization/offline compilation may use chess-aware control flow.
3. Runtime never performs semantic scalar reads, `.item`, CPU copies, dynamic
   candidate dispatch, state-dependent weights, masks, sparse plans or matrix
   selection.
4. Structural fixed loops are allowed until replaced by a captured device
   schedule; they may not inspect `HALT`, legality, moves or counters.
4a. The final GPU-resident claim requires one post-initialization device-schedule
    launch for the declared fixed horizon, with no D2H synchronization or host
    kernel scheduling between ticks. A shorter captured horizon may report
    `CAPACITY_EXHAUSTED` but may not be described as “runs every game to end”.
5. Runtime operations are 2D self-attention, deterministic ST selection,
   matrix projections/writes and necessary tensor arithmetic for the canonical
   ST expression.
6. Candidate tokens contain raw identity only. They never contain a legality
   label prepared by the host.
7. Python chess code exists only in test-oracle adapters.
7a. Chess-aware sources build a versioned, checksummed serialized
    `FrozenChessProgram` in a compiler-only target. The generic executor links
    only program loading, tensor execution, policy and tracing; its link closure
    and symbol map must contain no compiler or chess-rule objects.
7b. The serialized program contains machine-checkable interval-proof nodes and
    a `SelectorScoreCertificate` for every frozen selector. The generic loader
    recomputes bounds from canonical token domains and checksummed tensor values,
    and fails closed unless an eligible/sentinel score strictly dominates every
    ineligible score. Claimed numeric bounds alone are not trusted.

### Forward and backward

For every data-dependent selector a machine-checkable score contract constructs
a bounded score whose frozen eligibility/priority margins guarantee that its
maximum is an eligible row. Frozen rule selectors carry serialized interval
proofs; trainable policy selectors use the audited bounded parameterization in
invariant 34a. With `e` denoting the exact-forward `0/1` eligibility token:

```text
gated_scores = bounded_base_scores + frozen_margins(e, priority_tokens)
u = e * exp((gated_scores - rowmax(gated_scores)) / temperature)
soft = u / rowsum(u)
hard = one_hot(lowest_index_argmax(gated_scores))
selection = hard + soft - stop_gradient(soft)
```

The last line is the mathematical straight-through contract, not the literal
floating-point implementation. VM3 uses a custom autograd operation whose
forward returns the already-materialized `hard` tensor byte-for-byte and whose
backward applies the stable-softmax Jacobian saved or deterministically
recomputed from `scores`. A literal `hard + soft - soft.detach()` is rejected
because cancellation does not prove a bit-exact hard tensor in floating point.

8. `selection` is byte-exactly `hard` in forward execution.
9. Backward equals the declared softmax surrogate; this is explicitly a
   straight-through gradient, not the mathematical derivative of argmax.
9a. Evaluation/inference mode computes only exact hard routing. Training mode
    additionally computes stable-softmax surrogate tensors for backward, while
    the board/state forward value remains byte-exact hard. The site must show
    these as separate forward and surrogate-backward facts.
10. The canonical ST expression applies to rule lookups and policy selection,
    not only the final move.
11. Rule weights are `requires_grad=false`; recurrent activations are never
    detached between plies.
12. Only policy parameters are optimized.
13. The final terminal loss must produce finite, reproducible, non-zero
    gradients at first, middle and last policy decisions in a non-degenerate
    test game.
14. Additional fixed steps after terminal preserve state byte-for-byte, create
    no policy or trajectory contribution, and pass incoming state gradients
    through the identity route unchanged.
14a. Every selector has a compiler-provided finite sentinel/no-op row and a
     checked score-range certificate. Frozen margins exceed the certified base
     range, so at least one eligible row is always a global maximum and the
     denominator is positive. The executor performs only tensor arithmetic,
     `rowmax` and hardmax; it never constructs a state-dependent boolean mask.
     Temperature is a finite, strictly positive frozen program scalar validated
     before execution.
14b. Policy move/witness surrogate weights normalize only over the
     rule-eligible rows plus sentinel. Illegal descriptors therefore have zero
     surrogate transition weight. At the exact-forward boundary `e=0`, the
     declared counterfactual derivative is the derivative of the formula above.
     Holding gated scores fixed, with `a_i = exp((g_i-rowmax(g))/temperature)`,
     `Z = sum_i(e_i*a_i)` and `u_j=e_j*a_j`, it is
     `d soft_j / d e_i = a_i * (delta_ij*Z-u_j) / Z^2`; autograd also includes
     the declared path through frozen score margins. This eligibility gradient
     is tested analytically rather than hidden or called an argmax derivative.

### Exact move selection

15. Candidate and control policy scores are bounded probabilities in `[0,1]`
    produced by attention. Each candidate obtains its probability from the same
    local `ACCEPT/REJECT` attention comparison; each of the three control-token
    rows is scored by an analogous shared local comparison. A preliminary
    softmax over either the 4272-candidate axis or the three control rows is
    forbidden because a disabled row would then rescale enabled-row scores
    through its denominator. Cross-route normalization happens only once, in
    the eligibility-weighted ST selector.
15a. Candidate scoring is permutation-equivariant over trace rows; trainable
     parameter shapes and counts are independent of 4272, and candidate-ID
     embeddings, per-candidate biases or a flat `[d_model, 4272]` head are
     forbidden.
16. There are exactly 4272 chess-move candidates containing only source,
    target, promotion and raw coordinate one-hots. A separate three-way control
    head chooses `COMMIT_MOVE`, `CLAIM_NOW` or `CLAIM_AFTER_INTENDED_MOVE`;
    `HALT_IDENTITY` is a fourth rule-controlled route.
17. Candidate selectors use `2 * ELIGIBLE + candidate_probability` plus a
    fixed sentinel that dominates only when no candidate is eligible. Control
    routes use `2 * ALLOWED + control_probability`; auto-claim adds `4` to
    `CLAIM_NOW`, and halt receives `6 * HALT_REQUIRED`. Thus no illegal packet
    can affect the exact forward state, and halt dominates terminal state. The
    backward path follows the explicitly declared eligibility surrogate; it is
    not misreported as an argmax derivative or as exact chess semantics.
18. The committed or intended packet is computed by matrix multiplication of
    exact ST selection with fixed descriptors. Only `COMMIT_MOVE` updates the
    board and appends one token. Claim and halt routes preserve board/history.
18a. Integer winner indices exist only in tests/trace diagnostics. Runtime state
     transitions consume the floating exact/ST one-hot and never gather, index
     or cast a semantic predicate to integer.

### Chess state and history

The recurrent `ChessRingState` contains:

- `BOARD[64]` over `EMPTY, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK`;
- side to move;
- the exact 16-state `KQkq` castling-rights value;
- `EP = NONE | square`;
- halfmove clock and fullmove number;
- claim availability, terminal kind and game result;
- an initialization-bound adjudication mode (`LITERAL_CLAIM` or
  `AUTO_CLAIM_SELF_PLAY`), consumed as data by the same immutable graph;
- device-resident trajectory cursor;
- a logical append-only trajectory tape;
- fixed candidate and rule workspace rows.

19. One committed move appends exactly one logical `MOVE` token. Terminal steps
    append none.
20. Policy supplies only `source`, `target` and promotion choice. Capture,
    castling, en-passant, pawn-move and metadata effects are rule-emitted.
21. A move token carries its visible move fields and the exact post-position
    identity required for repetition, plus the rule-emitted ply address/side
    needed to preserve sequence order; no collision-prone hash is accepted as
    rule evidence. These remain fields of the same one logical token.
22. Position identity comprises board, side, castling rights and only an
    actually available en-passant right. Clocks are excluded. The initial
    position is occurrence one.
23. En passant removes the pawn from the captured square, not the target, and
    then passes the common own-king-safety test.
24. Castling rights are irreversibly removed by a king move, an original-rook
    move, or capture of that rook. Origin, transit and destination attack
    predicates are required.
25. Promotion always selects Q/R/B/N; a pawn cannot remain a pawn on the last
    rank.
26. The rule graph emits mate, stalemate, fivefold repetition and 75-move
    automatic terminals, plus threefold/50-move claim predicates. A separately
    named `MATERIAL_DEAD_SUBSET` may cover exactly oracle-validated material
    cases. General FIDE dead-position recognition is not part of the v10 claim:
    `python-chess.is_insufficient_material()` alone is not a complete oracle.
27. `LITERAL_CLAIM` supports both a claim in the current position and a claim
    accompanied by the legal intended move that would create the third
    occurrence or 100th halfmove. `AUTO_CLAIM_SELF_PLAY` force-selects only a
    currently available claim after it exists. The site and tokens must label
    auto-claim as an adjudication convention, not an automatic FIDE rule.
28. Mate on the final move has priority over a 75-move terminal.
29. Terminal state is absorbing and byte-identical under all remaining fixed
    unroll steps.
29a. Training represents each structural ply as a separate functional move/K/V
     slot in the static unrolled graph; it never mutates an autograd-visible
     history tensor in place and never recopies the full capacity per ply.
     Inference uses a separately verified device-paged append backend. Both
     expose the same logical active-token sequence and exact hard forward.
29b. Terminal classification is performed at the start of a ring tick on the
     current fully written position: legal-set emptiness/check first, then
     automatic draws and claims. A committed move updates state but cannot be
     pre-empted by a draw classification until the next tick, which preserves
     mate-over-75 priority without a host branch.
29c. Orthodox play has at most 96 pawn moves and 30 captures. With at most 149
     non-reset halfmoves before each reset and 150 after the final reset, a
     conservative 75-move-rule bound is 19,050 committed plies plus one terminal
     classification tick. Any deliberately shorter training horizon emits
     `CAPACITY_EXHAUSTED`, which is not a chess terminal and supplies no terminal
     result loss.

### Training and data

30. No external tree search, handcrafted evaluation, engine labels, tablebase
    labels or human-game data.
31. The policy ABI accepts a `PolicyPair`. It may bind shared storage for a
    side-conditioned experiment, but competitive training uses independent
    white/black modules and fixed tensor side routing; host code never selects a
    module from the current side.
32. The first training deliverable is a gradient-transport harness, not a claim
    of learned chess strength. It alternates optimization: with Black frozen,
    White minimizes `-(WHITE_WIN - BLACK_WIN)`; with White frozen, Black
    minimizes the opposite. Frozen opponent weights still pass gradients with
    respect to recurrent state. `CAPACITY_EXHAUSTED` produces no result loss or
    optimizer step.
33. Host training code may invoke forward, backward and optimizer steps and
    read final metrics after completion. It may not choose/apply moves or stop a
    rollout based on semantic tensors.

### 2D lookup and scaling

34. Every target lookup head has `d_head=2`; values and model state may be
    wider. Multi-head composition is allowed.
34a. Every one of the two physical coordinates of a trainable policy Q/K vector
     is produced as `2*stable_softmax([linear_projection_j, 0])[0]-1` before the
     dot product. Thus Q and K fill the bounded square `[-1,1]^2`, rather than
     collapsing onto a one-dimensional probability simplex, and every QK score
     remains in `[-2,2]` for all finite parameter values after any optimizer
     step. A frozen internal-attention eligibility margin greater than four
     makes an active history row dominate every inactive/PAD row. Local
     candidate/control `ACCEPT/REJECT` probabilities remain in `[0,1]` and use
     their own certified route margins. Policy attends
     all statically allocated history slots with tensor eligibility tokens; it
     never host-filters an active list. Non-finite policy parameters are a
     fail-closed training error, not a runtime fallback.
34b. A raw trainable parameter is never a matrix-multiply operand. Each raw
     scalar is first mapped by the same stable signed-binary construction into
     `[-1,1]`; canonical inputs are in `[-1,1]`; every Q/K/V, learned-token,
     output and readout projection is re-bounded before reuse. A checked
     `PolicyNumericContract` fixes dtype, maximum fan-in and outward-rounded
     accumulator bounds far below the dtype maximum. Inference checks raw
     parameter finiteness before its one post-initialization launch. Training
     checks it immediately after every optimizer step and again before scheduling
     the next rollout; failure prevents that rollout, so no damaged activation
     graph is executed or accepted as evidence.
35. Dense ST attention is the training/reference backend and computes exact
    stable-softmax surrogate gradients.
35a. The exact block backend consumes the same `Wq/Wk/Wv` as dense attention.
     Its immutable plan contains row routers, globally ordered key identifiers
     and contiguous query-group offsets. It performs a global two-pass stable
     normalization across every key block in a query group. Hard selection is
     lexicographic by global key index; `V/Wv` receive the hard-selection
     gradient while scores, Q and K receive the identical dense surrogate
     gradient. Block membership is structural and never selected from tensor
     values.
36. The convex-hull 2D KV backend is an inference backend for exact hard lookup
    and must match dense forward, including lowest-index ties.
37. No claim is made that exact dense softmax normalization is `O(log n)`.
    Million-token claims require separate measured hard-lookup evidence.
38. A future deterministic top-k surrogate is a separate approximation and
    cannot replace dense-gradient acceptance silently.
39. Full-game BPTT uses activation checkpointing/recomputation without
    detaching recurrent state.
39a. HullKV is eligible only for pure two-term dot-product heads over an active
     prefix within a compiler-declared static route class. Arbitrary additive
     query/key bias is ineligible; fixed masks become separate immutable hull
     classes. `q=(0,0)`, duplicate keys, collinear faces and lowest original
     token index have explicit metadata and tests.
39b. Exact-equivalence evidence disables TF32 and uses the same declared dtype
     and `q0*k0 + q1*k1` evaluation order. Append/build/query/end-to-end costs
     are reported separately; ordinary dense softmax remains the backward
     reference.

### Evidence and publication

40. Acceptance requires exact behavior, mutation purity, dense/hull forward
    equivalence, analytic ST-gradient checks, full-game gradient reachability,
    deterministic reruns, GPU-residency traces and measured memory/time.
41. The frontend only visualizes native artifacts. It never computes rules,
    winners, softmax values, gradients or board transitions.
42. Public claims advance only when the corresponding native acceptance gate
    and artifact are committed.

## Data flow of one ring step

```text
ChessRingState_t + MOVE_0..MOVE_t
  -> frozen candidate construction
  -> frozen pseudo-legal predicates
  -> frozen trial transitions
  -> frozen attack / own-king-safety predicates
  -> LEGAL_SET
  -> current-position mate / stalemate / automatic-terminal predicates
  -> trainable shared candidate probabilities and 3-way control probabilities
  -> legal move selector + claim-after intended-move witness selector
  -> control scores for COMMIT_MOVE / CLAIM_NOW / CLAIM_AFTER_INTENDED_MOVE
  -> HALT_IDENTITY score = 6*HALT_REQUIRED
  -> deterministic exact-forward/ST-backward candidate and control one-hots
  -> selected packet = selection^T * fixed raw candidate descriptors
  -> frozen board, metadata, clock and repetition writes
  -> append one MOVE token iff a move was committed
  -> absorbing gate
  -> ChessRingState_t+1
```

The output has the same tensor schema and allocated storage layout as the
input, so the next step consumes it directly. Terminal classification of that
resulting position is the first frozen phase of the next tick.

## Candidate bank

The correctness reference starts with a universal, fixed ordering:

- 4096 ordinary `source x target` candidates;
- 176 geometrically valid promotion variants;
- total `kMoveCandidateCount = 4272` chess moves;
- three control actions over the selected candidate/witness;
- one separate internal `HALT_IDENTITY` control route;
- one fixed sentinel row for an empty eligible-candidate selector.

The 176 promotion rows are 44 geometrically possible source/target pairs times
four promotion pieces. The corresponding ordinary source/target rows remain in
the universal bank but are rule-illegal when a promotion choice is required.

Optimization may later remove structurally impossible pairs, but candidate
ordering and behavior must remain exactly mapped and frozen. Castling and en
passant use ordinary source/target identities; their special effects are
derived by rules.

## Determinism contract

- lowest-index hard tie-break;
- fixed temperature and dtype per evidence profile;
- deterministic LibTorch algorithms and CUDA BLAS workspace configuration;
- no RNG in the initial acceptance policy;
- identical state/result bytes and gradient tensors on repeated runs on the
  same declared backend and hardware profile;
- cross-device numerical equivalence is tolerance-based unless bitwise identity
  is separately demonstrated.

## Milestone claim ladder

1. ST primitive only.
2. Canonical recurrent state and one-token append.
3. Unified pawn/knight ring.
4. Full pseudo-legal candidates.
5. Full legal candidates with own-king safety.
6. Exact metadata and special moves.
7. Exact terminal/claim state and absorbing ring.
8. Full-game gradient reaches first ply.
9. GPU-resident captured schedule.
10. Measured 2D hull lookup and updated public debugger.

No milestone inherits the claim of a later milestone.

“Exact terminal/claim” above means the explicitly enumerated v10 reasons, not a
claim to solve general FIDE dead-position reachability. That requires a separate
design and proof gate.

## Source boundaries

- [Percepta, “Can LLMs Be Computers?”](https://www.percepta.ai/blog/can-llms-be-computers)
  motivates executing a program inside transformer inference. It does not by
  itself prove any ChessMachineZero implementation claim; those come only from
  native artifacts and gates in this repository.
- [FIDE Laws of Chess, effective 1 January 2023](https://handbook.fide.com/chapter/E012023)
  are the rules source for move legality, current/intended-move draw claims,
  fivefold/75-move automatic draws and mate priority.
