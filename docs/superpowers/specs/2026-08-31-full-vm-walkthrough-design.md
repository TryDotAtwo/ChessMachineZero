# Full executable VM walkthrough

## Authorization and scope

The user asked for the website to show a complete VM step by step and then
directed inline implementation without subagents. Work stays in the existing
clean worktree. The chosen website mode is recorded genuine executions; an
internet inference service and arbitrary browser-side chess execution are not
part of this change. The current 45-operation position subgraph is not renamed
or advertised as a complete chess VM.

## Execution contract

The immutable artifact owns `context_0` and a generic tensor graph. Its input
is FP32 hard one-hot `[B,2048,128]`: three request rows followed by the previous
`[B,2045,128]` context. A request is FROM, TO, RESULT_PIECE. Output is the next
context: 1200 history rows, 768 ordered legal-set rows, one status row, 76
reserved rows. No board is carried in recurrent context. The initial position
and reusable rules are frozen weights; intermediate boards are permitted.

The externally supplied context is an initial context or an unmodified prior
VM output. A player can supply arbitrary hard-one-hot requests. History
legality is established inductively by VM transitions, not by a CPU checker.
Terminal contexts are absorbing; an illegal request leaves history unchanged;
capacity exhaustion is explicit. The recurrent VM matches the existing
oracle's `outcome(claim_draw=True)` behavior. The user confirmed on 2026-09-01
that claimable fifty-move and threefold positions terminate automatically.
There is no `CLAIM_DRAW` request token. Human arbitration, clocks and draw
offers remain outside this task.

## Boundaries

Production C++/CUDA may dispatch only generic operations. No piece-specific
branches, chess search, evaluation, Python or oracle calls enter execution.
Python compiles reusable matrix/circuit templates offline and provides test
oracles. Float hardmax forward remains exact; its backward remains the declared
surrogate. No native FP4 arithmetic, learning benefit or speedup is claimed.

The graph needs layout-only transpose/reshape and GEMM between two intermediate
values, in addition to existing frozen projections, residuals, row routing,
hardmax and attention. These are mathematical operations, not chess opcodes.
They preserve the batch axis and validate static dimensions before GPU access.

## Computation stages

1. Reconstruct position and history-derived side/rights/counters.
2. Match position-independent candidate triples against source/target occupancy,
   pawn geometry, sliding paths and special-move eligibility.
3. Exclude candidates leaving the moving side's king attacked, including
   en-passant discoveries and the castling start/transit/destination squares.
4. Compact legal triples in the fixed lexicographic candidate order using
   tensor arithmetic; no host list sorting/filtering.
5. Match the requested triple, select unchanged or appended history, reconstruct
   the resulting position, and generate the next side's legal set and status.
6. Pack a hard-one-hot next context and pass those same values into the next
   invocation. Terminal/illegal/overflow decisions are tensors produced by VM.

These are semantic group names in the trace, not procedural runtime phases.
Every group must expand to inspectable generic operations and frozen weights.
Each new subgraph is separately named and tested until the complete composition
passes; partial subgraphs must never return a plausible substitute full context.

## Trace and website

Use an artifact-driven, versioned execution manifest containing bootstrap,
request/prior/output summaries and every ordered operation. Each operation
identifies its inputs/output, frozen values, shapes, bilingual semantics and a
compact exact numeric window with scalar provenance. Show the context feedback
edge explicitly and verify exact equality across consecutive native calls.

Resolution recorded 2026-09-01: native acceptance compares complete output
contexts, recurrent feedback and backward without adding an intermediate-dump
API to production. Website intermediate windows come from the exact Python
reference executor for the same serialized artifact and must be labelled as
non-native. Publishing all conservative batch-one retained SSA values would be
roughly3GiB; the website intentionally uses compact windows. Generic derivations
of scores are labelled, and the site may decode/display tokens and verify
generic scalar arithmetic but may not apply chess rules or synthesize states.

The UI has stage navigation and all-operation navigation. Every operation shows
the selected output cell, exact contributors available in its compact window,
and frozen-matrix meaning. The nested45-op position microscope retains complete
COO and arbitrary scalar coordinates. Names, purposes, axes, status and controls
remain RU/EN. Validation stays fail-closed for topology, producer order, feedback,
status/channel consistency and displayed scalar values; removing fixed
`45/v45` assumptions must not weaken the retained position inspector.

## Acceptance and publication

Require independent exact complete legal sets and next contexts against the
development oracle, covering both colors, checks/pins, castling restrictions,
en passant, promotions, illegal requests, mate/draw and capacity boundaries.
Check generic derivatives and a connected native input-gradient path separately
from forward chess correctness; do not equate it with useful player learning.
Measure allocation/weight sizes before launching enlarged graphs on the GPU.

Do not publish a full-VM claim until the native complete-transition tests and
export consistency checks pass. Then verify every exported operation path,
context feedback, scalar provenance, RU/EN, responsive layouts and the deployed
content-versioned resources. Preserve the prior audit and all explicit limits.
