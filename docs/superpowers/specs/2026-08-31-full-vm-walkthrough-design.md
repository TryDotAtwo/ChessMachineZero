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
capacity exhaustion is explicit. The initial plan assumed compatibility with
the existing oracle's automatic exercise of claimable draws. That assumption
needs user confirmation: automatic claims versus a player claim changes game
behavior and potentially the request token language. The choice was raised on
2026-08-31 before implementing adjudication; no claim token/protocol change has
been made. Human arbitration, clocks and draw offers remain outside this task.

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

Use an artifact-driven, versioned execution manifest containing bootstrap and
ordered recurrent steps. Each step identifies request/prior context, operation
list, frozen values, intermediate tensors, named outputs, shape/axis semantics,
source/artifact identity and compute precision. Show the context feedback edge
explicitly and verify its exact equality across consecutive calls.

The native inspection path captures actual execution tensors. Inspection-only
CPU serialization is outside the production hot path; generic derivations of
unmaterialized attention scores must be labelled as derived views, not captured
native buffers. The site may decode/display tokens and verify generic scalar
arithmetic, but may not apply chess rules or synthesize game states.

Keep a small manifest plus content-versioned per-step data, loaded on demand.
The UI has iteration/stage navigation, then operation navigation, then arbitrary
scalar coordinates with exact contributing row/column/value provenance. Names,
purposes, axes, status and controls remain RU/EN. Validation stays fail-closed;
removing fixed `45/v45` assumptions must not weaken consistency checks.

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
