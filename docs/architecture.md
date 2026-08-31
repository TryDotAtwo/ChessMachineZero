# Pure frozen Transformer chess VM

This is a from-scratch branch, not a migration of the retired engine.

## Implemented scope versus target

The **executable artifact currently reconstructs a position from an already
valid chronological history**. `build_position_reconstruction_artifact()` emits
45 generic operations and a final `[B,64,128]` one-hot board. Its input has shape
`[B,2048,128]`, but only history rows participate. It ignores the requested move
in rows 0–2 and does not apply or validate that request.

The complete recurrent chess environment is a **target, not a completed
capability**. It must eventually map a request plus previous context to the next
`[B,2045,128]` context, generate the current legal set, and determine terminal or
illegal-move status. No executable artifact does that complete transition yet;
no trained player or demonstrated full-game learning result exists here.

`FrozenVm::execute_graph` exposes the position artifact for testing/composition.
`FrozenVm::forward` requires the recurrent output shape; it must not silently
accept a board-only artifact as a finished chess VM.

## Tensor protocol

A move has three hard one-hot rows `[FROM, TO, RESULT_PIECE]`. Square channels
encode file/rank, and channels 96–107 identify the 12 color-specific resulting
pieces. A promotion directly supplies the promoted result piece.

The target recurrent context comprises 1200 history rows (400 plies), 768
`LEGAL_SET` rows (256 triples), one status row (channels 89–94), and 76 reserved
rows. Request plus context gives 2048 rows of width 128. This layout does not
itself establish that legal-set/status execution exists.

The compiled `context_0` contains padded history, 20 sorted opening moves, OK and
padding service rows. It has no board rows. Bootstrap and reusable rule-relation
images contain frozen constants but have empty operation lists; they are not
executable complete chess artifacts. The position artifact separately contains
its frozen initial-position constants.

## Runtime boundary

Production C++/CUDA loads immutable tensor records and executes generic
operations: row routing, projection/GEMM, addition, concatenation, batch
expansion, hardmax and 2D attention. The generic executor also supports a gated
FFN for other graphs. No opcode has chess-specific semantics.

Chess rules have not disappeared: their structure is compiled **offline into
weights and graph connections**, not evaluated by piece-specific C++ branches.
Python and python-chess are development/compiler/oracle tools, not production
inference dependencies. The browser is a static inspector, not a replacement
runtime.

## Position reconstruction

Three strided views select the history's source, destination and result-piece
rows. Four castling patterns and 28 en-passant patterns are recognized through
frozen projections, additions and hardmax. These recognize events in a valid
history; they are **not legality checks**. This subgraph trusts the declared
result piece.

There are 2064 candidate events: 64 initial-square events plus five streams of
400 source-empty, destination-piece, castling-rook-source,
castling-rook-destination and en-passant-clear events. Targets and payloads are
separate tensors. Each square selects its latest event; the board is not carried
as recurrent state.

For compact square index `s`, `x = 1 + s/64`. Query `(2*xq,-1)` and key
`(xk,xk²-epsilon*t)` give:

```text
score = xq² - (xk-xq)² + epsilon*t
epsilon = 2^-21
0 <= t <= 400
```

The nearest different square has penalty `2^-12`; maximum time bonus is
`400*2^-21`. Their separation is `112*2^-21 > 0`. Materialized FP32-score
tests check every square/timestamp and strict chronological ordering. The
earlier epsilon `1e-6` violated this bound and produced incorrect boards.

## Numerical representation and backward

The artifact stores packed E2M1/FP4 codes **plus FP32 scales**. Several records
use one FP32 scale per element: 4.5 bytes per element before metadata, not
uniform four-bit weight memory.

The native loader decodes records into **FP32 CUDA tensors**. Native attention
also requires FP32. Native FP4 compute, exact FP16/BF16 fallback, and TF32
equivalence are not established. Reduced precision needs its own numerical
encoding and task-level acceptance; blindly casting these weights is invalid.

Hardmax forward selects the lowest-index eligible maximum. Its first-order
backward is a softmax-Jacobian surrogate at the specified temperature. Attention
forward returns the winning value exactly; backward differentiates the
softmax-weighted value over selected top-k competitors, with gradients to Q, K
**and V**. Candidate selection itself is discrete. The development reference
uses this same specified surrogate, not full-candidate softmax.

The standalone FP4 primitive uses quantized forward and an identity surrogate;
it is not evidence that the executor computes GEMMs in FP4.

A surrogate gradient is not the true derivative of chess and does not prove
that a player learns faster or better. Tests concern forward correctness and
first-order derivatives. Useful learning, long-rollout gradient quality and
higher-order derivative equivalence require separate experiments.

## Attention and performance boundaries

Attention keys/queries have width two. Static key sets can use offline nested
convex-layer candidate metadata with stable original-index ties. Dynamic Q/K/V
uses batch-local candidate indices.

The position artifact provides **all 2064 candidates**. The native kernel scans
them while maintaining top-8. It avoids constructing a dense score matrix, but
is not a demonstrated sublinear geometric lookup or measured speed advantage.
Runtime allocations remain; there is no allocation-free hot-path claim. CUDA
device guards/current streams must be preserved, with no host readback in
steady-state inference.

## Development evidence and site

The numerical website trace is exported from the Python/PyTorch development
executor, not captured from every native CUDA intermediate. Native tests are
separate evidence and must identify their exact fixtures and build.

The initial audit, including counterexamples and limits of the old tests, is
[test_results/audit_2026-08-31.md](../test_results/audit_2026-08-31.md).
Build commands and current acceptance evidence are documented separately.
Passing tests are not proof of the unimplemented complete chess environment.
