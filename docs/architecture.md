# Pure frozen Transformer chess VM

This branch is a from-scratch implementation, not a migration of the retired
engine. The production boundary is one serialized frozen tensor graph executed
by the generic C++/CUDA runtime.

## Executable recurrent contract

`build_recurrent_artifact()` emits the complete one-ply transition:

```text
input  [B, 2048, 128] = requested move [B, 3, 128]
                         + previous context [B, 2045, 128]

output [B, 2045, 128] = next context
```

The request is three hard one-hot rows `[FROM, TO, RESULT_PIECE]`. Square
channels use the numeric square language (`a1=11`, `a2=12`, `b1=21`);
channels 96–107 identify the twelve color-specific result pieces. Promotions
supply the promoted result piece directly. There is no separate human-readable
move object at runtime.

The context is fixed:

- 1200 rows: 400 chronological plies, three rows per ply;
- 768 rows: at most 256 sorted legal triples, token-0 padded;
- one row: `OK`, `ILLEGAL_MOVE`, `WHITE_WIN`, `BLACK_WIN`, `DRAW`, or
  `HISTORY_OVERFLOW` in channels 89–94;
- 76 reserved service rows.

The artifact contains `context_0` with empty history, the exact twenty opening
moves and `OK`. For every accepted move, the output context is the next input
context byte-for-byte; C++ only needs to concatenate the next three-row request.
Terminal win/draw contexts are absorbing. Illegal requests do not append to the
history. A request at the 400-ply capacity emits `HISTORY_OVERFLOW` explicitly.

The current artifact has 2,877 ordered operations and 192 immutable tensor
records. Its final output is the full `[B,2045,128]` context, not a board-only or
legal-set-only substitute.

## Where the chess rules live

Chess semantics exist in the offline compiler. It synthesizes fixed relation
matrices, candidate tables and graph topology; the C++ runtime does not discover
or branch on pieces, colors, squares, castling or en passant.

Position recovery is the original 45-operation subgraph. It reconstructs the
current 64-square one-hot position from chronological result-piece events. Four
castling patterns and 28 en-passant patterns become frozen projections and hard
selection. Latest-event attention uses 2D parabolic keys over 2,064 events with
`epsilon=2^-21`; materialized FP32 tests prove square separation and time order
through all 400 plies.

The legal circuit starts from 7,780 position-independent move candidates. Frozen
geometry and tensor gates check side/source/target/path constraints, promotions,
castling rights and transit, en passant, and post-move king safety. Castling rook
relocation and en-passant victim removal are included in each candidate's
post-board. Candidate order is canonical at compile time; two-level prefix GEMMs,
hard rank routing and payload GEMM perform stable compaction into 256 triples.
There is no runtime sort, CPU filtering or external policy head.

Request legality is exact membership in the previous context's `LEGAL_SET`.
After an accepted request, the graph appends the triple, reconstructs the new
position, builds the opposing side's next `LEGAL_SET`, and selects the status.
Mate and stalemate derive from the same legal circuit, not from a second engine.

## Automatic draw policy

For now the VM deliberately matches the development oracle's
`outcome(claim_draw=True)`. There is no `CLAIM_DRAW` request token. The graph
automatically emits `DRAW` for:

- stalemate;
- insufficient material;
- an already claimable fifty-move or threefold position;
- a requested move that makes fifty moves or a third repetition claimable.

The repetition key includes board, side, castling rights and only an *effective*
legal en-passant right. A raw expired en-passant square is not treated as a
different key. The halfmove-99 lookahead also verifies that the candidate leaves
at least one legal reply; a move that immediately stalemates is adjudicated by
stalemate, not by a fictitious future fifty-move claim. Reply existence is
factorized into `[256,candidate]` and `[256,64,64]` tensors instead of allocating
a `[256,candidate,64]` cube.

`python-chess` is used only by development fixtures and the oracle comparison.
It is not linked or called by production inference.

## Generic runtime boundary

The serialized graph uses only these generic tensor operations:

- row routing and row concatenation;
- frozen expansion and token/frozen projections (GEMM);
- residual/position addition;
- hardmax with the declared STE;
- 2D hard attention;
- batch-preserving reshape and matrix transpose;
- intermediate matrix GEMM;
- grouped intermediate matrix GEMM.

Grouped GEMM is wire opcode 14. It performs the same rank-three row-by-column
operation independently for each declared group; static ranks, extents,
contraction axes and products are validated before CUDA access. It is a layout
and arithmetic primitive, not a chess opcode. Production sources contain no
board class, move generator, chess oracle or per-piece dispatch.

## Storage, compute and backward

Tensor records are stored as packed E2M1/FP4 codes plus FP32 scales. Records with
one scale per element cost 4.5 bytes per element before metadata, so the file
format must not be advertised as a uniform four-bit memory footprint. The
current native loader materializes FP32 tensors and computes in FP32 with TF32
disabled. Native FP4, FP16 and BF16 execution are not established.

Hardmax has exact lowest-index hard forward and a specified floating softmax
surrogate backward. 2D attention returns the selected value exactly; its
first-order backward differentiates a selected-top-k softmax-weighted surrogate
to Q, K and V. The legal-set loss has a finite nonzero gradient through the full
recurrent graph to the three-row request (`144` nonzero request components in
the recorded native acceptance). This proves gradient transport, not that a
player will learn useful chess or outperform a conventional environment.

For batch one, the compiler's conservative logical estimate is 269,563,016
bytes of decoded frozen FP32 payload plus 3,020,409,648 bytes of retained SSA
values. It counts views conservatively and excludes autograd saves, allocator
and workspace. A sampled RTX 3070 Laptop run of 47 full contexts, recurrent
feedback and one backward reached 3,736 MiB global device usage and took
56,284 ms. This is an acceptance workload, not a per-move latency benchmark or
a speedup claim.

## Acceptance evidence

Fresh native build `build/recurrent-native-20260901-c`:

- loaded `context_0` from the artifact;
- executed 47 complete recurrent contexts in FP32 with TF32 disabled;
- verified five `context_0` bindings and fed 40 sequential output contexts
  directly into the following request;
- matched every complete oracle context exactly;
- backpropagated generated `LEGAL_SET` to the request with `144` nonzero finite
  components and absolute gradient sum `7056.8`.

The accepted artifact hash is
`63d4bbd5abfbe555b4e4240445d42631fab78912f467d6ccc34d21bf564d7710`.
The 47 native cases cover ordinary play, Fool's Mate absorption, next-move
threefold, effective-en-passant repetition, stalemate, illegal requests and a
wrong result-piece request. Python exact tests additionally cover history
overflow, the fifty-move edge and insufficient material. These are strong
fixture and invariant checks, not an exhaustive proof over every reachable
chess history.

## Website evidence boundary

`site/recurrent_trace.json` covers all 2,877 operations in nine stages. Every
operation exposes its opcode, inputs/output, shapes, bilingual semantic purpose,
frozen-matrix meaning, compact exact numeric windows and a scalar proof for the
selected output cell. The top-level context cards show request, previous state,
next status/legal count and the exact output-to-input feedback hash. The nested
45-operation position microscope retains full COO and arbitrary-coordinate
producer navigation.

The recurrent intermediate numbers are generated by the exact Python reference
executor for the same serialized artifact. They are not a capture of every
native CUDA intermediate. Publishing every retained batch-one SSA cell would be
roughly 3 GiB, so the static trace intentionally publishes compact windows and
proofs instead. Native evidence separately compares full output contexts and
the backward path. The browser validates topology, producer order, feedback
identity and scalar proofs; it neither runs `.cmz` nor computes chess.

Real-browser acceptance covers GEMM, hardmax, attention, grouped GEMM, the final
operation, RU/EN switching, 1440 px desktop, 390×844 mobile, and an empty fresh
console. In GEMM, only the left input row is green, only the right input column
is amber, and the output marks only the selected cell. `ARGMAX` is rendered as
the centered operation between input and output.

## Deliberate non-claims

- No player transformer is included.
- No useful training result, long-rollout gradient study or speed advantage is
  demonstrated.
- No native low-precision compute path is accepted yet.
- The static site is an inspector, not a second runtime and not native capture
  of all intermediates.
- The test corpus is not a formal exhaustive verification of chess.

Current commands, hashes, timings and remaining boundaries are recorded in
`test_results/full_recurrent_vm_and_site_2026-09-01.md`.
