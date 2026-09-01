# Full recurrent VM and website acceptance — 2026-09-01

## Accepted scope

This report covers the complete one-ply recurrent artifact, not the earlier
position-only or legal-set-only subgraphs:

```text
[B,3,128] request + [B,2045,128] prior context
    -> 2,877 generic frozen tensor operations
    -> [B,2045,128] next context
```

Final artifact facts from
`build/recurrent-fixtures-2026-09-01-final/manifest.json`:

- operations: **2,877**
- immutable tensor records: **192** (including `context_0`)
- complete native fixture contexts: **47**
- `known_pending`: **[]**
- `recurrent.cmz` SHA-256:
  `63d4bbd5abfbe555b4e4240445d42631fab78912f467d6ccc34d21bf564d7710`
- `recurrent.bin` SHA-256:
  `9a93c53a447e8437217a6fd0f1cac93b327294851316f0d7eeb91dc2a060bcd0`

The final bundle hashes are identical to the bundle used by the fresh native
build and memcheck; regenerating the manifest did not substitute a new graph.

## Runtime-purity audit

The production C++ source dispatches only generic tensor opcodes. This change
adds grouped intermediate matrix GEMM (wire opcode14) to row routing,
projection/GEMM, add, concat, expansion, hardmax/STE,2D attention,
transpose/reshape and intermediate matrix GEMM. Static ranks, extents,
contraction axes and products are rejected before CUDA access.

No C++ chess board, piece-specific branch, move generator, oracle call, Python
dependency, evaluation, policy head or external search was added. Chess
relations and candidate order are compiled offline into immutable weights and
graph connections. `python-chess` remains confined to development/oracle code.

## Exact transition coverage

Python/reference equality tests cover:

- legal initial `e2e4`, illegal `e2e5`, false result-piece declaration;
- unchanged output-to-next-input context feedback for a black reply;
- Fool's Mate `BLACK_WIN` and absorbing terminal context;
- stalemate and empty next `LEGAL_SET`;
- full 400-ply history overflow without silent truncation;
- automatic current/next threefold and effective legal en-passant key;
- automatic fifty-move claim including the halfmove-99 legal-reply edge;
- insufficient material;
- complete sorted legal triples and padding, not only counts;
- a finite nonzero gradient from generated `LEGAL_SET` to the requested move.

The full graph uses the confirmed automatic
`outcome(claim_draw=True)` policy. There is no `CLAIM_DRAW` request token.

## Full integrated gate

Command:

```powershell
python -m pytest -q
```

Result:

```text
312 passed in 243.37s (0:04:03)
```

This includes Python/reference exactness, serialization/validation, negative
mutation tests, Node browser-model validation and legacy-site regressions.

## Fresh native acceptance

Build: `build/recurrent-native-20260901-c`, compiled after the grouped-GEMM and
sequential-feedback source changes. Its build manifest records the exact source
hashes, MSVC14.44, CUDA12.5, LibTorch2.8.0+cu128 and sm86 target.

The C++ fixture checks five sequence starts against artifact `context_0`, feeds
40 output contexts unchanged into the next request of their sequence, and
compares every element of every complete context. The remaining two cases are
independent invalid-request checks. Result:

```text
PASS 47 full recurrent tensors; native feedback, FP32, TF32 disabled, exact equality
PASS generated LEGAL_SET backward to requested move; nonzero=144 abs_sum=7056.8
```

The 47 cases cover ordinary play, Fool's Mate, next-move threefold, the
effective-en-passant repetition edge, stalemate, an illegal request and a wrong
result-piece request. The longer fifty-move, insufficient-material and capacity
fixtures are exact Python/reference gates, not claimed as members of the
47-case native file.

A timed run of all 47 contexts, sequential feedback and one backward reported:

```text
elapsed_ms=56284
sampled_global_peak_mib=3736
```

Hardware: NVIDIA GeForce RTX3070 Laptop GPU, 8,192 MiB. The memory sample is
global device usage, not allocator-exclusive peak. The workload is acceptance,
not a per-move latency benchmark.

Compiler logical batch-one estimate:

```text
decoded frozen FP32 payload: 269,563,016 bytes
retained logical SSA values: 3,020,409,648 bytes
```

The estimate conservatively counts views and excludes autograd saved tensors,
allocator and workspace.

## Compute Sanitizer

Compute Sanitizer12.5 memcheck was run outside the restricted launcher sandbox
against the same native binary and the hash-identical 47-case artifact/fixture:

```text
--tool memcheck --error-exitcode 99 --check-exit-code yes --require-cuda-init yes
```

Result:

```text
exit 0
ERROR SUMMARY: 0 errors
```

The first restricted-sandbox attempts returned only `Error launching target
app`; they are not counted as CUDA evidence. The authorized instrumentation run
attached to the active target and completed the full forward/feedback/backward
workload.

## Static full-VM inspector

Generated artifacts:

- `site/recurrent_trace.json`: **7,986,472 bytes**, all 2,877 operations in nine
  contiguous stages, raw SHA-256
  `0f5a8e7be7b6408bbbe19544e2331d3dc745c46531294dea48fd73f316fec0f1`
- `site/numeric_trace.json`: **4,805,888 bytes**, nested 45-op complete COO,
  raw SHA-256
  `aefcc53c11c4ae8b2ba2166f48e61b0bdd844246fd3a6111236047c07f34c386`

The full trace exposes every operation's topology, shapes, bilingual meaning,
frozen-matrix meaning, compact exact numeric windows and scalar proof. The
context header exposes request, prior/output status, history/legal counts and
the exact feedback hash. It explicitly labels intermediate numbers as Python
reference execution of the same artifact and native full outputs as separate
evidence. It does not claim native capture of roughly 3 GiB of retained SSA.

Browser trace validation rejects operation-count/stage gaps, undefined
producers, unknown opcodes, feedback mismatch, status/channel mismatch,
history/legal count mismatch, missing samples, forged proof results, and values
that disagree with the displayed selected matrix cell.

## Real-browser acceptance

Playwright loaded the final content-versioned static site from a local HTTP
server and used hard assertions. Checked operations:

- op9 matrix GEMM: operators `×`, `=`; left card has row highlighting only,
  right card has column highlighting only, output has neither;
- op13 hardmax: centered `ARGMAX` (`display:grid`, both axes centered);
- op140 attention: `×`, `ARGMAX →`, `=`;
- op1464 grouped matrix GEMM and exact grouped scalar terms;
- op2877 final recurrent context and feedback identity.

Also checked RU→EN state, English full-trace status, 1440 px desktop,390×844
mobile, no document overflow, and an empty fresh error/warning/pageerror list.
The initial browser audit found a missing favicon and a Russian full-trace status
after EN switching; both were fixed and the final asserted reload passed.

## Final requirement-by-requirement self-review

- **Complete recurrent contract:** the current generated artifact is
  `[B,2048,128] -> [B,2045,128]`, ends at `v2877`, embeds `context_0`, and its
  manifest has 2,877 operations, 192 records and `known_pending: []`.
- **Runtime purity:** a case-insensitive scan of current production `src/` and
  `include/cmz/` for board/piece/move/chess/castling/en-passant/legal terms
  returned no matches. The C++ diff adds only generic grouped rank-three GEMM
  validation and execution.
- **Native identity:** every source hash recorded by
  `build/recurrent-native-20260901-c/build-manifest.json` matches the current
  source file. The final artifact and fixture hashes match the accepted bundle.
- **Chess transition behavior:** full-context oracle assertions cover legal,
  illegal, terminal, draw-policy and capacity cases; native acceptance covers
  complete contexts and unmodified recurrent feedback, not only selected rows.
- **Differentiability evidence:** both reference and native backward produce a
  finite nonzero gradient from generated `LEGAL_SET` to all three request rows.
  This remains a transport test, not a learning-quality claim.
- **Website completeness and honesty:** all 2,877 artifact operations belong to
  exactly one of nine contiguous stages and have topology, shapes, semantic
  RU/EN labels, exact compact windows and scalar provenance. The UI explicitly
  separates Python-reference intermediates from native full-output evidence;
  JavaScript neither applies moves nor computes legality.
- **Rendered behavior:** final Playwright assertions reached operations 9, 13,
  140, 1464 and 2877; checked multiplication highlighting, centered ARGMAX,
  attention/grouped-GEMM operators, SSA/frozen explanations, native evidence,
  RU/EN, 390-pixel layout and an empty console/problem list.

The remaining gate after this review is ordinary fast-forward publication and
verification of the deployed revision and content-versioned resources.

## Explicit limits

- Packed E2M1/FP4 is storage plus FP32 scales; current compute is FP32. No
  accepted native FP4/FP16/BF16 compute result exists.
- Surrogate backward proves transport of a finite gradient, not useful player
  learning, long-rollout quality or a true derivative of discrete chess.
- No player transformer or training result is included.
- No throughput or speedup over a conventional chess environment is claimed.
- The fixture corpus and invariants are not a formal exhaustive proof over all
  reachable chess histories.
- The website is a static inspector, not a second chess runtime and not a dump
  of every native intermediate.
