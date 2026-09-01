# Project Memory

Current-state record for `codex/pure-frozen-transformer-vm-clean`, updated
2026-09-01. Historical entries are not current acceptance evidence. The
authoritative boundary is [architecture.md](architecture.md).

## Repository and boundary

- implementation=fresh branch/worktree; no retired runtime migration
- production=generic LibTorch C++/CUDA tensor graph only; no chess types,
  procedural replay, move generator, oracle, evaluation, labels, tree search or
  Python dependency
- compiler=offline Python synthesizes frozen chess relation tensors and generic
  graph topology; chess rules are compiled, not absent
- oracle=python-chess only in development/test wrappers
- player=separate future component; not present here
- publication=content commit3b1dac6 fast-forwarded to clean branch and main;
  Pages run33490289502 succeeded; no force/history rewrite or unrelated changes

## Executable recurrent VM

- artifact_builder=`vm_compiler.recurrent_circuit.build_recurrent_artifact`
- artifact=2877 operations,192 immutable tensor records, final full context
  `[B,2045,128]`; `context_0` is embedded
- input=`[B,2048,128]` = three-row request + prior2045-row context
- request=`FROM,TO,RESULT_PIECE` hard one-hot; result piece includes color and
  promotion identity
- context=1200 history rows +768 sorted/padded legal rows +1 status +76 service
  rows
- statuses=OK89, ILLEGAL_MOVE90, WHITE_WIN91, BLACK_WIN92, DRAW93,
  HISTORY_OVERFLOW94
- feedback=output context is passed unchanged as the next prior context;
  terminal win/draw contexts are absorbing
- capacity=400 plies,256 legal moves; overflow is explicit, never truncated
- position=45-op latest-event reconstruction from history; no board rows are
  carried recurrently
- legal=7780 fixed ordered candidate patterns, exact post-board king safety,
  castling, en passant, promotion and stable tensor compaction to256 triples
- request_application=membership in previous LEGAL_SET, conditional append,
  new position, opposing LEGAL_SET and status, all inside the artifact
- draw_policy=automatic `outcome(claim_draw=True)` parity: stalemate,
  insufficient material, current/future fifty-move and current/future threefold;
  effective legal en-passant key and halfmove-99 reply-existence edge included

## Generic operations and numeric contract

- runtime_opcodes=row route/concat, frozen expand, projection/GEMM, add,
  hardmax/STE,2D attention, matrix transpose/reshape/matmul and grouped matrix
  matmul
- grouped_gemm=wire14, rank-three grouped row-by-column multiplication with
  static validation before CUDA; no chess semantics
- storage=packed E2M1/FP4 code plus FP32 scales; block-size1 records cost4.5
  bytes/element before metadata
- compute=current native/reference FP32, TF32 disabled; no accepted native FP4,
  FP16 or BF16 compute path
- backward=hard forward plus declared first-order softmax/selected-top-k
  surrogates; Q/K/V gradients and full LEGAL_SET-to-request gradient accepted
- memory_B1=269563016 decoded frozen FP32 bytes +3020409648 conservative retained
  logical SSA bytes; excludes autograd/workspace/allocator and counts views
- performance_boundary=no speedup claim; acceptance run is not a latency
  benchmark

## Current acceptance evidence

- artifact_sha256=63d4bbd5abfbe555b4e4240445d42631fab78912f467d6ccc34d21bf564d7710
- fixtures=47 complete recurrent contexts; ordinary moves, Fool's Mate,
  next-move threefold, effective-EP key, stalemate, illegal request and false
  result-piece request
- native_build=`build/recurrent-native-20260901-c`; all47 full contexts exact,
  40 output-to-next-input edges +5 context_0 bindings exact, FP32/TF32-off
- native_backward=request gradient144 nonzero components, abs_sum7056.8
- native_timed=56284ms for all47 contexts +feedback +one backward; sampled global
  GPU peak3736MiB on RTX3070 Laptop8GiB
- python_edges=exact history overflow, fifty-move reply edge, insufficient
  material, current/next repetition and effective legal en-passant regressions
- evidence=`test_results/full_recurrent_vm_and_site_2026-09-01.md`

## Website

- source=static `site/`
- recurrent_trace=`site/recurrent_trace.json`,7986472 bytes,2877 operations,9
  contiguous stages,sha256=0f5a8e7be7b6408bbbe19544e2331d3dc745c46531294dea48fd73f316fec0f1
- inspector=all operations navigable; compact exact reference windows, scalar
  proofs, bilingual tensor/frozen meanings, context/status/legal summary and
  feedback identity
- position_microscope=nested45-op exact trace retains full COO,
  arbitrary-coordinate values and producer navigation
- evidence_boundary=recurrent intermediates are Python-reference execution of
  the same artifact; native acceptance compares full outputs/backward, not every
  intermediate; browser runs no chess or `.cmz`
- visual_contract=GEMM highlights only left row green and right column amber;
  output only selected cell; centered ARGMAX; attention stages explicit
- browser_acceptance=1440 desktop and390x844 mobile; no body overflow; RU/EN,
  operations9/13/140/1464/2877, feedback and empty fresh console passed
- cache=CSS,five JS files and both JSON traces use generator-derived content
  versions
- public=https://trydotatwo.github.io/ChessMachineZero/; deployed HTML and all
  eight versioned assets match local content; public Playwright acceptance
  reached operations9/13/2877, RU/EN and390x844 with empty console

## Deliberate unknowns / future work

- no trained player or proof of useful end-to-end learning
- no long-rollout gradient-quality study
- no performance comparison with a conventional chess environment
- no accepted native low-precision compute
- fixtures/invariants are not an exhaustive formal proof over all chess games
- recurrent site publishes compact selected windows, not roughly3GiB of every
  retained SSA cell and not native intermediate capture
