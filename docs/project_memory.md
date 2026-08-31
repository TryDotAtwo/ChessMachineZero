# Project Memory

Current-state record, updated during the 2026-08-31 audit corrections. Historical change entries are not current acceptance evidence. Authoritative boundary: [architecture.md](architecture.md).

## Repository and architecture

- active_branch=codex/pure-frozen-transformer-vm-clean; fresh implementation, no migration of legacy runtime code
- production_boundary=generic LibTorch C++/CUDA tensor operators only; no chess types, procedural board replay, move decoder, evaluation, external search, labels, or Python dependency
- compiler_boundary=offline Python synthesizes reusable frozen rule matrices and generic graph operations; chess rules exist in those matrices, not magically absent from the system
- development_oracle=python-chess is restricted to development/test oracle wrappers; its full transition behavior is not a shipped native VM capability
- move_language=three hard one-hot rows FROM, TO, RESULT_PIECE; square a1=11, a2=12, b1=21; result-piece channels96..107 include color and promoted identity

## Executable now

- position_artifact=45 generic operations; eight opcode kinds; consumes FP32 [B,2048,128] and reconstructs FP32 hard one-hot [B,64,128]
- legal_artifact=549 generic operations,91 deduplicated frozen records,7780 position-independent candidate patterns; returns [B,768,128] ordered legal triples plus padding; includes source/color/path/king safety, castling rights and en passant; not request application or adjudication
- legal_compaction=two-level64-wide prefix projections, hard rank routing and routes-times-payload GEMM; no runtime sorting/filtering; named count/presence/overflow values available for later composition
- generic_matrix_ops=wire11 transpose,12 batch-preserving reshape,13 intermediate-matrix GEMM; rank/size/overflow validation occurs before CUDA; exact two-batch native forward/gradient tests pass
- legal_native=79 complete independent legal-set tensors exactly match in freshly compiled FP32 CUDA executor with TF32 disabled; artifact3a06b48ecc48d9abd8739ecc638d8afc4b7daf6577d9f39cfc957e09ed137e50
- foundation_acceptance=284 Python/reference/Node tests passed in103.35s; native generic matrix forward/gradients and52 boards passed; legal artifact79 exact tensors and memcheck exit0/zero errors; evidence=test_results/full_vm_foundation_2026-08-31.md
- legal_memory=45,219,348 frozen FP32 bytes and263,740,908 retained logical value bytes forB1; excludes autograd/workspaces/allocator, not a measured peak or speed result
- input_usage=only chronological history rows3..1202 are consumed; request rows0..2 and remaining service/legal/status rows are ignored by this artifact
- history_capacity=400 already-valid plies; declared result-piece identity must be correct; this subgraph does not validate history legality
- board_layout=64 rows in file-major order a1,a2,...,a8,b1,...,h8; 128 vocabulary columns; EMPTY=95
- latest_event=64 initial events plus five400-row streams: source clearing, destination result piece, castling rook source, castling rook destination, en-passant clearing
- special_events=frozen linear/residual/hardmax pattern blocks recognize four castling triples and28 adjacent-ply en-passant patterns, assuming valid histories
- latest_event_addressing=2D parabolic keys, query(2x,-1), x=1+compact/64; epsilon=2^-21, corrected from1e-6 after audit's legal400-ply counterexample
- fp32_score_invariant=all64 queries x64 addresses x401 timestamps checked on materialized weights; min cross-square margin5.340576171875e-05 and min chronological step4.76837158203125e-07 in current reference

## Target, not implemented

- recurrent_target=input [B,2048,128] = request3 + context2045; output context[B,2045,128]
- context_target=1200 history +768 legal-set +1 status +76 service rows; max256 legal moves and explicit overflow policy
- missing_execution=requested-move validation/application, terminal adjudication and next recurrent context; position and legal artifacts return partial outputs, not that context
- draw_policy_pending=existing test oracle uses outcome(claim_draw=True); whether the VM automatically exercises a claim or requires a player request was raised for user choice on2026-08-31; no token-language change authorized yet
- context_0=bootstrap record exists with initial20 legal moves and OK status; position artifact does not contain it
- rule_image=parametric relation tensors exist, but rule-image/bootstrap operation lists remain empty
- player=separate future trained transformer; no learning-benefit or complete differentiable game-loop evidence yet

## Numerical and cache contracts

- artifact=CMZVM001 v1 SHA-256 container, explicit shapes/names/opcodes, E2M1 two-nibble storage with FP32 scales
- compute_precision=current native/reference arithmetic FP32; no verified native FP4, FP16, BF16 or TF32-equivalent position execution
- storage_cost=block-size1 scaled records cost0.5 byte code +4 bytes scale per element; packed FP4 is not evidence of four-bit runtime memory or speed
- hardmax=exact lowest-original-index hard forward; floating softmax surrogate backward; empty allowed sets are errors
- attention_backward=first-order selected-top-k softmax surrogate for Q,K,V, aligned between development reference and native contract; not the true derivative of discrete chess and not proven useful learning
- fp4_primitive=finite extremes saturate before nearest-lattice selection; identity STE backward is an explicit surrogate
- hull_metadata=offline nested convex layers preserve collinear boundary points, duplicates and original-index ties
- position_cache=all2064 candidate rows are scanned for every query; top8 competitors selected; dense QK storage avoided natively, but no sublinear HullKV performance demonstrated
- performance=no throughput or whole-system speedup claim; no comparison proving advantage over a conventional engine

## Evidence and website

- audit=[../test_results/audit_2026-08-31.md](../test_results/audit_2026-08-31.md), historical findings with reproducible counterexamples
- python_corrections=commit5202cbd; initial corrected gate123 passed; final integrated gate through721db71 is257 passed in20.92s
- site_source=static site under site/; numeric_trace.json is an offline Python reference execution export, not native CUDA intermediates or browser inference
- site_fixture=three legal plies e2e4,d7d5,e4d5; token triples [[52,54,96],[47,45,102],[54,45,96]]
- site_full_vm_pending=site remains a45-operation position reference inspector; full recurrent/native-trace walkthrough is not implemented or published; local numeric source fingerprint refreshed after generic reference-op additions without changing exported matrix values
- publication=GitHub Pages workflow publishes site/ from main; scoped ordinary push allowed, no force/history rewrite/unrelated changes
- publication_verified=code/site revision721db7129a176890d8b4b530aa8de6b75581adb0; Pages33356317808 success; public45operations/72tensors, exact fixture/values, RU/EN and fresh console verified after cache correction; later report-only commits do not change deployed site bytes
- native_corrections=commit b3ba731; fresh direct Windows build, 15 checked process exits, 52 exact FP32 full boards, independent selected-top-k Q/K/V derivatives, malformed metadata/mask guards; independent review Approved
- native_memcheck=controller Compute Sanitizer 2024.2 on valid attention and all 52 board fixtures: exit0 and zero reported errors; integrated position backward and CMake success remain unverified
- site_inspector=commits58284fa,baf94a5,102e599; single reference fixture, semantic metadata for24 frozen/46 SSA/2 derived tensors; arbitrary-coordinate values and producer links; correct GEMM row/column, explicit ARGMAX and real K-transpose stages; explicit loading/ready/error lifecycle
- site_validation=rejects inconsistent stored generic FP32 arithmetic before render, including malformed/missing data; not independent chess-legality proof or cryptographic authenticity
- site_cache=commit721db71; generator fingerprints CSS/all4JS/numericJSON by normalized content; app fetch/download share the versioned URL; this fixes mixed old-script/new-HTML cache failure found during live acceptance
- site_browser=all45 operations at1536px desktop and390x844 mobile; exact cells/producer navigation, RU/EN and invalid-export failure state verified; console errors/warnings empty
- release_evidence=[../test_results/audit_corrections_2026-08-31.md](../test_results/audit_corrections_2026-08-31.md); final review/publication status is recorded there
