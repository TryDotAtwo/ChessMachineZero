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
- missing_execution=full LEGAL_SET enumeration, requested-move validation/application, terminal status and next recurrent context; the current position artifact returns a board, not that context
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
- python_corrections=commit5202cbd; reported full gate123 passed; final integrated evidence is recorded separately after all tasks
- site_source=static site under site/; numeric_trace.json is an offline Python reference execution export, not native CUDA intermediates or browser inference
- site_fixture=three legal plies e2e4,d7d5,e4d5; token triples [[52,54,96],[47,45,102],[54,45,96]]
- publication=GitHub Pages workflow publishes site/ from main; scoped ordinary push allowed, no force/history rewrite/unrelated changes
- native_corrections=commit b3ba731; fresh direct Windows build, 15 checked process exits, 52 exact FP32 full boards, independent selected-top-k Q/K/V derivatives, malformed metadata/mask guards; independent review Approved
- native_memcheck=controller Compute Sanitizer 2024.2 on valid attention and all 52 board fixtures: exit0 and zero reported errors; integrated position backward and CMake success remain unverified
- in_progress=single-fixture whole-page RU/EN inspector and final integration/publication are being verified; do not treat implementation plans as completed evidence
