# Percepta Policy-Only Full Code Audit, 2026-05-28

## Metadata

- audit_id=percepta_policy_only_full_code_audit_2026-05-28
- source_request=verify implementation against `docs/cmz_percepta_policy_only_architecture_review.md`
- source_architecture_doc=`docs/cmz_percepta_policy_only_architecture_review.md`
- workspace=C:/Users/Иван Литвак/Documents/ChessMachineZero
- reviewed_scope=all tracked and untracked source/config/test/doc files visible to `rg --files`, excluding historical `test_results/**`, cache folders, and binary screenshots
- audit_method=static source inspection plus targeted grep evidence; runtime pytest verification recorded separately
- primary_question=whether current code actually implements all rule and decoder behavior as 2D frozen self-attention with HullKV/NestedHull/CUDA/CUTLASS and policy-only decoder semantics

## Executive Verdict

- verdict=not_fully_conformant
- native_contract_claim=declares_full_conformance
- implementation_truth=partial_conformance_with_overclaimed_contract_flags
- highest_risk_gap=`full_frozen_attention_only=true` is asserted in native metadata and tests, but several CUDA kernels still contain hand-coded chess rule branches, serial loops, and custom control-flow kernels rather than uniformly expressed `QK -> hardmax/select -> V/write` attention layers.
- second_risk_gap=legacy Python strategy modules still contain ranker, baseline, actor, softmax selection, and negamax lookahead paths; current architecture doc requires policy-only decoder without external critic/eval/search wrapper.
- third_risk_gap=Python dashboard self-play uses deterministic legal-move indexing from a trace, not the native policy-only decoder command policy.
- satisfied_core=append-only TracePacket formats exist; runtime python-chess import is confined to oracle/verifier/test helpers; native API enforces decoder head_dim=2 and reports no value/critic head; legal-filter v2 contains explicit 2D QK hardmax helper use; CUTLASS HullHardmax2D path exists.

## Requirement Matrix

| requirement | verdict | evidence |
|---|---:|---|
| transformer-computer; rules executed as computation, not policy head | partial | Native rule API separates rule VM from decoder (`native/cpp/src/cmz_engine.cpp:227-245`); Python dashboard step selects `legal_moves[(seed+ply)%len]` instead of decoder policy (`src/chess_machine_zero/model/percepta_parametric_selfplay.py:139-150`). |
| frozen VM; chess rules in frozen attention layers | partial | Contract reports frozen attention graph (`native/cpp/src/cmz_engine.cpp:250-315`); legal-filter v2 uses `cmz_qk2_hardmax_select_u32` (`native/cpp/src/cmz_cuda_kernels.cu:448,585,632,674,774,917,964`); candidate/terminal kernels still use rule-specific loops and branches (`native/cpp/src/cmz_cuda_kernels.cu:1030-1126`, `1241-1279`, `1351-1401`, `1404-1522`). |
| `full_frozen_attention_only=true`; hot path without Python/control-flow fallback | fail as semantic claim | Contract declares `full_frozen_attention_only=true` (`native/cpp/src/cmz_engine.cpp:307`), `python_hot_path=false` and `fallback_allowed=false` (`native/cpp/src/cmz_engine.cpp:236-237`); CUDA code still contains custom chess control flow, so claim is metadata-compliant but semantic-overclaim. |
| 2D attention heads; `head_dim=2` contract | mostly satisfied | Native contract reports `executor_head_dim=2` (`native/cpp/src/cmz_engine.cpp:227`); Rust decoder scaffold rejects non-2D heads (`native/crates/cmz-engine-sys/src/lib.rs:271-274`); Python `CMZMachineTransformer` enforces `d_model/n_heads==2` (`src/chess_machine_zero/model/machine_transformer.py:37-38`). |
| `QK -> hardmax/select -> V/write` for rules | partial | `DenseHardmax2D` implements dot/argmax/value select (`src/chess_machine_zero/model/hardmax_attention.py:17-39`); native qk helper exists (`native/cpp/src/cmz_cuda_kernels.cu:132-146`); many candidate/terminal/resolve kernels are loops with conditionals, not generic QK-hardmax-V layer bodies (`native/cpp/src/cmz_cuda_kernels.cu:1241-1279`, `1542-1566`). |
| `HullKVCache` long-context lookup | partial/scaffold | Python `HullKVCache.hardmax` delegates to convex hull support (`src/chess_machine_zero/hullkv/cache.py:23-45`); contract names HullKVCache (`native/cpp/src/cmz_engine.cpp:233`); no production rule hot path was found where board/trace VM lookup depends on HullKVCache rather than custom trace/board code. |
| `NestedHullTopK2D` sparse top-k lookup | partial/scaffold | Python implementation rebuilds hulls over remaining points (`src/chess_machine_zero/hullkv/nested_hulls.py:17-37`); native API has CPU implementation (`native/cpp/src/cmz_engine.cpp:1967-2001`); not integrated as CUDA/CUTLASS rule hot path. |
| append-only trace stream | satisfied for representation | TracePacket width and variants exist in Python/Rust (`src/chess_machine_zero/vm/trace_packet.py:1-72`, `native/crates/cmz-engine-sys/src/packets.rs:176-264`); move traces append candidate/legal/commit/write/terminal/halt packets (`native/cpp/src/cmz_engine.cpp:1459-1491`, `1667-1702`). |
| token streaming; VM streams trace packets | partial | Native exposes begin/next stream API (`native/cpp/include/cmz_engine.h:102-103`, `native/crates/cmz-engine-sys/src/lib.rs:785-795`); `cmz_engine_legal_trace_begin` precomputes the whole legal trace (`native/cpp/src/cmz_engine.cpp:2452-2471`), so streaming is buffered playback rather than generation-on-demand. |
| CUDA/CUTLASS substrate for attention semantics | partial | CUTLASS QK score and CUDA hardmax select path exists (`native/cpp/src/cmz_cuda_kernels.cu:1667-1754`); many rule kernels are CUDA device code but not CUTLASS generic attention semantics (`native/cpp/src/cmz_cuda_kernels.cu:1030-1522`). |
| policy-only decoder; no critic/eval/search wrapper | native yes; repo-wide partial | Native contract and Rust wrapper disable critic/value (`native/cpp/src/cmz_engine.cpp:241-245`, `native/crates/cmz-engine-sys/src/lib.rs:954-957`); legacy Python baseline/ranker/negamax remain (`src/chess_machine_zero/model/baseline.py:11-26`, `src/chess_machine_zero/model/ranker.py:17-50`, `src/chess_machine_zero/vm/lookahead.py:44-83`, `src/chess_machine_zero/train/losses.py:24-59`). |
| python-chess oracle in tests only | satisfied with caveat | Only direct `import chess` is in oracle wrapper (`src/chess_machine_zero/chess/rules_oracle.py:1-5`); tests/verifier import the oracle (`src/chess_machine_zero/trace/verifier.py:5-18`, tests grep results); runtime dashboard reports `runtime_oracle_used=false` (`src/chess_machine_zero/model/percepta_parametric_selfplay.py:195-197`). |

## Native C++/CUDA Findings

### `native/cpp/src/cmz_engine.cpp`

- lines=227-245; finding=contract reports 2D, HullKVCache, no Python hot path, no fallback, policy-only decoder, no value/critic.
- lines=250-315; finding=frozen rule graph reports `board_projection=latest_write_hardmax_2d`, `trace_select=cursor_hardmax_2d`, CUDA QK hardmax backends, `full_frozen_attention_only=true`, `strict_qk_layer_split_remaining=none`.
- lines=1110-1151; finding=`pseudo_legal_moves` calls `cmz_cuda_candidate_moves_attention`, then host code converts CUDA move records into `std::vector<Move>`; host conversion is orchestration, but semantic proof depends entirely on CUDA kernel internals.
- lines=1154-1212; finding=`frozen_make_move_board_layer` uses CUDA board and metadata attention calls, then host code copies tokens/metadata into C++ `Board`; no fallback was found.
- lines=1214-1290; finding=legal filter single/batch call v2 CUDA attention wrappers; batch route prepares host vectors and calls CUDA batch filter.
- lines=1301-1366; finding=`frozen_candidate_move_from_request` validates requested UCI using C++ chess-specific branches before make-move resolution; classification=remaining_non_attention_host_validation_path.
- lines=1459-1491; finding=`frozen_legal_trace_attention_tokens` emits `CANDIDATE` and `LEGAL_SET` packets in a host loop after CUDA candidate and legal-filter outputs; append-only trace exists, but trace construction loop is not itself attention.
- lines=1498-1538; finding=`resolve_legal_move` converts moves/legal bits to records, calls `cmz_cuda_resolve_move_attention`, then returns selected C++ move.
- lines=1545-1582; finding=`insufficient_material` C++ helper remains in source; direct usage not found in current native terminal route, but source presence is legacy risk.
- lines=1584-1612; finding=terminal predicate route calls `cmz_cuda_terminal_status_attention`.
- lines=1621-1665; finding=board transition trace writes are emitted by host loop comparing before/after board squares.
- lines=1667-1702; finding=make-move trace concatenates resolve, board update, commit packet, transition packets, terminal status, and halt packet; composition is host C++ orchestration over CUDA layer calls.
- lines=1883-1916; finding=native decoder hidden state uses 2D board/query/key tensors but line 1907 applies `torch::softmax(scores, 0)`, not hardmax/select; policy decoder can be softmax policy; classification=not_frozen_rule_attention.
- lines=1967-2001; finding=native `nested_hull_topk_2d` CPU loop rebuilds `ConvexHull2D` and erases selected points; classification=not_CUDA_CUTLASS_topk.
- lines=2392,2427,2434; finding=count-only trace calls derive packet counts from `sorted_pseudo_legal_moves(board)` and full `frozen_legal_trace_attention_tokens`; trace stream count path still invokes generation logic.
- lines=2452-2471; finding=`cmz_engine_legal_trace_begin` precomputes all legal trace tokens into `engine->stream_tokens`.
- lines=2471-2490; finding=`cmz_engine_legal_trace_next` returns next packet from precomputed buffer; stream semantics are buffer playback.
- lines=2625-2669; finding=decoder forward and policy-gradient APIs expose policy logits and REINFORCE-style loss only; no value output found.

### `native/cpp/src/cmz_cuda_kernels.cu`

- lines=15-41; finding=2D dot score and CUDA hardmax select primitive exist.
- lines=72-100; finding=latest-write board projection scans trace packets in CUDA; named attention semantics exist but implementation uses packet loop and conditionals.
- lines=128-146; finding=`cmz_qk2_score_u32` and `cmz_qk2_hardmax_select_u32` provide explicit 2D QK hardmax helper.
- lines=150-193; finding=target/ray helper functions use fixed chess offsets and ray loops.
- lines=322-381; finding=attack detection uses loops over squares and slider ray loops.
- lines=445-463,585-641,674,774-789,917-964; finding=legal-filter v2 contains real qk2 hardmax helper calls for write/select stages.
- lines=1002-1026 and 1107-1126; finding=castling uses explicit board-square conditions and attack checks, not generic attention table lookup.
- lines=1030-1105; finding=`cmz_candidate_target_mask_value` is branch-coded per piece type with loops/conditions.
- lines=1241-1279; finding=`cmz_candidate_record_emit_attention_kernel` serially loops from-square, to-square, promotion; not a QK-hardmax-select layer.
- lines=1302-1348; finding=`cmz_candidate_move_is_legal` constructs next board and checks king safety with loops.
- lines=1351-1401; finding=`cmz_any_legal_candidate_move` enumerates candidates and returns on first legal candidate; classification=control_flow_search_inside_terminal_predicate.
- lines=1404-1440; finding=`cmz_insufficient_material_value` counts material by branch logic.
- lines=1542-1566; finding=`cmz_resolve_move_attention_kernel` loops records and selects first legal matching request; kernel name says attention but body is scan/filter.
- lines=1667-1754; finding=`cmz_cutlass_hardmax2d_values` uses CUTLASS GEMM for QK scores then CUDA hardmax select; classification=strongest_direct_CUTLASS_attention_evidence.
- lines=1991-2118; finding=candidate move attention wrapper launches named candidate layers; layer names do not by themselves prove inner QK-hardmax semantics because called kernels above contain custom loops.
- lines=2119-2190; finding=terminal status wrapper launches named terminal layers; inner legal-presence/check/material logic still contains custom loops.
- lines=2293-2460 and 2620-2757; finding=legal-filter v2 single/batch wrappers launch layered CUDA pipeline; best current conformance area.

### `native/cpp/include/cmz_engine.h`

- lines=14-21; finding=contract/graph runtime metadata APIs exist.
- lines=43-72; finding=counters exist for CUDA trace select, CUTLASS hardmax, board projection, attack/candidate/ray/legal/castle/make/terminal attention.
- lines=73-89; finding=HullHardmax2D and NestedHullTopK2D native APIs exist.
- lines=91-103; finding=legal trace packets and legal trace streaming APIs exist.
- lines=131-145; finding=decoder forward and policy-gradient APIs expose policy commands only; no value/critic output C API found.

## Native Rust Findings

### `native/crates/cmz-engine-sys/src/lib.rs`

- lines=15-213; finding=FFI surface mirrors C API and includes contract, counters, trace, decoder calls.
- lines=238-246; finding=`DecoderForwardOutput` includes command logits, command names, attention head dim, tracepacket_backprop=false, value/critic booleans.
- lines=255-307; finding=`PerceptaDecoderScaffold::new` requires `head_dim==2`, disallows separate white/black weights, and exposes no value/critic head.
- lines=667-688; finding=NestedHullTopK2D wrapper validates input and calls native CPU top-k implementation.
- lines=708-795; finding=trace packet collection and begin/next stream wrappers exist.
- lines=933-957; finding=decoder forward returns command logits and hardcoded `attention_head_dim=2`, `value_head_enabled=false`, `critic_head_enabled=false`.
- lines=961-970; finding=policy-gradient step wraps native pure policy update.
- lines=1315-1328; finding=tests assert contract strings for 2D/HullKV/no fallback/policy-only.
- lines=1354-1380; finding=tests assert CUTLASS hardmax route and graph metadata.
- lines=1423-1455; finding=tests assert decoder head_dim=2 and no value/critic.
- lines=1503-1513; finding=tests assert `full_frozen_attention_only=true` metadata; tests do not inspect CUDA kernel bodies for QK-only semantics.

### `native/crates/cmz-engine-sys/src/packets.rs`

- lines=176-195; finding=TraceOp includes audit and policy events, including `ScoreSet` and `SampleSet`; score/sample ops are useful for legacy policy traces and should be reviewed under policy-only requirement.
- lines=253-264; finding=TracePacket width is 7, matching architecture handoff.
- lines=301-319; finding=token decoding validates packet width and field ranges.

### `native/crates/cmz-dashboard/src/lib.rs`

- lines=1-21; finding=dashboard uses native Engine and reports `python_hot_path=false`.
- lines=27-42; finding=snapshot reads native legal trace stream packets until ProgramHalt.
- lines=103-117; finding=packet rendering converts TracePacket to readable/tokens strings.
- lines=189-193; finding=browser JS embedded in Rust dashboard displays HullKVCache/2D token metrics.
- lines=201-208; finding=test asserts native dashboard snapshot contract contains `python_hot_path=false` and `HullKVCache`.

### `native/crates/cmz-cli/src/main.rs`

- lines=1-58; finding=CLI creates native engine and prints legal trace; no policy decoder/training behavior.

### `native/crates/cmz-engine-sys/build.rs`, `native/cpp/CMakeLists.txt`, `native/Cargo.toml`, crate Cargo files

- finding=build orchestration for Rust/C++/CUDA; no architecture-semantic issue except dependency on CUDA/CUTLASS build availability.

## Python Rule/VM Findings

### `src/chess_machine_zero/model/percepta_parametric_selfplay.py`

- lines=126-132; finding=legal moves are decoded from `decode_legal_tensor_trace_host_append_only` and `legal_moves_from_trace`.
- lines=134-150; finding=self-play step selects `legal_moves[(seed+ply)%len]`; no native policy decoder or learned strategy is used.
- lines=165-204; finding=snapshot reports `single_shared_transformer_selfplay`, `shared_model_instance_count=1`, trace legal verification, and `runtime_oracle_used=false`.
- lines=214-236; finding=commit verifies selected move by membership in LEGAL_SET-derived legal list, then decodes make-move trace.
- lines=237-266; finding=last white/black trace streams are stored, but `transformer_id=actor`, and actor for automatic move is `shared_transformer`; current milestone text requiring two independent white/black transformers is not met by `src/chess_machine_zero/model/percepta_parametric_selfplay.py`.

### `src/chess_machine_zero/model/percepta_frozen_attention_vm.py`

- lines=235; finding=packet conversion uses `.tolist()` and host TracePacket construction.
- lines=451; finding=board reconstruction delegates back to tensor kernel conversion.
- lines=598-624; finding=decode step uses attention-select wrapper for next packet.
- lines=704-732; finding=logarithmic 2D attention converts keys to Python tuples and uses Python control-flow search over angles/candidates; exact 2D lookup exists, but implementation is not CUDA/CUTLASS and not frozen-layer-only hot path.

### `src/chess_machine_zero/model/percepta_attention_rule_kernels.py`

- lines=106-136; finding=board/tensor conversion uses Python list and `.tolist()` conversions.
- lines=138-186; finding=piece dispatch, ray scan, and attack test use PyTorch logical reductions, not explicit 2D QK-hardmax-V/write primitives.
- lines=188-213; finding=legal filter uses vectorized tensor logic, `.any`, and `argmax`.
- lines=243-247; finding=terminal checks use `candidates.legal.any()` and king-square `argmax`.
- lines=268-330; finding=candidate generation uses tensor boolean formulas per piece and castling conditions.
- lines=389-403; finding=insufficient material uses tensor reductions plus Python-derived bishop color condition; not generic attention lookup.

### `src/chess_machine_zero/model/percepta_attention_block_stack.py`

- lines=109-158; finding=trace-to-board and legal trace tensor generation use tensor max/stack/cat operations.
- lines=160-208; finding=make-move and terminal trace creation are tensor-stack orchestration over rule kernels.
- lines=235-269; finding=resolve/legal move extraction uses tensor masks/nonzero rather than pure attention layer streaming.
- lines=274-523; finding=compiled executor duplicates rule-kernel style logic with tensor formulas; not all operations are explicit 2D QK-hardmax-V/write.
- lines=542-687; finding=wrapper reports frozen attention block stack backend and delegates matrix interpreter operations; semantics remain partially declarative/metadata-driven.

### `src/chess_machine_zero/model/percepta_matrix_attention_runtime.py`

- lines=81-91; finding=entrypoint routing uses Python control flow.
- lines=204-216; finding=board-after-move execution delegates to tensor rule graph.
- lines=587-600; finding=mask/value checks are interpreter mechanics; file is a matrix interpreter, not CUDA/CUTLASS.

### `src/chess_machine_zero/model/percepta_rule_compiler.py`

- lines=76-146; finding=compiles chess ISA microprogram rows into frozen tensors and reports metadata; supports target architecture but does not enforce CUDA QK-only kernels.
- lines=193-219; finding=program serialization and residual write extraction use Python loops.

### `src/chess_machine_zero/model/percepta_rule_layer_graph.py`

- lines=43-81; finding=graph methods convert prompt traces to board tensors and call rule kernels.
- lines=117-122; finding=legal move checks build Python dict of UCI to legal booleans.
- lines=158-181; finding=tensor candidate rows are converted via `.tolist()` and yielded as MovePacket objects.

### `src/chess_machine_zero/model/percepta_tensor_trace_runtime.py`

- finding=tensor trace facade delegates to block stack; no python-chess import; no independent CUDA/HullKV implementation.

### `src/chess_machine_zero/hullkv/cache.py`

- lines=23-45; finding=HullKVCache enforces equal key/value lengths and delegates hardmax query to ConvexHull2D support.

### `src/chess_machine_zero/hullkv/convex_hull_2d.py`

- lines=33-73; finding=convex hull support is exact but computes all hull vertex scores in a Python list and `max`; not GPU/CUTLASS.

### `src/chess_machine_zero/hullkv/nested_hulls.py`

- lines=17-37; finding=NestedHullTopK2D rebuilds hulls over remaining keys in a Python while loop; correct sparse retrieval scaffold, not native hot path.

### `src/chess_machine_zero/hullkv/equivalence.py`

- lines=14-20; finding=dense equivalence oracle computes dot scores and top-k by Python sorting; test oracle only.

### `src/chess_machine_zero/model/hardmax_attention.py`

- lines=17-39; finding=exact 2D dense hardmax attention exists and enforces query/key final dimension 2.

### `src/chess_machine_zero/model/sparse_topk_attention.py`

- lines=20-31; finding=NestedHullTopK2D retrieval is followed by local `torch.softmax`; appropriate for a trainable/ranking head but not strict hardmax-only rule execution.

### `src/chess_machine_zero/model/machine_transformer.py`

- lines=23-40; finding=next-packet transformer enforces head_dim=2.
- lines=62-85; finding=uses PyTorch TransformerEncoderLayer and field logits; not frozen rule VM, used for experimental next packet prediction.

### `src/chess_machine_zero/model/hosted_vm.py`

- lines=21-48; finding=hosted VM decodes next packets from trainable transformer logits using argmax; not current native frozen rule path.

### `src/chess_machine_zero/model/percepta_e2e_decoder.py`

- lines=28-56; finding=model-only legal trace decoder stores prompt keys and continuation tensors; head keys are 2D.
- lines=69-87,104-128; finding=builds examples from FEN/prompt traces and memorized continuations; useful experiment but not the target parametric rule-computer proof.
- lines=149-171; finding=lookup returns stored continuation tokens; no chess evaluation, but finite-prompt lookup conflicts with Percepta rule-weight target if treated as production.

### `src/chess_machine_zero/model/analytic_rules.py`

- lines=48-79; finding=legal trace generated by Python loops over pseudo-legal moves and `is_legal_move`.
- lines=85-178; finding=make-move, terminal, and resolve use analytic host logic; legacy/non-target under full frozen attention requirement.

### `src/chess_machine_zero/model/analytic_machine.py`

- lines=45-73; finding=uses `CMZMoveRanker`, softmax probabilities, SCORE_SET/SAMPLE_SET, and host selection; conflicts with policy-only decoder requirement if active.

### `src/chess_machine_zero/model/weight_compiled_rules.py`

- lines=78-100; finding=trace generation loops over generated moves and emits CANDIDATE/LEGAL_SET.
- lines=434-478; finding=board transition and board reconstruction use Python loops.
- lines=567-642; finding=static tables built by Python loops; acceptable as compile-time scaffolding but not runtime frozen attention hot path.

### `src/chess_machine_zero/vm/interpreter.py`

- lines=44-73; finding=host Python legal trace generator loops over candidates and calls `is_legal_move`.
- lines=85-124; finding=make-move and terminal trace use host Python logic.
- lines=222-242; finding=terminal status enumerates legal moves and calls insufficient-material helper.
- lines=253-549; finding=complete chess move generation and attack logic implemented in Python control flow; legacy/test oracle-adjacent VM, not acceptable hot path for `full_frozen_attention_only=true`.

### `src/chess_machine_zero/vm/lookahead.py`

- lines=44-83; finding=depth-limited negamax wrapper exists and calls baseline at depth zero; violates policy-only/no external search wrapper if reachable in production.

### `src/chess_machine_zero/vm/decision_program.py`

- lines=19-24; finding=declares negamax decision program with `CALL_MOVE_RANKER`; legacy search/ranker concept remains.

### `src/chess_machine_zero/model/baseline.py`

- lines=11-26; finding=CMZOutcomeBaseline exists and predicts outcome from side/ply; conflicts with no required critic/value baseline if imported in active training.

### `src/chess_machine_zero/model/ranker.py`

- lines=17-50; finding=CMZMoveRanker scores legal moves; classification=separate_move_ranking_strategy_module_not_native_policy_only_command_decoder.

### `src/chess_machine_zero/selfplay/actor.py`

- lines=110-145; finding=SelfPlayActor selects moves using ranker scores, temperature probabilities, SCORE_SET/SAMPLE_SET, and COMMIT_MOVE.
- lines=180-185; finding=temperature path uses argmax or `torch.softmax`; legacy policy path not compatible with current policy-only decoder target.

### `src/chess_machine_zero/train/losses.py`

- lines=24-59; finding=trains ranker plus baseline with policy loss, baseline MSE loss, and entropy; conflicts with architecture handoff stating no prescribed critic/evaluator/baseline.

### `src/chess_machine_zero/train/trainer.py`

- lines=21-78; finding=next-packet trace prediction training; not rule hot path and not native policy-only decoder.

## Dashboard Findings

### `src/chess_machine_zero/dashboard/state.py`

- lines=24-41; finding=dashboard session wraps `PerceptaParametricSelfPlaySession.create`.
- lines=63-83; finding=legal moves, transformer step, human move, and snapshot delegate to Python session; no direct python-chess import.
- lines=32 and 51; finding=deterministic temperature 0.0 is enforced.

### `src/chess_machine_zero/dashboard/server.py`

- lines=44-60; finding=HTTP routes expose snapshot/reset/step/move/static files.
- lines=86-94; finding=step endpoint loops count and calls dashboard session; no oracle use found.

### `src/chess_machine_zero/dashboard/static/dashboard.js`

- lines=103-127; finding=snapshot renders active transformer, rule execution mode, token streaming, legal counts, last emitter, trace verification, and op counts.
- lines=201-424; finding=side trace journals and packet display render CANDIDATE/LEGAL_SET/COMMIT/TOKEN arrays.
- lines=484-527; finding=human/selfplay mode control and autoplay logic are UI-level only.

### `src/chess_machine_zero/dashboard/static/index.html`

- lines=28-75; finding=board, mode selector, legal/trace counters, and trace sections exist.

### `src/chess_machine_zero/dashboard/static/dashboard.css`, `favicon.svg`, `dashboard/__init__.py`

- finding=UI presentation/static package files; no rule/decoder semantics.

## Chess/Trace/Data Utility Findings

### `src/chess_machine_zero/chess/rules_oracle.py`

- lines=1-5; finding=only direct `import chess`; docstring marks python-chess oracle as development/test-only.

### `src/chess_machine_zero/chess/board_io.py`

- finding=FEN parse/serialize utility; no oracle dependency; host data parsing is outside attention semantics.

### `src/chess_machine_zero/chess/move_packet.py`

- finding=MovePacket/Promo/flag codec; supports trace representation.

### `src/chess_machine_zero/chess/outcome.py`

- finding=ResultCode/TerminalReason data model; no decoder/eval issue.

### `src/chess_machine_zero/trace/verifier.py`

- lines=1-18; finding=development verifier imports oracle; acceptable for tests but must remain outside runtime hot path.

### `src/chess_machine_zero/trace/compiler.py`

- lines=39-77; finding=compiles legal trace examples from host VM traces; training-data scaffolding for trace models, not production rule execution.

### `src/chess_machine_zero/trace/datasets.py`, `trace/reconstruct.py`, `trace/windows.py`, `vm/trace_packet.py`, `vm/trace_hash.py`, `rng.py`

- finding=packet tensor conversion, board reconstruction, trace windowing, codecs, hashes, deterministic RNG; supports auditability and tests.

## Root/Docs/Test Coverage Matrix

| file | audit_state | main finding |
|---|---:|---|
| `AGENTS.md` | read | standing rules require memory/history/test_results and restrict python-chess, search/eval/labels. |
| `README.md` | read | Python dashboard section still documents Python Percepta runtime as current dashboard path; native status section says Python runtime remains until parity, which conflicts with current native full-conformance claim. |
| `pyproject.toml` | read | `python-chess` is a project dependency, but direct source import is confined to oracle wrapper. |
| `docs/cmz_percepta_policy_only_architecture_review.md` | read | target says `full_frozen_attention_only=true`, `python_hot_path=false`, `fallback_allowed=false`, `decoder policy-only`; code does not fully prove semantic target. |
| `docs/chess_machine_zero_percepta_architecture.md` | read | broader architecture source; current milestone in project memory still mentions two independent Percepta instances. |
| `docs/native_rust_cpp_cuda_architecture.md` | read | native architecture doc should be updated later if overclaim is accepted as finding. |
| `docs/cutlass_acceleration_plan.md` | read | plan doc only; no implementation semantics. |
| `docs/percepta_can_llms_be_computers.md` | read | conceptual doc only. |
| `docs/project_memory.md`, `docs/change_history.md`, `docs/prompt_history.md` | read/update_target | history currently records `full_frozen_attention_only=true`; audit appends truth-state note. |
| `docker/native/*.ps1`, `docker/native/Dockerfile` | read | native container helpers; no rule semantics. |
| `tests/test_hullkv_equivalence.py`, `tests/test_dense_hardmax_2d.py`, `tests/test_sparse_topk_attention.py` | read | verify local 2D hardmax/top-k equivalence; do not prove production hot-path integration. |
| `tests/test_dashboard.py` | read | verifies dashboard metadata and trace display; does not force native policy decoder usage. |
| `tests/test_percepta_frozen_attention_vm.py`, `tests/test_percepta_rule_compiler.py`, `tests/test_percepta_parametric_rules.py`, `tests/test_percepta_e2e_decoder.py` | read | verify Python Percepta behavior vs oracle and declared metadata; do not inspect CUDA semantic lowering. |
| `tests/test_selfplay_training_step.py`, `tests/test_trace_lookahead.py`, `tests/test_select_move_trace.py`, `tests/test_selfplay_*`, `tests/test_analytic_*` | read | preserve legacy ranker/baseline/search/selfplay paths conflicting with current policy-only target unless marked non-production. |
| `tests/test_vm_*`, `tests/test_weight_compiled_rules.py`, `tests/test_transformer_hosted_vm.py`, `tests/test_trace_*` | read | validate legacy Python VM, trace codecs, oracle equivalence, and next-packet transformer experiments. |

## Test Suite Adequacy

- current_tests_strength=behavioral legality, oracle parity, API metadata, counters, dashboard rendering, packet codecs
- current_tests_gap=no static or dynamic proof for expression of every native CUDA rule layer as `QK -> hardmax/select -> V/write`
- current_tests_gap=no test rejects chess-specific branch/loop kernels under names containing `attention`
- current_tests_gap=no test fails when contract string says `full_frozen_attention_only=true` while CUDA source contains `cmz_candidate_target_mask_value`, `cmz_any_legal_candidate_move`, or `cmz_resolve_move_attention_kernel` serial scans
- current_tests_gap=no test ensures Python dashboard uses native policy-only decoder rather than deterministic legal-move index
- current_tests_gap=no test quarantines legacy ranker/baseline/negamax modules from active architecture

## Recommended Remediation Order

1. Set truthful contract fields until kernel bodies are actually lowered: `full_frozen_attention_only=false`, `strict_qk_layer_split_remaining=[candidate_target_mask,record_emit,terminal_legal_presence,terminal_material,resolve_move,trace_projection]`, or equivalent exact fields.
2. Add static source tests failing on custom chess-control-flow helper use in native hot-path kernels unless helper is explicitly classified as compile-time or non-production.
3. Lower candidate generation to uniform frozen 2D attention layers: piece dispatch table lookup, ray/blocker lookup, target mask selection, promotion expansion, and record emission.
4. Lower terminal legal-presence/check/material predicates to uniform frozen 2D attention layers; remove early-exit candidate search semantics.
5. Lower resolve-move and trace selection to true QK hardmax selection over legal-set trace records.
6. Integrate native policy-only decoder into self-play/dashboard or explicitly mark Python dashboard as trace-rule demo, not policy decoder demo.
7. Quarantine or delete legacy ranker/baseline/negamax modules and tests from the current Percepta policy-only acceptance target.
8. Integrate HullKVCache/NestedHullTopK2D into actual rule trace/workspace lookup path or remove "long_context_cache=HullKVCache" from production contract until true.
9. Replace precomputed trace stream with incremental token generation if architecture requires true streaming generation.

## Final Audit Statement

- claim=`all requirements are implemented exactly as written`
- result=false
- reason=metadata and API contract overstate semantic lowering; actual code contains a mixture of frozen-attention metadata, CUDA kernels, PyTorch tensor rule kernels, Python host rule kernels, legacy ranker/baseline/search modules, and scaffolding HullKV/NestedHull utilities.
- claim=`native has no Python runtime fallback for rule calls`
- result=true_with_scope
- scope=native C/Rust route; CUDA failures return errors rather than silently using Python; Python package still contains separate rule runtimes.
- claim=`policy-only native decoder exists`
- result=true
- caveat=Python dashboard self-play and legacy selfplay/training code do not use native_policy_only_decoder as the active strategy.
