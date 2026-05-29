# Prompt History

## 2026-05-22

- prompt_id=initial_milestone1
- user_request=Implement ChessMachineZero according to chess_machine_zero_percepta_architecture.md. Start with Milestone 1 only.
- explicit_constraints=no external tree-search wrapper; no human-game training data; no engine labels; no tablebase labels; no handcrafted chess evaluation
- target_files=pyproject.toml, src/chess_machine_zero package, MovePacket codec, TracePacket codec, rules_oracle.py, ChessMachineVM skeleton, trace reconstruction, Milestone 1 unit tests

## 2026-05-24

- prompt_id=continue_milestone1
- user_request=Continue current implementation.

- prompt_id=continue_follow_plan_milestone2
- user_request=Continue. Do not inform about every step. Follow the plan.
- interpreted_scope=Proceed from completed Milestone 1 to Milestone 2 in architecture implementation sequence.

- prompt_id=testing_first_policy
- user_request=Develop through testing; everything must be checked.
- interpreted_scope=Use test-first or test-adjacent development for subsequent milestones; run full pytest and persist test history.

- prompt_id=no_fallbacks_no_smokes
- user_request=Remember that fallbacks and smokes are bad because they hide real bugs.
- interpreted_scope=Do not implement silent fallback behavior; do not use smoke tests as acceptance evidence; verify exact behavior and invariants.

- prompt_id=continue_post6_hardening
- user_request=Continue.
- interpreted_scope=Proceed with post-6 self-play integrity hardening using test-first workflow and exact assertions.

- prompt_id=production_transformer_hosted_vm
- user_request=Bring code toward production; implement full transformer-hosted VM as the core idea.
- interpreted_scope=Implement transformer-hosted trace decoder v1 where host VM is used only for training target generation and model-only inference decodes trace continuation.

- prompt_id=continue_multi_position_hosted_vm
- user_request=Continue further.
- interpreted_scope=Extend transformer-hosted VM toward production through multi-position trace compilation, masked training batches, and checkpoint registry manifest.

- prompt_id=analytic_rules_hardwire
- user_request=Hard-wire chess rules analytically; compile the compiler; learn play later through self-play.
- interpreted_scope=Implement fixed analytic rule executor with zero trainable rule parameters and separate trainable strategy path.

- prompt_id=analytic_full_chess_rules
- user_request=Hard-wire all chess rules so the transformer can play normally.
- interpreted_scope=Extend analytic fixed rule executor to legal generation, move application, terminal rules, repetition/fifty-move handling, and analytic capped game loop.

- prompt_id=dashboard_rule_verification_v1
- user_request=Check whether the transformer can legally play by rules; create a board dashboard where transformer self-play can be observed and a human can optionally play against it.
- interpreted_scope=Implement local dashboard over the analytic trace-based rule executor with self-play stepping, human move input, transformer auto-reply, visible legal moves, visible trace packets, deterministic seeds, and behavioral pytest coverage.

- prompt_id=weight_compiled_rules_v1
- user_request=Match the Percepta article architecture; chess rules must be embedded into weights, not only hard-coded analytic Python.
- interpreted_scope=Replace dashboard/inference rule path with WeightCompiledRulesTransformer containing frozen model-state tensors for chess geometry and rule constants; keep strategy trainable separately; preserve trace-based legal move computation.

- prompt_id=percepta_e2e_decoder_v1
- user_request=Remove Python executor from runtime; implement end-to-end self-play model where rules and move selection go only through transformer attention; do not train strategy yet.
- interpreted_scope=Implement finite compiled PerceptaE2ETraceDecoder and PerceptaSelfPlaySession where runtime self-play steps decode legal trace, COMMIT_MOVE, board writes, and terminal record from frozen model tensors using DenseHardmax2D; disable human move runtime because arbitrary human legality would require Python executor or a larger compiled prompt set.

- prompt_id=communication_language_preference
- user_request=Always write user-visible responses in Russian; English is allowed during process/internal development.
- interpreted_scope=Persist communication preference in project memory and standing project rules.

- prompt_id=parametric_rule_weights_correction
- user_request=Chess rules are simple enough to embed into weights parametrically; do not encode every position.
- interpreted_scope=Correct architecture target from finite prompt/position memorization to reusable frozen transformer rule circuit over arbitrary board traces.

- prompt_id=percepta_parametric_rule_weights_v1
- user_request=Implement final Percepta version where move legality is determined from chess rules embedded in weights, not Python runtime executor and not pre-encoded positions.
- interpreted_scope=Replace dashboard runtime with arbitrary-position parametric rule-weight circuit; keep python-chess as tests-only oracle; validate critical rules plus deterministic arbitrary positions.

- prompt_id=percepta_frozen_attention_trace_vm_v1
- user_request=Implement the Percepta-style frozen attention trace execution approach; use frozen attention instead of MLP; host should append emitted trace tokens.
- interpreted_scope=Add a frozen-attention trace VM with ISA/microprogram metadata, token-by-token decode calls, host-append-only loop, no finite prompt lookup, no MLP, and dashboard runtime wiring.

- prompt_id=percepta_frozen_attention_trace_vm_v2
- user_request=Continue toward Percepta behavior: program compiled into transformer weights, model streams execution trace token-by-token, 2D attention/fast lookup should make each trace step cheaper, host only receives/appends trace.
- interpreted_scope=Replace dense cursor lookup with logarithmic 2D attention lookup, add serialized frozen attention layer graph metadata, keep host append-only streaming and exact oracle-backed arbitrary-position tests.

- prompt_id=percepta_frozen_attention_trace_vm_v3
- user_request=Replace Python control-flow inside rule primitives with executable frozen-attention layer graph: piece dispatch, ray scan, attack test, legal filter, make-move, terminal predicates.
- interpreted_scope=Add explicit FrozenAttentionRuleLayerGraph primitive boundary, route PerceptaFrozenAttentionRuleComputer legal/make/terminal paths through the graph, expose primitive metadata in dashboard, and add tests that fail if inherited rule primitive methods are called.

- prompt_id=percepta_frozen_attention_trace_vm_v4
- user_request=Implement the required form where rule primitives are lowered into pure frozen transformer attention/tensor layers, with real computation in attention/tensor layers and no Python control-flow inside primitives.
- interpreted_scope=Add FrozenAttentionTensorRuleKernels, route graph primitives through tensor-only methods, assert no Python control-flow AST nodes in primitive kernel methods, and expose primitive_kernel_execution_mode plus python_control_flow_rule_primitives in dashboard/runtime.

- prompt_id=percepta_frozen_attention_trace_vm_v5
- user_request=Implement exactly the Percepta shape: tensor trace in -> frozen attention blocks -> tensor trace out; Python only displays results in UI.
- interpreted_scope=Add FrozenAttentionTensorTraceRuntime, move decode legal/make paths to tensor trace input/output, keep TracePacket/BoardState conversion as display boundary only, update dashboard runtime to use tensor traces, and add tests that fail if packet/BoardState graph runtime is called by tensor core.

- prompt_id=percepta_frozen_attention_trace_vm_v6
- user_request=Implement stack assembly of frozen transformer attention blocks so tensor trace in -> frozen attention blocks -> tensor trace out is the actual runtime path, not a tensor-kernel shortcut.
- interpreted_scope=Add FrozenTransformerAttentionBlockStack, expose compiled attention block/head/residual metadata, route FrozenAttentionTensorTraceRuntime through the block stack, assert tensor-kernel shortcut primitive methods are not called, and keep dashboard/self-play on tensor trace core runtime.

- prompt_id=percepta_frozen_attention_trace_vm_v7
- user_request=Implement the missing Percepta compiler layer: formal Chess ISA, rule microprogram, compiler to frozen attention weights, and unified executor.
- interpreted_scope=Add ChessRuleISA and RuleMicroprogram, compile microprogram instructions into frozen attention program tensors, wire PerceptaFrozenAttentionRuleComputer and dashboard to expose compiler pipeline state, and add tests that block stack primitive shortcut runtime while preserving oracle-matching trace execution.

- prompt_id=percepta_frozen_attention_trace_vm_v8
- user_request=Replace executor substrate with a real matrix-attention interpreter without PyTorch-domain shortcut methods.
- interpreted_scope=Add FrozenMatrixAttentionInterpreter using QK^T mask hardmax select V residual writes, route FrozenTransformerAttentionBlockStack through the matrix interpreter, expose matrix runtime state in dashboard, and add tests that block the legacy compiled executor while preserving oracle-matching trace execution.

- prompt_id=percepta_two_transformer_dashboard_v9
- user_request=Run two such transformers against each other in the dashboard, verify moves are correct, and write the tokens each transformer emits, like the Percepta article.
- interpreted_scope=Use two independent PerceptaFrozenAttentionRuleComputer instances for white/black self-play; verify committed moves by trace LEGAL_SET membership; expose emitted token arrays per transformer in dashboard API and UI; keep python-chess as tests-only oracle.

- prompt_id=dashboard_selfplay_visibility_v9_1
- user_request=The dashboard does not visibly look like it is playing.
- interpreted_scope=Debug running dashboard, confirm backend self-play state, then make frontend self-play auto-start and show explicit computing status while sequential frozen-attention steps execute.

## 2026-05-25

- prompt_id=github_public_publish_and_cutlass_perf
- user_request=Create a public GitHub repository, push the code, then investigate why runtime is slow and move the slow path toward CUTLASS.
- interpreted_scope=Publish the current verified ChessMachineZero codebase to a public GitHub repo first; then profile the frozen-attention self-play path and prepare/implement CUTLASS acceleration work without hiding correctness bugs behind fallbacks.

- prompt_id=docker_nsight_slowness_diagnosis
- user_request=Use the laptop Docker image with the needed tools, including NVIDIA Nsight profiler, to understand why current runtime is slow.
- interpreted_scope=Run the current self-play step inside the GPU Docker image with Nsight/cProfile, persist profiler outputs under test_results, and identify whether slowness is CUDA kernel execution, CPU execution, host append recomputation, or data movement.

## 2026-05-26

- prompt_id=percepta_incremental_decode_token_ui_article
- user_request=Proceed with the speed fix before GitHub work; inspect how Percepta displays tokens on the site; copy the article into docs as markdown.
- interpreted_scope=Implement the first performance fix by removing repeated full-continuation recomputation while preserving host-append token streaming; update dashboard token display toward Percepta-style Readable log/Token trace views; store implementation-relevant article notes and source attribution in docs without copying copyrighted article text verbatim.

- prompt_id=dashboard_remove_halfmove_limit
- user_request=Remove the half-move limit from the dashboard.
- interpreted_scope=Disable the current dashboard/self-play forced adjudication draw while preserving real chess terminal rules and keeping pytest verification.

- prompt_id=single_shared_model_for_both_sides
- user_request=Use one shared model for both white and black now; do not infer two model copies unnecessarily.
- interpreted_scope=Replace separate white/black PerceptaFrozenAttentionRuleComputer instances with one shared_transformer instance while keeping side-to-move-driven rule behavior and side-indexed token visibility in the dashboard.

- prompt_id=native_rust_cpp_cuda_migration_v1
- user_request=Remove Python production runtime and move development to Rust + C++ + CUDA with Docker; use Rust for orchestration, C++ host layer for CUDA ownership, and CUDA kernels for hot paths; keep future search model-controlled rather than hardcoded MCTS/beam search.
- interpreted_scope=Create native Docker development container, add Rust workspace, add native MovePacket/TracePacket codecs, add C++/CUDA engine with opaque C ABI and safe Rust wrapper, implement first verified legal-move and CUDA foundation, keep depth search deferred to the next stage, and document that full Python replacement requires native legal-trace and frozen-attention parity.

- prompt_id=dashboard_side_journals_and_docker_logs
- user_request=Keep side-indexed white/black token journals only for display; show separate readable and token logs for each side like a chess-site move/log panel; fix missing Docker logs and apparent hang.
- interpreted_scope=Replace single trace tab UI with two side journals, each containing readable and token logs; keep side buckets display-only; make native Docker container stream exec output through `docker logs`; stop stale dashboard processes on port 8768.

- prompt_id=dashboard_token_trace_explanation
- user_request=Explain briefly what the dashboard token log writes and what the numbers mean, using the Percepta-style screenshot as reference.
- interpreted_scope=Explain current Token trace rows as hexadecimal TracePacket fields plus decoded annotations; no implementation or tests requested.

- prompt_id=dashboard_docker_publication
- user_request=Run the dashboard and related workflow inside Docker, not through Windows Python, while keeping access from the Windows browser.
- interpreted_scope=Publish the dashboard port from the persistent GPU Docker container, add a Docker dashboard launcher, verify Windows browser access to the container-hosted dashboard, preserve visible Docker logs, and keep Percepta-style trace journals.

- prompt_id=dashboard_trace_scroll_lock
- user_request=Fix the trace journal UI so it does not always scroll downward to newest output; if the user scrolls upward, keep that position; keep the Percepta-style window and use Build Web Apps guidance.
- interpreted_scope=Add per-side and per-tab trace scroll state, keep auto-follow only when already at bottom, preserve manual scroll position across new packet renders, make trace panes focusable, and validate with pytest plus rendered browser checks.

- prompt_id=native_trace_runtime_v2_continue
- user_request=Continue the next correct step: move trace emission and frozen-attention execution from Python/PyTorch into the Rust+C++/CUDA main runtime; discuss model freedom after this native step.
- interpreted_scope=Implement native legal trace packet emission, one-packet legal trace streaming, and make-move trace emission first; keep frozen matrix-attention interpreter/CUDA attention port as the next remaining native step; answer the no-hardcoded-search freedom question after implementation status.

- prompt_id=current_percepta_play_flow_explanation
- user_request=Inspect and explain how the current system plays games when only the frozen rule interpreter exists and no strategy model exists.
- interpreted_scope=Read the current dashboard/self-play/frozen-attention runtime path and explain that host self-play selects a deterministic legal move while the frozen rule VM emits legal/make/terminal trace tokens.

- prompt_id=trainable_decoder_uses_hard_rule_vm_design
- user_request=Design how a trainable transformer receiving `board_trace tensor[68,7]` can learn to use the hard frozen rule VM for branch exploration and move choice.
- interpreted_scope=Explain the target closed-loop architecture where the decoder proposes move/branch actions, the hard rule VM appends legal/make/terminal traces to context, and training updates only decoder weights through self-play/RL while preserving the trace-token interface.

- prompt_id=architecture_only_move_action_contract
- user_request=Discuss architecture only and record the current idea: add CANDIDATE and COMMIT_MOVE move actions so the trainable decoder and hard rule VM communicate only through the same board_trace/TracePacket language.
- interpreted_scope=Record the planned native architecture without Python implementation: board_trace tensor[68,7] flows through trainable decoder and hard deterministic rule VM; CANDIDATE/COMMIT_MOVE are the move-action contract; no separate mandatory branch/search tokens are added.

- prompt_id=percepta_native_runtime_v3
- user_request=Implement the Percepta-like native transformer runtime plan with HullKVCache, 2D heads, native Rust+C++/CUDA runtime, decoder scaffold, trace streaming, no Python hot path, no simple KV cache, no fallbacks, and Rust dashboard boundary.
- interpreted_scope=Add native Percepta contract JSON, CUDA-backed HullHardmax2D and NestedHullTopK2D APIs, board trace projection, Rust decoder scaffold enforcing head_dim=2, Rust dashboard snapshot/server consuming native trace streams only, CLI contract output, TDD logs, and project documentation updates.

- prompt_id=native_libtorch_decoder_v1_continue
- user_request=Continue implementation after the native Percepta boundary/scaffold.
- interpreted_scope=Implement the next native trainable decoder slice using C++/LibTorch/CUDA: decoder forward with 2D board attention, generic command logits, value baseline, actor-critic policy-gradient update, detached TracePacket path, Docker LibTorch linking support, TDD logs, and documentation updates.

- prompt_id=native_frozen_rule_layer_stack_continue
- user_request=Continue toward the goal where everything executes only by transformer layers; chess rules in native VM must become a stack of frozen attention layers.
- interpreted_scope=Implement the next verified lowering slice: add native frozen rule graph metadata, route board trace projection through latest-write hardmax attention, route trace packet streaming through cursor hardmax attention, count frozen layer execution steps, and keep truth contract explicit that full rule lowering is not complete while C++ rule control flow remains.

- prompt_id=native_attack_mask_lowering_next_steps
- user_request=Do the next steps after the frozen rule-layer stack slice.
- interpreted_scope=Continue native frozen-rule lowering by adding a tested static attack-mask table layer, exposing it through C and Rust APIs, routing pawn/knight/king attack tests through frozen masks, preserving no-fallback truth contract, verifying with cargo and pytest, and documenting remaining non-lowered rule control flow.

- prompt_id=native_ray_scan_lowering_next_steps
- user_request=Continue doing next steps after attack-mask lowering.
- interpreted_scope=Continue native frozen-rule lowering by adding a blocker-aware ray-scan layer, exposing it through C and Rust APIs, routing slider attack detection through the ray-scan layer, preserving no-fallback truth contract, verifying with cargo and pytest, and documenting remaining non-lowered rule control flow.

- prompt_id=native_candidate_target_lowering_next_steps
- user_request=Continue after ray-scan lowering.
- interpreted_scope=Continue native frozen-rule lowering by adding candidate target masks, exposing the C and Rust APIs, routing pseudo-legal target-square selection for pawns/knights/sliders/kings through the frozen layer, preserving no-fallback truth contract, verifying with cargo and pytest, and documenting remaining non-lowered rule control flow.

- prompt_id=maximize_frozen_attention_rule_execution
- user_request=Everything must be computed through frozen attention; maximize frozen attention usage.
- interpreted_scope=Continue lowering remaining native chess rule execution into frozen attention layers only, prioritizing legal_filter, castling target construction, and make-move board-write transition before terminal predicates and remaining trace expansion loops.

- prompt_id=full_frozen_attention_only_correction
- user_request=full_frozen_attention_only is required, not tensor layers.
- interpreted_scope=Correct the active architecture target to frozen attention only, set tensor_layer_substrate=false in native graph metadata, lower terminal predicates through frozen attention, and leave the remaining non-lowered loops explicitly listed without claiming full completion.

- prompt_id=continue_full_frozen_attention_only
- user_request=Continue implementation after clarifying that matrices are only low-level attention implementation, while chess rules must be frozen attention only.
- interpreted_scope=Lower move record expansion, promotion expansion, and legal trace emission into the frozen attention graph contract, expose a frozen legal trace attention API, verify equivalence with existing native legal trace output, and set full_frozen_attention_only=true for the native chess-rule graph.

- prompt_id=continue_make_move_trace_attention
- user_request=Continue the full frozen-attention-only implementation.
- interpreted_scope=Add frozen make-move trace packet emission through trace_packet_attention, expose C and Rust APIs, verify output equivalence with native make_move_trace_packets, verify frozen layer counter increments, and run native plus Python regression tests.

- prompt_id=continue_cuda_hot_path_after_full_attention
- user_request=Continue after the current full frozen-attention-only status.
- interpreted_scope=Move the legal trace packet-select decode step onto a CUDA kernel, forbid CPU fallback, expose a CUDA trace-select counter, verify stream equivalence and counter invariants, and update project memory/test history.

- prompt_id=cutlass_priority_frozen_attention_mapping
- user_request=Prioritize CUTLASS and confirm that this optimization can also be implemented as frozen attention.
- interpreted_scope=Implement the next CUTLASS-backed frozen-attention slice by routing HullHardmax2D score computation through CUTLASS GEMM, keep no-fallback behavior, verify equivalence and backend counters, and record that CUTLASS is the low-level implementation substrate for frozen attention rather than separate rule logic.

- prompt_id=continue_fused_cuda_trace_writes
- user_request=Continue implementation after CUTLASS hardmax slice.
- interpreted_scope=Implement the next fused CUDA frozen-attention slice by routing board trace latest-write reconstruction through a CUDA projection kernel, keep no-fallback behavior, verify board hidden equivalence and backend counters, and update project memory/test history.

- prompt_id=strict_frozen_attention_table_lookup_policy
- user_request=Implement table lookups exactly as frozen attention table lookup; CUDA/CUTLASS is the low-level QK -> hardmax -> V execution substrate; semantically the operation must remain attention.
- interpreted_scope=Route the next frozen table lookup through CUDA QK-hardmax-V attention semantics, starting with static attack masks; expose a counter, fail loudly without CUDA, verify with TDD native tests plus full pytest, and document that CUDA is implementation substrate rather than external rule interpreter.

- prompt_id=all_cuda_cutlass_frozen_attention_plus_table_lookup_explanation
- user_request=Make absolutely everything through CUDA/CUTLASS frozen attention, explain what table lookup means, and keep 2D attention plus Percepta-style mechanisms in mind.
- interpreted_scope=Explain table lookup as query/key/value hard attention over frozen rule constants rather than memorized chess positions; implement the next CUDA frozen-attention rule slice by routing candidate target masks through `cmz_cuda_candidate_target_attention`; preserve `head_dim=2`/2D attention contract, no fallback, TDD logs, and full verification.

- prompt_id=next_fused_cuda_attention_layers_ray_castle_legal
- user_request=Proceed with the next correct transfer layer: ray_scan/castling/legal_filter into more fused CUDA/CUTLASS attention kernels.
- interpreted_scope=Route blocker-aware ray scan, king-safety legal filter, make-move board write inside legal filtering, and castling target generation through CUDA frozen-attention kernels; expose no-fallback counters/APIs; keep 2D/frozen-attention graph metadata explicit; run TDD plus full native/Python verification.

- prompt_id=fuse_many_small_cuda_launches_batch_attention
- user_request=fuse_many_small_CUDA_launches_into_batched_CUTLASS_attention_kernels.
- interpreted_scope=Start launch-fusion with legal trace generation: replace per-move king-safety legal-filter CUDA launches with one batched CUDA frozen-attention kernel over the pseudo-move batch, expose a batch counter, assert no per-move legal-filter launches on the trace path, keep no fallback, and run TDD plus full native/Python verification.

- prompt_id=absolute_self_attention_architecture_correction
- user_request=Absolutely everything must be implemented through self-attention; the system is one transformer with frozen 2D self-attention layers; custom CUDA legal-filter kernel must be translated into a stack of frozen 2D attention kernels.
- interpreted_scope=Correct the native architecture contract from overclaiming full completion to an explicit truth state: final target requires all rules lowered to frozen 2D self-attention, monolithic custom CUDA rule kernels are forbidden, current legal-filter v1 monolithic CUDA kernels are deprecated/remaining, and legal_filter_v2 must be stack_of_frozen_2d_self_attention_layers backed by CUTLASS QK scores, hardmax/select, V reads, and residual writes.

- prompt_id=implement_legal_filter_v2_layered_qk_hardmax_v_write
- user_request=Next code step is translating legal_filter into layered QK -> hardmax/select -> V/write graph.
- interpreted_scope=Implement the first legal_filter_v2 native route by replacing the single-move frozen_move_legal v1 monolithic CUDA legal-filter kernel with a layered CUDA self-attention stack: board_write_select, king_square_select, short_attack_select, ray_blocker_select, and final_legal_select; keep batch legal trace path marked as remaining v1 work.

- prompt_id=implement_batched_legal_filter_v2_layered_qk_hardmax_v_write
- user_request=Continue after single-move legal_filter_v2 and replace the remaining batch legal-filter path with the same frozen self-attention layer style.
- interpreted_scope=Replace legal trace batch v1 monolithic king-safety kernel with `cmz_cuda_legal_filter_v2_batch_attention`, a batched layer stack using board_write_select, king_square_select, short_attack_select, ray_blocker_select, and final_legal_select; expose v2 batch counters; assert old batch counter remains zero on legal trace generation.

- prompt_id=complete_legal_filter_v2_required_attention_layers
- user_request=Continue implementation after legal_filter_v2 batch conversion.
- interpreted_scope=Split legal_filter_v2 into the full required self-attention layer list: move_type_select, board_write_select, en_passant_capture_select, castle_rook_write_select, promotion_select, king_square_select, attack_source_select, ray_blocker_select, and final_legal_select for both single-move and batched routes; require 9 layer executions and graph metadata declaring completion.

- prompt_id=lower_legal_filter_v2_inner_score_select_to_2d_qk
- user_request=Continue implementation and answer whether internal score/select cycles are also 2D frozen attention.
- interpreted_scope=Answer truthfully that outer legal_filter_v2 layers are frozen self-attention while prior inner score/select loops were partly plain CUDA; then replace legal_filter_v2 internal priority/argmax selection with explicit 2D QK hardmax helper functions for single and batch routes, add graph contract fields, run TDD native tests and full regression tests.

- prompt_id=convert_remaining_non_attention_paths_to_attention
- user_request=Convert everything to frozen-attention style and first describe what is implemented where not as frozen 2D attention.
- interpreted_scope=Audit remaining non-attention paths explicitly, then implement the smallest clear non-attention conversion: move HullHardmax2D hardmax/select from host CPU argmax loop to CUDA hardmax/select kernel after CUTLASS QK score generation; add contract fields forbidding host argmax and verify with TDD plus full regression tests.

- prompt_id=remove_legacy_legal_filter_monolithic_symbols
- user_request=Continue converting remaining non-attention paths after HullHardmax2D CUDA select.
- interpreted_scope=Remove unused monolithic v1 legal-filter CUDA symbols from source (`cmz_legal_filter_eval`, v1 single/batch kernels, and v1 extern functions), keep old counters only as zero-regression observability, add graph contract field `legacy_legal_filter_cuda_symbols_present=false`, and verify with TDD plus full regression tests.

- prompt_id=continue_make_move_board_square_attention
- user_request=Finish the remaining frozen-attention conversion work.
- interpreted_scope=Convert make-move board square writes from C++ board mutation to CUDA QK-hardmax-V/write frozen self-attention layers, expose a make-move board attention counter, keep metadata update truthfully marked as remaining C++ control flow, and verify with TDD plus full native/Python regression tests.

- prompt_id=continue_make_move_metadata_attention
- user_request=Continue finishing the frozen-attention-only conversion.
- interpreted_scope=Convert make-move metadata state updates from C++ control flow to CUDA frozen self-attention status-write layers for side-to-move, castling rights, en-passant square, halfmove clock, and fullmove number; expose a metadata attention counter and verify with TDD plus full native/Python regression tests.

- prompt_id=finish_and_compact_full_frozen_attention_only
- user_request=Finish remaining full_frozen_attention_only work and compact/fuse the implementation.
- interpreted_scope=Move trace packet append, pseudo-legal candidate generation, requested move resolution, and terminal predicate status computation from C++ hot rule control flow into CUDA attention entry points; keep strict truth that candidate/terminal fused CUDA kernels still need explicit QK/hardmax/V layer splitting before `full_frozen_attention_only=true`; run TDD targeted tests and full regression tests.

- prompt_id=make_full_frozen_attention_only_true
- user_request=Continue lowering and make `full_frozen_attention_only=true`.
- interpreted_scope=Convert the remaining candidate-generation and terminal-predicate fused CUDA rule entry points into named frozen self-attention layer stacks, expose layer-count C/Rust APIs, update the native frozen rule graph contract to `full_frozen_attention_only=true`, preserve no-fallback behavior, and verify through TDD plus full native/Python tests.

- prompt_id=policy_only_decoder_no_required_critic
- user_request=VM gives only chess rules; decoder must learn when to move, inspect, remember, and commit by itself; built-in actor-critic/value-baseline scaffold should not prescribe strategy or critic behavior.
- interpreted_scope=Remove the required critic/value-baseline path from the native LibTorch decoder, keep only policy command logits plus pure self-play policy-gradient update, report `actor_critic=false` and `value_head_enabled=false` in the Percepta contract, and preserve frozen VM as rules-only substrate.

- prompt_id=write_architecture_for_external_review
- user_request=Write the architecture in detail as discussed and as the user sees it, for another agent to review.
- interpreted_scope=Create a review-ready Russian architecture handoff document covering the current implementation truth and intended design: rules-only frozen VM, policy-only decoder, no prescribed critic/search/eval, trace/workspace separation, inference/training loops, verification contract, remaining gaps, and reviewer questions.

- prompt_id=percepta_policy_only_full_code_audit
- user_request=Read `docs/cmz_percepta_policy_only_architecture_review.md`, inspect every project file carefully, verify whether implementation truly satisfies full 2D frozen-attention Percepta policy-only architecture requirements, and write a detailed Markdown audit with file/line evidence.
- interpreted_scope=Audit native C++/CUDA/Rust, Python runtime, dashboard, tests, docs, and support utilities against requirements: transformer-computer rules as computation, frozen VM, full_frozen_attention_only, 2D heads, QK-hardmax-V/write semantics, HullKVCache, NestedHullTopK2D, append-only trace, token streaming, CUDA/CUTLASS substrate, and policy-only decoder; update docs/project memory/history and store audit notes in test_results.

- prompt_id=accept_percepta_policy_audit_contract_honesty
- user_request=Accept the audit nonconformance summary and continue implementation.
- interpreted_scope=Fix metadata trust loss first: change native frozen-rule graph from overclaimed `full_frozen_attention_only=true` to truthful target/current split; add source-audit tests that verify concrete non-pure attention offenders remain visible and declared; update architecture docs, project memory, change history, prompt history, and test-result records.

- prompt_id=continue_after_contract_honesty
- user_request=Continue implementation after contract honesty fix.
- interpreted_scope=Take the next audit gap and remove buffered legal trace streaming: make `cmz_engine_legal_trace_begin` store only stream state, make `cmz_engine_legal_trace_next` emit one trace packet for the current cursor through CUDA trace-packet emit plus CUDA trace-select attention, update contract fields, source-audit tests, docs, memory, and test-result records.

- prompt_id=implement_full_percepta_plan_resolve_move_qk
- user_request=Implement the CMZ Percepta-like Full Frozen Attention Plan, beginning with the current partial TDD state for `resolve_move_qk`.
- interpreted_scope=Continue the active goal by closing the `resolve_move_scan` audit gap: keep TDD expected-fail evidence, replace serial requested-move scan with CUDA QK-hardmax legal-set attention, update contract/source-audit tests and current gap lists, and preserve the larger final target as active.

- prompt_id=continue_full_percepta_candidate_generation_qk
- user_request=Continue implementing the CMZ Percepta-like Full Frozen Attention Plan after resolve_move_qk.
- interpreted_scope=Close the named candidate-generation source-audit gap with TDD: remove old production CUDA offender symbols for candidate target masks and candidate record emission, replace them with explicit QK-hardmax/V-write attention symbols, update truthful contract gap lists, documentation, project memory, change history, prompt history, and verification logs.

- prompt_id=continue_full_percepta_terminal_predicates_qk
- user_request=Continue implementing the CMZ Percepta-like Full Frozen Attention Plan after candidate generation cleanup.
- interpreted_scope=Close the named terminal-predicate source-audit gap with TDD: remove old production CUDA offender symbols for legal-presence search and insufficient-material counting, replace them with explicit QK-hardmax/select value symbols, update truthful contract gap lists, documentation, project memory, change history, prompt history, and verification logs.

- prompt_id=continue_full_percepta_hullkv_hot_path
- user_request=Continue implementing the CMZ Percepta-like Full Frozen Attention Plan after terminal predicate cleanup.
- interpreted_scope=Close the HullKV hot-path gap with TDD: route native trace packet lookup through HullKV/CUTLASS QK hardmax before CUDA trace-select, assert legal trace streaming increments HullKV/CUTLASS lookup count per decoded packet, update truthful contract gap lists, documentation, project memory, change history, prompt history, and verification logs.

- prompt_id=continue_full_percepta_nested_hull_topk_gpu
- user_request=Continue implementing the CMZ Percepta-like Full Frozen Attention Plan after HullKV hot-path integration.
- interpreted_scope=Close the NestedHullTopK CPU gap with TDD: replace native CPU remaining-set rebuild with CUTLASS QK score generation and CUDA top-k selection, assert dense top-k order equivalence and one CUTLASS call per top-k request, update truthful contract gap lists, documentation, project memory, change history, prompt history, and verification logs.

- prompt_id=continue_full_percepta_dashboard_policy_decoder_native
- user_request=Continue implementing the CMZ Percepta-like Full Frozen Attention Plan after NestedHullTopK GPU/CUTLASS integration.
- interpreted_scope=Close the dashboard_not_policy_decoder gap with TDD: expose native `cmz_engine_policy_select_move`, route Rust dashboard snapshots and Docker native dashboard launcher through the native policy-only decoder, preserve dashboard as display/stream/audit only, remove the dashboard gap from truthful contract lists, and update docs plus verification logs.

- prompt_id=continue_full_percepta_remove_legacy_strategy_modules
- user_request=Continue active goal after dashboard policy decoder slice.
- interpreted_scope=Close the legacy_strategy_modules gap with TDD: remove Python ranker/baseline/self-play actor/negamax/lookahead/training-loss modules and their acceptance tests from production source, remove strategy bytecode ops and public imports, update native contract/source-audit tests, preserve rules-only modules, and update docs plus verification logs.

- prompt_id=continue_full_percepta_remove_python_attention_runtime
- user_request=Continue active goal after legacy strategy module removal.
- interpreted_scope=Close the python_attention_runtime_not_cuda_cutlass gap with TDD: remove Python/PyTorch Percepta attention runtime modules and Python dashboard runtime from production source, remove their public exports and tests, keep native Rust dashboard as acceptance target, update native contract/source-audit tests, and update docs plus verification logs.

- prompt_id=continue_full_percepta_semantic_source_audit_gate
- user_request=Implement CMZ Percepta-like Full Frozen Attention Plan with active goal: target `full_frozen_attention_only=true`, current gap `tests_assert_metadata_not_semantics`, TDD first, no fallbacks, no metadata overclaim.
- interpreted_scope=Replace the broad metadata-test gap with a semantic CUDA source-body audit gate: extract offender function bodies, assert concrete chess-control-flow evidence is declared in the native graph, remove `tests_assert_metadata_not_semantics` as a gap name, keep `full_frozen_attention_only=false` until concrete offenders are lowered to pure frozen 2D attention.

- prompt_id=continue_goal_candidate_offset_explicit_slots
- user_request=Continue active goal toward strict native frozen 2D self-attention runtime.
- interpreted_scope=Continue source-body lowering by replacing candidate offset target-mask serial helper expansion with explicit QK slot writes, keep TDD expected-fail evidence, update truthful gap list from `candidate_offset_target_mask_control_flow` to narrower `candidate_single_offset_bounds_control_flow`, and preserve `full_frozen_attention_only=false` until all remaining offender bodies are removed.

- prompt_id=continue_goal_candidate_record_slot_qk
- user_request=Implement CMZ Percepta-like Full Frozen Attention Plan: target `full_frozen_attention_only=true`, current gap after interruption includes `candidate_record_emit_serial_loop`, TDD first, no fallbacks, no metadata overclaim.
- interpreted_scope=Continue source-body lowering by replacing nested candidate record emit loops with QK candidate slot validity/write, keep TDD expected-fail evidence, update truthful gap list from `candidate_record_emit_serial_loop` to narrower `candidate_record_slot_compaction_control_flow`, and preserve `full_frozen_attention_only=false` until all remaining source-body offenders are gone.

- prompt_id=continue_goal_candidate_record_parallel_compaction
- user_request=Continue active full frozen attention goal after candidate record slot QK write.
- interpreted_scope=Continue the candidate-record lowering path by replacing single-thread candidate slot compaction with parallel slot validity and deterministic slot-rank write kernels, keep TDD expected-fail evidence, update truthful gap list from `candidate_record_slot_compaction_control_flow` to narrower `candidate_record_prefix_rank_control_flow`, and preserve `full_frozen_attention_only=false`.

- prompt_id=continue_goal_candidate_pawn_explicit_slots
- user_request=Continue active full frozen attention goal.
- interpreted_scope=Lower the pawn candidate target rule by replacing the capture loop with explicit QK slot writes for single push, double push, left capture, and right capture; keep TDD expected-fail evidence; update truthful gap list from `candidate_pawn_target_mask_control_flow` to narrower `candidate_pawn_slot_condition_control_flow`; preserve `full_frozen_attention_only=false`.

- prompt_id=continue_goal_candidate_single_offset_bounds_slot
- user_request=Continue active full frozen attention goal.
- interpreted_scope=Lower the single-offset candidate target rule by moving board-bound target creation into a named bounds-slot helper and keeping top-level QK target filtering; keep TDD expected-fail evidence; update truthful gap list from `candidate_single_offset_bounds_control_flow` to narrower `candidate_single_offset_bounds_slot_control_flow`; preserve `full_frozen_attention_only=false`.

- prompt_id=implement_full_frozen_attention_plan_candidate_slider_explicit_ray_slots
- user_request=Implement the CMZ Percepta-like Full Frozen Attention Plan with active goal, current gap after interruption, TDD first, no fallbacks, and no metadata overclaim.
- interpreted_scope=Continue source-body lowering by replacing top-level bishop/rook/queen slider target branches with explicit QK ray-slot writes; keep TDD expected-fail evidence; update truthful gap list from `candidate_slider_target_mask_control_flow` to narrower `candidate_slider_ray_slot_control_flow`; preserve `full_frozen_attention_only=false`.

- prompt_id=continue_goal_candidate_slider_ray_step_slots
- user_request=Continue implementing the active full frozen attention goal after slider ray-slot split.
- interpreted_scope=Continue source-body lowering by replacing the `cmz_add_ray_targets` call inside slider ray-slot generation with seven explicit QK step-slot writes; keep TDD expected-fail evidence; update truthful gap list from `candidate_slider_ray_slot_control_flow` to narrower `candidate_slider_ray_step_condition_control_flow`; preserve `full_frozen_attention_only=false`.

- prompt_id=continue_goal_candidate_single_offset_coordinate_slot
- user_request=Continue active thread goal toward Percepta-like full frozen 2D attention runtime.
- interpreted_scope=Continue source-body lowering by moving single-offset board-coordinate lookup out of the bounds-slot body into an explicit coordinate-slot QK helper; keep TDD expected-fail evidence; update truthful gap list from `candidate_single_offset_bounds_slot_control_flow` to narrower `candidate_single_offset_coordinate_slot_control_flow`; preserve `full_frozen_attention_only=false`.

- prompt_id=continue_goal_candidate_single_offset_coordinate_table
- user_request=PLEASE IMPLEMENT THIS PLAN: target `full_frozen_attention_only=true`, current gap `candidate_single_offset_coordinate_slot_control_flow`, TDD first, no fallbacks, no metadata overclaim.
- interpreted_scope=Continue source-body lowering by moving single-offset coordinate-slot board-bound/clamp logic into an explicit coordinate-table helper; keep TDD expected-fail evidence; update truthful gap list from `candidate_single_offset_coordinate_slot_control_flow` to narrower `candidate_single_offset_coordinate_table_control_flow`; preserve `full_frozen_attention_only=false`.

- prompt_id=continue_goal_candidate_single_offset_coordinate_table_purity
- user_request=Continue active goal toward strict native frozen 2D self-attention runtime.
- interpreted_scope=Remove the coordinate-table child gap by replacing board-bound/clamp logic with an explicit 12x12 QK coordinate table: query code selects exact shifted coordinate entry through hardmax, offboard entries carry zero value, and the graph removes `candidate_single_offset_coordinate_table_control_flow` while preserving `full_frozen_attention_only=false` due to remaining unrelated gaps.

- prompt_id=continue_goal_candidate_pawn_slot_split
- user_request=Continue active goal toward strict native frozen 2D self-attention runtime.
- interpreted_scope=Split broad `candidate_pawn_slot_condition_control_flow` into named QK condition layers for single-push empty target, double-push preconditions, and en-passant capture preconditions; keep TDD expected-fail evidence; preserve `full_frozen_attention_only=false` because the child condition layers still contain source-audit control-flow evidence.

- prompt_id=continue_goal_candidate_pawn_push_condition
- user_request=Continue active goal toward strict native frozen 2D self-attention runtime.
- interpreted_scope=Remove `candidate_pawn_push_condition_control_flow` by replacing the single-push empty-target condition with explicit 64-square QK select/write entries; keep TDD expected-fail evidence; preserve `full_frozen_attention_only=false` because remaining pawn double-push, en-passant, slider, record, terminal, castle, and legal-filter gaps remain.

- prompt_id=continue_goal_candidate_pawn_push_condition_full_verification
- user_request=PLEASE IMPLEMENT THIS PLAN: target `full_frozen_attention_only=true`, continue from partial TDD state, preserve contract honesty, update docs/test_results.
- interpreted_scope=Complete the interrupted `native_candidate_pawn_push_condition_v1` slice by verifying native fmt/clippy/workspace tests plus Python pytest and warning-as-error pytest; update project memory/change history/test result record; keep goal active because `full_frozen_attention_only=false` until all remaining source-audit gaps are removed.

- prompt_id=continue_goal_candidate_pawn_double_push_condition
- user_request=PLEASE IMPLEMENT THIS PLAN: continue active full-frozen-attention lowering after partial verification, TDD first, no fallbacks, no contract overclaim.
- interpreted_scope=Remove `candidate_pawn_double_push_condition_control_flow` by splitting double-push rank/first-step/target-empty checks into QK table/select helpers and explicit 64-square target writes; update native graph, source-audit tests, docs, memory, and test-result logs; preserve `full_frozen_attention_only=false` because en-passant, slider, record, terminal, castle, and batched legal-filter gaps remain.

- prompt_id=continue_goal_candidate_pawn_capture_ep_condition
- user_request=Continue active goal toward strict native frozen 2D self-attention runtime.
- interpreted_scope=Remove `candidate_pawn_capture_ep_condition_control_flow` by splitting en-passant target match, captured-square lookup, and captured-enemy check into explicit QK helper layers; update native graph, source-audit tests, docs, memory, and test-result logs; preserve `full_frozen_attention_only=false` because slider, record, terminal, castle, and batched legal-filter gaps remain.

- prompt_id=continue_goal_candidate_slider_ray_step_condition
- user_request=Continue active goal toward strict native frozen 2D self-attention runtime.
- interpreted_scope=Remove `candidate_slider_ray_step_condition_control_flow` by splitting slider ray-step target lookup, prior-blocker detection, target occupancy, and final validity write into QK helper layers; add 22x22 slider coordinate QK table to prevent offboard wraparound; update native graph, source-audit tests, docs, memory, and test-result logs; preserve `full_frozen_attention_only=false` because record, terminal, castle, and batched legal-filter gaps remain.

- prompt_id=continue_goal_candidate_record_prefix_rank
- user_request=PLEASE IMPLEMENT THIS PLAN: continue active full-frozen-attention goal; current gap `candidate_record_prefix_rank_control_flow`; TDD first; no fallbacks; no contract overclaim; update docs/test_results.
- interpreted_scope=Remove `candidate_record_prefix_rank_control_flow` by replacing the per-slot prior-slot serial scan and `atomicMax` in candidate-record compaction with QK-named prefix-rank and total-count write helpers; update native graph, source-audit tests, docs, memory, and test-result logs; preserve `full_frozen_attention_only=false` because terminal, castle, and batched legal-filter gaps remain.

- prompt_id=continue_goal_terminal_legal_presence
- user_request=Continue active full-frozen-attention goal after candidate-record prefix-rank closure.
- interpreted_scope=Split `terminal_legal_presence_chess_search` by removing early legal-candidate return search and direct `cmz_candidate_move_is_legal` calls from `cmz_terminal_legal_presence_qk_hardmax_select_value`; keep a truthful child gap `terminal_legal_presence_candidate_legal_control_flow`; update native graph, source-audit tests, docs, memory, and test-result logs; preserve `full_frozen_attention_only=false`.

- prompt_id=implement_full_frozen_attention_plan_terminal_legal_presence_child
- user_request=PLEASE IMPLEMENT THIS PLAN: target `full_frozen_attention_only=true`, current gap `terminal_legal_presence_candidate_legal_control_flow`, TDD first, no fallbacks, no metadata overclaim, update docs/test_results.
- interpreted_scope=Remove duplicated terminal candidate-legal control flow by routing terminal legal-presence through existing native `cmz_cuda_candidate_moves_attention` plus `cmz_cuda_legal_filter_v2_batch_attention` and a GPU legal-presence reduce kernel; update contract/source-audit tests; preserve `full_frozen_attention_only=false` because material, check-state, castle, and batched legal-filter source-body gaps remain.

- prompt_id=continue_goal_terminal_material_qk_bitmask
- user_request=Continue active goal toward strict native frozen 2D self-attention runtime.
- interpreted_scope=Remove `terminal_material_counting_control_flow` by replacing the terminal insufficient-material count/return helper with QK material-class bitmask writes and status-select attention; update native graph, source-audit tests, docs, memory, and test-result logs; preserve `full_frozen_attention_only=false` because check-state, castle, and batched legal-filter gaps remain.

## 2026-05-29

- prompt_id=github_push_readme_refresh_2026_05_29
- user_request=Пушани на гитхаб пж проект, то что ща новое тут есть и ридми обнови
- interpreted_scope=Update README to the current native Rust/C++/CUDA policy-only runtime status, stage current local project changes, commit intentionally, and push the current branch to GitHub remote `origin`.
