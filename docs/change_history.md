# Change History

## 2026-05-24

- change_id=milestone1_init
- summary=Created Milestone 1 foundation files for ChessMachineZero.
- files_added=pyproject.toml, AGENTS.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md, src/chess_machine_zero package, tests package
- scope=project skeleton, MovePacket codec, TracePacket codec, rules oracle wrapper, host VM legal enumeration trace, board write reconstruction, Milestone 1 tests
- verification=Milestone 1 acceptance suite passed; full pytest suite passed; python-chess direct import confined to rules_oracle.py

- change_id=milestone2_deterministic_program
- summary=Added deterministic chess program foundation and trace-based make-move/terminal paths.
- files_added=.gitignore, src/chess_machine_zero/chess/outcome.py, src/chess_machine_zero/vm/bytecode.py, src/chess_machine_zero/vm/chess_program.py, tests/test_vm_make_move.py, tests/test_vm_terminal.py
- files_updated=AGENTS.md, src/chess_machine_zero/chess/rules_oracle.py, src/chess_machine_zero/vm/interpreter.py, src/chess_machine_zero/vm/trace_packet.py, src/chess_machine_zero/trace/reconstruct.py, src/chess_machine_zero/trace/verifier.py
- scope=Milestone 2 only; deterministic legal generation, make-move transitions, terminal checks, random legal game verification
- verification=Full pytest suite passed with 28 tests; 1000 deterministic random positions matched python-chess oracle; direct python-chess import confined to rules_oracle.py

- change_id=milestone3_dense_executor
- summary=Added DenseHardmax2D, CMZMachineTransformer, trace dataset utilities, next-packet overfit trainer, and trace executor facade.
- files_added=src/chess_machine_zero/model/__init__.py, src/chess_machine_zero/model/hardmax_attention.py, src/chess_machine_zero/model/machine_transformer.py, src/chess_machine_zero/model/trace_executor.py, src/chess_machine_zero/trace/datasets.py, src/chess_machine_zero/train/__init__.py, src/chess_machine_zero/train/trainer.py, tests/test_dense_hardmax_2d.py, tests/test_machine_transformer_shapes.py, tests/test_trace_dataset.py, tests/test_trace_next_packet_training.py, tests/test_trace_executor.py
- files_updated=pyproject.toml, AGENTS.md, docs/project_memory.md
- scope=Milestone 3 only; no move strategy labels; next-packet training uses deterministic trace packets only
- verification=Tests added before implementation; first new-test run failed with expected import errors; full pytest passed with 36 tests; pytest warning-as-error passed with 36 tests; direct python-chess import confined to rules_oracle.py

- change_id=milestone4_selfplay_ranker
- summary=Added trace-based legal move scorer, self-play actor, replay records, and self-play training step.
- files_added=src/chess_machine_zero/model/ranker.py, src/chess_machine_zero/model/baseline.py, src/chess_machine_zero/vm/decision_program.py, src/chess_machine_zero/vm/trace_hash.py, src/chess_machine_zero/selfplay/__init__.py, src/chess_machine_zero/selfplay/game_record.py, src/chess_machine_zero/selfplay/actor.py, src/chess_machine_zero/selfplay/replay.py, src/chess_machine_zero/selfplay/temperature.py, src/chess_machine_zero/train/losses.py, tests/test_select_move_trace.py, tests/test_selfplay_no_illegal_moves.py, tests/test_selfplay_training_step.py
- files_updated=src/chess_machine_zero/model/__init__.py, src/chess_machine_zero/vm/bytecode.py, AGENTS.md, docs/project_memory.md
- scope=Milestone 4 only; legal moves come from VM trace records; ranker scores only legal candidate packets; no external tree search, no human-game data, no engine/tablebase labels, no handcrafted chess evaluation
- verification=Tests added before implementation; first new-test run failed with expected import errors; full pytest passed with 39 tests; pytest warning-as-error passed with 39 tests; direct python-chess import confined to rules_oracle.py

- change_id=milestone5_internal_lookahead
- summary=Added internal trace lookahead with baseline leaf values and trace-window controls.
- files_added=src/chess_machine_zero/trace/windows.py, src/chess_machine_zero/vm/lookahead.py, tests/test_trace_lookahead.py
- files_updated=src/chess_machine_zero/vm/decision_program.py, AGENTS.md, docs/project_memory.md
- scope=Milestone 5 only; TRACE_NEGAMAX represented as trace records; leaf values come from CMZOutcomeBaseline; no external evaluator
- verification=Tests added before implementation; first new-test run failed with expected import error; full pytest passed with 42 tests; pytest warning-as-error passed with 42 tests; direct python-chess import confined to rules_oracle.py

- change_id=milestone6_hullkv
- summary=Added exact convex-hull-backed hardmax and nested top-k retrieval with cache and sparse attention integration.
- files_added=src/chess_machine_zero/hullkv/__init__.py, src/chess_machine_zero/hullkv/convex_hull_2d.py, src/chess_machine_zero/hullkv/cache.py, src/chess_machine_zero/hullkv/nested_hulls.py, src/chess_machine_zero/hullkv/equivalence.py, src/chess_machine_zero/model/sparse_topk_attention.py, tests/test_hullkv_equivalence.py, tests/test_sparse_topk_attention.py
- files_updated=AGENTS.md, docs/project_memory.md, docs/prompt_history.md
- scope=Milestone 6 only; convex hull support query and nested hull top-k; dense helpers isolated as test/equivalence oracle, not runtime fallback
- verification=Tests added before implementation; first new-test run failed with expected import error; full pytest passed with 47 tests; pytest warning-as-error passed with 47 tests; deterministic HullKV benchmark equivalent=true with speedup=530.77; direct python-chess import confined to rules_oracle.py

- change_id=post6_selfplay_audit_hardening
- summary=Added exact self-play audit helpers and deterministic game trace hash verification.
- files_added=src/chess_machine_zero/selfplay/audit.py, tests/test_selfplay_audit.py
- files_updated=src/chess_machine_zero/selfplay/__init__.py, AGENTS.md, docs/project_memory.md
- scope=Post-6 hardening; audit generated self-play game integrity; no silent fallback; no smoke tests
- verification=Tests added before implementation; first new-test run failed with expected import error; self-play audit test verified at least 10000 generated plies; full pytest passed with 49 tests; pytest warning-as-error passed with 49 tests; direct python-chess import confined to rules_oracle.py; source/tests contain no fallback or smoke strings

- change_id=transformer_hosted_vm_v1
- summary=Added transformer-hosted VM autoregressive trace decoding, halt-delimited decoding, and checkpoint roundtrip.
- files_added=src/chess_machine_zero/model/hosted_vm.py, src/chess_machine_zero/model/checkpoint.py, tests/test_transformer_hosted_vm.py
- files_updated=src/chess_machine_zero/model/__init__.py, src/chess_machine_zero/train/trainer.py, src/chess_machine_zero/vm/interpreter.py, src/chess_machine_zero/vm/trace_packet.py, AGENTS.md, docs/project_memory.md
- scope=Transformer-hosted VM v1; host VM allowed only to generate training target traces; inference path uses CMZMachineTransformer only; no external tree search; no silent fallback; no smoke-test acceptance
- verification=Tests added before implementation; first new-test run failed with expected missing module; transformer-hosted decoder exact-matches trained trace continuation; decode_until_halt stops on PROGRAM_HALT; checkpoint-loaded model preserves decode; full pytest passed with 51 tests; pytest warning-as-error passed with 51 tests; direct python-chess import confined to rules_oracle.py

- change_id=multi_position_hosted_vm
- summary=Added multi-position legal-trace compiler, padded masked next-packet batches, and checkpoint registry manifest.
- files_added=src/chess_machine_zero/trace/compiler.py, tests/test_trace_compiler_multi_position.py, tests/test_multi_position_transformer_hosted_vm.py, tests/test_checkpoint_registry.py
- files_updated=src/chess_machine_zero/trace/__init__.py, src/chess_machine_zero/model/checkpoint.py, AGENTS.md, docs/project_memory.md
- scope=Production transformer-hosted VM training support; host VM generates training targets only; inference still model-only; no fallback and no smoke-test acceptance
- verification=Tests added before implementation; first new-test run failed with expected import errors; targeted tests passed with 5 tests; full pytest passed with 56 tests; pytest warning-as-error passed with 56 tests; direct python-chess import confined to rules_oracle.py

- change_id=analytic_rules_compilation_v1
- summary=Added fixed analytic rule executor and analytic machine shell separating hardcoded rules from trainable strategy.
- files_added=src/chess_machine_zero/model/analytic_rules.py, src/chess_machine_zero/model/analytic_machine.py, tests/test_analytic_rules_compiler.py
- files_updated=src/chess_machine_zero/model/__init__.py, AGENTS.md, docs/project_memory.md
- scope=Analytic rule compilation v1; legal generator is fixed and zero-trainable; strategy remains ranker-trained separately; python-chess remains oracle-only
- verification=Tests added before implementation; first new-test run failed with expected missing module; targeted tests passed with 3 tests; full pytest passed with 59 tests; pytest warning-as-error passed with 59 tests; direct python-chess import confined to rules_oracle.py

- change_id=analytic_full_rules_v2
- summary=Extended analytic fixed rules from legal generation to full move, terminal, and capped game-loop rule execution.
- files_added=tests/test_analytic_full_rules.py
- files_updated=src/chess_machine_zero/model/analytic_rules.py, src/chess_machine_zero/model/analytic_machine.py, AGENTS.md, docs/project_memory.md
- scope=Analytic full chess rules v2; rules are fixed code with zero trainable rule parameters; ranker remains the trainable strategy component
- verification=Tests added before implementation; first new-test run failed with expected missing methods; targeted tests passed with 18 tests; full pytest passed with 77 tests; pytest warning-as-error passed with 77 tests; direct python-chess import confined to rules_oracle.py; source/tests contain no fallback or smoke strings

- change_id=dashboard_rule_verification_v1
- summary=Added local interactive dashboard for observing trace-based transformer self-play and human-vs-transformer play.
- files_added=src/chess_machine_zero/dashboard/__init__.py, src/chess_machine_zero/dashboard/state.py, src/chess_machine_zero/dashboard/server.py, src/chess_machine_zero/dashboard/static/index.html, src/chess_machine_zero/dashboard/static/dashboard.css, src/chess_machine_zero/dashboard/static/dashboard.js, src/chess_machine_zero/dashboard/static/favicon.svg, tests/test_dashboard.py, test_results/dashboard_rule_verification_v1_2026-05-24.md
- files_updated=pyproject.toml, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Dashboard rule verification v1; stdlib HTTP server; local static frontend; self-play, human move entry, trace packet observability, deterministic session reset
- verification=Tests added before implementation; first new-test run failed with expected missing dashboard package; targeted dashboard tests passed with 8 tests; full pytest passed with 85 tests; pytest warning-as-error passed with 85 tests; Playwright desktop/mobile checks passed; python-chess direct import confined to rules_oracle.py; no fallback/smoke matches in src/tests; dashboard uses no external assets

- change_id=weight_compiled_rules_v1
- summary=Moved chess rule execution from an analytic rule module into frozen model-state rule weights.
- files_added=src/chess_machine_zero/model/weight_compiled_rules.py, src/chess_machine_zero/model/weight_compiled_machine.py, tests/test_weight_compiled_rules.py, test_results/weight_compiled_rules_v1_2026-05-24.md
- files_updated=src/chess_machine_zero/model/__init__.py, src/chess_machine_zero/dashboard/state.py, src/chess_machine_zero/dashboard/static/dashboard.js, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Weight-compiled rules v1; frozen nn.Parameter tensors store chess geometry/rule constants; legal generation, make-move, terminal traces, and dashboard session use WeightCompiledRulesTransformer
- verification=Tests added before implementation; first new-test run failed with expected missing module; targeted weight/dashboard tests passed with 28 tests; full pytest passed with 105 tests; pytest warning-as-error passed with 105 tests; python-chess direct import confined to rules_oracle.py; fallback/smoke terms absent from src/tests; Playwright dashboard check reports weight_compiled engine and compiled=5258

- change_id=percepta_e2e_decoder_v1
- summary=Replaced dashboard/self-play runtime rule path with a model-only Percepta trace decoder.
- files_added=src/chess_machine_zero/model/percepta_e2e_decoder.py, src/chess_machine_zero/model/percepta_selfplay.py, tests/test_percepta_e2e_decoder.py, test_results/percepta_e2e_decoder_v1_2026-05-24.md
- files_updated=src/chess_machine_zero/model/__init__.py, src/chess_machine_zero/dashboard/state.py, src/chess_machine_zero/dashboard/server.py, src/chess_machine_zero/dashboard/static/index.html, tests/test_dashboard.py, tests/test_weight_compiled_rules.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Percepta E2E decoder v1; runtime steps use PerceptaE2ETraceDecoder frozen tensors and DenseHardmax2D lookup; Python chess executor is compile/test-only, not dashboard step runtime
- verification=Tests added before implementation; first new-test run failed with expected missing module; targeted Percepta/dashboard tests passed with 14 tests; full pytest passed with 111 tests; pytest warning-as-error passed with 111 tests; dashboard state imports no analytic or weight-compiled rule executor; python-chess direct import confined to rules_oracle.py; fallback/smoke terms absent from src/tests

- change_id=percepta_parametric_rule_weights_v1
- summary=Replaced finite dashboard runtime with reusable parametric chess rule weights over arbitrary board prompts.
- files_added=src/chess_machine_zero/model/percepta_parametric_rules.py, src/chess_machine_zero/model/percepta_parametric_selfplay.py, tests/test_percepta_parametric_rules.py, test_results/percepta_parametric_rule_weights_v1_2026-05-24.md
- files_updated=src/chess_machine_zero/dashboard/state.py, src/chess_machine_zero/dashboard/static/index.html, src/chess_machine_zero/model/__init__.py, tests/test_dashboard.py, tests/test_weight_compiled_rules.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Percepta parametric rule weights v1; frozen nn.Parameter tensors store reusable chess rule geometry and predicates; dashboard/self-play accepts arbitrary board traces; no finite position lookup; human legal move validation restored through parametric rule weights
- verification=Tests added before implementation; first new-test run failed with expected missing module; targeted parametric/dashboard tests passed with 22 tests; full pytest passed with 125 tests; pytest warning-as-error passed with 125 tests; python-chess direct import confined to rules_oracle.py; fallback/smoke terms absent from src/tests; parametric runtime has no finite E2E prompt-continuation lookup imports

- change_id=percepta_frozen_attention_trace_vm_v1
- summary=Added frozen-attention token-streaming trace VM and wired dashboard runtime to host-append-only decode.
- files_added=src/chess_machine_zero/model/percepta_frozen_attention_vm.py, tests/test_percepta_frozen_attention_vm.py, test_results/percepta_frozen_attention_trace_vm_v1_2026-05-24.md
- files_updated=src/chess_machine_zero/model/percepta_parametric_selfplay.py, src/chess_machine_zero/model/__init__.py, src/chess_machine_zero/dashboard/static/dashboard.js, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Frozen attention trace VM v1; ISA and microprogram metadata are frozen parameters; decode_next emits one TracePacket per call through DenseHardmax2D cursor attention; host loop appends packets only; no finite position lookup and no MLP in the VM module
- verification=Tests added before implementation; first new-test run failed with expected missing module; targeted frozen-attention/dashboard tests passed with 13 tests; full pytest passed with 130 tests; pytest warning-as-error passed with 130 tests; python-chess direct import confined to rules_oracle.py; fallback/smoke terms absent from src/tests; frozen-attention runtime has no finite E2E prompt-continuation lookup imports and no nn.Linear

- change_id=percepta_frozen_attention_trace_vm_v2
- summary=Replaced dense cursor lookup with logarithmic 2D attention lookup and added serialized frozen attention layer graph metadata.
- files_added=test_results/percepta_frozen_attention_trace_vm_v2_2026-05-24.md
- files_updated=src/chess_machine_zero/model/percepta_frozen_attention_vm.py, src/chess_machine_zero/model/percepta_parametric_selfplay.py, tests/test_percepta_frozen_attention_vm.py, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Frozen attention trace VM v2; decode cursor lookup uses unit-circle 2D keys plus logarithmic angle search; runtime exposes O(log n), no dense scan, no MLP, serialized layer graph parameters, and host append-only token streaming
- verification=Tests updated before implementation; targeted test run failed with expected backend/missing-attribute assertions; targeted frozen-attention/dashboard tests passed with 14 tests; full pytest passed with 131 tests; pytest warning-as-error passed with 131 tests; python-chess direct import confined to rules_oracle.py; fallback/smoke terms absent from src/tests; frozen-attention runtime has no DenseHardmax2D, no nn.Linear, and no finite E2E continuation lookup imports

- change_id=percepta_frozen_attention_trace_vm_v3
- summary=Moved frozen-attention rule primitive runtime behind an explicit executable rule layer graph and blocked inherited primitive calls by test.
- files_added=src/chess_machine_zero/model/percepta_rule_layer_graph.py, test_results/percepta_frozen_attention_trace_vm_v3_2026-05-24.md
- files_updated=src/chess_machine_zero/model/percepta_frozen_attention_vm.py, src/chess_machine_zero/model/percepta_parametric_selfplay.py, tests/test_percepta_frozen_attention_vm.py, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Frozen attention trace VM v3; PerceptaFrozenAttentionRuleComputer delegates piece dispatch, ray scan, attack test, legal filter, make-move, and terminal predicates to FrozenAttentionRuleLayerGraph; dashboard exposes rule_core_execution_mode and graph primitive metadata; inherited DenseHardmax2D runtime instance removed from frozen-attention computer
- verification=Tests updated before implementation; targeted test run failed with expected missing-attribute, inherited-primitive-call, and hidden-DenseHardmax2D assertions; targeted tests passed with 15 tests; full pytest passed with 132 tests; pytest warning-as-error passed with 132 tests; direct python-chess import confined to rules_oracle.py; fallback/smoke terms absent from src/tests; frozen-attention VM and rule graph have no DenseHardmax2D, no nn.Linear, and no finite E2E continuation lookup imports

- change_id=percepta_frozen_attention_trace_vm_v4
- summary=Lowered chess rule primitives into frozen tensor-attention kernels and removed Python control-flow from primitive kernel methods.
- files_added=src/chess_machine_zero/model/percepta_attention_rule_kernels.py, test_results/percepta_frozen_attention_trace_vm_v4_2026-05-24.md
- files_updated=src/chess_machine_zero/model/percepta_rule_layer_graph.py, src/chess_machine_zero/model/percepta_frozen_attention_vm.py, src/chess_machine_zero/model/percepta_parametric_selfplay.py, tests/test_percepta_frozen_attention_vm.py, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Frozen attention trace VM v4; FrozenAttentionTensorRuleKernels implements piece dispatch, ray scan, attack test, legal filter, make-move, and terminal predicates through tensor operations over frozen parameters; graph layer now formats trace around tensor-kernel outputs
- verification=Tests updated before implementation; first targeted run failed with expected missing kernel module; targeted tests passed with 16 tests; full pytest passed with 133 tests; pytest warning-as-error passed with 133 tests; AST assertions verify no Python For/While/If/IfExp/Match/Try nodes in primitive kernel methods; direct python-chess import confined to rules_oracle.py; fallback/smoke terms absent from src/tests; frozen-attention VM, graph, and kernels have no DenseHardmax2D, no nn.Linear, and no finite E2E continuation lookup imports

- change_id=percepta_frozen_attention_trace_vm_v5
- summary=Moved core runtime to tensor trace input/output so TracePacket and BoardState are display/API boundary formats only.
- files_added=src/chess_machine_zero/model/percepta_tensor_trace_runtime.py, test_results/percepta_frozen_attention_trace_vm_v5_2026-05-24.md
- files_updated=src/chess_machine_zero/model/percepta_frozen_attention_vm.py, src/chess_machine_zero/model/percepta_parametric_selfplay.py, src/chess_machine_zero/model/__init__.py, tests/test_percepta_frozen_attention_vm.py, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Frozen attention trace VM v5; decode legal/make paths accept tensor trace packets and emit tensor trace packets; dashboard/self-play use tensor traces internally; packet and board object conversion is limited to UI/test display boundaries
- verification=Tests updated before implementation; first targeted run failed with expected missing tensor trace runtime module; targeted tests passed with 17 tests; full pytest passed with 134 tests; pytest warning-as-error passed with 134 tests; tensor core monkeypatch test blocks packet/BoardState graph runtime; direct python-chess import confined to rules_oracle.py; fallback/smoke terms absent from src/tests; tensor trace runtime has no BoardState or TracePacket references

- change_id=percepta_frozen_attention_trace_vm_v6
- summary=Routed tensor trace execution through a frozen transformer attention block stack and disabled tensor-kernel shortcut runtime.
- files_added=src/chess_machine_zero/model/percepta_attention_block_stack.py, test_results/percepta_frozen_attention_trace_vm_v6_2026-05-24.md
- files_updated=src/chess_machine_zero/model/percepta_tensor_trace_runtime.py, src/chess_machine_zero/model/percepta_frozen_attention_vm.py, src/chess_machine_zero/model/percepta_parametric_selfplay.py, src/chess_machine_zero/model/__init__.py, tests/test_percepta_frozen_attention_vm.py, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Frozen attention trace VM v6; FrozenTransformerAttentionBlockStack provides compiled block/head/residual metadata and owns legal trace, make-move trace, board transition, terminal, and legal move resolution execution; FrozenAttentionTensorTraceRuntime delegates to the stack and no longer calls FrozenAttentionTensorRuleKernels primitive shortcut methods.
- verification=Tests updated before implementation; first targeted run failed with expected missing block-stack module; targeted frozen-attention/dashboard tests passed with 18 tests; full pytest passed with 135 tests; pytest warning-as-error passed with 135 tests; monkeypatch test blocks all FrozenAttentionTensorRuleKernels primitive shortcuts; direct python-chess import confined to rules_oracle.py; fallback/smoke terms absent from src/tests; tensor trace runtime has no BoardState or TracePacket references

- change_id=percepta_frozen_attention_trace_vm_v7
- summary=Added formal ChessRuleISA microprogram compiler that emits frozen attention program weights and wired runtime metadata/execution through the compiled program.
- files_added=src/chess_machine_zero/model/percepta_rule_compiler.py, tests/test_percepta_rule_compiler.py, test_results/percepta_frozen_attention_trace_vm_v7_2026-05-24.md
- files_updated=src/chess_machine_zero/model/percepta_attention_block_stack.py, src/chess_machine_zero/model/percepta_frozen_attention_vm.py, src/chess_machine_zero/model/percepta_parametric_selfplay.py, src/chess_machine_zero/model/__init__.py, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Frozen attention trace VM v7; ChessRuleMicroprogramCompiler builds a chess_rule_isa microprogram, CompiledAttentionProgramWeights stores opcode/source/target/entrypoint/Q/K/V/residual tensors as frozen parameters, and FrozenTransformerAttentionBlockStack delegates runtime to a unified compiled executor rather than stack primitive shortcuts.
- verification=Tests added before implementation; first targeted run failed with expected missing compiler module; targeted compiler/frozen-attention/dashboard tests passed with 21 tests; full pytest passed with 138 tests; pytest warning-as-error passed with 138 tests; compiler tensors are frozen; monkeypatch test blocks stack primitive shortcuts; direct python-chess import confined to rules_oracle.py; fallback/smoke terms absent from src/tests

- change_id=percepta_frozen_attention_trace_vm_v8
- summary=Replaced the low-level executor substrate with a matrix-attention interpreter using QK hardmax select V residual writes.
- files_added=src/chess_machine_zero/model/percepta_matrix_attention_runtime.py, test_results/percepta_frozen_attention_trace_vm_v8_2026-05-24.md
- files_updated=src/chess_machine_zero/model/percepta_rule_compiler.py, src/chess_machine_zero/model/percepta_attention_block_stack.py, src/chess_machine_zero/model/percepta_frozen_attention_vm.py, src/chess_machine_zero/model/percepta_parametric_selfplay.py, src/chess_machine_zero/model/__init__.py, tests/test_percepta_rule_compiler.py, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Frozen attention trace VM v8; FrozenMatrixAttentionInterpreter executes compiled rule-program reads and writes through `QK^T -> mask -> hardmax/select -> V -> residual write`; FrozenTransformerAttentionBlockStack no longer owns `compiled_executor`; dashboard exposes matrix-interpreter runtime metadata.
- verification=Tests updated before implementation; first targeted run failed with expected missing matrix runtime module; targeted compiler/frozen-attention/dashboard tests passed with 22 tests; full pytest passed with 139 tests; pytest warning-as-error passed with 139 tests; monkeypatch test blocks legacy compiled executor methods; direct python-chess import confined to rules_oracle.py; fallback/smoke terms absent from src/tests

- change_id=percepta_two_transformer_dashboard_v9
- summary=Updated dashboard self-play to run separate white and black Percepta transformer rule computers and display side-specific emitted token streams.
- files_added=test_results/percepta_two_transformer_dashboard_v9_2026-05-24.md
- files_updated=src/chess_machine_zero/model/percepta_parametric_selfplay.py, src/chess_machine_zero/dashboard/static/index.html, src/chess_machine_zero/dashboard/static/dashboard.js, tests/test_dashboard.py, AGENTS.md, README.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Two-transformer dashboard self-play; actor ids are transformer_white and transformer_black; each move summary stores emitted trace tokens and trace membership verification; dashboard shows active transformer, last emitter, verified flag, and token arrays.
- verification=Tests added before implementation and failed on missing two-transformer fields; dashboard targeted tests passed with 9 tests; compiler/frozen-attention/dashboard targeted tests passed with 23 tests; full pytest passed with 140 tests; pytest warning-as-error passed with 140 tests; API self-play check verified 2 plies, white_tokens=117, black_tokens=117, illegal_commits=0; Playwright desktop/mobile check passed; direct python-chess import confined to rules_oracle.py; fallback/smoke terms absent from src/tests

- change_id=percepta_two_transformer_dashboard_v9_1
- summary=Made dashboard self-play visibly auto-run and serialized frontend step requests.
- files_updated=src/chess_machine_zero/dashboard/static/dashboard.js, src/chess_machine_zero/dashboard/static/index.html, tests/test_dashboard.py, AGENTS.md, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md, test_results/percepta_two_transformer_dashboard_v9_2026-05-24.md
- scope=Frontend self-play visibility; selfplay mode auto-starts on page load; Play uses sequential `setTimeout` loop instead of overlapping `setInterval`; status line reports `transformer_* computing move`.
- verification=Tests added before implementation and failed on missing `state.busy`; dashboard targeted tests passed with 10 tests; dashboard warning-as-error tests passed with 10 tests; Playwright check verified `/static/dashboard.js?v=9.1`, automatic Pause state, and ply advance 0->1 with transformer token log.

## 2026-05-25

- change_id=github_public_publish_v1
- summary=Initialized local git repository, created public GitHub repository, and pushed the ChessMachineZero codebase.
- files_added=test_results/github_public_publish_v1_2026-05-25.md
- files_updated=.gitignore, docs/project_memory.md, docs/change_history.md, docs/prompt_history.md
- scope=Public GitHub repository at https://github.com/TryDotAtwo/ChessMachineZero; initial branch main; initial commit cbdee31; generated screenshots/cache/bytecode artifacts excluded.
- verification=GitHub CLI authenticated as TryDotAtwo; `gh repo create ... --public --source . --remote origin --push` succeeded; origin/main tracks local main; public repository URL returned by GitHub CLI.
