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
