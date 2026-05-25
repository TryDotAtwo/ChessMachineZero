# ChessMachineZero

ChessMachineZero is a trace-based chess machine prototype following
`docs/chess_machine_zero_percepta_architecture.md`.

Current scope:

- Fixed MovePacket and TracePacket codecs.
- `python-chess` development oracle confined to `src/chess_machine_zero/chess/rules_oracle.py`.
- Deterministic VM traces for legal move enumeration, move application, terminal checks, and board writes.
- Transformer-hosted trace decoder experiments and analytic fixed-rule executor.
- Frozen-weight rule executor for legal generation, make-move, terminal rules, repetition, fifty-move, and capped games.
- Local dashboard for observing trace-based self-play or playing human-vs-transformer.

Current dashboard rule path:

- `PerceptaFrozenAttentionRuleComputer`
- `rule_execution_mode=percepta_frozen_attention_trace_vm`
- `attention_backend=logarithmic_2d_attention`
- `lookup_complexity=O(log n)`
- `rule_core_execution_mode=executable_frozen_attention_layer_graph`
- `primitive_kernel_execution_mode=pure_frozen_attention_tensor_layers`
- `core_trace_runtime=tensor_trace_in_frozen_attention_blocks_tensor_trace_out`
- `core_rule_compute_backend=frozen_transformer_attention_block_stack`
- `tensor_kernel_shortcut_runtime=false`
- `compiled_attention_block_stack=true`
- `compiled_attention_block_count=6`
- `compiled_attention_head_count=11`
- `residual_trace_write_count=3`
- `percepta_compiler_pipeline=chess_isa_microprogram_to_frozen_attention_weights`
- `rule_compiler_backend=rule_microprogram_to_frozen_attention_weights`
- `rule_microprogram_source=chess_rule_isa`
- `rule_microprogram_instruction_count=21`
- `compiled_rule_program_weight_count=408`
- `unified_rule_executor_runtime=true`
- `handwritten_stack_primitive_runtime=false`
- `matrix_attention_interpreter_runtime=true`
- `executor_substrate=matrix_attention_interpreter`
- `attention_step_operator=QK^T_mask_hardmax_select_V_residual_write`
- `pytorch_domain_shortcut_runtime=false`
- `python_host_boundary_role=display_only`
- `tensor_trace_core_runtime=true`
- `tracepacket_core_runtime=false`
- `python_rule_primitive_runtime=false`
- `python_control_flow_rule_primitives=false`
- `compiled_rule_primitives=[PIECE_DISPATCH,RAY_SCAN,ATTACK_TEST,LEGAL_FILTER,MAKE_MOVE,TERMINAL_PREDICATES]`
- `tensor_kernel_count=6`
- `compiled_layer_graph_serialized=true`
- `parametric_rule_weights=true`
- `host_append_only=true`
- `token_streaming=true`
- `uses_mlp=false`
- `position_lookup=false`
- `compiled_prompt_count=0`
- `python_rule_executor_runtime=false`
- `strategy_module=none`
- `strategy_training=false`
- `two_transformer_selfplay=true`
- `transformer_white=PerceptaFrozenAttentionRuleComputer`
- `transformer_black=PerceptaFrozenAttentionRuleComputer`
- `trace_legal_verification=selected_move_in_trace_LEGAL_SET`
- `dashboard_token_log=emitted_trace_packet_tokens_per_transformer`

Run tests:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider -W error
```

Run dashboard:

```powershell
$env:PYTHONPATH='src'
python -m chess_machine_zero.dashboard.server --host 127.0.0.1 --port 8768
```

Dashboard URL: `http://127.0.0.1:8768`
