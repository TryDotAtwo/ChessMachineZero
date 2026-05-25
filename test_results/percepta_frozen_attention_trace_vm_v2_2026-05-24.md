# Percepta Frozen Attention Trace VM V2 Test Results

## Scope

- task_id=percepta_frozen_attention_trace_vm_v2
- runtime_model=PerceptaFrozenAttentionRuleComputer
- runtime_session=PerceptaParametricSelfPlaySession
- rule_execution_mode=percepta_frozen_attention_trace_vm
- attention_backend=logarithmic_2d_attention
- lookup_complexity=O(log n)
- host_append_only=true
- token_streaming=true
- uses_mlp=false
- uses_dense_scan=false
- compiled_layer_graph_serialized=true
- position_lookup=false
- finite_prompt_lookup=false
- compiled_prompt_count=0
- compiled_isa_instruction_count=14
- compiled_microprogram_step_count=18
- compiled_attention_layer_count=26
- compiled_parameter_count=13740
- python_rule_executor_runtime=false
- strategy_training=false
- strategy_module=none

## TDD Record

- initial_command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- initial_result=failed as expected
- initial_failures=backend still `dense_hardmax_2d`; missing `attention_select_decode_step`; missing `max_lookup_steps`

## Pytest Verification

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- result=`14 passed in 163.79s (0:02:43)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider`
- result=`131 passed in 292.37s (0:04:52)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -W error`
- result=`131 passed in 355.52s (0:05:55)`

## Boundary Scans

- command=`rg "^import chess$|^from chess(\.|\s)" src tests`
- result=`src\chess_machine_zero\chess\rules_oracle.py:import chess`

- command=`rg -i "fallback|smoke" src tests`
- result=no matches

- command=`rg "DenseHardmax2D|nn\.Linear|PerceptaE2ETraceDecoder|prompt_fingerprints|continuation_tokens|rules_oracle|ChessMachineVM|AnalyticRuleCompiler" src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_parametric_selfplay.py src\chess_machine_zero\dashboard\state.py`
- result=no matches

## Runtime Verification

- server_url=http://127.0.0.1:8768
- server_pid=59132
- API_snapshot_engine=PerceptaFrozenAttentionRuleComputer
- API_snapshot_rule_execution_mode=percepta_frozen_attention_trace_vm
- API_snapshot_attention_backend=logarithmic_2d_attention
- API_snapshot_lookup_complexity=O(log n)
- API_snapshot_host_append_only=true
- API_snapshot_token_streaming=true
- API_snapshot_uses_mlp=false
- API_snapshot_uses_dense_scan=false
- API_snapshot_compiled_layer_graph_serialized=true
- API_snapshot_python_rule_executor_runtime=false
- API_snapshot_compiled_prompt_count=0
- API_snapshot_position_lookup=false
- API_snapshot_finite_prompt_lookup=false
- API_snapshot_compiled_parameter_count=13740
- API_snapshot_compiled_attention_layer_count=26
- API_snapshot_max_lookup_steps=16
- API_snapshot_legal_count=20

## Known Limit

- v2_done=trace cursor lookup is logarithmic frozen 2D attention; host append-only token streaming is active; layer graph metadata is serialized in frozen weights
- next_required=replace chess rule primitive Python loops with executable frozen-attention layer graph operations for piece dispatch, ray scan, attack test, legal filter, make-move, and terminal predicates
