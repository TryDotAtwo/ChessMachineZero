# Percepta Frozen Attention Trace VM V1 Test Results

## Scope

- task_id=percepta_frozen_attention_trace_vm_v1
- runtime_model=PerceptaFrozenAttentionRuleComputer
- runtime_session=PerceptaParametricSelfPlaySession
- rule_execution_mode=percepta_frozen_attention_trace_vm
- attention_backend=dense_hardmax_2d
- host_append_only=true
- token_streaming=true
- uses_mlp=false
- position_lookup=false
- finite_prompt_lookup=false
- compiled_prompt_count=0
- compiled_isa_instruction_count=14
- compiled_microprogram_step_count=18
- compiled_parameter_count=13610
- python_rule_executor_runtime=false
- strategy_training=false
- strategy_module=none

## TDD Record

- initial_command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- initial_result=failed as expected
- initial_failure=`ModuleNotFoundError: No module named 'chess_machine_zero.model.percepta_frozen_attention_vm'`

## Pytest Verification

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- result=`13 passed in 54.17s`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider`
- result=`130 passed in 140.92s (0:02:20)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -W error`
- result=`130 passed in 136.85s (0:02:16)`

## Boundary Scans

- command=`rg "^import chess$|^from chess(\.|\s)" src tests`
- result=`src\chess_machine_zero\chess\rules_oracle.py:import chess`

- command=`rg -i "fallback|smoke" src tests`
- result=no matches

- command=`rg "PerceptaE2ETraceDecoder|prompt_fingerprints|continuation_tokens|rules_oracle|ChessMachineVM|AnalyticRuleCompiler|nn\.Linear" src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_parametric_selfplay.py src\chess_machine_zero\dashboard\state.py`
- result=no matches

## Runtime Verification

- server_url=http://127.0.0.1:8768
- server_pid=22300
- API_snapshot_engine=PerceptaFrozenAttentionRuleComputer
- API_snapshot_rule_execution_mode=percepta_frozen_attention_trace_vm
- API_snapshot_attention_backend=dense_hardmax_2d
- API_snapshot_host_append_only=true
- API_snapshot_token_streaming=true
- API_snapshot_uses_mlp=false
- API_snapshot_python_rule_executor_runtime=false
- API_snapshot_compiled_prompt_count=0
- API_snapshot_position_lookup=false
- API_snapshot_finite_prompt_lookup=false
- API_snapshot_compiled_parameter_count=13610
- API_snapshot_legal_count=20
- API_snapshot_trace_ops=WRITE_SQ, WRITE_REG, WRITE_CASTLE, WRITE_EP, WRITE_CLOCK, CANDIDATE, LEGAL_SET, PROGRAM_HALT

## Known Limit

- v1_scope=trace streaming and cursor selection run as frozen attention; chess rule primitives remain Python control flow over frozen tensors
- next_required=serialize each rule predicate into explicit per-layer frozen attention graph
