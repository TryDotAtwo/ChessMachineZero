# Percepta Parametric Rule Weights V1 Test Results

## Scope

- task_id=percepta_parametric_rule_weights_v1
- runtime_model=PerceptaParametricRulesTransformer
- runtime_session=PerceptaParametricSelfPlaySession
- rule_execution_mode=percepta_parametric_rule_weights
- attention_backend=dense_hardmax_2d
- parametric_rule_weights=true
- position_lookup=false
- finite_prompt_lookup=false
- compiled_prompt_count=0
- compiled_parameter_count=5386
- python_rule_executor_runtime=false
- strategy_training=false
- strategy_module=none
- dashboard_human_runtime=true

## TDD Record

- initial_command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_parametric_rules.py tests\test_dashboard.py`
- initial_result=failed as expected
- initial_failure=`ModuleNotFoundError: No module named 'chess_machine_zero.model.percepta_parametric_rules'`

## Pytest Verification

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_parametric_rules.py tests\test_dashboard.py`
- result=`22 passed in 3.81s`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider`
- result=`125 passed in 123.82s (0:02:03)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -W error`
- result=`125 passed in 86.91s (0:01:26)`

## Boundary Scans

- command=`rg "^import chess$|^from chess(\.|\s)" src tests`
- result=`src\chess_machine_zero\chess\rules_oracle.py:import chess`

- command=`rg -i "fallback|smoke" src tests`
- result=no matches

- command=`rg "PerceptaE2ETraceDecoder|PerceptaSelfPlaySession|prompt_fingerprints|continuation_tokens" src\chess_machine_zero\dashboard src\chess_machine_zero\model\percepta_parametric_rules.py src\chess_machine_zero\model\percepta_parametric_selfplay.py`
- result=no matches

- command=`rg "rules_oracle|ChessMachineVM|AnalyticRuleCompiler|AnalyticRulesTransformer" src\chess_machine_zero\model\percepta_parametric_rules.py src\chess_machine_zero\model\percepta_parametric_selfplay.py src\chess_machine_zero\dashboard\state.py`
- result=no matches

## Runtime Verification

- server_url=http://127.0.0.1:8768
- server_pid=42716
- API_snapshot_engine=PerceptaParametricRulesTransformer
- API_snapshot_rule_execution_mode=percepta_parametric_rule_weights
- API_snapshot_attention_backend=dense_hardmax_2d
- API_snapshot_python_rule_executor_runtime=false
- API_snapshot_compiled_prompt_count=0
- API_snapshot_position_lookup=false
- API_snapshot_finite_prompt_lookup=false
- API_snapshot_compiled_parameter_count=5386
- API_snapshot_legal_count=20
- API_human_move=e2e4
- API_human_move_result_fen=`rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1`
- API_snapshot_trace_ops=WRITE_SQ, WRITE_REG, WRITE_CASTLE, WRITE_EP, WRITE_CLOCK, CANDIDATE, LEGAL_SET, COMMIT_MOVE, TERMINAL_SET, PROGRAM_HALT

## Known Limit

- v1_scope=rules are stored as reusable frozen tensors and executed through model-side tensor/table operations
- remaining_gap=full constructive transformer block decomposition for every predicate into explicit per-layer attention heads is not yet represented as a serialized layer graph
