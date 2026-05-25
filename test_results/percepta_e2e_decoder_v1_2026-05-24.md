# Percepta E2E Decoder V1 Test Results

## Scope

- task_id=percepta_e2e_decoder_v1
- runtime_model=PerceptaE2ETraceDecoder
- runtime_session=PerceptaSelfPlaySession
- attention_backend=DenseHardmax2D
- runtime_rule_executor=false
- strategy_training=false
- strategy_module=none
- dashboard_human_runtime=false
- default_compiled_prompt_count=64
- default_compiled_parameter_count=49930

## TDD Record

- initial_command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_e2e_decoder.py`
- initial_result=failed as expected
- initial_failure=`ModuleNotFoundError: No module named 'chess_machine_zero.model.percepta_e2e_decoder'`
- dashboard_tests_rewritten=expected `PerceptaE2ETraceDecoder`, `percepta_e2e_trace_decoder`, `python_rule_executor_runtime=false`, human move disabled

## Pytest Verification

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_e2e_decoder.py tests\test_dashboard.py`
- result=`14 passed in 49.24s`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider`
- result=`111 passed in 214.86s (0:03:34)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -W error`
- result=`111 passed in 214.45s (0:03:34)`

## Boundary Scans

- command=`rg "^import chess$|^from chess(\.|\s)" src tests`
- result=`src\chess_machine_zero\chess\rules_oracle.py:import chess`

- command=`rg -i "fallback|smoke" src tests`
- result=no matches

- command=`rg "WeightCompiledRuleCompiler|CMZWeightCompiledMachine|AnalyticRuleCompiler|CMZAnalyticMachine" src\chess_machine_zero\dashboard\state.py src\chess_machine_zero\dashboard\server.py src\chess_machine_zero\dashboard\static`
- result=no matches

- command=`rg "SAMPLE_SET|CMZMoveRanker|train_ranker|policy_gradient" src\chess_machine_zero\dashboard src\chess_machine_zero\model\percepta_selfplay.py tests\test_percepta_e2e_decoder.py tests\test_dashboard.py`
- result=only assertions that SAMPLE_SET count is zero in `tests\test_dashboard.py`

## Runtime Verification

- server_url=http://127.0.0.1:8768
- server_pid=16820
- API_snapshot_engine=PerceptaE2ETraceDecoder
- API_snapshot_rule_execution_mode=percepta_e2e_trace_decoder
- API_snapshot_python_rule_executor_runtime=false
- API_snapshot_compiled_prompt_count=64
- API_snapshot_compiled_parameter_count=49930
- API_snapshot_legal_count=20
- API_snapshot_trace_ops=WRITE_SQ, WRITE_REG, WRITE_CASTLE, WRITE_EP, WRITE_CLOCK, CANDIDATE, LEGAL_SET, COMMIT_MOVE, TERMINAL_SET, PROGRAM_HALT

## Known Limit

- v1_scope=finite compiled prompt trajectory, not arbitrary-position universal chess interpreter
- next_required=constructive transformer-weight compiler or large trace-training pipeline for arbitrary prompts
