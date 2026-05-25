# Weight Compiled Rules V1 Test Results

## Scope

- task_id=weight_compiled_rules_v1
- source_reference=https://www.percepta.ai/blog/can-llms-be-computers
- goal=move chess rule execution into frozen nn.Parameter model weights
- rule_module=WeightCompiledRulesTransformer
- rule_execution_mode=weight_compiled
- compiled_rule_parameters=5258
- trainable_rule_parameters=0
- strategy_module=CMZMoveRanker
- forbidden_scope=external tree search, human-game data, engine labels, tablebase labels, handcrafted evaluation

## TDD Record

- initial_command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_weight_compiled_rules.py tests\test_dashboard.py`
- initial_result=failed as expected
- initial_failure=`ModuleNotFoundError: No module named 'chess_machine_zero.model.weight_compiled_rules'`
- tests_added=source-boundary test, legal generation oracle match, make-move board reconstruction, terminal oracle match, threefold rule, capped game, dashboard weight-rule path

## Pytest Verification

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_weight_compiled_rules.py tests\test_dashboard.py`
- result=`28 passed in 9.78s`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider`
- result=`105 passed in 108.97s (0:01:48)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -W error`
- result=`105 passed in 177.43s (0:02:57)`

## Boundary Scans

- command=`rg "^import chess$|^from chess(\.|\s)" src tests`
- result=`src\chess_machine_zero\chess\rules_oracle.py:import chess`

- command=`rg -i "fallback|smoke" src tests`
- result=no matches

## Browser Verification

- server_url=http://127.0.0.1:8768
- server_pid=57796
- check=dashboard engine line
- result=`WeightCompiledRulesTransformer mode=weight_compiled rule_params=0 compiled=5258`
- check=board render
- result=squares=64, legal_count=20, packet_count=108
- check=selfplay step
- result=ply=1, side_to_move=b, history=`transformer w f2f4`, illegal_count=0
- check=mobile viewport
- viewport=390x840
- result=horizontal_overflow=false, squares=64
