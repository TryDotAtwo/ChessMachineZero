# Dashboard Rule Verification V1 Test Results

## Scope

- task_id=dashboard_rule_verification_v1
- implementation=local dashboard for trace-based self-play and human-vs-transformer play
- rule_path=AnalyticRulesTransformer emits legal move traces and make-move traces
- strategy_path=CMZMoveRanker chooses only legal moves emitted by trace records
- forbidden_scope=external tree search, human-game data, engine labels, tablebase labels, handcrafted evaluation, CDN/external assets

## TDD Record

- initial_command=`python -m pytest -p no:cacheprovider tests\test_dashboard.py`
- initial_result=failed as expected
- initial_failure=`ModuleNotFoundError: No module named 'chess_machine_zero.dashboard'`
- regression_added=deterministic reset SAMPLE_SET seed replay
- regression_failure=`assert 1584337146 == 446883271`
- regression_fix=reset dashboard RNG to configured deterministic seed on session reset

## Pytest Verification

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_dashboard.py`
- result=`8 passed in 2.23s`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider`
- result=`85 passed in 77.16s (0:01:17)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -W error`
- result=`85 passed in 91.97s (0:01:31)`

## Boundary Scans

- command=`rg "^import chess$|^from chess(\.|\s)" src tests`
- result=`src\chess_machine_zero\chess\rules_oracle.py:import chess`

- command=`rg -i "fallback|smoke" src tests`
- result=no matches

- command=`rg "https?://|cdn" src\chess_machine_zero\dashboard\static tests\test_dashboard.py`
- result=only negative assertions in `tests\test_dashboard.py`

## Browser Verification

- server_command=`$env:PYTHONPATH='src'; python -m chess_machine_zero.dashboard.server --host 127.0.0.1 --port 8768`
- server_url=http://127.0.0.1:8768
- server_pid=58264
- Playwright desktop check=loaded page; title `ChessMachineZero Dashboard`; 64 board squares; legal_count=20; packet_count=108; no console errors after favicon route fix
- Playwright selfplay check=Step button advanced ply to 1; side_to_move=b; history contained transformer legal move; illegal_count=0
- Playwright human check=mode human_white; e2 then e4 produced human move e2e4 plus transformer reply; ply=2; side_to_move=w; illegal_count=0
- Playwright mobile check=390x840 viewport; 64 board squares; horizontal_overflow=false; board_width=319; viewport_width=375
- screenshots=cmz-dashboard-desktop.png, cmz-dashboard-mobile.png
