# Post-6 Hardening Test Result

- date=2026-05-24
- task_id=post6_hardening
- development_mode=test_first
- quality_policy=no_fallbacks; no_smoke_tests

## Expected Failure Before Implementation

- command=`python -m pytest tests/test_selfplay_audit.py`
- result=expected_fail
- summary=1 import error for missing selfplay audit module

## New Tests After Implementation

- command=`python -m pytest tests/test_selfplay_audit.py`
- result=pass
- summary=2 passed in 64.26s
- verified=10000_generated_selfplay_plies_without_illegal_commits
- verified=same_seed_same_ranker_same_game_trace_hash

## Full Suite

- command=`python -m pytest`
- result=pass
- summary=49 passed in 63.39s

## Warning Check

- command=`python -m pytest -W error`
- result=pass
- summary=49 passed in 64.03s

## Boundary Checks

- python_chess_import=src/chess_machine_zero/chess/rules_oracle.py only
- fallback_smoke_scan=no matches in src/tests
- non_ascii_python_scan=no matches outside filesystem path context
