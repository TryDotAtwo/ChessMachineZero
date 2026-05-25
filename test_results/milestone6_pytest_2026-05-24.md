# Milestone 6 Test Result

- date=2026-05-24
- task_id=milestone6
- development_mode=test_first
- quality_policy=no_fallbacks; no_smoke_tests

## Expected Failure Before Implementation

- command=`python -m pytest tests/test_hullkv_equivalence.py`
- result=expected_fail
- summary=1 import error for missing hullkv package

## New Tests After Implementation

- command=`python -m pytest tests/test_hullkv_equivalence.py`
- result=pass
- summary=4 passed

## Full Suite

- command=`python -m pytest`
- result=pass
- summary=47 passed in 13.80s

## Warning Check

- command=`python -m pytest -W error`
- result=pass
- summary=47 passed in 13.70s

## Boundary Check

- command=`rg "^\s*(import chess|from chess(\s|\.))" src tests`
- result=pass
- summary=direct python-chess import found only in `src/chess_machine_zero/chess/rules_oracle.py`

## No Fallback/Smoke String Check

- command=`rg -n "fallback|smoke" src tests`
- result=pass
- summary=no matches in source or tests

## HullKV Benchmark

- key_count=20004
- query_count=1000
- dense_seconds=3.079185
- hull_seconds=0.005801
- equivalent=true
- speedup=530.77
