# Analytic Rules Compilation v1 Test Result

- date=2026-05-24
- task_id=analytic_rules_compilation_v1
- development_mode=test_first
- quality_policy=no_fallbacks; no_smoke_tests

## Expected Failure Before Implementation

- command=`python -m pytest -p no:cacheprovider tests/test_analytic_rules_compiler.py`
- result=expected_fail
- summary=missing analytic machine / analytic rules modules before implementation

## Targeted Tests After Implementation

- command=`python -m pytest -p no:cacheprovider tests/test_analytic_rules_compiler.py`
- result=pass
- summary=3 passed in 1.99s
- verified=AnalyticRulesTransformer has zero trainable parameters
- verified=analytic legal trace from prompt trace equals deterministic reference trace
- verified=CMZAnalyticMachine keeps rules fixed and ranker trainable

## Full Suite

- command=`python -m pytest -p no:cacheprovider`
- result=pass
- summary=59 passed in 94.16s

## Warning Check

- command=`python -m pytest -p no:cacheprovider -W error`
- result=pass
- summary=59 passed in 100.94s

## Boundary Checks

- python_chess_import=src/chess_machine_zero/chess/rules_oracle.py only
- fallback_smoke_scan=no matches in src/tests
- non_ascii_python_scan=no matches
- pytest_cacheprovider=disabled due previous disk-full cache write error
