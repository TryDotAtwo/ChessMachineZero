# Multi-Position Transformer-Hosted VM Test Result

- date=2026-05-24
- task_id=multi_position_hosted_vm
- development_mode=test_first
- quality_policy=no_fallbacks; no_smoke_tests

## Expected Failure Before Implementation

- command=`python -m pytest -p no:cacheprovider tests/test_trace_compiler_multi_position.py tests/test_multi_position_transformer_hosted_vm.py tests/test_checkpoint_registry.py`
- result=expected_fail
- summary=3 import errors for missing trace compiler and checkpoint registry

## Targeted Tests After Implementation

- command=`python -m pytest -p no:cacheprovider tests/test_trace_compiler_multi_position.py tests/test_multi_position_transformer_hosted_vm.py tests/test_checkpoint_registry.py`
- result=pass
- summary=5 passed in 17.09s
- verified=multi-position examples preserve prompt-prefix and PROGRAM_HALT continuation
- verified=batch loss masks select continuation targets only
- verified=single TransformerHostedVM exact-decodes multiple board prompts
- verified=checkpoint registry manifest tracks ordered versions and latest pointer

## Full Suite

- command=`python -m pytest -p no:cacheprovider`
- result=pass
- summary=56 passed in 104.59s

## Warning Check

- command=`python -m pytest -p no:cacheprovider -W error`
- result=pass
- summary=56 passed in 87.30s

## Boundary Checks

- python_chess_import=src/chess_machine_zero/chess/rules_oracle.py only
- fallback_smoke_scan=no matches in src/tests
- non_ascii_python_scan=no matches
- pytest_cacheprovider=disabled due previous disk-full cache write error
