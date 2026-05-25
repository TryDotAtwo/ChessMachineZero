# Transformer-Hosted VM v1 Test Result

- date=2026-05-24
- task_id=transformer_hosted_vm_v1
- development_mode=test_first
- quality_policy=no_fallbacks; no_smoke_tests

## Expected Failure Before Implementation

- command=`python -m pytest -p no:cacheprovider tests/test_transformer_hosted_vm.py`
- result=expected_fail
- summary=missing hosted VM / checkpoint modules before implementation

## Targeted Tests After Implementation

- command=`python -m pytest -p no:cacheprovider tests/test_transformer_hosted_vm.py`
- result=pass
- summary=2 passed in 12.55s
- verified=TransformerHostedVM has no `vm` or `host_vm` attribute
- verified=decode_until_halt exact-matches expected legal-trace continuation including PROGRAM_HALT
- verified=decoded trace passes oracle verifier
- verified=checkpoint-loaded model preserves decode

## Full Suite

- command=`python -m pytest -p no:cacheprovider`
- result=pass
- summary=51 passed in 62.68s

## Warning Check

- command=`python -m pytest -p no:cacheprovider -W error`
- result=pass
- summary=51 passed in 61.99s

## Boundary Checks

- python_chess_import=src/chess_machine_zero/chess/rules_oracle.py only
- fallback_smoke_scan=no matches in src/tests
- non_ascii_python_scan=no matches
- pytest_cacheprovider=disabled due previous disk-full cache write error
