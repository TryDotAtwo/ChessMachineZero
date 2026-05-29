# Single Shared Transformer Runtime 2026-05-26

## Scope

- change_id=single_shared_transformer_runtime_v12
- user_request=Use one shared model for both white and black instead of inferring two model copies.
- behavior_change=PerceptaParametricSelfPlaySession owns one PerceptaFrozenAttentionRuleComputer instance and uses board side-to-move state for both sides.
- token_observability=White and black token stream buckets remain in dashboard as side-indexed display history, not as separate model instances.

## TDD Evidence

- initial_targeted_tests=`python -m pytest -p no:cacheprovider tests\test_dashboard.py::test_dashboard_snapshot_exposes_transformer_rule_state_from_trace tests\test_dashboard.py::test_dashboard_single_shared_transformer_plays_both_sides_and_emits_verified_tokens -q` => failed because runtime still exposed two_transformer_selfplay and transformer_white/transformer_black actors.
- post_change_targeted_tests=same command => 2 passed.

## Verification

- dashboard_tests=`python -m pytest -p no:cacheprovider tests\test_dashboard.py -q` => 12 passed.
- targeted_percepta_dashboard_tests=`python -m pytest -p no:cacheprovider tests\test_percepta_rule_compiler.py tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py -q` => 28 passed.
- full_pytest=`python -m pytest -p no:cacheprovider -q` => 145 passed.
- dashboard_warning_check=`python -m pytest -p no:cacheprovider -W error tests\test_dashboard.py -q` => 12 passed.

## Runtime Probe

- probe_log=test_results/single_shared_transformer_perf_2026-05-26.txt
- two_ply_seconds=0.337271
- first_actor=shared_transformer
- first_side=w
- second_actor=shared_transformer
- second_side=b
- mode=single_shared_transformer_selfplay
- shared_model_instance_count=1
- white_trace_packets=117
- black_trace_packets=117
- illegal_commit_count=0

## Boundary Scan

- search=`rg -n "two_transformer_selfplay|black_rules|white_rules|_rules_for_side|_active_rules|_transformer_id_for_side|transformer_white|transformer_black" src/chess_machine_zero/model/percepta_parametric_selfplay.py tests/test_dashboard.py`
- result=no matches.

## Files

- updated=src/chess_machine_zero/model/percepta_parametric_selfplay.py
- updated=tests/test_dashboard.py
- updated=README.md
