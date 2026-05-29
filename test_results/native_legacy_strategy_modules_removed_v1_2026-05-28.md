# Native Legacy Strategy Modules Removed V1

```text
date=2026-05-28
slice=native_legacy_strategy_modules_removed_v1
goal=remove legacy_strategy_modules from truthful Percepta-like native policy-only contract
tdd_expected_fail_log=test_results/native_container_logs/cargo_test_legacy_strategy_modules_expected_fail_2026-05-28.txt
targeted_pass_log=test_results/native_container_logs/cargo_test_legacy_strategy_modules_targeted_2026-05-28.txt
```

## Scope

```text
removed_modules=ranker.py,baseline.py,analytic_machine.py,weight_compiled_machine.py,selfplay/actor.py,train/losses.py,vm/lookahead.py,vm/decision_program.py
removed_tests=test_select_move_trace.py,test_selfplay_no_illegal_moves.py,test_selfplay_audit.py,test_selfplay_training_step.py,test_trace_lookahead.py
policy_only_decoder=true
hardcoded_MCTS=false
hardcoded_beam=false
hardcoded_negamax=false
handcrafted_eval=false
required_value_baseline=false
```

## Contract Delta

```text
before_remaining_non_attention_paths=legacy_strategy_modules,python_attention_runtime_not_cuda_cutlass,tests_assert_metadata_not_semantics
after_remaining_non_attention_paths=python_attention_runtime_not_cuda_cutlass,tests_assert_metadata_not_semantics
strict_qk_layer_split_remaining=python_attention_runtime,semantic_tests
legacy_strategy_modules=false
```

## Verification

```text
expected_fail=passed_as_failure; reason=graph_still_declared_legacy_strategy_modules_and_ranker_py_existed
targeted_native=cargo test -p cmz-engine-sys native_frozen_rule_graph -- --nocapture
targeted_native_result=passed; tests=2
targeted_pytest=passed; tests=40
cargo_fmt_check=passed
cargo_clippy_workspace_all_targets_D_warnings=passed
cargo_test_workspace=passed; tests=49
pytest_q=passed; tests=137
pytest_werror_q=passed; tests=137
```

## Logs

```text
expected_fail=test_results/native_container_logs/cargo_test_legacy_strategy_modules_expected_fail_2026-05-28.txt
targeted_native=test_results/native_container_logs/cargo_test_legacy_strategy_modules_targeted_2026-05-28.txt
targeted_pytest=test_results/legacy_strategy_modules_targeted_pytest_2026-05-28.txt
fmt_apply=test_results/native_container_logs/cargo_fmt_apply_legacy_strategy_modules_2026-05-28.txt
fmt_check=test_results/native_container_logs/cargo_fmt_legacy_strategy_modules_2026-05-28.txt
clippy=test_results/native_container_logs/cargo_clippy_legacy_strategy_modules_2026-05-28.txt
cargo_test=test_results/native_container_logs/cargo_test_legacy_strategy_modules_2026-05-28.txt
pytest=test_results/legacy_strategy_modules_pytest_2026-05-28.txt
pytest_werror=test_results/legacy_strategy_modules_pytest_werror_2026-05-28.txt
```
