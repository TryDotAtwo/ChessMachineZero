# Native Resolve Move QK V1 - 2026-05-28

## Scope

```text
change_id=native_resolve_move_qk_v1
source_gap=resolve_move_scan
goal=replace requested-move serial scan with frozen 2D QK hardmax legal-set lookup
old_kernel=cmz_resolve_move_attention_kernel
new_kernel=cmz_resolve_move_qk_hardmax_legal_set_attention_kernel
```

## Contract

```text
resolve_move_backend=cuda_resolve_move_qk_hardmax_legal_set_attention
resolve_move_cpp_loop_remaining=false
resolve_move_scan=false
resolve_move_qk_hardmax_2d=true
full_frozen_attention_only=false
target_full_frozen_attention_only=true
```

## TDD Evidence

```text
expected_fail_log=test_results/native_container_logs/cargo_test_resolve_move_qk_expected_fail_2026-05-28.txt
expected_fail_result=failed_before_contract_and_kernel_edit
expected_fail_reason=graph still reported resolve_move_backend=cuda_resolve_move_attention
```

## Targeted Verification

```text
contract_log=test_results/native_container_logs/cargo_test_resolve_move_qk_contract_2026-05-28.txt
contract_result=passed
source_audit_log=test_results/native_container_logs/cargo_test_resolve_move_qk_source_audit_2026-05-28.txt
source_audit_result=passed
make_move_behavior_log=test_results/native_container_logs/cargo_test_resolve_move_qk_make_move_2026-05-28.txt
make_move_behavior_result=passed
```

## Full Verification

```text
cargo_fmt_apply_log=test_results/native_container_logs/cargo_fmt_apply_resolve_move_qk_2026-05-28.txt
cargo_fmt_apply_result=passed
cargo_fmt_check_log=test_results/native_container_logs/cargo_fmt_resolve_move_qk_2026-05-28.txt
cargo_fmt_check_result=passed
cargo_clippy_log=test_results/native_container_logs/cargo_clippy_resolve_move_qk_2026-05-28.txt
cargo_clippy_result=passed
cargo_test_workspace_log=test_results/native_container_logs/cargo_test_resolve_move_qk_2026-05-28.txt
cargo_test_workspace_result=passed_48_native_tests
pytest_log=test_results/resolve_move_qk_pytest_2026-05-28.txt
pytest_result=passed_146_tests
pytest_werror_log=test_results/resolve_move_qk_pytest_werror_2026-05-28.txt
pytest_werror_result=passed_146_tests
```
