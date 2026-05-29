# Native Candidate Target Dispatch QK V1 - 2026-05-28

```text
entity_id=native_candidate_target_dispatch_qk_v1
status=implemented_verified
target_full_frozen_attention_only=true
full_frozen_attention_only=false
semantic_attention_purity=false
candidate_target_dispatch_backend=qk_hardmax_piece_family_select
```

## Scope

```text
removed_gap=candidate_target_mask_chess_control_flow
new_child_gaps=candidate_pawn_target_mask_control_flow,candidate_offset_target_mask_control_flow,candidate_slider_target_mask_control_flow
top_level_change=cmz_candidate_target_mask_qk_hardmax_v_attention_value now performs QK hardmax over piece-family value masks instead of a direct piece branch tree
contract_truth=child value-mask functions still contain chess-specific control flow and remain declared gaps
```

## TDD Evidence

```text
expected_fail_log=test_results/native_container_logs/cargo_test_candidate_target_dispatch_expected_fail_2026-05-28.txt
expected_fail_reason=top-level candidate target mask did not contain cmz_qk2_hardmax_select_u64 before implementation
targeted_pass_log=test_results/native_container_logs/cargo_test_candidate_target_dispatch_targeted_2026-05-28.txt
package_pass_log=test_results/native_container_logs/cargo_test_candidate_target_dispatch_package_2026-05-28.txt
package_pass_result=cmz-engine-sys package tests passed; 51 passed
```

## Final Verification

```text
cargo_fmt_apply=test_results/native_container_logs/cargo_fmt_apply_candidate_target_dispatch_2026-05-28.txt; status=passed
cargo_fmt_check=test_results/native_container_logs/cargo_fmt_candidate_target_dispatch_2026-05-28.txt; status=passed
cargo_clippy=test_results/native_container_logs/cargo_clippy_candidate_target_dispatch_2026-05-28.txt; status=passed
cargo_test_workspace=test_results/native_container_logs/cargo_test_candidate_target_dispatch_2026-05-28.txt; status=passed; native_tests=52
pytest=test_results/candidate_target_dispatch_pytest_2026-05-28.txt; status=passed
pytest_werror=test_results/candidate_target_dispatch_pytest_werror_2026-05-28.txt; status=passed
```
