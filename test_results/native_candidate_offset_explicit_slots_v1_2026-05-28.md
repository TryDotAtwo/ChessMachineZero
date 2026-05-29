# Native Candidate Offset Explicit Slots V1 - 2026-05-28

```text
entity_id=native_candidate_offset_explicit_slots_v1
status=implemented_verified
target_full_frozen_attention_only=true
full_frozen_attention_only=false
semantic_attention_purity=false
candidate_offset_targets_backend=qk_explicit_offset_slot_writes
```

## Scope

```text
removed_gap=candidate_offset_target_mask_control_flow
new_child_gap=candidate_single_offset_bounds_control_flow
top_level_change=cmz_candidate_offset_target_mask_attention_value no longer calls cmz_add_offset_targets and no longer uses serial for-loop offset expansion
slot_write_semantics=8 explicit offset slots; each slot writes through cmz_qk2_select_or_write_u64
contract_truth=single-offset bounds/friendly-mask logic still contains source-level control flow and remains declared
```

## TDD Evidence

```text
expected_fail_log=test_results/native_container_logs/cargo_test_candidate_offset_explicit_slots_expected_fail_2026-05-28.txt
expected_fail_reason=offset target body did not contain cmz_qk2_select_or_write_u64 before implementation
targeted_pass_log=test_results/native_container_logs/cargo_test_candidate_offset_explicit_slots_targeted_2026-05-28.txt
package_pass_log=test_results/native_container_logs/cargo_test_candidate_offset_explicit_slots_package_2026-05-28.txt
package_pass_result=cmz-engine-sys package tests passed; 52 passed
```

## Final Verification

```text
cargo_fmt_apply=test_results/native_container_logs/cargo_fmt_apply_candidate_offset_explicit_slots_2026-05-28.txt; status=passed
cargo_fmt_check=test_results/native_container_logs/cargo_fmt_candidate_offset_explicit_slots_2026-05-28.txt; status=passed
cargo_clippy=test_results/native_container_logs/cargo_clippy_candidate_offset_explicit_slots_2026-05-28.txt; status=passed
cargo_test_workspace=test_results/native_container_logs/cargo_test_candidate_offset_explicit_slots_2026-05-28.txt; status=passed; native_tests=53
pytest=test_results/candidate_offset_explicit_slots_pytest_2026-05-28.txt; status=passed
pytest_werror=test_results/candidate_offset_explicit_slots_pytest_werror_2026-05-28.txt; status=passed
```
