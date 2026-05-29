# Native Candidate Pawn Explicit Slots V1

## Scope

- slice_id=native_candidate_pawn_explicit_slots_v1
- target_gap=candidate_pawn_target_mask_control_flow
- implementation=single_push_slot + double_push_slot + left_capture_slot + right_capture_slot
- contract_update=candidate_pawn_targets_backend=qk_explicit_pawn_slot_writes
- removed_gap=candidate_pawn_target_mask_control_flow
- new_gap=candidate_pawn_slot_condition_control_flow
- full_frozen_attention_only=false
- semantic_attention_purity=false

## TDD

- expected_fail_log=test_results/native_container_logs/cargo_test_candidate_pawn_explicit_slots_expected_fail_2026-05-28.txt
- expected_fail_reason=pawn target body lacked explicit pawn slot helpers before implementation
- targeted_pass_log=test_results/native_container_logs/cargo_test_candidate_pawn_explicit_slots_targeted_2026-05-28.txt
- package_pass_log=test_results/native_container_logs/cargo_test_candidate_pawn_explicit_slots_package_2026-05-28.txt

## Verification

- cargo_fmt=passed; log=test_results/native_container_logs/cargo_fmt_candidate_pawn_explicit_slots_2026-05-28.txt
- cargo_fmt_apply=passed; log=test_results/native_container_logs/cargo_fmt_apply_candidate_pawn_explicit_slots_2026-05-28.txt
- cargo_clippy=passed; log=test_results/native_container_logs/cargo_clippy_candidate_pawn_explicit_slots_2026-05-28.txt
- cargo_test_workspace=passed; log=test_results/native_container_logs/cargo_test_candidate_pawn_explicit_slots_2026-05-28.txt
- pytest=passed; log=test_results/candidate_pawn_explicit_slots_pytest_2026-05-28.txt
- pytest_werror=passed; log=test_results/candidate_pawn_explicit_slots_pytest_werror_2026-05-28.txt
