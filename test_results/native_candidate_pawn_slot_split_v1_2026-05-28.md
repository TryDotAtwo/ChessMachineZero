# Native Candidate Pawn Slot Split V1

## Scope

- change_id=native_candidate_pawn_slot_split_v1
- target_gap=candidate_pawn_slot_condition_control_flow
- replacement_gaps=candidate_pawn_push_condition_control_flow,candidate_pawn_double_push_condition_control_flow,candidate_pawn_capture_ep_condition_control_flow
- contract_truth=full_frozen_attention_only=false; semantic_attention_purity=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_candidate_pawn_slot_split_expected_fail_2026-05-28.txt
- expected_fail_result=failed before implementation because pawn slot helpers did not call target-slot and QK condition helper layers

## Implementation

- updated=native/cpp/src/cmz_cuda_kernels.cu
- updated=native/cpp/src/cmz_engine.cpp
- updated=native/crates/cmz-engine-sys/src/lib.rs
- summary=single-push, double-push, and capture slot helpers now delegate target generation and condition checks to named helpers; broad pawn condition gap is replaced by narrower child condition gaps

## Verification

- targeted_log=test_results/native_container_logs/cargo_test_candidate_pawn_slot_split_targeted_2026-05-28.txt; status=failed_compile_missing_forward_declaration
- targeted_retry_log=test_results/native_container_logs/cargo_test_candidate_pawn_slot_split_targeted_retry_2026-05-28.txt; status=passed
- source_audit_log=test_results/native_container_logs/cargo_test_candidate_pawn_slot_split_source_audit_2026-05-28.txt; status=passed
- package_retry_log=test_results/native_container_logs/cargo_test_candidate_pawn_slot_split_package_retry_2026-05-28.txt; status=passed; tests=62
- cargo_fmt_initial_log=test_results/native_container_logs/cargo_fmt_candidate_pawn_slot_split_2026-05-28.txt; status=failed_before_rustfmt
- cargo_fmt_apply_log=test_results/native_container_logs/cargo_fmt_apply_candidate_pawn_slot_split_2026-05-28.txt; status=passed
- cargo_fmt_log=test_results/native_container_logs/cargo_fmt_candidate_pawn_slot_split_2026-05-28.txt; status=passed_after_rerun
- cargo_clippy_log=test_results/native_container_logs/cargo_clippy_candidate_pawn_slot_split_2026-05-28.txt; status=passed
- cargo_test_workspace_log=test_results/native_container_logs/cargo_test_candidate_pawn_slot_split_2026-05-28.txt; status=passed
- pytest_log=test_results/candidate_pawn_slot_split_pytest_2026-05-28.txt; status=passed
- pytest_werror_log=test_results/candidate_pawn_slot_split_pytest_werror_2026-05-28.txt; status=passed
- full_verification_status=passed

## Remaining Gaps

- candidate_pawn_push_condition_control_flow
- candidate_pawn_double_push_condition_control_flow
- candidate_pawn_capture_ep_condition_control_flow
- candidate_slider_ray_step_condition_control_flow
- candidate_record_prefix_rank_control_flow
- terminal_legal_presence_chess_search
- terminal_material_counting_control_flow
- terminal_check_state_king_scan
- castle_target_chess_control_flow
- legal_filter_batch_attack_chess_control_flow
- legal_filter_batch_ray_scan_control_flow
