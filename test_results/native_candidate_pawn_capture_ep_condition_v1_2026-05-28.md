# Native Candidate Pawn En-Passant Condition V1

## Scope

- change_id=native_candidate_pawn_capture_ep_condition_v1
- target_gap=candidate_pawn_capture_ep_condition_control_flow
- replacement=ep_target_match_QK_table + side_ep_captured_slot_QK_table + captured_enemy_QK_table + final_QK_write
- contract_truth=full_frozen_attention_only=false; semantic_attention_purity=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_candidate_pawn_capture_ep_condition_expected_fail_2026-05-28.txt
- expected_fail_result=failed before implementation because `cmz_candidate_pawn_ep_target_match_attention_value` was absent

## Implementation

- updated=native/cpp/src/cmz_cuda_kernels.cu
- updated=native/cpp/src/cmz_engine.cpp
- updated=native/crates/cmz-engine-sys/src/lib.rs
- summary=`cmz_candidate_pawn_capture_ep_condition_attention_value` now composes QK helper layers and no longer owns ep-square range logic, target-slot construction, captured-square arithmetic, captured-valid logic, or direct captured-slot enemy masking

## Verification

- targeted_log=test_results/native_container_logs/cargo_test_candidate_pawn_capture_ep_condition_targeted_2026-05-28.txt; status=passed
- source_audit_log=test_results/native_container_logs/cargo_test_candidate_pawn_capture_ep_condition_source_audit_2026-05-28.txt; status=passed
- package_log=test_results/native_container_logs/cargo_test_candidate_pawn_capture_ep_condition_package_2026-05-28.txt; status=passed; tests=65
- cargo_fmt_log=test_results/native_container_logs/cargo_fmt_candidate_pawn_capture_ep_condition_2026-05-28.txt; status=passed
- cargo_clippy_log=test_results/native_container_logs/cargo_clippy_candidate_pawn_capture_ep_condition_2026-05-28.txt; status=passed
- cargo_workspace_test_log=test_results/native_container_logs/cargo_test_candidate_pawn_capture_ep_condition_2026-05-28.txt; status=passed; tests=65
- pytest_log=test_results/candidate_pawn_capture_ep_condition_pytest_2026-05-28.txt; status=passed
- pytest_werror_log=test_results/candidate_pawn_capture_ep_condition_pytest_werror_2026-05-28.txt; status=passed
- full_verification_status=passed

## Remaining Gaps

- candidate_slider_ray_step_condition_control_flow
- candidate_record_prefix_rank_control_flow
- terminal_legal_presence_chess_search
- terminal_material_counting_control_flow
- terminal_check_state_king_scan
- castle_target_chess_control_flow
- legal_filter_batch_attack_chess_control_flow
- legal_filter_batch_ray_scan_control_flow
