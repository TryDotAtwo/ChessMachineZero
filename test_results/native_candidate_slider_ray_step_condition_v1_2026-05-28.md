# Native Candidate Slider Ray-Step Condition V1

## Scope

- change_id=native_candidate_slider_ray_step_condition_v1
- target_gap=candidate_slider_ray_step_condition_control_flow
- replacement=slider_coordinate_QK_table + prior_step_enabled_QK_table + square_occupied_QK_table + slot_nonzero_QK_table + final_QK_write
- contract_truth=full_frozen_attention_only=false; semantic_attention_purity=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_condition_expected_fail_2026-05-28.txt
- expected_fail_result=failed before implementation because `cmz_candidate_slider_ray_step_target_slot_attention_value` was absent

## Implementation

- updated=native/cpp/src/cmz_cuda_kernels.cu
- updated=native/cpp/src/cmz_engine.cpp
- updated=native/crates/cmz-engine-sys/src/lib.rs
- summary=`cmz_candidate_slider_ray_step_attention_value` now delegates target-square lookup, prior blocker detection, target-exists detection, own-piece rejection, and final write to QK helpers; the helper body no longer contains `cmz_on_board`, `for`, `blocked_before_step`, direct own-occupancy masking, or prior-step loops
- regression_fix=first package run produced extra start-position slider moves because the single-offset 12x12 coordinate table was reused outside its valid input range; `cmz_candidate_slider_coordinate_table_attention_value` now uses a 22x22 QK table for slider ray coordinates -7..14 and returns zero for offboard coordinates

## Verification

- targeted_log=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_condition_targeted_2026-05-28.txt; status=passed
- source_audit_log=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_condition_source_audit_2026-05-28.txt; status=passed
- first_package_log=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_condition_package_2026-05-28.txt; status=failed_expected_regression; failure=extra_offboard_slider_moves
- targeted_retry_log=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_condition_targeted_retry2_2026-05-28.txt; status=passed
- package_retry_log=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_condition_package_retry_2026-05-28.txt; status=passed; tests=66
- cargo_fmt_apply_log=test_results/native_container_logs/cargo_fmt_apply_candidate_slider_ray_step_condition_2026-05-28.txt; status=passed
- cargo_fmt_log=test_results/native_container_logs/cargo_fmt_candidate_slider_ray_step_condition_2026-05-28.txt; status=passed
- cargo_clippy_log=test_results/native_container_logs/cargo_clippy_candidate_slider_ray_step_condition_2026-05-28.txt; status=passed
- cargo_workspace_test_log=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_condition_2026-05-28.txt; status=passed; tests=66
- pytest_log=test_results/candidate_slider_ray_step_condition_pytest_2026-05-28.txt; status=passed
- pytest_werror_log=test_results/candidate_slider_ray_step_condition_pytest_werror_2026-05-28.txt; status=passed
- full_verification_status=passed

## Remaining Gaps

- candidate_record_prefix_rank_control_flow
- terminal_legal_presence_chess_search
- terminal_material_counting_control_flow
- terminal_check_state_king_scan
- castle_target_chess_control_flow
- legal_filter_batch_attack_chess_control_flow
- legal_filter_batch_ray_scan_control_flow
