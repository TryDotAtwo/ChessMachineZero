# Native Candidate Single-Offset Coordinate Table Purity V1

## Scope

- change_id=native_candidate_single_offset_coordinate_table_purity_v1
- target_gap=candidate_single_offset_coordinate_table_control_flow
- replacement=explicit_12x12_shifted_coordinate_QK_table
- contract_truth=full_frozen_attention_only=false; semantic_attention_purity=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_candidate_single_offset_coordinate_table_purity_expected_fail_2026-05-28.txt
- expected_fail_result=failed before implementation because `#define CMZ_COORDINATE_TABLE_ATTEND_ENTRY` was absent

## Implementation

- updated=native/cpp/src/cmz_cuda_kernels.cu
- updated=native/cpp/src/cmz_engine.cpp
- updated=native/crates/cmz-engine-sys/src/lib.rs
- summary=`cmz_candidate_single_offset_coordinate_table_attention_value` now uses explicit shifted-coordinate table entries selected by QK hardmax; offboard entries select zero value; helper body has no `cmz_on_board`, no clamp logic, no `if`, and no `for`

## Verification

- targeted_log=test_results/native_container_logs/cargo_test_candidate_single_offset_coordinate_table_purity_targeted_2026-05-28.txt; status=passed
- source_audit_log=test_results/native_container_logs/cargo_test_candidate_single_offset_coordinate_table_purity_source_audit_2026-05-28.txt; status=passed
- package_log=test_results/native_container_logs/cargo_test_candidate_single_offset_coordinate_table_purity_package_2026-05-28.txt; status=passed; tests=61
- cargo_fmt_log=test_results/native_container_logs/cargo_fmt_candidate_single_offset_coordinate_table_purity_2026-05-28.txt; status=passed
- cargo_clippy_log=test_results/native_container_logs/cargo_clippy_candidate_single_offset_coordinate_table_purity_2026-05-28.txt; status=passed
- cargo_test_workspace_log=test_results/native_container_logs/cargo_test_candidate_single_offset_coordinate_table_purity_2026-05-28.txt; status=passed
- pytest_log=test_results/candidate_single_offset_coordinate_table_purity_pytest_2026-05-28.txt; status=passed
- pytest_werror_log=test_results/candidate_single_offset_coordinate_table_purity_pytest_werror_2026-05-28.txt; status=passed
- full_verification_status=passed

## Remaining Gaps

- candidate_pawn_slot_condition_control_flow
- candidate_slider_ray_step_condition_control_flow
- candidate_record_prefix_rank_control_flow
- terminal_legal_presence_chess_search
- terminal_material_counting_control_flow
- terminal_check_state_king_scan
- castle_target_chess_control_flow
- legal_filter_batch_attack_chess_control_flow
- legal_filter_batch_ray_scan_control_flow
