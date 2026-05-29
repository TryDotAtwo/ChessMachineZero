# Native Candidate Pawn Double Push Condition V1

## Scope

- change_id=native_candidate_pawn_double_push_condition_v1
- target_gap=candidate_pawn_double_push_condition_control_flow
- replacement=start_rank_QK_table + single_push_nonzero_QK_selector + explicit_64_square_target_empty_QK_writes
- contract_truth=full_frozen_attention_only=false; semantic_attention_purity=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_candidate_pawn_double_push_condition_expected_fail_2026-05-28.txt
- expected_fail_result=failed before implementation because `cmz_candidate_pawn_start_rank_match_attention_value` was absent

## Implementation

- updated=native/cpp/src/cmz_cuda_kernels.cu
- updated=native/cpp/src/cmz_engine.cpp
- updated=native/crates/cmz-engine-sys/src/lib.rs
- summary=`cmz_candidate_pawn_double_push_condition_attention_value` now uses condition bits produced by QK helper layers and writes the final target through explicit 64-square QK entries; no direct `from_rank`, `start_rank`, `single_push_mask`, `target_exists`, `target_empty`, or `occupancy_mask & target_slot` logic remains in the double-push condition body

## Verification

- targeted_log=test_results/native_container_logs/cargo_test_candidate_pawn_double_push_condition_targeted_retry_2026-05-28.txt; status=passed
- source_audit_log=test_results/native_container_logs/cargo_test_candidate_pawn_double_push_condition_source_audit_2026-05-28.txt; status=passed
- package_log=test_results/native_container_logs/cargo_test_candidate_pawn_double_push_condition_package_2026-05-28.txt; status=passed; tests=64
- cargo_fmt_log=test_results/native_container_logs/cargo_fmt_candidate_pawn_double_push_condition_2026-05-28.txt; status=passed
- cargo_clippy_log=test_results/native_container_logs/cargo_clippy_candidate_pawn_double_push_condition_2026-05-28.txt; status=passed
- cargo_workspace_test_log=test_results/native_container_logs/cargo_test_candidate_pawn_double_push_condition_2026-05-28.txt; status=passed; tests=64
- pytest_log=test_results/candidate_pawn_double_push_condition_pytest_2026-05-28.txt; status=passed
- pytest_werror_log=test_results/candidate_pawn_double_push_condition_pytest_werror_2026-05-28.txt; status=passed
- full_verification_status=passed

## Remaining Gaps

- candidate_pawn_capture_ep_condition_control_flow
- candidate_slider_ray_step_condition_control_flow
- candidate_record_prefix_rank_control_flow
- terminal_legal_presence_chess_search
- terminal_material_counting_control_flow
- terminal_check_state_king_scan
- castle_target_chess_control_flow
- legal_filter_batch_attack_chess_control_flow
- legal_filter_batch_ray_scan_control_flow
