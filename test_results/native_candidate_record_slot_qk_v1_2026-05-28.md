# Native Candidate Record Slot QK V1

## Scope

- slice_id=native_candidate_record_slot_qk_v1
- target_gap=candidate_record_emit_serial_loop
- implementation=cmz_candidate_record_slot_qk_write_value + cmz_qk2_select_or_write_u64
- contract_update=candidate_record_emit_backend=qk_candidate_slot_write_attention
- removed_gap=candidate_record_emit_serial_loop
- new_gap=candidate_record_slot_compaction_control_flow
- full_frozen_attention_only=false
- semantic_attention_purity=false

## TDD

- expected_fail_log=test_results/native_container_logs/cargo_test_candidate_record_emit_expected_fail_2026-05-28.txt
- expected_fail_reason=record emit body lacked cmz_candidate_record_slot_qk_write_value before implementation
- targeted_pass_log=test_results/native_container_logs/cargo_test_candidate_record_emit_targeted_2026-05-28.txt
- package_pass_log=test_results/native_container_logs/cargo_test_candidate_record_emit_package_2026-05-28.txt

## Verification

- cargo_fmt=passed; log=test_results/native_container_logs/cargo_fmt_candidate_record_slot_qk_2026-05-28.txt
- cargo_clippy=passed; log=test_results/native_container_logs/cargo_clippy_candidate_record_slot_qk_2026-05-28.txt
- cargo_test_workspace=passed; log=test_results/native_container_logs/cargo_test_candidate_record_slot_qk_2026-05-28.txt
- pytest=passed; log=test_results/candidate_record_slot_qk_pytest_2026-05-28.txt
- pytest_werror=passed; log=test_results/candidate_record_slot_qk_pytest_werror_2026-05-28.txt
