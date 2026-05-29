# native_candidate_record_prefix_rank_v1_2026-05-28

- slice=native_candidate_record_prefix_rank_v1
- target_gap=candidate_record_prefix_rank_control_flow
- status=implemented
- code_scope=native CUDA candidate-record compaction now uses `cmz_candidate_record_prefix_rank_attention_value` and `cmz_candidate_record_total_count_attention_value`; `cmz_candidate_record_slot_rank_write_attention_kernel` no longer contains the per-slot `for (uint32_t prior_slot = 0...)` scan and no longer uses `atomicMax`
- contract_scope=`candidate_record_compaction_backend=qk_prefix_rank_slot_write_attention`; `candidate_moves_layers=context_select,piece_dispatch,target_mask_select,castle_merge,promotion_expand,record_emit,prefix_rank_select,record_order_select`; `candidate_record_prefix_rank_control_flow` removed from `strict_qk_layer_split_remaining` and `remaining_non_attention_paths`
- contract_honesty=`full_frozen_attention_only=false`; `semantic_attention_purity=false`; remaining gaps are `terminal_legal_presence_chess_search`, `terminal_material_counting_control_flow`, `terminal_check_state_king_scan`, `castle_target_chess_control_flow`, `legal_filter_batch_attack_chess_control_flow`, `legal_filter_batch_ray_scan_control_flow`

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_candidate_record_prefix_rank_expected_fail_2026-05-28.txt
- expected_fail_result=failed before implementation because `cmz_candidate_record_prefix_rank_attention_value` was absent
- targeted_pass_log=test_results/native_container_logs/cargo_test_candidate_record_prefix_rank_targeted_2026-05-28.txt
- package_pass_log=test_results/native_container_logs/cargo_test_candidate_record_prefix_rank_package_2026-05-28.txt
- invalid_command_log=test_results/native_container_logs/cargo_test_candidate_record_prefix_rank_source_audit_2026-05-28.txt
- invalid_command_reason=manual cargo invocation passed multiple test filters; package test superseded the invalid command

## Required Verification

- `cd /work/native && cargo fmt --all -- --check` => passed after `cargo fmt --all`
- fmt_apply_log=test_results/native_container_logs/cargo_fmt_apply_candidate_record_prefix_rank_2026-05-28.txt
- fmt_check_log=test_results/native_container_logs/cargo_fmt_candidate_record_prefix_rank_2026-05-28.txt
- `cd /work/native && cargo clippy --workspace --all-targets -- -D warnings` => passed
- clippy_log=test_results/native_container_logs/cargo_clippy_candidate_record_prefix_rank_2026-05-28.txt
- `cd /work/native && cargo test --workspace` => passed with 67 `cmz-engine-sys` tests plus Rust dashboard test
- cargo_workspace_log=test_results/native_container_logs/cargo_test_candidate_record_prefix_rank_2026-05-28.txt
- `python -m pytest -p no:cacheprovider -q` => passed on Windows host Python 3.11 with 112 tests
- pytest_log=test_results/candidate_record_prefix_rank_pytest_2026-05-28.txt
- `python -m pytest -p no:cacheprovider -W error -q` => passed on Windows host Python 3.11 with 112 tests
- pytest_werror_log=test_results/candidate_record_prefix_rank_pytest_werror_2026-05-28.txt
- docker_python_note=container lacks `python` binary and `python3` lacks `python-chess`; Docker-side Python pytest failed at collection for missing oracle dependency, while native Rust/CUDA tests passed inside Docker and host required Python checks passed
- docker_python_missing_log=test_results/native_container_logs/candidate_record_prefix_rank_pytest_2026-05-28.txt
- docker_python3_missing_chess_log=test_results/native_container_logs/candidate_record_prefix_rank_pytest_python3_2026-05-28.txt
