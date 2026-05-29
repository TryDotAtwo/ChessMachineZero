# Native Terminal Legal-Presence Child Closure V1 - 2026-05-28

- status=passed
- scope=removed `terminal_legal_presence_candidate_legal_control_flow`
- implementation=`cmz_cuda_terminal_status_attention` routes legal-presence through `cmz_cuda_candidate_moves_attention` + `cmz_cuda_legal_filter_v2_batch_attention` + `cmz_terminal_legal_presence_from_batch_attention_kernel`
- removed_symbols=`cmz_terminal_candidate_move_legal_attention_value`, `cmz_terminal_legal_presence_accumulate_attention_value`, `cmz_terminal_legal_presence_qk_hardmax_select_value`
- remaining_non_attention_paths=terminal_material_counting_control_flow,terminal_check_state_king_scan,castle_target_chess_control_flow,legal_filter_batch_attack_chess_control_flow,legal_filter_batch_ray_scan_control_flow
- full_frozen_attention_only=false
- semantic_attention_purity=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_terminal_legal_presence_child_expected_fail_2026-05-28.txt
- expected_fail_reason=source audit detected remaining `cmz_terminal_candidate_move_legal_attention_value`
- targeted_pass_log=test_results/native_container_logs/cargo_test_terminal_legal_presence_child_targeted_2026-05-28.txt
- package_pass_log=test_results/native_container_logs/cargo_test_terminal_legal_presence_child_package_2026-05-28.txt

## Verification

- `cd /work/native && cargo fmt --all -- --check` => passed after `cargo fmt --all`
- fmt_fail_log=test_results/native_container_logs/cargo_fmt_terminal_legal_presence_child_2026-05-28.txt
- fmt_apply_log=test_results/native_container_logs/cargo_fmt_apply_terminal_legal_presence_child_2026-05-28.txt
- fmt_pass_log=test_results/native_container_logs/cargo_fmt_check_terminal_legal_presence_child_2026-05-28.txt
- `cd /work/native && cargo clippy --workspace --all-targets -- -D warnings` => passed
- clippy_log=test_results/native_container_logs/cargo_clippy_terminal_legal_presence_child_final_2026-05-28.txt
- `cd /work/native && cargo test --workspace` => passed
- workspace_test_log=test_results/native_container_logs/cargo_test_workspace_terminal_legal_presence_child_final_2026-05-28.txt
- `python -m pytest -p no:cacheprovider -q` => passed, 112 tests
- pytest_log=test_results/terminal_legal_presence_child_pytest_2026-05-28.txt
- `python -m pytest -p no:cacheprovider -W error -q` => passed, 112 tests
- pytest_werror_log=test_results/terminal_legal_presence_child_pytest_werror_2026-05-28.txt
