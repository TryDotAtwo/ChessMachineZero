# native_terminal_legal_presence_v1_2026-05-28

- slice=native_terminal_legal_presence_v1
- target_gap=terminal_legal_presence_chess_search
- status=implemented_as_gap_split
- code_scope=`cmz_terminal_legal_presence_qk_hardmax_select_value` no longer calls `cmz_candidate_move_is_legal` and no longer uses early `return true` / `return false` search; legal-presence accumulation now goes through `cmz_terminal_legal_presence_accumulate_attention_value`
- contract_scope=removed_gap=terminal_legal_presence_chess_search; new_child_gap=terminal_legal_presence_candidate_legal_control_flow
- contract_honesty=`full_frozen_attention_only=false`; `semantic_attention_purity=false`; remaining gaps are `terminal_legal_presence_candidate_legal_control_flow`, `terminal_material_counting_control_flow`, `terminal_check_state_king_scan`, `castle_target_chess_control_flow`, `legal_filter_batch_attack_chess_control_flow`, `legal_filter_batch_ray_scan_control_flow`

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_terminal_legal_presence_expected_fail_2026-05-28.txt
- expected_fail_result=failed before implementation because `cmz_terminal_candidate_move_legal_attention_value` was absent
- targeted_pass_log=test_results/native_container_logs/cargo_test_terminal_legal_presence_targeted_2026-05-28.txt
- package_pass_log=test_results/native_container_logs/cargo_test_terminal_legal_presence_package_2026-05-28.txt

## Required Verification

- `cd /work/native && cargo fmt --all -- --check` => passed
- fmt_check_log=test_results/native_container_logs/cargo_fmt_terminal_legal_presence_2026-05-28.txt
- `cd /work/native && cargo clippy --workspace --all-targets -- -D warnings` => passed
- clippy_log=test_results/native_container_logs/cargo_clippy_terminal_legal_presence_2026-05-28.txt
- `cd /work/native && cargo test --workspace` => passed with 68 `cmz-engine-sys` tests plus Rust dashboard test
- cargo_workspace_log=test_results/native_container_logs/cargo_test_terminal_legal_presence_2026-05-28.txt
- `python -m pytest -p no:cacheprovider -q` => passed on Windows host Python 3.11 with 112 tests
- pytest_log=test_results/terminal_legal_presence_pytest_2026-05-28.txt
- `python -m pytest -p no:cacheprovider -W error -q` => passed on Windows host Python 3.11 with 112 tests
- pytest_werror_log=test_results/terminal_legal_presence_pytest_werror_2026-05-28.txt
