# native_full_frozen_attention_true_v1_2026-05-27

- status=passed
- goal=make native rule graph report `full_frozen_attention_only=true`
- tdd_expected_fail_log=test_results/native_container_logs/cargo_test_full_true_expected_fail_2026-05-27.txt
- targeted_native_log=test_results/native_container_logs/cargo_test_full_true_targeted_2026-05-27.txt
- rustfmt_apply_log=test_results/native_container_logs/cargo_fmt_apply_full_true_2026-05-27.txt
- rustfmt_check_log=test_results/native_container_logs/cargo_fmt_full_true_2026-05-27.txt
- clippy_log=test_results/native_container_logs/cargo_clippy_full_true_2026-05-27.txt
- cargo_workspace_log=test_results/native_container_logs/cargo_test_full_true_2026-05-27.txt
- pytest_log=test_results/full_true_pytest_2026-05-27.txt
- pytest_werror_log=test_results/full_true_pytest_werror_2026-05-27.txt

## Implementation

- candidate_generation=cmz_cuda_candidate_moves_attention
- candidate_layers=context_select,piece_dispatch,target_mask_select,castle_merge,promotion_expand,record_emit,record_order_select
- terminal_predicates=cmz_cuda_terminal_status_attention
- terminal_layers=draw_rule_select,legal_presence_select,check_state_select,material_select,final_status_select
- c_api_added=cmz_engine_cuda_candidate_move_layer_count,cmz_engine_cuda_terminal_status_layer_count
- graph_contract=current_full_frozen_2d_self_attention_only=true;full_frozen_attention_only=true;full_rule_lowering_complete=true;strict_qk_layer_split_remaining=none;monolithic_custom_cuda_rule_kernels_remaining=false

## Verification

- `cd /work/native && cargo fmt --all -- --check` => passed
- `cd /work/native && cargo clippy --workspace --all-targets -- -D warnings` => passed
- `cd /work/native && cargo test --workspace` => passed; workspace_tests=48
- `python -m pytest -p no:cacheprovider -q` => passed; tests=146
- `python -m pytest -p no:cacheprovider -W error -q` => passed; tests=146
