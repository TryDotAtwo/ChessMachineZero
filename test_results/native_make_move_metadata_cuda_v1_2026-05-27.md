# Native Make-Move Metadata CUDA Attention V1 - 2026-05-27

- scope=make-move metadata updates lowered to CUDA frozen self-attention status-write layers
- cuda_symbol=cmz_cuda_make_move_metadata_attention
- c_counter=cmz_engine_cuda_make_move_metadata_attention_count
- layers=side_toggle_select, castling_rights_select, ep_square_select, halfmove_clock_select, fullmove_number_select
- graph_contract=make_move_board_metadata_backend=cuda_make_move_metadata_attention
- full_frozen_attention_only=false

## TDD Log

- expected_fail=test_results/native_container_logs/cargo_test_make_move_metadata_cuda_expected_fail_2026-05-27.txt
- expected_fail_result=failed before implementation because `cmz_engine_cuda_make_move_metadata_attention_count` symbol was missing
- targeted_pass=test_results/native_container_logs/cargo_test_make_move_metadata_cuda_targeted_2026-05-27.txt
- targeted_pass_result=1 targeted native test passed

## Verification

- cargo_fmt=test_results/native_container_logs/cargo_fmt_make_move_metadata_cuda_2026-05-27.txt
- cargo_fmt_result=passed
- cargo_clippy=test_results/native_container_logs/cargo_clippy_make_move_metadata_cuda_2026-05-27.txt
- cargo_clippy_result=passed with `-D warnings`
- cargo_test=test_results/native_container_logs/cargo_test_make_move_metadata_cuda_2026-05-27.txt
- cargo_test_result=47 native tests passed
- pytest=test_results/make_move_metadata_cuda_pytest_2026-05-27.txt
- pytest_result=passed; 146 tests reached 100%
- pytest_werror=test_results/make_move_metadata_cuda_pytest_werror_2026-05-27.txt
- pytest_werror_result=passed; 146 tests reached 100%

## Remaining Work

- remaining=pseudo_legal_moves/sorted_pseudo_legal_moves/resolve_legal_move C++ control-flow loops
- remaining=frozen_terminal_predicate_layer C++ terminal logic
- remaining=trace packet append loops in C++
