# Native Legal Filter Lowering v1 - 2026-05-27

## Scope

- change_id=native_legal_filter_lowering_v1
- goal=continue replacing C++ chess-rule control flow with executable frozen attention/table layers inside the native VM
- implemented=frozen move-legality layer
- c_api=cmz_engine_frozen_move_legal
- rust_api=Engine::frozen_move_legal
- lowered_layers=castling_targets=castle_path_attention; legal_filter=king_safety_attention; make_move=board_write_attention
- routed_runtime=legal_after_king_filter now calls frozen_legal_filter_layer; add_castles now calls frozen_castle_target_mask_layer; make_move board transition uses frozen_make_move_board_layer
- tested_cases=start legal move, illegal non-candidate move, pinned en-passant self-check, castling through attack, legal queenside castling, in-check legal king escape, in-check illegal king move
- counter_contract=frozen_move_legal increments cmz_engine_frozen_layer_step_count
- invalid_contract=bad FEN fails loudly

## Remaining Boundary

- full_rule_lowering_complete=false
- cpp_control_flow_rule_vm_remaining=true
- remaining_rule_control_flow=Move record expansion, promotion expansion, trace packet emission loops, terminal_predicates
- python_hot_path=false for native production contract
- fallback_allowed=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_legal_filter_expected_fail_2026-05-27.txt
- expected_fail_reason=cmz_engine_frozen_move_legal symbol absent before implementation

## Verification

- command=`cd /work/native && cargo fmt --all -- --check`
- result=passed
- log=test_results/native_container_logs/cargo_fmt_legal_filter_final_2026-05-27.txt

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- result=passed
- log=test_results/native_container_logs/cargo_clippy_legal_filter_2026-05-27.txt

- command=`cd /work/native && cargo test --workspace`
- result=passed
- native_test_count=33
- counted_runtime_tests=32_engine_sys_plus_1_dashboard
- log=test_results/native_container_logs/cargo_test_legal_filter_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider tests\test_trace_packet.py tests\test_move_packet.py -q`
- result=passed
- python_test_count=5
- log=test_results/legal_filter_packet_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -q`
- result=passed
- python_test_count=146
- log=test_results/legal_filter_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -W error -q`
- result=passed
- python_test_count=146
- log=test_results/legal_filter_pytest_werror_2026-05-27.txt

## Files Changed

- native/cpp/include/cmz_engine.h
- native/cpp/src/cmz_engine.cpp
- native/crates/cmz-engine-sys/src/lib.rs
- docs/native_rust_cpp_cuda_architecture.md
- docs/project_memory.md
- docs/change_history.md
- docs/prompt_history.md
