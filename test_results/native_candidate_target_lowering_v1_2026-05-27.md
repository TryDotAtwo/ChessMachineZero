# Native Candidate Target Lowering v1 - 2026-05-27

## Scope

- change_id=native_candidate_target_lowering_v1
- goal=continue replacing C++ chess-rule control flow with executable frozen attention/table layers inside the native VM
- implemented=piece candidate target mask layer
- c_api=cmz_engine_frozen_candidate_target_mask
- rust_api=Engine::frozen_candidate_target_mask
- lowered_layers=candidate_targets=target_mask_attention
- routed_runtime=pawn, knight, bishop, rook, queen, and king pseudo-legal target selection now uses frozen_candidate_target_mask_layer
- retained_cpp_append_details=piece iteration, Move struct expansion, promotion multiplicity, en-passant flag construction, castling append path
- tested_geometry=knight friendly-filter/enemy-keep; rook blocker-inclusive target mask; pawn push, double-push, capture, blocker, and en-passant target masks
- counter_contract=frozen_candidate_target_mask increments cmz_engine_frozen_layer_step_count
- invalid_contract=invalid piece token fails loudly

## Remaining Boundary

- full_rule_lowering_complete=false
- cpp_control_flow_rule_vm_remaining=true
- remaining_rule_control_flow=Move record expansion, castling candidate construction, legal_filter, make_move, terminal_predicates
- python_hot_path=false for native production contract
- fallback_allowed=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_candidate_targets_expected_fail_2026-05-27.txt
- expected_fail_reason=cmz_engine_frozen_candidate_target_mask symbol absent before implementation

## Verification

- command=`cd /work/native && cargo fmt --all -- --check`
- result=passed
- log=test_results/native_container_logs/cargo_fmt_candidate_targets_final_2026-05-27.txt

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- result=passed
- log=test_results/native_container_logs/cargo_clippy_candidate_targets_2026-05-27.txt

- command=`cd /work/native && cargo test --workspace`
- result=passed
- native_test_count=31
- counted_runtime_tests=30_engine_sys_plus_1_dashboard
- log=test_results/native_container_logs/cargo_test_candidate_targets_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider tests\test_trace_packet.py tests\test_move_packet.py -q`
- result=passed
- python_test_count=5
- log=test_results/candidate_targets_packet_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -q`
- result=passed
- python_test_count=146
- log=test_results/candidate_targets_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -W error -q`
- result=passed
- python_test_count=146
- log=test_results/candidate_targets_pytest_werror_2026-05-27.txt

## Files Changed

- native/cpp/include/cmz_engine.h
- native/cpp/src/cmz_engine.cpp
- native/crates/cmz-engine-sys/src/lib.rs
- docs/native_rust_cpp_cuda_architecture.md
- docs/project_memory.md
- docs/change_history.md
- docs/prompt_history.md
