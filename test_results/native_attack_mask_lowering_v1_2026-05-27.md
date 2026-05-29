# Native Attack Mask Lowering v1 - 2026-05-27

## Scope

- change_id=native_attack_mask_lowering_v1
- goal=continue replacing C++ chess-rule control flow with executable frozen attention/table layers inside the native VM
- implemented=static piece attack-mask table layer
- c_api=cmz_engine_frozen_attack_mask
- rust_api=Engine::frozen_attack_mask
- lowered_layers=piece_dispatch=frozen_table_attention; attack_masks=static_attack_mask_table_attention
- lowered_attack_path=pawn_knight_king
- tested_static_geometry=white_pawn, black_pawn, knight, bishop_empty_board_rays, rook_empty_board_rays, king
- counter_contract=frozen_attack_mask increments cmz_engine_frozen_layer_step_count
- invalid_contract=invalid piece token fails loudly

## Remaining Boundary

- full_rule_lowering_complete=false
- cpp_control_flow_rule_vm_remaining=true
- remaining_rule_control_flow=slider blocker-aware ray_scan, legal_filter, make_move, terminal_predicates, full move candidate generation
- python_hot_path=false for native production contract
- fallback_allowed=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_attack_mask_expected_fail_2026-05-27.txt
- expected_fail_reason=cmz_engine_frozen_attack_mask symbol absent before implementation

## Verification

- command=`cd /work/native && cargo fmt --all -- --check`
- result=passed
- log=test_results/native_container_logs/cargo_fmt_attack_mask_2026-05-27.txt

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- result=passed
- log=test_results/native_container_logs/cargo_clippy_attack_mask_final_2026-05-27.txt

- command=`cd /work/native && cargo test --workspace`
- result=passed
- native_test_count=26
- log=test_results/native_container_logs/cargo_test_attack_mask_final_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider tests\test_trace_packet.py tests\test_move_packet.py -q`
- result=passed
- python_test_count=5
- log=test_results/attack_mask_packet_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -q`
- result=passed
- python_test_count=146
- log=test_results/attack_mask_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -W error -q`
- result=passed
- python_test_count=146
- log=test_results/attack_mask_pytest_werror_2026-05-27.txt

## Files Changed

- native/cpp/include/cmz_engine.h
- native/cpp/src/cmz_engine.cpp
- native/crates/cmz-engine-sys/src/lib.rs
- docs/native_rust_cpp_cuda_architecture.md
- docs/project_memory.md
- docs/change_history.md
- docs/prompt_history.md
