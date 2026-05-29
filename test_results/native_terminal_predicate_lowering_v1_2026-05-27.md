# Native Terminal Predicate Lowering v1 - 2026-05-27

## Scope

- change_id=native_terminal_predicate_lowering_v1
- user_correction=full_frozen_attention_only; tensor_layer_substrate=false
- goal=continue replacing remaining C++ chess-rule control flow with executable frozen attention layers only
- implemented=frozen terminal status layer
- c_api=cmz_engine_frozen_terminal_status
- rust_api=Engine::frozen_terminal_status
- lowered_layers=terminal_predicates=terminal_status_attention
- contract=attention_only_rule_substrate=true; tensor_layer_substrate=false
- tested_cases=ongoing, threefold repetition, fifty-move rule, insufficient material, checkmate, stalemate
- counter_contract=frozen_terminal_status increments cmz_engine_frozen_layer_step_count
- invalid_contract=bad FEN fails loudly

## Remaining Boundary

- full_rule_lowering_complete=false
- cpp_control_flow_rule_vm_remaining=true
- remaining_rule_control_flow=Move record expansion, promotion expansion, trace packet emission loops
- python_hot_path=false for native production contract
- fallback_allowed=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_terminal_predicates_expected_fail_2026-05-27.txt
- expected_fail_reason=cmz_engine_frozen_terminal_status symbol absent before implementation

## Verification

- command=`cd /work/native && cargo fmt --all -- --check`
- result=passed
- log=test_results/native_container_logs/cargo_fmt_terminal_predicates_final_2026-05-27.txt

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- result=passed
- log=test_results/native_container_logs/cargo_clippy_terminal_predicates_2026-05-27.txt

- command=`cd /work/native && cargo test --workspace`
- result=passed
- native_test_count=35
- counted_runtime_tests=34_engine_sys_plus_1_dashboard
- log=test_results/native_container_logs/cargo_test_terminal_predicates_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider tests\test_trace_packet.py tests\test_move_packet.py -q`
- result=passed
- python_test_count=5
- log=test_results/terminal_predicates_packet_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -q`
- result=passed
- python_test_count=146
- log=test_results/terminal_predicates_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -W error -q`
- result=passed
- python_test_count=146
- log=test_results/terminal_predicates_pytest_werror_2026-05-27.txt

## Files Changed

- native/cpp/include/cmz_engine.h
- native/cpp/src/cmz_engine.cpp
- native/crates/cmz-engine-sys/src/lib.rs
- docs/native_rust_cpp_cuda_architecture.md
- docs/project_memory.md
- docs/change_history.md
- docs/prompt_history.md
