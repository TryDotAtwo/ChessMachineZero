# Native Ray Scan Lowering v1 - 2026-05-27

## Scope

- change_id=native_ray_scan_lowering_v1
- goal=continue replacing C++ chess-rule control flow with executable frozen attention/table layers inside the native VM
- implemented=blocker-aware ray scan mask layer
- c_api=cmz_engine_frozen_ray_scan_mask
- rust_api=Engine::frozen_ray_scan_mask
- lowered_layers=ray_scan=blocker_aware_ray_scan_attention
- lowered_attack_path=pawn_knight_king_slider_ray_scan
- routed_runtime=slider attack detection now uses frozen_ray_scan_mask_layer inside native is_attacked
- tested_geometry=east_ray, northeast_ray, blocker_inclusive_east_ray, blocker_inclusive_west_ray
- counter_contract=frozen_ray_scan_mask increments cmz_engine_frozen_layer_step_count
- invalid_contract=zero direction and non-unit direction fail loudly

## Remaining Boundary

- full_rule_lowering_complete=false
- cpp_control_flow_rule_vm_remaining=true
- remaining_rule_control_flow=full move candidate generation, legal_filter, make_move, terminal_predicates
- python_hot_path=false for native production contract
- fallback_allowed=false

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_ray_scan_expected_fail_2026-05-27.txt
- expected_fail_reason=cmz_engine_frozen_ray_scan_mask symbol absent before implementation

## Verification

- command=`cd /work/native && cargo fmt --all -- --check`
- result=passed
- log=test_results/native_container_logs/cargo_fmt_ray_scan_2026-05-27.txt

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- result=passed
- log=test_results/native_container_logs/cargo_clippy_ray_scan_2026-05-27.txt

- command=`cd /work/native && cargo test --workspace`
- result=passed
- native_test_count=28
- counted_runtime_tests=27_engine_sys_plus_1_dashboard
- log=test_results/native_container_logs/cargo_test_ray_scan_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider tests\test_trace_packet.py tests\test_move_packet.py -q`
- result=passed
- python_test_count=5
- log=test_results/ray_scan_packet_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -q`
- result=passed
- python_test_count=146
- log=test_results/ray_scan_pytest_2026-05-27.txt

- command=`python -m pytest -p no:cacheprovider -W error -q`
- result=passed
- python_test_count=146
- log=test_results/ray_scan_pytest_werror_2026-05-27.txt

## Files Changed

- native/cpp/include/cmz_engine.h
- native/cpp/src/cmz_engine.cpp
- native/crates/cmz-engine-sys/src/lib.rs
- docs/native_rust_cpp_cuda_architecture.md
- docs/project_memory.md
- docs/change_history.md
- docs/prompt_history.md
