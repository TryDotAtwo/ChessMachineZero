# Native Make-Move Trace Emit Lowering v1 - 2026-05-27

## Scope

- change_id=native_make_move_trace_emit_lowering_v1
- user_target=full_frozen_attention_only
- goal=emit make-move trace packets through frozen trace-packet attention
- c_api=cmz_engine_frozen_make_move_trace_attention_packets
- rust_api=Engine::frozen_make_move_trace_attention_packets
- graph_layer=make_move_trace_emission=trace_packet_attention
- emitted_packets=COMMIT_MOVE, WRITE_SQ, WRITE_REG, WRITE_CASTLE, WRITE_EP, WRITE_CLOCK, TERMINAL_SET, PROGRAM_HALT
- equivalence=frozen_make_move_trace_attention_packets equals native make_move_trace_packets for start position e2e4
- counter_contract=frozen make-move trace attention increments cmz_engine_frozen_layer_step_count by at least emitted packet count
- fallback_allowed=false
- python_hot_path=false

## Verification

- tdd_red=test_results/native_container_logs/cargo_test_make_trace_emit_expected_fail_2026-05-27.txt
- tdd_red_result=expected failure on missing cmz_engine_frozen_make_move_trace_attention_packets symbol
- cargo_fmt=test_results/native_container_logs/cargo_fmt_make_trace_emit_2026-05-27.txt
- cargo_fmt_result=passed
- cargo_clippy=test_results/native_container_logs/cargo_clippy_make_trace_emit_2026-05-27.txt
- cargo_clippy_result=passed
- cargo_test_targeted=test_results/native_container_logs/cargo_test_make_trace_emit_2026-05-27.txt
- cargo_test_targeted_result=passed; tests=2
- cargo_test_workspace=test_results/native_container_logs/cargo_test_make_trace_emit_workspace_2026-05-27.txt
- cargo_test_workspace_result=passed; cmz-engine-sys=38 passed; cmz-dashboard=1 passed; native_total=39 passed
- pytest_full=test_results/make_trace_emit_pytest_2026-05-27.txt
- pytest_full_result=passed; tests=146
- pytest_werror=test_results/make_trace_emit_pytest_werror_2026-05-27.txt
- pytest_werror_result=passed; tests=146

## Files

- native/cpp/include/cmz_engine.h
- native/cpp/src/cmz_engine.cpp
- native/crates/cmz-engine-sys/src/lib.rs
- docs/native_rust_cpp_cuda_architecture.md
- docs/project_memory.md
- docs/change_history.md
- docs/prompt_history.md
